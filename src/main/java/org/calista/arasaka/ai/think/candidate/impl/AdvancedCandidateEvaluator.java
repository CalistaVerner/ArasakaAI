package org.calista.arasaka.ai.think.candidate.impl;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.TokenOverlapScorer;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Advanced deterministic evaluator:
 * - structure contract (sections) + formatting sanity
 * - groundedness aggregation over context statements
 * - query coverage (does it address user input)
 * - novelty / unsupported-assertion risk (tokens not backed by context)
 * - repetition / low-information penalties
 * - length/verbosity guardrails
 *
 * No randomness, no domain hard-coded phrases. Only general heuristics/policy thresholds.
 *
 * Supports sectioning contracts:
 *  - Legacy: 1) 2) 3)
 *  - Markdown: ## / ### headers (>= 3 headers)
 */
public final class AdvancedCandidateEvaluator implements CandidateEvaluator {

    // --- policy thresholds (tunable) ---
    private final double minGroundedness;
    private final double maxContradictionRisk;

    private final double minQueryCoverage;      // candidate should overlap with user query
    private final double maxNovelty;            // too many tokens not in context => likely hallucination
    private final double maxRepetition;         // excessive repetition => low quality / loop
    private final int minChars;                 // too short => usually useless
    private final int maxCharsSoft;             // soft penalty after this
    private final int maxCharsHard;             // invalid after this (prevents runaway verbosity)

    private final Tokenizer tokenizer;
    private final TokenOverlapScorer overlapScorer;

    // Legacy numeric section markers at line start.
    private static final Pattern SECTION_MARKERS = Pattern.compile("(?m)^(\\s*)([1-3])\\)\\s+");
    private static final Pattern LEGACY_1 = Pattern.compile("(?m)^\\s*1\\)\\s+");
    private static final Pattern LEGACY_2 = Pattern.compile("(?m)^\\s*2\\)\\s+");
    private static final Pattern LEGACY_3 = Pattern.compile("(?m)^\\s*3\\)\\s+");

    // Markdown section markers (line-start headers)
    private static final Pattern MD_HEADER = Pattern.compile("(?m)^\\s*#{2,3}\\s+.+$");

    private static final Pattern BULLETISH = Pattern.compile("(?m)^\\s*([-*•]|\\d+\\.|\\d+\\))\\s+");
    private static final Pattern WORDLIKE = Pattern.compile("[\\p{L}\\p{Nd}_]{2,}");

    public AdvancedCandidateEvaluator(Tokenizer tokenizer) {
        this(
                tokenizer,
                new TokenOverlapScorer(Objects.requireNonNull(tokenizer, "tokenizer")),
                0.30, 0.70,
                0.20,
                0.70,
                0.35,
                40,
                1600,
                3500
        );
    }

    public AdvancedCandidateEvaluator(
            Tokenizer tokenizer,
            TokenOverlapScorer overlapScorer,
            double minGroundedness,
            double maxContradictionRisk,
            double minQueryCoverage,
            double maxNovelty,
            double maxRepetition,
            int minChars,
            int maxCharsSoft,
            int maxCharsHard
    ) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.overlapScorer = Objects.requireNonNull(overlapScorer, "overlapScorer");

        this.minGroundedness = clamp01(minGroundedness);
        this.maxContradictionRisk = clamp01(maxContradictionRisk);
        this.minQueryCoverage = clamp01(minQueryCoverage);
        this.maxNovelty = clamp01(maxNovelty);
        this.maxRepetition = clamp01(maxRepetition);

        this.minChars = Math.max(0, minChars);
        this.maxCharsSoft = Math.max(this.minChars + 1, maxCharsSoft);
        this.maxCharsHard = Math.max(this.maxCharsSoft + 1, maxCharsHard);
    }

    @Override
    public double score(String userText, String candidateText, List<Statement> context) {
        return evaluate(userText, candidateText, context).effectiveScore();
    }

    @Override
    public Evaluation evaluate(String userText, String candidateText, List<Statement> context) {

        final String q = userText == null ? "" : userText.trim();
        final String a = candidateText == null ? "" : candidateText.trim();
        final List<Statement> ctx = context == null ? List.of() : context;

        if (a.isEmpty()) {
            return invalid(-1.0, "err=empty");
        }
        if (a.length() < minChars) {
            return invalid(-0.8, "err=too_short;len=" + a.length());
        }
        if (a.length() > maxCharsHard) {
            return invalid(-0.6, "err=too_long_hard;len=" + a.length());
        }

        // ---- 1) Structure quality ----
        Structure s = structureOf(a);

        // ---- 2) Groundedness to context (max + topK mean) ----
        GroundingAgg g = groundednessAgg(a, ctx);
        double groundedness = g.groundedness;

        // ---- 3) Query coverage (is it addressing user question) ----
        double queryCoverage = queryCoverage(q, a);

        // ---- 4) Novelty vs context (unsupported token mass) ----
        Novelty n = noveltyVsContext(a, ctx);

        // ---- 5) Repetition / low-information signals ----
        Repetition rep = repetition(a);

        // ---- 6) Contradiction / hallucination risk heuristic (general) ----
        double numericDensity = (double) countDigits(a) / (double) Math.max(1, a.length());
        double punctuationDensity = (double) countPunctuation(a) / (double) Math.max(1, a.length());

        double contradictionRisk = clamp01(
                0.15
                        + 0.40 * clamp01(numericDensity / 0.18)          // lots of numbers => risk unless grounded
                        + 0.15 * clamp01(punctuationDensity / 0.22)      // noisy text => risk
                        + 0.45 * clamp01(n.novelty / Math.max(1e-9, maxNovelty)) * 0.5
                        + (s.hasAllSections ? 0.0 : 0.10)
                        - 0.55 * groundedness
        );

        // ---- 7) Style / verbosity soft penalty ----
        double verbosityPenalty = 0.0;
        if (a.length() > maxCharsSoft) {
            double over = (double) (a.length() - maxCharsSoft) / (double) Math.max(1, (maxCharsHard - maxCharsSoft));
            verbosityPenalty = clamp01(over) * 0.35;
        }

        double stylePenalty = 0.0;
        stylePenalty += s.structurePenalty;
        stylePenalty += verbosityPenalty;
        stylePenalty += rep.repetitionPenalty;

        // ---- 8) Validity decision ----
        boolean valid =
                s.hasAllSections
                        && groundedness >= minGroundedness
                        && contradictionRisk <= maxContradictionRisk
                        && queryCoverage >= minQueryCoverage
                        && n.novelty <= maxNovelty
                        && rep.repetition <= maxRepetition;

        // ---- 9) Final score (ordering) ----
        // Weighted to prefer: grounded + on-topic + structured; penalize risk + noise.
        double score =
                (1.15 * groundedness)
                        + (0.55 * queryCoverage)
                        - (1.05 * contradictionRisk)
                        - (1.00 * stylePenalty)
                        - (0.35 * clamp01(n.novelty)) // extra nudge against unsupported mass
                        + (0.10 * s.actionability);   // small bonus if it looks step-like

        // ---- 10) Legacy fields mapping (kept stable) ----
        double coverage = queryCoverage;
        double contextSupport = groundedness;

        String telemetry = telemetry(
                s, g, groundedness,
                queryCoverage,
                n,
                rep,
                contradictionRisk,
                numericDensity,
                punctuationDensity,
                stylePenalty,
                valid
        );

        return new Evaluation(
                score,
                telemetry,
                coverage,
                contextSupport,
                stylePenalty,
                groundedness,
                contradictionRisk,
                valid,
                telemetry
        );
    }

    // -------------------- structure --------------------

    private static final class Structure {
        final boolean hasAllSections;
        final boolean has1, has2, has3;   // legacy markers at line-start
        final int mdHeaders;             // markdown header count
        final boolean usesMarkdown;      // true if markdown headers drive the contract
        final double structurePenalty;   // 0..~1
        final double actionability;      // 0..1 (heuristic)
        final int sectionCount;          // best-effort indicator: legacy count OR md header count

        private Structure(boolean hasAllSections,
                          boolean has1, boolean has2, boolean has3,
                          int mdHeaders, boolean usesMarkdown,
                          double structurePenalty, double actionability,
                          int sectionCount) {
            this.hasAllSections = hasAllSections;
            this.has1 = has1;
            this.has2 = has2;
            this.has3 = has3;
            this.mdHeaders = mdHeaders;
            this.usesMarkdown = usesMarkdown;
            this.structurePenalty = structurePenalty;
            this.actionability = actionability;
            this.sectionCount = sectionCount;
        }
    }

    private static Structure structureOf(String answer) {
        // legacy markers must be line-start to avoid false positives in prose
        boolean has1 = LEGACY_1.matcher(answer).find();
        boolean has2 = LEGACY_2.matcher(answer).find();
        boolean has3 = LEGACY_3.matcher(answer).find();
        boolean legacyAll = has1 && has2 && has3;

        int legacyCount = 0;
        Matcher m = SECTION_MARKERS.matcher(answer);
        while (m.find()) legacyCount++;

        int mdHeaders = 0;
        Matcher mh = MD_HEADER.matcher(answer);
        while (mh.find()) mdHeaders++;

        boolean mdAll = mdHeaders >= 3;
        boolean hasAll = legacyAll || mdAll;
        boolean usesMarkdown = (!legacyAll) && mdAll;

        // penalties: missing contract, or malformed section ordering density
        double p = 0.0;

        if (!hasAll) {
            // no recognizable schema at all
            p += 0.70;
        } else {
            // recognized schema but weak signals
            if (legacyAll) {
                if (legacyCount < 3) p += 0.20; // section markers not really present at line-start
            } else {
                // markdown schema
                if (mdHeaders < 3) p += 0.20; // should not happen due to mdAll, but keep defensive
            }
        }

        // actionability: presence of bullet/step patterns in body
        int bullets = 0;
        Matcher b = BULLETISH.matcher(answer);
        while (b.find()) bullets++;

        double actionability = clamp01(bullets / 8.0); // saturates near 8 bullet-like lines

        int sectionCount = legacyAll ? legacyCount : mdHeaders;
        return new Structure(
                hasAll,
                has1, has2, has3,
                mdHeaders, usesMarkdown,
                clamp01(p),
                actionability,
                sectionCount
        );
    }

    // -------------------- groundedness --------------------

    private record GroundingAgg(
            double groundedness,
            int bestIdx,
            double bestScore,
            int topK,
            double topKMean
    ) {}

    private GroundingAgg groundednessAgg(String answer, List<Statement> ctx) {
        if (ctx == null || ctx.isEmpty()) {
            return new GroundingAgg(0.0, -1, 0.0, 0, 0.0);
        }

        double best = 0.0;
        int bestIdx = -1;

        final int K = Math.min(4, ctx.size());
        double[] top = new double[K];

        for (int i = 0; i < ctx.size(); i++) {
            Statement st = ctx.get(i);
            if (st == null || st.text == null || st.text.isBlank()) continue;

            double s = overlapScorer.score(answer, st);
            if (!Double.isFinite(s)) s = 0.0;

            // scorer can exceed 1 because st.weight; normalize for stability.
            s = clamp01(s);

            if (s > best) {
                best = s;
                bestIdx = i;
            }

            for (int j = 0; j < K; j++) {
                if (s > top[j]) {
                    for (int k = K - 1; k > j; k--) top[k] = top[k - 1];
                    top[j] = s;
                    break;
                }
            }
        }

        double sum = 0.0;
        int cnt = 0;
        for (double v : top) {
            if (v > 0) { sum += v; cnt++; }
        }
        double mean = cnt == 0 ? 0.0 : sum / cnt;

        // Primary signal: max support.
        // Secondary can be used later: mean of topK (telemetry).
        return new GroundingAgg(best, bestIdx, best, K, mean);
    }

    // -------------------- query coverage --------------------

    private double queryCoverage(String userText, String candidateText) {
        if (userText == null || userText.isBlank()) return 0.5; // no query => neutral
        List<String> q = tokenizer.tokenize(userText);
        List<String> a = tokenizer.tokenize(candidateText);
        if (q.isEmpty() || a.isEmpty()) return 0.0;

        Set<String> qset = new HashSet<>(q);
        int hit = 0;
        for (String tok : a) if (qset.contains(tok)) hit++;

        // normalize similarly to overlap scorer, but bounded [0..1]
        double cov = (double) hit / Math.sqrt((double) (q.size() * a.size()));
        return clamp01(cov);
    }

    // -------------------- novelty vs context --------------------

    private static final class Novelty {
        final double novelty;     // 0..1 (share of tokens not in context token set)
        final int totalTokens;
        final int novelTokens;

        private Novelty(double novelty, int totalTokens, int novelTokens) {
            this.novelty = novelty;
            this.totalTokens = totalTokens;
            this.novelTokens = novelTokens;
        }
    }

    private Novelty noveltyVsContext(String answer, List<Statement> ctx) {
        List<String> aTok = tokenizer.tokenize(answer);
        if (aTok.isEmpty()) return new Novelty(1.0, 0, 0);

        Set<String> ctxTok = new HashSet<>();
        if (ctx != null) {
            for (Statement st : ctx) {
                if (st == null || st.text == null || st.text.isBlank()) continue;
                ctxTok.addAll(tokenizer.tokenize(st.text));
            }
        }

        if (ctxTok.isEmpty()) {
            // No context: everything is "novel" (but risk should be handled elsewhere).
            return new Novelty(1.0, aTok.size(), aTok.size());
        }

        int novel = 0;
        for (String t : aTok) {
            if (!ctxTok.contains(t)) novel++;
        }

        // We don't want novelty to punish simple stopwords too much, but tokenizer already normalizes.
        double ratio = (double) novel / (double) Math.max(1, aTok.size());
        return new Novelty(clamp01(ratio), aTok.size(), novel);
    }

    // -------------------- repetition --------------------

    private static final class Repetition {
        final double repetition;          // 0..1 (1 = very repetitive)
        final double repetitionPenalty;   // 0..~1
        final int uniqueTokens;
        final int totalTokens;

        private Repetition(double repetition, double repetitionPenalty, int uniqueTokens, int totalTokens) {
            this.repetition = repetition;
            this.repetitionPenalty = repetitionPenalty;
            this.uniqueTokens = uniqueTokens;
            this.totalTokens = totalTokens;
        }
    }

    private Repetition repetition(String answer) {
        // Token-based repetition
        List<String> tok = tokenizer.tokenize(answer);
        if (tok.isEmpty()) return new Repetition(1.0, 0.30, 0, 0);

        Set<String> uniq = new HashSet<>(tok);
        int total = tok.size();
        int unique = uniq.size();

        // repetition = 1 - (unique/total)
        double rep = 1.0 - ((double) unique / (double) Math.max(1, total));
        rep = clamp01(rep);

        // Also detect degenerate loops at character level (low unique wordlike chunks)
        int wordLike = countMatches(WORDLIKE, answer);
        double loopSignal = 0.0;
        if (wordLike > 0) {
            double wlUniqueRatio = (double) unique / (double) Math.max(1, wordLike);
            loopSignal = clamp01(1.0 - wlUniqueRatio);
        }

        double penalty = clamp01((rep * 0.65) + (loopSignal * 0.35)) * 0.55;

        return new Repetition(rep, penalty, unique, total);
    }

    // -------------------- telemetry + helpers --------------------

    private static Evaluation invalid(double score, String note) {
        return new Evaluation(
                score,
                note,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                false,
                note
        );
    }

    private static String telemetry(
            Structure s,
            GroundingAgg g,
            double groundedness,
            double queryCoverage,
            Novelty n,
            Repetition rep,
            double contradictionRisk,
            double numericDensity,
            double punctuationDensity,
            double stylePenalty,
            boolean valid
    ) {
        return "sec=" + s.sectionCount +
                ";md=" + s.mdHeaders +
                ";md_on=" + (s.usesMarkdown ? 1 : 0) +
                ";s1=" + (s.has1 ? 1 : 0) +
                ";s2=" + (s.has2 ? 1 : 0) +
                ";s3=" + (s.has3 ? 1 : 0) +
                ";act=" + fmt2(s.actionability) +
                ";g=" + fmt2(groundedness) +
                ";g_best_i=" + g.bestIdx +
                ";g_best=" + fmt2(g.bestScore) +
                ";g_topk=" + g.topK +
                ";g_topk_mean=" + fmt2(g.topKMean) +
                ";qc=" + fmt2(queryCoverage) +
                ";nov=" + fmt2(n.novelty) +
                ";nov_n=" + n.novelTokens +
                ";nov_t=" + n.totalTokens +
                ";rep=" + fmt2(rep.repetition) +
                ";rep_u=" + rep.uniqueTokens +
                ";rep_t=" + rep.totalTokens +
                ";r=" + fmt2(contradictionRisk) +
                ";nd=" + fmt4(numericDensity) +
                ";pd=" + fmt4(punctuationDensity) +
                ";sp=" + fmt2(stylePenalty) +
                ";v=" + (valid ? 1 : 0);
    }

    private static int countDigits(String s) {
        int c = 0;
        for (int i = 0; i < s.length(); i++) if (Character.isDigit(s.charAt(i))) c++;
        return c;
    }

    private static int countPunctuation(String s) {
        int c = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (!Character.isLetterOrDigit(ch) && !Character.isWhitespace(ch)) c++;
        }
        return c;
    }

    private static int countMatches(Pattern p, String s) {
        int c = 0;
        Matcher m = p.matcher(s);
        while (m.find()) c++;
        return c;
    }

    private static double clamp01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    private static String fmt2(double v) {
        return String.format(Locale.ROOT, "%.2f", v);
    }

    private static String fmt4(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }
}