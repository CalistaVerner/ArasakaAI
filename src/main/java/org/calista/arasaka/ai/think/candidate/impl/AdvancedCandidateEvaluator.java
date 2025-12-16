package org.calista.arasaka.ai.think.candidate.impl;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.scorer.impl.TokenOverlapScorer;
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
 *
 * IMPORTANT: No-context mode:
 *  - When context is empty, do NOT punish groundedness/novelty harshly.
 *  - Instead punish echoing the question and repetition. This fixes the "Кто ты?" -> "Кто ты?" loop.
 */
public final class AdvancedCandidateEvaluator implements CandidateEvaluator {

    // --- policy thresholds (tunable) ---
    private final double minGroundedness;
    private final double maxContradictionRisk;

    private final double minQueryCoverage;      // candidate should overlap with user query
    private final double maxNovelty;            // too many tokens not in context => likely hallucination
    private final double maxRepetition;         // excessive repetition => low quality / loop
    private final int minChars;                 // too short => usually useless (context mode)
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
        if (a.length() > maxCharsHard) {
            return invalid(-0.6, "err=too_long_hard;len=" + a.length());
        }

        // ---- 1) Structure quality ----
        Structure s = structureOf(a);

        // ---- No-context mode (chat/identity/smalltalk) ----
        // Do not invalidate short answers here; short can be fine without evidence.
        if (ctx.isEmpty()) {
            double queryCoverage = queryCoverage(q, a);
            Repetition rep = repetition(a);
            double echoPenalty = echoPenalty(q, a);

            // In no-context mode, schema isn't mandatory; keep soft penalty only.
            double structureSoft = clamp01(s.structurePenalty) * 0.25;

            double verbosityPenalty = 0.0;
            if (a.length() > maxCharsSoft) {
                double over = (double) (a.length() - maxCharsSoft) / (double) Math.max(1, (maxCharsHard - maxCharsSoft));
                verbosityPenalty = clamp01(over) * 0.35;
            }

            double stylePenalty = structureSoft + verbosityPenalty + rep.repetitionPenalty + echoPenalty;

            // Treat “risk” as “uselessness/loop risk” here.
            double contradictionRisk = clamp01(
                    0.10
                            + 0.35 * echoPenalty
                            + 0.25 * rep.repetition
                            + 0.10 * (s.hasAllSections ? 0.0 : 1.0)
            );

            boolean valid =
                    // allow lower query coverage in no-context mode, but avoid pure echo
                    queryCoverage >= (minQueryCoverage * 0.50)
                            && echoPenalty <= 0.60
                            && rep.repetition <= maxRepetition
                            && a.length() <= maxCharsHard;

            double score =
                    (0.75 * queryCoverage)
                            - (0.95 * contradictionRisk)
                            - (0.90 * stylePenalty)
                            + (0.10 * s.actionability);

            String telemetry = "noctx"
                    + ";qc=" + fmt2(queryCoverage)
                    + ";echo=" + fmt2(echoPenalty)
                    + ";rep=" + fmt2(rep.repetition)
                    + ";sec=" + s.sectionCount
                    + ";md=" + s.mdHeaders
                    + ";v=" + (valid ? 1 : 0);

            return new Evaluation(
                    score,
                    telemetry,
                    queryCoverage,
                    0.0,
                    stylePenalty,
                    0.0,                 // groundedness not applicable
                    contradictionRisk,    // in noctx: loop/uselessness risk proxy
                    valid,
                    telemetry
            );
        }

        // ---- Context mode: enforce minimal usefulness ----
        if (a.length() < minChars) {
            return invalid(-0.8, "err=too_short;len=" + a.length());
        }

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

        // ---- 9) Final score ----
        double score =
                (1.15 * groundedness)
                        + (0.55 * queryCoverage)
                        - (1.05 * contradictionRisk)
                        - (1.00 * stylePenalty)
                        - (0.35 * clamp01(n.novelty))
                        + (0.10 * s.actionability);

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
        final int sectionCount;          // legacy count OR md header count

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

        double p = 0.0;
        if (!hasAll) {
            p += 0.70;
        } else {
            if (legacyAll) {
                if (legacyCount < 3) p += 0.20;
            } else {
                if (mdHeaders < 3) p += 0.20;
            }
        }

        int bullets = 0;
        Matcher b = BULLETISH.matcher(answer);
        while (b.find()) bullets++;

        double actionability = clamp01(bullets / 8.0);
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

        return new GroundingAgg(best, bestIdx, best, K, mean);
    }

    // -------------------- query coverage --------------------

    private double queryCoverage(String userText, String candidateText) {
        if (userText == null || userText.isBlank()) return 0.5;
        List<String> q = tokenizer.tokenize(userText);
        List<String> a = tokenizer.tokenize(candidateText);
        if (q.isEmpty() || a.isEmpty()) return 0.0;

        Set<String> qset = new HashSet<>(q);
        int hit = 0;
        for (String tok : a) if (qset.contains(tok)) hit++;

        double cov = (double) hit / Math.sqrt((double) (q.size() * a.size()));
        return clamp01(cov);
    }

    // -------------------- novelty vs context --------------------

    private static final class Novelty {
        final double novelty;
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
            return new Novelty(1.0, aTok.size(), aTok.size());
        }

        int novel = 0;
        for (String t : aTok) {
            if (!ctxTok.contains(t)) novel++;
        }

        double ratio = (double) novel / (double) Math.max(1, aTok.size());
        return new Novelty(clamp01(ratio), aTok.size(), novel);
    }

    // -------------------- repetition --------------------

    private static final class Repetition {
        final double repetition;         // 0..1
        final double repetitionPenalty;  // 0..~1
        final int uniqueTokens;
        final int totalTokens;

        private Repetition(double repetition, double repetitionPenalty, int uniqueTokens, int totalTokens) {
            this.repetition = repetition;
            this.repetitionPenalty = repetitionPenalty;
            this.uniqueTokens = uniqueTokens;
            this.totalTokens = totalTokens;
        }
    }

    private Repetition repetition(String text) {
        List<String> toks = tokenizer.tokenize(text);
        if (toks.isEmpty()) return new Repetition(0.0, 0.0, 0, 0);

        Map<String, Integer> freq = new HashMap<>();
        int total = 0;
        for (String t : toks) {
            if (t == null || t.isBlank()) continue;
            if (t.length() < 2) continue;
            freq.merge(t, 1, Integer::sum);
            total++;
        }
        int unique = freq.size();
        if (total <= 0) return new Repetition(0.0, 0.0, unique, total);

        // repetition: how much token mass is repeats (beyond first occurrence)
        int repeatMass = 0;
        for (int c : freq.values()) if (c > 1) repeatMass += (c - 1);

        double rep = clamp01((double) repeatMass / (double) Math.max(1, total));
        double penalty = clamp01(rep / Math.max(1e-9, maxRepetition)) * 0.55;
        return new Repetition(rep, penalty, unique, total);
    }

    // -------------------- echo penalty (no-context) --------------------

    /**
     * Penalize answers that mostly repeat the question ("echo").
     */
    private double echoPenalty(String userText, String candidateText) {
        if (userText == null || candidateText == null) return 0.0;
        String q = userText.trim();
        String a = candidateText.trim();
        if (q.isEmpty() || a.isEmpty()) return 0.0;

        Set<String> qt = new LinkedHashSet<>(tokenizer.tokenize(q));
        Set<String> at = new LinkedHashSet<>(tokenizer.tokenize(a));
        qt.removeIf(t -> t.length() < 2);
        at.removeIf(t -> t.length() < 2);
        if (qt.isEmpty() || at.isEmpty()) return 0.0;

        int inter = 0;
        for (String t : qt) if (at.contains(t)) inter++;
        int union = qt.size() + at.size() - inter;
        double j = union <= 0 ? 0.0 : (double) inter / (double) union;

        if (j < 0.75) return 0.0;
        double shortness = clamp01((double) (Math.max(0, (q.length() * 2) - a.length())) / (double) Math.max(1, q.length()));
        double p = clamp01((j - 0.75) / 0.25) * (0.25 + 0.35 * shortness);
        return clamp01(p);
    }

    // -------------------- invalid helper --------------------

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

    // -------------------- telemetry --------------------

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