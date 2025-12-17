package org.calista.arasaka.ai.think.candidate.impl;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.scorer.impl.TokenOverlapScorer;
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
 *  - Numbered sections: 1) ... 2) ... 3) ... (ANY 3+ numeric sections)
 *  - Markdown: ## / ### headers (>= 3 headers)
 *
 * IMPORTANT: No-context mode:
 *  - When context is empty, do NOT punish groundedness/novelty harshly.
 *  - Instead punish echoing the question and repetition. This fixes the "Кто ты?" -> "Кто ты?" loop.
 */
public final class CandidateEvaluator implements org.calista.arasaka.ai.think.candidate.CandidateEvaluator {

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

    // Numbered section markers at line start (accept any 1..99).
    private static final Pattern SECTION_MARKERS = Pattern.compile("(?m)^(\\s*)(\\d{1,2})\\)\\s+");
    private static final Pattern LEGACY_1 = Pattern.compile("(?m)^\\s*1\\)\\s+");
    private static final Pattern LEGACY_2 = Pattern.compile("(?m)^\\s*2\\)\\s+");
    private static final Pattern LEGACY_3 = Pattern.compile("(?m)^\\s*3\\)\\s+");

    // Markdown section markers (line-start headers)
    private static final Pattern MD_HEADER = Pattern.compile("(?m)^\\s*#{2,3}\\s+.+$");

    private static final Pattern BULLETISH = Pattern.compile("(?m)^\\s*([-*•]|\\d+\\.|\\d+\\))\\s+");

    public CandidateEvaluator(Tokenizer tokenizer) {
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

    public CandidateEvaluator(
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
        final long t0 = System.nanoTime();

        final String q = userText == null ? "" : userText.trim();
        final String a = candidateText == null ? "" : candidateText.trim();
        final List<Statement> ctx = context == null ? List.of() : context;

        final int tokens = safeTokenCount(a);

        if (a.isEmpty()) {
            return invalid(-1.0, "err=empty", tokens, System.nanoTime() - t0);
        }
        if (a.length() > maxCharsHard) {
            return invalid(-0.6, "err=too_long_hard;len=" + a.length(), tokens, System.nanoTime() - t0);
        }

        // ---- 1) Structure quality ----
        Structure s = structureOf(a);
        double structureScore = structureScoreOf(s); // (0..1)

        // ---- No-context mode ----
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
                    queryCoverage >= (minQueryCoverage * 0.50)
                            && echoPenalty <= 0.60
                            && rep.repetition <= maxRepetition
                            && a.length() <= maxCharsHard;

            double score =
                    (0.75 * queryCoverage)
                            - (0.95 * contradictionRisk)
                            - (0.90 * stylePenalty)
                            + (0.15 * structureScore)
                            + (0.10 * s.actionability);

            String telemetry = "noctx"
                    + ";qc=" + fmt2(queryCoverage)
                    + ";echo=" + fmt2(echoPenalty)
                    + ";rep=" + fmt2(rep.repetition)
                    + ";sec=" + s.sectionCount
                    + ";md=" + s.mdHeaders
                    + ";st=" + fmt2(structureScore)
                    + ";v=" + (valid ? 1 : 0);

            long nanos = System.nanoTime() - t0;
            return new Evaluation(
                    score,
                    telemetry,
                    queryCoverage,
                    0.0,
                    stylePenalty,
                    0.0,
                    contradictionRisk,
                    structureScore,
                    valid,
                    telemetry,
                    tokens,
                    nanos
            );
        }

        // ---- Context mode ----
        if (a.length() < minChars) {
            return invalid(-0.8, "err=too_short;len=" + a.length(), tokens, System.nanoTime() - t0);
        }

        // ---- 2) Groundedness to context ----
        GroundingAgg g = groundednessAgg(a, ctx);
        double groundedness = g.groundedness;

        // ---- 3) Query coverage ----
        double queryCoverage = queryCoverage(q, a);

        // ---- 4) Novelty vs context ----
        Novelty n = noveltyVsContext(a, ctx);

        // ---- 5) Repetition ----
        Repetition rep = repetition(a);

        // ---- 6) Contradiction / hallucination risk ----
        double numericDensity = (double) countDigits(a) / (double) Math.max(1, a.length());
        double punctuationDensity = (double) countPunctuation(a) / (double) Math.max(1, a.length());

        double contradictionRisk = clamp01(
                0.15
                        + 0.40 * clamp01(numericDensity / 0.18)
                        + 0.15 * clamp01(punctuationDensity / 0.22)
                        + 0.45 * clamp01(n.novelty / Math.max(1e-9, maxNovelty)) * 0.5
                        + (s.hasAllSections ? 0.0 : 0.10)
                        - 0.55 * groundedness
        );

        // ---- 7) Style / verbosity ----
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
                        && structureScore >= 0.35
                        && groundedness >= minGroundedness
                        && contradictionRisk <= maxContradictionRisk
                        && queryCoverage >= minQueryCoverage
                        && n.novelty <= maxNovelty
                        && rep.repetition <= maxRepetition
                        && a.length() <= maxCharsHard;

        // ---- 9) Score ----
        double score =
                (0.50 * groundedness)
                        + (0.25 * queryCoverage)
                        + (0.15 * structureScore)
                        + (0.10 * s.actionability)
                        - (0.55 * contradictionRisk)
                        - (0.55 * stylePenalty)
                        - (0.35 * n.novelty)
                        - (0.35 * rep.repetition);

        // Penalize missing schema stronger in context-mode
        if (!s.hasAllSections) score -= 0.35;

        String telemetry = "qc=" + fmt2(queryCoverage)
                + ";g=" + fmt2(groundedness)
                + ";nov=" + fmt2(n.novelty)
                + ";rep=" + fmt2(rep.repetition)
                + ";sec=" + s.sectionCount
                + ";md=" + s.mdHeaders
                + ";st=" + fmt2(structureScore)
                + ";risk=" + fmt2(contradictionRisk)
                + ";v=" + (valid ? 1 : 0);

        long nanos = System.nanoTime() - t0;

        return new Evaluation(
                score,
                telemetry,
                queryCoverage,
                groundedness,
                stylePenalty,
                n.novelty,
                contradictionRisk,
                structureScore,
                valid,
                telemetry,
                tokens,
                nanos
        );
    }

    // -------------------- structure --------------------

    private record Structure(
            boolean hasAllSections,
            boolean has1,
            boolean has2,
            boolean has3,
            int mdHeaders,
            boolean usesMarkdown,
            double structurePenalty,
            double actionability,
            int sectionCount
    ) {}

    private static Structure structureOf(String answer) {
        boolean has1 = LEGACY_1.matcher(answer).find();
        boolean has2 = LEGACY_2.matcher(answer).find();
        boolean has3 = LEGACY_3.matcher(answer).find();

        boolean legacyAll;

        int legacyCount = 0;
        Matcher m = SECTION_MARKERS.matcher(answer);
        while (m.find()) legacyCount++;

        // New: accept any 3+ numbered sections (not only 1/2/3). Keeps legacy flags for telemetry.
        legacyAll = legacyCount >= 3;

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

    private static double structureScoreOf(Structure s) {
        if (s == null) return 0.0;
        double base = 1.0 - clamp01(s.structurePenalty);
        double score = (0.80 * base) + (0.20 * clamp01(s.actionability));
        if (s.hasAllSections) score += 0.05;
        return clamp01(score);
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

        double mean = cnt == 0 ? 0.0 : (sum / cnt);
        double agg = clamp01(0.70 * best + 0.30 * mean);

        return new GroundingAgg(agg, bestIdx, best, cnt, mean);
    }

    // -------------------- novelty --------------------

    private record Novelty(double novelty, double contextSupport, int unsupportedTokens, int totalTokens) {}

    private Novelty noveltyVsContext(String answer, List<Statement> ctx) {
        Set<String> ctxTok = new HashSet<>(2048);
        for (Statement s : ctx) {
            if (s == null || s.text == null || s.text.isBlank()) continue;
            ctxTok.addAll(tokenSet(s.text));
            if (ctxTok.size() > 20_000) break;
        }

        List<String> aTok = tokenize(answer);
        if (aTok.isEmpty()) return new Novelty(0.0, 1.0, 0, 0);

        int unsupported = 0;
        for (String t : aTok) {
            if (t == null) continue;
            if (!ctxTok.contains(t)) unsupported++;
        }

        double novelty = clamp01((double) unsupported / (double) Math.max(1, aTok.size()));
        double support = 1.0 - novelty;

        return new Novelty(novelty, support, unsupported, aTok.size());
    }

    // -------------------- query coverage --------------------

    private double queryCoverage(String query, String answer) {
        if (query == null || query.isBlank()) return 0.0;
        if (answer == null || answer.isBlank()) return 0.0;

        List<String> qTok = tokenize(query);
        List<String> aTok = tokenize(answer);

        if (qTok.isEmpty() || aTok.isEmpty()) return 0.0;

        Set<String> aSet = new HashSet<>(aTok);
        int inter = 0;
        for (String t : qTok) if (aSet.contains(t)) inter++;

        return clamp01((double) inter / (double) Math.max(1, qTok.size()));
    }

    // -------------------- repetition --------------------

    private record Repetition(double repetition, double repetitionPenalty) {}

    private Repetition repetition(String answer) {
        List<String> tok = tokenize(answer);
        if (tok.isEmpty()) return new Repetition(0.0, 0.0);

        Map<String, Integer> c = new HashMap<>(tok.size() * 2);
        for (String t : tok) c.merge(t, 1, Integer::sum);

        int max = 1;
        for (int v : c.values()) if (v > max) max = v;

        double rep = clamp01((double) max / (double) Math.max(1, tok.size() / 6));
        double pen = clamp01(rep / Math.max(1e-9, maxRepetition)) * 0.35;

        return new Repetition(rep, pen);
    }

    private double echoPenalty(String q, String a) {
        if (q == null || q.isBlank() || a == null || a.isBlank()) return 0.0;
        List<String> qt = tokenize(q);
        List<String> at = tokenize(a);
        if (qt.isEmpty() || at.isEmpty()) return 0.0;

        Set<String> qs = new HashSet<>(qt);
        Set<String> as = new HashSet<>(at);

        int inter = 0;
        for (String t : qs) if (as.contains(t)) inter++;

        double jacc = (double) inter / (double) Math.max(1, (qs.size() + as.size() - inter));
        return clamp01(jacc);
    }

    // -------------------- token utils --------------------

    private List<String> tokenize(String s) {
        if (s == null || s.isBlank()) return List.of();
        List<String> out = tokenizer.tokenize(s);
        if (out == null || out.isEmpty()) return List.of();
        if (out.size() > 512) return out.subList(0, 512);
        return out;
    }

    private Set<String> tokenSet(String s) {
        List<String> t = tokenize(s);
        if (t.isEmpty()) return Set.of();
        return new HashSet<>(t);
    }

    private int safeTokenCount(String s) {
        if (s == null || s.isBlank()) return 0;
        List<String> t = tokenizer.tokenize(s);
        return t == null ? 0 : t.size();
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

    // -------------------- invalid helper --------------------

    private Evaluation invalid(double score, String note, int tokens, long nanos) {
        return new Evaluation(
                score,
                note,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                false,
                note,
                tokens,
                nanos
        );
    }

    // -------------------- misc math --------------------

    private static double clamp01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    private static String fmt2(double v) {
        if (!Double.isFinite(v)) return "0.00";
        double x = Math.round(v * 100.0) / 100.0;
        String s = Double.toString(x);
        int dot = s.indexOf('.');
        if (dot < 0) return s + ".00";
        int dec = s.length() - dot - 1;
        if (dec == 0) return s + "00";
        if (dec == 1) return s + "0";
        if (dec > 2) return s.substring(0, dot + 3);
        return s;
    }
}