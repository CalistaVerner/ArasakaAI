package org.calista.arasaka.ai.think.candidate.impl;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.TokenOverlapScorer;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

@Deprecated
public final class SimpleCandidateEvaluator implements CandidateEvaluator {

    private final TokenOverlapScorer overlapScorer;

    // Thresholds are constants (policy), not "phrases".
    private final double minGroundedness;
    private final double maxContradictionRisk;

    // Lightweight "factuality" checks: penalize claim-like fragments with low context support.
    // These are generic heuristics (no domain lexicons).
    private final double factualityMinSupport;
    private final double factualityPenaltyWeight;

    public SimpleCandidateEvaluator(Tokenizer tokenizer) {
        this(new TokenOverlapScorer(tokenizer), 0.25, 0.70, 0.12, 0.80);
    }

    public SimpleCandidateEvaluator(TokenOverlapScorer overlapScorer,
                                    double minGroundedness,
                                    double maxContradictionRisk,
                                    double factualityMinSupport,
                                    double factualityPenaltyWeight) {
        this.overlapScorer = Objects.requireNonNull(overlapScorer, "overlapScorer");
        this.minGroundedness = clamp01(minGroundedness);
        this.maxContradictionRisk = clamp01(maxContradictionRisk);
        this.factualityMinSupport = clamp01(factualityMinSupport);
        this.factualityPenaltyWeight = clamp01(factualityPenaltyWeight);
    }

    @Override
    public double score(String userText, String candidateText, List<Statement> context) {
        return evaluate(userText, candidateText, context).effectiveScore();
    }

    @Override
    public Evaluation evaluate(String userText, String candidateText, List<Statement> context) {

        final String answer = candidateText == null ? "" : candidateText.trim();
        final List<Statement> ctx = context == null ? List.of() : context;

        if (answer.isEmpty()) {
            return new Evaluation(
                    -1.0,
                    "err=empty",
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    false,
                    "err=empty"
            );
        }

        // --- 1) Structural validity (format contract) ---
        final boolean hasS1 = answer.contains("1)");
        final boolean hasS2 = answer.contains("2)");
        final boolean hasS3 = answer.contains("3)");
        final boolean hasStructure = hasS1 && hasS2 && hasS3;

        // --- 2) Context grounding (aggregation over Statements) ---
        final GroundingAgg agg = groundednessAgg(answer, ctx);
        final double groundedness = agg.groundedness;

        // --- 3) Unsupported-assertion / contradiction risk heuristic ---
        final double numericDensity = (double) countDigits(answer) / (double) Math.max(1, answer.length());

        // --- 3a) Factuality (low-support claims) ---
        final FactualityFacts facts = factualityFacts(answer, ctx);
        final double factualityPenalty = clamp01(facts.unsupportedRatio * factualityPenaltyWeight);

        final double contradictionRisk = clamp01(riskFrom(numericDensity, groundedness, hasStructure) + factualityPenalty);

        // --- 4) Additional soft metrics (kept compatible with legacy fields) ---
        final double coverage = groundedness;
        final double contextSupport = groundedness;
        final double stylePenalty = hasStructure ? 0.0 : 0.50;

        // --- 5) Validity decision ---
        final boolean valid = hasStructure
                && groundedness >= minGroundedness
                && contradictionRisk <= maxContradictionRisk;

        // --- 6) Score (primary ordering) ---
        final double score = (groundedness * 1.0)
                - (contradictionRisk * 1.0)
                - (factualityPenalty * 0.75)
                - stylePenalty;

        // --- 7) Machine telemetry ---
        final String telemetry = telemetry(
                hasS1, hasS2, hasS3,
                groundedness, contradictionRisk, numericDensity, valid,
                agg.bestIdx, agg.bestScore, agg.topK, agg.topKMean,
                facts.fragments, facts.unsupported, facts.unsupportedRatio, factualityPenalty
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

    // -------------------- groundedness aggregation --------------------

    private record GroundingAgg(
            double groundedness,
            int bestIdx,
            double bestScore,
            int topK,
            double topKMean
    ) {}

    /**
     * Aggregation policy (deterministic):
     * - compute overlap(answer, each statement)
     * - groundedness = max overlap (strongest support in context)
     * - also compute top-k mean (telemetry; can be used later if you want)
     */
    private GroundingAgg groundednessAgg(String answer, List<Statement> ctx) {
        if (ctx == null || ctx.isEmpty()) {
            return new GroundingAgg(0.0, -1, 0.0, 0, 0.0);
        }

        double best = 0.0;
        int bestIdx = -1;

        // keep a tiny fixed topK (no allocations of large heaps)
        final int K = Math.min(3, ctx.size());
        double[] top = new double[K]; // init 0

        for (int i = 0; i < ctx.size(); i++) {
            Statement st = ctx.get(i);
            if (st == null || st.text == null || st.text.isBlank()) continue;

            double s = overlapScorer.score(answer, st); // <-- correct signature
            if (!Double.isFinite(s)) s = 0.0;
            s = clamp01(s); // scorer can exceed 1 because of weight; we cap to [0..1] for stability

            if (s > best) {
                best = s;
                bestIdx = i;
            }

            // insert into top-K array
            for (int j = 0; j < K; j++) {
                if (s > top[j]) {
                    // shift right
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

        // primary: max overlap
        return new GroundingAgg(best, bestIdx, best, K, mean);
    }

    // -------------------- internals --------------------

    private static int countDigits(String s) {
        int c = 0;
        for (int i = 0; i < s.length(); i++) {
            if (Character.isDigit(s.charAt(i))) c++;
        }
        return c;
    }

    private static double riskFrom(double numericDensity, double groundedness, boolean hasStructure) {
        double r = 0.20;

        r += clamp01(numericDensity / 0.20) * 0.50;
        r -= groundedness * 0.40;
        if (!hasStructure) r += 0.10;

        return r;
    }

    private static String telemetry(
            boolean hasS1, boolean hasS2, boolean hasS3,
            double groundedness, double risk, double numericDensity, boolean valid,
            int bestIdx, double bestScore, int topK, double topKMean,
            int fragments, int unsupported, double unsupportedRatio, double factualityPenalty
    ) {
        return "s1=" + (hasS1 ? 1 : 0) +
                ";s2=" + (hasS2 ? 1 : 0) +
                ";s3=" + (hasS3 ? 1 : 0) +
                ";g=" + fmt2(groundedness) +
                ";r=" + fmt2(risk) +
                ";nd=" + fmt4(numericDensity) +
                ";v=" + (valid ? 1 : 0) +
                ";f_frag=" + fragments +
                ";f_ub=" + unsupported +
                ";f_ub_r=" + fmt2(unsupportedRatio) +
                ";f_pen=" + fmt2(factualityPenalty) +
                ";g_best_i=" + bestIdx +
                ";g_best=" + fmt2(bestScore) +
                ";g_topk=" + topK +
                ";g_topk_mean=" + fmt2(topKMean);
    }

    // -------------------- factuality heuristic --------------------

    private record FactualityFacts(int fragments, int unsupported, double unsupportedRatio) {}

    /**
     * Deterministic heuristic:
     * - split answer into fragments (sentences / bullets)
     * - treat each non-trivial fragment as a "claim"
     * - compute its max context overlap; if below threshold -> unsupported
     */
    private FactualityFacts factualityFacts(String answer, List<Statement> ctx) {
        if (ctx == null || ctx.isEmpty()) return new FactualityFacts(0, 0, 0.0);
        if (answer == null || answer.isBlank()) return new FactualityFacts(0, 0, 0.0);

        String[] parts = answer.split("[\\n\\r\\t]+|(?<=[\\.\\!\\?;])\\s+|•");
        int fragments = 0;
        int unsupported = 0;

        for (String raw : parts) {
            if (raw == null) continue;
            String frag = raw.trim();
            if (frag.length() < 24) continue; // too small to be a claim
            if (frag.startsWith("1)") || frag.startsWith("2)") || frag.startsWith("3)")) continue;

            int tok = approxTokenCount(frag);
            if (tok < 5) continue;

            fragments++;

            double best = 0.0;
            for (Statement st : ctx) {
                if (st == null || st.text == null || st.text.isBlank()) continue;
                double s = overlapScorer.score(frag, st);
                if (!Double.isFinite(s)) continue;
                s = clamp01(s);
                if (s > best) best = s;
            }

            if (best < factualityMinSupport) unsupported++;
        }

        if (fragments == 0) return new FactualityFacts(0, 0, 0.0);
        return new FactualityFacts(fragments, unsupported, (double) unsupported / (double) fragments);
    }

    private static int approxTokenCount(String s) {
        int c = 0;
        boolean in = false;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            boolean w = Character.isLetterOrDigit(ch) || ch == '_';
            if (w && !in) {
                c++;
                in = true;
            } else if (!w) {
                in = false;
            }
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