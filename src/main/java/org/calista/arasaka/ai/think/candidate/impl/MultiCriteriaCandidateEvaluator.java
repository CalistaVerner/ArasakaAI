// FILE: MultiCriteriaCandidateEvaluator.java
package org.calista.arasaka.ai.think.candidate.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * MultiCriteriaCandidateEvaluator (Quantum-like):
 * - multi-channel scoring
 * - coherence & entropy penalties
 * - deterministic (no randomness)
 *
 * Added:
 *  - strict verify-pass mode via evaluateStrict(...)
 */
public final class MultiCriteriaCandidateEvaluator implements org.calista.arasaka.ai.think.candidate.CandidateEvaluator {

    private static final Logger log = LogManager.getLogger(MultiCriteriaCandidateEvaluator.class);

    private final CandidateEvaluator base;

    // --- quantum policy knobs ---
    private final double minCoherence;        // 0..1
    private final double entropyPenaltyWeight;
    private final double coherenceWeight;

    // --- strict verify knobs ---
    private final double strictMinCoherenceBoost;     // +0.12
    private final double strictEntropyBoost;          // +0.10
    private final double strictRiskWeight;            // 0.45 (vs 0.30)

    // --- debug knobs ---
    private final boolean debugEnabled;
    private final int debugSnippetChars;

    public MultiCriteriaCandidateEvaluator(Tokenizer tokenizer) {
        this(
                new CandidateEvaluator(tokenizer),
                0.45,
                0.35,
                0.40,
                0.12,
                0.10,
                0.45,
                false,
                180
        );
    }

    public MultiCriteriaCandidateEvaluator(
            CandidateEvaluator base,
            double minCoherence,
            double entropyPenaltyWeight,
            double coherenceWeight
    ) {
        this(base, minCoherence, entropyPenaltyWeight, coherenceWeight, 0.12, 0.10, 0.45, false, 180);
    }

    public MultiCriteriaCandidateEvaluator(
            CandidateEvaluator base,
            double minCoherence,
            double entropyPenaltyWeight,
            double coherenceWeight,
            boolean debugEnabled,
            int debugSnippetChars
    ) {
        this(base, minCoherence, entropyPenaltyWeight, coherenceWeight, 0.12, 0.10, 0.45, debugEnabled, debugSnippetChars);
    }

    public MultiCriteriaCandidateEvaluator(
            CandidateEvaluator base,
            double minCoherence,
            double entropyPenaltyWeight,
            double coherenceWeight,
            double strictMinCoherenceBoost,
            double strictEntropyBoost,
            double strictRiskWeight,
            boolean debugEnabled,
            int debugSnippetChars
    ) {
        this.base = Objects.requireNonNull(base, "base");
        this.minCoherence = clamp01(minCoherence);
        this.entropyPenaltyWeight = clamp01(entropyPenaltyWeight);
        this.coherenceWeight = clamp01(coherenceWeight);

        this.strictMinCoherenceBoost = clamp01(strictMinCoherenceBoost);
        this.strictEntropyBoost = clamp01(strictEntropyBoost);
        this.strictRiskWeight = clamp01(strictRiskWeight);

        this.debugEnabled = debugEnabled;
        this.debugSnippetChars = Math.max(40, debugSnippetChars);
    }

    @Override
    public double score(String userText, String candidateText, List<Statement> context) {
        return evaluate(userText, candidateText, context).effectiveScore();
    }

    @Override
    public Evaluation evaluate(String userText, String candidateText, List<Statement> context) {
        return evaluateInternal(userText, candidateText, context, false);
    }

    /**
     * Strict verify-pass evaluation:
     * - higher min coherence
     * - stronger entropy + contradiction risk penalty
     */
    public Evaluation evaluateStrict(String userText, String candidateText, List<Statement> context) {
        return evaluateInternal(userText, candidateText, context, true);
    }

    private Evaluation evaluateInternal(String userText, String candidateText, List<Statement> context, boolean strict) {
        final long t0 = System.nanoTime();

        final String q = userText == null ? "" : userText.trim();
        final String a = candidateText == null ? "" : candidateText.trim();
        final int ctxSize = (context == null) ? 0 : context.size();

        // ---- Base (classical) evaluation ----
        Evaluation e = base.evaluate(q, a, context);

        // ---- Quantum channels (superposition) ----
        double factual = clamp01(e.groundedness);
        double structure = clamp01(e.structureScore);
        double coverage = clamp01(e.coverage);
        double actionability = clamp01(estimateActionability(e));
        double risk = clamp01(e.contradictionRisk);

        double[] channels = { factual, structure, coverage, actionability };

        // ---- Entropy (disorder of signals) ----
        double entropy = entropy(channels);

        // ---- Coherence (signals agree with each other) ----
        double coherence = coherence(channels);

        // strict adjustments
        double minCoh = strict ? clamp01(this.minCoherence + strictMinCoherenceBoost) : this.minCoherence;
        double entW = strict ? clamp01(this.entropyPenaltyWeight + strictEntropyBoost) : this.entropyPenaltyWeight;
        double riskW = strict ? this.strictRiskWeight : 0.30;

        // ---- Quantum-adjusted score ----
        double quantumScore =
                e.score
                        + coherenceWeight * coherence
                        - entW * entropy
                        - riskW * risk;

        boolean valid = e.valid && coherence >= minCoh;

        String telemetry =
                e.validationNotes
                        + ";q_f=" + fmt2(factual)
                        + ";q_s=" + fmt2(structure)
                        + ";q_c=" + fmt2(coverage)
                        + ";q_a=" + fmt2(actionability)
                        + ";q_r=" + fmt2(risk)
                        + ";q_ent=" + fmt2(entropy)
                        + ";q_coh=" + fmt2(coherence)
                        + ";q_minCoh=" + fmt2(minCoh)
                        + ";q_entW=" + fmt2(entW)
                        + ";q_rW=" + fmt2(riskW)
                        + ";q_strict=" + (strict ? 1 : 0)
                        + ";q_v=" + (valid ? 1 : 0);

        long nanos = System.nanoTime() - t0;

        Evaluation out = new Evaluation(
                quantumScore,
                telemetry,
                e.coverage,
                e.contextSupport,
                e.stylePenalty,
                e.groundedness,
                e.contradictionRisk,
                e.structureScore,
                valid,
                telemetry,
                e.tokens,
                nanos
        );

        if (log.isDebugEnabled()) {
            boolean flip = (e.valid != valid);
            boolean lowC = coherence < minCoh;

            if (debugEnabled || flip || lowC || strict) {
                log.debug(
                        "QuantumEval | strict={} ctx={} tok={} baseScore={} qScore={} eff={} valid={} (baseValid={}) " +
                                "| g={} st={} cov={} act={} risk={} coh={} ent={} " +
                                "| q='{}' a='{}'",
                        strict ? 1 : 0,
                        ctxSize,
                        e.tokens,
                        fmt4(e.score),
                        fmt4(quantumScore),
                        fmt4(out.effectiveScore()),
                        valid,
                        e.valid,
                        fmt3(factual),
                        fmt3(structure),
                        fmt3(coverage),
                        fmt3(actionability),
                        fmt3(risk),
                        fmt3(coherence),
                        fmt3(entropy),
                        snippet(q, debugSnippetChars),
                        snippet(a, debugSnippetChars)
                );
            }
        }

        if (debugEnabled && log.isDebugEnabled()) {
            log.debug("QuantumEvalTelemetry | {}", telemetry);
        }

        return out;
    }

    // -------------------- quantum math --------------------

    private static double entropy(double[] v) {
        double sum = 0.0;
        for (double x : v) sum += x;
        if (sum <= 1e-9) return 1.0;

        double h = 0.0;
        for (double x : v) {
            double p = x / sum;
            if (p > 1e-9) {
                h -= p * Math.log(p);
            }
        }
        return clamp01(h / Math.log(v.length));
    }

    private static double coherence(double[] v) {
        double mean = 0.0;
        for (double x : v) mean += x;
        mean /= v.length;

        double var = 0.0;
        for (double x : v) {
            double d = x - mean;
            var += d * d;
        }
        var /= v.length;

        return clamp01(1.0 - Math.sqrt(var));
    }

    private static double estimateActionability(Evaluation e) {
        double a = 0.0;
        a += 0.5 * clamp01(e.structureScore);
        a += 0.5 * clamp01(e.coverage);
        return clamp01(a);
    }

    // -------------------- utils --------------------

    private static String snippet(String s, int max) {
        if (s == null) return "";
        String x = s.replace('\n', ' ').replace('\r', ' ').trim();
        if (x.length() <= max) return x;
        return x.substring(0, Math.max(0, max - 1)) + "…";
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

    private static String fmt3(double v) {
        return String.format(Locale.ROOT, "%.3f", v);
    }

    private static String fmt4(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }
}