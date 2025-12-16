package org.calista.arasaka.ai.think.candidate.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * QuantumCandidateEvaluator
 *
 * Enterprise / Neural evaluator:
 * - multi-channel (superposition) scoring
 * - coherence & entropy penalties
 * - deterministic, no randomness
 *
 * Designed for NeuralQuantumEngine.
 */
public final class MultiCriteriaCandidateEvaluator implements CandidateEvaluator {

    private static final Logger log = LogManager.getLogger(MultiCriteriaCandidateEvaluator.class);

    private final AdvancedCandidateEvaluator base;

    // --- quantum policy knobs ---
    private final double minCoherence;        // 0..1
    private final double entropyPenaltyWeight;
    private final double coherenceWeight;

    // --- debug knobs ---
    private final boolean debugEnabled;
    private final int debugSnippetChars;

    public MultiCriteriaCandidateEvaluator(Tokenizer tokenizer) {
        this(
                new AdvancedCandidateEvaluator(tokenizer),
                0.45,
                0.35,
                0.40,
                false,
                180
        );
    }

    public MultiCriteriaCandidateEvaluator(
            AdvancedCandidateEvaluator base,
            double minCoherence,
            double entropyPenaltyWeight,
            double coherenceWeight
    ) {
        this(base, minCoherence, entropyPenaltyWeight, coherenceWeight, false, 180);
    }

    public MultiCriteriaCandidateEvaluator(
            AdvancedCandidateEvaluator base,
            double minCoherence,
            double entropyPenaltyWeight,
            double coherenceWeight,
            boolean debugEnabled,
            int debugSnippetChars
    ) {
        this.base = Objects.requireNonNull(base, "base");
        this.minCoherence = clamp01(minCoherence);
        this.entropyPenaltyWeight = clamp01(entropyPenaltyWeight);
        this.coherenceWeight = clamp01(coherenceWeight);
        this.debugEnabled = debugEnabled;
        this.debugSnippetChars = Math.max(40, debugSnippetChars);
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

        // ---- Quantum-adjusted score ----
        double quantumScore =
                e.score
                        + coherenceWeight * coherence
                        - entropyPenaltyWeight * entropy
                        - 0.30 * risk;

        boolean valid = e.valid && coherence >= minCoherence;

        String telemetry =
                e.validationNotes
                        + ";q_f=" + fmt2(factual)
                        + ";q_s=" + fmt2(structure)
                        + ";q_c=" + fmt2(coverage)
                        + ";q_a=" + fmt2(actionability)
                        + ";q_r=" + fmt2(risk)
                        + ";q_ent=" + fmt2(entropy)
                        + ";q_coh=" + fmt2(coherence)
                        + ";q_minCoh=" + fmt2(minCoherence)
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

        // ---- DEBUG LOGS ----
        // 1) Always: if result flips validity or coherence is low => log at DEBUG (useful to tune thresholds)
        if (log.isDebugEnabled()) {
            boolean flip = (e.valid != valid);
            boolean lowC = coherence < minCoherence;

            if (debugEnabled || flip || lowC) {
                log.debug(
                        "QuantumEval | ctx={} tok={} baseScore={} qScore={} eff={} valid={} (baseValid={}) " +
                                "| g={} st={} cov={} act={} risk={} coh={} ent={} " +
                                "| q='{}' a='{}'",
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

        // 2) TRACE-like details via DEBUG: only when explicitly enabled
        if (debugEnabled && log.isDebugEnabled()) {
            log.debug("QuantumEvalTelemetry | {}", telemetry);
        }

        return out;
    }

    // -------------------- quantum math --------------------

    /**
     * Entropy of channel distribution.
     * High entropy = chaotic / unbalanced answer.
     */
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

    /**
     * Coherence = 1 - normalized variance.
     * High when channels are aligned.
     */
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

    /**
     * Actionability proxy:
     * reuse structure/coverage signals without re-parsing text.
     */
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