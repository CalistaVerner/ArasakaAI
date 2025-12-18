package org.calista.arasaka.ai.think.candidate.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.utils.Tags;
import org.calista.arasaka.ai.think.candidate.CandidateControlSignals;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;

import java.util.*;

/**
 * MultiCriteriaCandidateEvaluator:
 * - multi-channel scoring layered on top of BaseCandidateEvaluator
 * - coherence & entropy penalties
 * - deterministic (no randomness)
 *
 * Strict verify-pass mode:
 * - enabled by state.tags (engine-driven) using Tags.VERIFY_STRICT / Tags.REPAIR_STRICT,
 *   or by calling evaluateStrict(...).
 *
 * NOTE:
 * - Must conform to CandidateEvaluator.evaluate(userText, context, state, draft)
 * - Uses only fields present in CandidateEvaluator.Evaluation (current project)
 */
public final class MultiCriteriaCandidateEvaluator implements CandidateEvaluator {

    private static final Logger log = LogManager.getLogger(MultiCriteriaCandidateEvaluator.class);

    private final BaseCandidateEvaluator base;

    // --- quantum policy knobs ---
    private final double minCoherence;           // 0..1
    private final double entropyPenaltyWeight;   // 0..1
    private final double coherenceWeight;        // 0..1

    // --- strict verify knobs ---
    private final double strictMinCoherenceBoost; // +0.12
    private final double strictEntropyBoost;      // +0.10
    private final double strictRiskWeight;        // 0..1
    private final double baseRiskWeight;          // 0..1

    // --- debug knobs ---
    private final boolean debugEnabled;
    private final int debugSnippetChars;

    /**
     * IMPORTANT:
     * In current project BaseCandidateEvaluator does NOT have a (Tokenizer-only) ctor.
     * Provide a fully constructed BaseCandidateEvaluator to avoid hidden hardcode.
     */
    public MultiCriteriaCandidateEvaluator(BaseCandidateEvaluator base) {
        this(base, 0.45, 0.35, 0.40, 0.12, 0.10, 0.45, 0.30, false, 180);
    }

    public MultiCriteriaCandidateEvaluator(
            BaseCandidateEvaluator base,
            double minCoherence,
            double entropyPenaltyWeight,
            double coherenceWeight
    ) {
        this(base, minCoherence, entropyPenaltyWeight, coherenceWeight, 0.12, 0.10, 0.45, 0.30, false, 180);
    }

    public MultiCriteriaCandidateEvaluator(
            BaseCandidateEvaluator base,
            double minCoherence,
            double entropyPenaltyWeight,
            double coherenceWeight,
            boolean debugEnabled,
            int debugSnippetChars
    ) {
        this(base, minCoherence, entropyPenaltyWeight, coherenceWeight, 0.12, 0.10, 0.45, 0.30, debugEnabled, debugSnippetChars);
    }

    public MultiCriteriaCandidateEvaluator(
            BaseCandidateEvaluator base,
            double minCoherence,
            double entropyPenaltyWeight,
            double coherenceWeight,
            double strictMinCoherenceBoost,
            double strictEntropyBoost,
            double strictRiskWeight,
            double baseRiskWeight,
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
        this.baseRiskWeight = clamp01(baseRiskWeight);

        this.debugEnabled = debugEnabled;
        this.debugSnippetChars = Math.max(40, debugSnippetChars);
    }

    @Override
    public Evaluation evaluate(String userText, List<Statement> context, ThoughtState state, String draft) {
        boolean strict = isStrict(state);
        return evaluateInternal(userText, context, state, draft, strict);
    }

    /** Optional strict verify-pass entrypoint (engine may call explicitly). */
    public Evaluation evaluateStrict(String userText, List<Statement> context, ThoughtState state, String draft) {
        return evaluateInternal(userText, context, state, draft, true);
    }

    private Evaluation evaluateInternal(String userText, List<Statement> context, ThoughtState state, String draft, boolean strict) {
        final String q = (userText == null) ? "" : userText.trim();
        final String a = (draft == null) ? "" : draft.trim();
        final int ctxSize = (context == null) ? 0 : context.size();

        // ---- Base evaluation ----
        Evaluation baseEv;
        try {
            baseEv = base.evaluate(q, context, state, a);
        } catch (Exception ex) {
            log.warn("MultiCriteriaCandidateEvaluator: base.evaluate failed", ex);
            Evaluation z = Evaluation.invalid(Double.NEGATIVE_INFINITY, "err=base_eval_exception", "");
            z.valid = false;
            z.syncNotes();
            return z;
        }

        if (baseEv == null) {
            Evaluation z = Evaluation.invalid(Double.NEGATIVE_INFINITY, "err=null_eval", "");
            z.valid = false;
            z.syncNotes();
            return z;
        }
        baseEv.syncNotes();

        // ---- Channels ----
        double factual = clamp01(baseEv.groundedness);
        double structure = clamp01(baseEv.structureScore);

        // Prefer evaluator’s own qc if present (we added queryCoverage to Evaluation),
        // fall back to proxy if old evaluator didn't fill it.
        double coverage = (baseEv.queryCoverage > 0.0) ? clamp01(baseEv.queryCoverage) : clamp01(coverageProxy(q, a));

        // Actionability: proxy + structure
        double actionability = clamp01(actionabilityProxy(a, structure));

        double risk = clamp01(baseEv.contradictionRisk);

        double[] channels = {factual, structure, coverage, actionability};

        // ---- Entropy & coherence ----
        double entropy = entropy(channels);
        double coherence = coherence(channels);

        // strict adjustments
        double minCoh = strict ? clamp01(this.minCoherence + strictMinCoherenceBoost) : this.minCoherence;
        double entW = strict ? clamp01(this.entropyPenaltyWeight + strictEntropyBoost) : this.entropyPenaltyWeight;
        double riskW = strict ? this.strictRiskWeight : this.baseRiskWeight;

        // ---- Quantum-adjusted score ----
        // IMPORTANT:
        // - baseEv.score is already clamped to [0..1] in BaseCandidateEvaluator
        // - here we keep it stable but re-clamp final score into [0..1] for engine thresholds.
        double quantumScore =
                baseEv.score
                        + coherenceWeight * coherence
                        - entW * entropy
                        - riskW * risk;

        quantumScore = clamp01(quantumScore);

        boolean valid = baseEv.valid && coherence >= minCoh;

        // ---- Output: keep base metrics, add quantum telemetry ----
        Evaluation out = new Evaluation();
        out.score = quantumScore;
        out.valid = valid;

        out.groundedness = baseEv.groundedness;
        out.contradictionRisk = baseEv.contradictionRisk;
        out.structureScore = baseEv.structureScore;

        out.queryCoverage = coverage;
        out.coherence = coherence;
        out.repetition = baseEv.repetition;
        out.novelty = baseEv.novelty;

        // Keep generator-safe critique: only actionable short hint.
        if (!valid) {
            StringBuilder c = new StringBuilder(96);
            if (!baseEv.valid) c.append("base_fail ");
            if (coherence < minCoh) c.append("low_coherence ");
            if (risk > 0.60) c.append("high_risk ");
            out.critique = c.toString().trim();
        } else {
            out.critique = ""; // don't spam generator
        }

        out.validationNotes =
                safe(baseEv.validationNotes)
                        + ";q_f=" + fmt2(factual)
                        + ";q_s=" + fmt2(structure)
                        + ";q_qc=" + fmt2(coverage)
                        + ";q_act=" + fmt2(actionability)
                        + ";q_r=" + fmt2(risk)
                        + ";q_ent=" + fmt2(entropy)
                        + ";q_coh=" + fmt2(coherence)
                        + ";q_minCoh=" + fmt2(minCoh)
                        + ";q_entW=" + fmt2(entW)
                        + ";q_rW=" + fmt2(riskW)
                        + ";q_strict=" + (strict ? 1 : 0)
                        + ";q_v=" + (valid ? 1 : 0);

        out.syncNotes();

        if (log.isDebugEnabled()) {
            boolean flip = (baseEv.valid != valid);
            boolean lowC = coherence < minCoh;

            if (debugEnabled || flip || lowC || strict) {
                log.debug(
                        "QuantumEval | strict={} ctx={} baseScore={} qScore={} valid={} (baseValid={}) " +
                                "| g={} st={} qc={} act={} risk={} coh={} ent={} " +
                                "| q='{}' a='{}'",
                        strict ? 1 : 0,
                        ctxSize,
                        fmt4(baseEv.score),
                        fmt4(quantumScore),
                        valid,
                        baseEv.valid,
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
            log.debug("QuantumEvalTelemetry | {}", out.validationNotes);
        }

        return out;
    }

    // -------------------- strict detection --------------------

    private static boolean isStrict(ThoughtState state) {
        if (state == null) return false;

        Map<String, String> tags = state.tags;
        if (tags != null && !tags.isEmpty()) {
            if (isTrue(tags.get(Tags.VERIFY_STRICT))) return true;
            if (isTrue(tags.get(Tags.REPAIR_STRICT))) return true;
        }

        // fallback heuristic: verify/repair often happens later
        // (keep conservative to avoid surprising score shifts early)
        return state.iteration >= 2 && state.phase == CandidateControlSignals.Phase.REPAIR.ordinal();
    }

    private static boolean isTrue(String v) {
        if (v == null) return false;
        String x = v.trim();
        return "1".equals(x) || "true".equalsIgnoreCase(x) || "yes".equalsIgnoreCase(x);
    }

    // -------------------- proxies (deterministic, no tokenizer coupling) --------------------

    private static double coverageProxy(String q, String a) {
        if (q == null || q.isBlank() || a == null || a.isBlank()) return 0.0;

        Set<String> qs = cheapTokenSet(q);
        if (qs.isEmpty()) return 0.0;

        Set<String> as = cheapTokenSet(a);
        if (as.isEmpty()) return 0.0;

        int inter = 0;
        for (String t : qs) if (as.contains(t)) inter++;

        return clamp01((double) inter / (double) Math.max(1, qs.size()));
    }

    private static double actionabilityProxy(String a, double structureScore) {
        if (a == null || a.isBlank()) return 0.0;

        int bullets = countBulletLikeLines(a);
        double bulletScore = clamp01(bullets / 8.0);

        return clamp01(0.55 * bulletScore + 0.45 * clamp01(structureScore));
    }

    private static int countBulletLikeLines(String s) {
        int n = 0;
        int i = 0;
        while (i < s.length()) {
            int lineEnd = s.indexOf('\n', i);
            if (lineEnd < 0) lineEnd = s.length();
            String line = s.substring(i, lineEnd).trim();

            if (!line.isEmpty()) {
                char c0 = line.charAt(0);
                if (c0 == '-' || c0 == '*' || c0 == '•') n++;
                else if (startsWithNumberedBullet(line)) n++;
            }

            i = lineEnd + 1;
        }
        return n;
    }

    private static boolean startsWithNumberedBullet(String line) {
        int i = 0;
        int n = line.length();
        while (i < n && Character.isWhitespace(line.charAt(i))) i++;
        int j = i;
        while (j < n && Character.isDigit(line.charAt(j))) j++;
        if (j == i) return false;
        if (j >= n) return false;
        char c = line.charAt(j);
        return c == '.' || c == ')';
    }

    private static Set<String> cheapTokenSet(String s) {
        HashSet<String> set = new HashSet<>();
        StringBuilder tok = new StringBuilder(24);

        String x = s.toLowerCase(Locale.ROOT);
        for (int i = 0; i < x.length(); i++) {
            char c = x.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '_') {
                tok.append(c);
            } else {
                if (tok.length() >= 3) set.add(tok.toString());
                tok.setLength(0);
                if (set.size() > 256) break;
            }
        }
        if (tok.length() >= 3 && set.size() <= 256) set.add(tok.toString());
        return set;
    }

    // -------------------- quantum math --------------------

    private static double entropy(double[] v) {
        double sum = 0.0;
        for (double x : v) sum += x;
        if (sum <= 1e-9) return 1.0;

        double h = 0.0;
        for (double x : v) {
            double p = x / sum;
            if (p > 1e-9) h -= p * Math.log(p);
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

    // -------------------- utils --------------------

    private static String safe(String s) {
        return s == null ? "" : s;
    }

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