package org.calista.arasaka.ai.think.candidate.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;

import java.util.*;

/**
 * MultiCriteriaCandidateEvaluator (Quantum-like):
 * - multi-channel scoring on top of BaseCandidateEvaluator
 * - coherence & entropy penalties
 * - deterministic (no randomness)
 *
 * Strict verify-pass mode:
 * - can be enabled by setting state.tags (engine-driven), or by calling evaluateStrict(...)
 *
 * NOTE:
 * - Must conform to CandidateEvaluator.evaluate(userText, context, state, draft)
 * - Uses only fields present in CandidateEvaluator.Evaluation (current project)
 */
public final class MultiCriteriaCandidateEvaluator implements CandidateEvaluator {

    private static final Logger log = LogManager.getLogger(MultiCriteriaCandidateEvaluator.class);

    private final BaseCandidateEvaluator base;

    // --- quantum policy knobs ---
    private final double minCoherence;        // 0..1
    private final double entropyPenaltyWeight;
    private final double coherenceWeight;

    // --- strict verify knobs ---
    private final double strictMinCoherenceBoost;     // +0.12
    private final double strictEntropyBoost;          // +0.10
    private final double strictRiskWeight;            // 0.45 (vs 0.30)
    private final double baseRiskWeight;              // default risk weight in non-strict

    // --- debug knobs ---
    private final boolean debugEnabled;
    private final int debugSnippetChars;

    public MultiCriteriaCandidateEvaluator(org.calista.arasaka.ai.tokenizer.Tokenizer tokenizer) {
        this(
                new BaseCandidateEvaluator(tokenizer),
                0.45,
                0.35,
                0.40,
                0.12,
                0.10,
                0.45,
                0.30,
                false,
                180
        );
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
        // Strict mode can be injected by engine via tags, but to avoid coupling
        // we use a conservative rule: strict if state != null and state.phase suggests repair/verify.
        boolean strict = isStrict(state);
        return evaluateInternal(userText, context, state, draft, strict);
    }

    /**
     * Optional strict verify-pass entrypoint (engine may call explicitly).
     * Does NOT change interface; just a convenience method.
     */
    public Evaluation evaluateStrict(String userText, List<Statement> context, ThoughtState state, String draft) {
        return evaluateInternal(userText, context, state, draft, true);
    }

    private Evaluation evaluateInternal(String userText, List<Statement> context, ThoughtState state, String draft, boolean strict) {
        final long t0 = System.nanoTime();

        final String q = userText == null ? "" : userText.trim();
        final String a = draft == null ? "" : draft.trim();
        final int ctxSize = (context == null) ? 0 : context.size();

        // ---- Base (classical) evaluation ----
        Evaluation e = base.evaluate(q, context, state, a);
        if (e == null) {
            Evaluation z = new Evaluation();
            z.score = -1.0;
            z.valid = false;
            z.critique = "err=null_eval";
            z.validationNotes = "err=null_eval";
            z.syncNotes();
            return z;
        }

        // ---- Quantum channels (superposition) ----
        double factual = clamp01(e.groundedness);
        double structure = clamp01(e.structureScore);

        // coverage: cheap lexical overlap proxy (deterministic)
        double coverage = clamp01(coverageProxy(q, a));

        // actionability: proxies + structure (deterministic)
        double actionability = clamp01(actionabilityProxy(a, structure));

        double risk = clamp01(e.contradictionRisk);

        double[] channels = { factual, structure, coverage, actionability };

        // ---- Entropy & coherence ----
        double entropy = entropy(channels);
        double coherence = coherence(channels);

        // strict adjustments
        double minCoh = strict ? clamp01(this.minCoherence + strictMinCoherenceBoost) : this.minCoherence;
        double entW = strict ? clamp01(this.entropyPenaltyWeight + strictEntropyBoost) : this.entropyPenaltyWeight;
        double riskW = strict ? this.strictRiskWeight : this.baseRiskWeight;

        // ---- Quantum-adjusted score ----
        double quantumScore =
                e.score
                        + coherenceWeight * coherence
                        - entW * entropy
                        - riskW * risk;

        boolean valid = e.valid && coherence >= minCoh;

        // ---- Output: keep base metrics, add quantum telemetry ----
        Evaluation out = new Evaluation();
        out.score = quantumScore;
        out.valid = valid;

        out.groundedness = e.groundedness;
        out.contradictionRisk = e.contradictionRisk;
        out.structureScore = e.structureScore;

        // keep base aux metrics if base computed them
        out.coherence = coherence;
        out.repetition = e.repetition;
        out.novelty = e.novelty;

        String telemetry =
                safe(e.validationNotes)
                        + ";q_f=" + fmt2(factual)
                        + ";q_s=" + fmt2(structure)
                        + ";q_cov=" + fmt2(coverage)
                        + ";q_act=" + fmt2(actionability)
                        + ";q_r=" + fmt2(risk)
                        + ";q_ent=" + fmt2(entropy)
                        + ";q_coh=" + fmt2(coherence)
                        + ";q_minCoh=" + fmt2(minCoh)
                        + ";q_entW=" + fmt2(entW)
                        + ";q_rW=" + fmt2(riskW)
                        + ";q_strict=" + (strict ? 1 : 0)
                        + ";q_v=" + (valid ? 1 : 0);

        // critique: keep it short & generator-safe
        out.critique = "quantum"
                + ";coh=" + fmt2(coherence)
                + ";ent=" + fmt2(entropy)
                + ";risk=" + fmt2(risk)
                + ";v=" + (valid ? 1 : 0);

        out.validationNotes = telemetry;
        out.syncNotes();

        if (log.isDebugEnabled()) {
            boolean flip = (e.valid != valid);
            boolean lowC = coherence < minCoh;

            if (debugEnabled || flip || lowC || strict) {
                log.debug(
                        "QuantumEval | strict={} ctx={} baseScore={} qScore={} valid={} (baseValid={}) " +
                                "| g={} st={} cov={} act={} risk={} coh={} ent={} " +
                                "| q='{}' a='{}'",
                        strict ? 1 : 0,
                        ctxSize,
                        fmt4(e.score),
                        fmt4(quantumScore),
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

    // -------------------- strict detection --------------------

    private static boolean isStrict(ThoughtState state) {
        if (state == null) return false;

        // If engine uses phase ordinals, "repair/verify" typically has higher values.
        // Keep it conservative: strict only when phase is non-zero AND tags indicate repair intent.
        if (state.tags != null) {
            String v = state.tags.get("verify.strict");
            if ("1".equals(v) || "true".equalsIgnoreCase(v) || "yes".equalsIgnoreCase(v)) return true;
            v = state.tags.get("repair.strict");
            if ("1".equals(v) || "true".equalsIgnoreCase(v)) return true;
        }

        // fallback heuristic: later iterations are more likely verify-pass
        return state.iteration >= 2 && state.phase != 0;
    }

    // -------------------- proxies (deterministic, no tokenizer coupling) --------------------

    private static double coverageProxy(String q, String a) {
        if (q == null || q.isBlank() || a == null || a.isBlank()) return 0.0;

        // cheap char-level overlap proxy: intersection of lowercased "wordish" tokens length>=3
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

        // slight bias to structured output
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
