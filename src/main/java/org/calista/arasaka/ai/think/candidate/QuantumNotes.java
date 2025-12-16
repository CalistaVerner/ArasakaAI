package org.calista.arasaka.ai.think.candidate;

import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;

import java.util.Locale;

public final class QuantumNotes {
    private QuantumNotes() {}

    /** Compact, retrieval-safe notes for refinement (NO telemetry keys). */
    public static String compact(CandidateEvaluator.Evaluation ev) {
        if (ev == null) return "";
        // только “сигналы”, без md_on/qc/nov/g_topk/... и без длинных строк
        StringBuilder b = new StringBuilder(64);

        double g = clamp01(ev.groundedness);
        double r = clamp01(ev.contradictionRisk);
        double st = clamp01(ev.structureScore);

        if (!ev.valid) b.append("invalid ");
        if (g < 0.35) b.append("more_evidence ");
        if (r > 0.65) b.append("reduce_claims ");
        if (st < 0.35) b.append("improve_structure ");

        String s = b.toString().trim();
        if (s.length() > 80) s = s.substring(0, 80).trim();
        return s;
    }

    private static double clamp01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    public static String fmt(double v) {
        return String.format(Locale.ROOT, "%.2f", v);
    }
}