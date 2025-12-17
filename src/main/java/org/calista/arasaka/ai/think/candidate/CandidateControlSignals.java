package org.calista.arasaka.ai.think.candidate;

import java.util.Locale;

public final class CandidateControlSignals {
    private CandidateControlSignals() {}

    /**
     * Compact, retrieval-safe numeric control signals for refinement.
     *
     * Format: key=value;key=value;...
     * Keys are generic (no domain semantics). Values are clamped.
     *
     * IMPORTANT:
     *  - never include free-form text (only numeric signals)
     *  - never include user text or context text
     *  - safe to store in Candidate.critique and feed into generator hint
     *  - engine query filters MUST block patterns like "g=0.42;" to prevent retriever poisoning
     */
    public static String compact(CandidateEvaluator.Evaluation ev) {
        if (ev == null) return "";

        // minimal yet useful control plane (all 0..1 except tok)
        double g = clamp01(ev.groundedness);
        double r = clamp01(ev.contradictionRisk);
        double st = clamp01(ev.structureScore);

        double cov = clamp01(ev.coverage);
        double cs = clamp01(ev.contextSupport);
        double sp = clamp01(ev.stylePenalty);

        int v = ev.valid ? 1 : 0;
        int tok = Math.max(0, ev.tokens);

        // deterministic order
        StringBuilder b = new StringBuilder(96);
        b.append("v=").append(v).append(';');
        b.append("g=").append(fmt2(g)).append(';');
        b.append("r=").append(fmt2(r)).append(';');
        b.append("st=").append(fmt2(st)).append(';');
        b.append("cov=").append(fmt2(cov)).append(';');
        b.append("cs=").append(fmt2(cs)).append(';');
        b.append("sp=").append(fmt2(sp)).append(';');
        if (tok > 0) b.append("tok=").append(tok).append(';');

        String s = b.toString();
        if (s.length() > 120) s = s.substring(0, 120);
        return s.replace(" ", "");
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