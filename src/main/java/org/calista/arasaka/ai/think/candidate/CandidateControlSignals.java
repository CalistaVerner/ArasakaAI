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


    // -------------------- structured planning (engine -> generator) --------------------

    /**
     * Phase of the thinking loop. Encoded into generationHint (retrieval-safe).
     */
    public enum Phase {
        EXPLORE,   // widen search, more drafts, more diversity
        EXPLOIT,   // tighten, improve bestSoFar
        VERIFY,    // strict verification pass
        REPAIR     // focused fix based on critique
    }

    /**
     * Deterministic diversity level (NOT randomness).
     */
    public enum Diversity {
        LOW, MEDIUM, HIGH
    }

    /**
     * Retrieval-safe plan (no free-form text). Intended to be embedded into ThoughtState.generationHint.
     *
     * <p>Encoding is compact and parseable:
     * phase/diversity are encoded as small ints, booleans as 0/1.</p>
     */
    public static final class Plan {
        public final Phase phase;
        public final Diversity diversity;
        public final long seed;
        public final int drafts;
        public final int beamWidth;
        public final int maxTokens;

        /** 0..1 how hard we require grounding on evidence. */
        public final double evidenceStrictness;

        /** 1 => enforce sectioned answer according to response.sections. */
        public final boolean requireSections;

        /** 1 => discourage generic filler. */
        public final boolean forbidGeneric;

        private Plan(Phase phase,
                     Diversity diversity,
                     long seed,
                     int drafts,
                     int beamWidth,
                     int maxTokens,
                     double evidenceStrictness,
                     boolean requireSections,
                     boolean forbidGeneric) {
            this.phase = phase == null ? Phase.EXPLORE : phase;
            this.diversity = diversity == null ? Diversity.MEDIUM : diversity;
            this.seed = seed;
            this.drafts = Math.max(1, drafts);
            this.beamWidth = Math.max(1, beamWidth);
            this.maxTokens = Math.max(16, maxTokens);
            this.evidenceStrictness = clamp01(evidenceStrictness);
            this.requireSections = requireSections;
            this.forbidGeneric = forbidGeneric;
        }

        public static Plan of(Phase phase,
                              Diversity diversity,
                              long seed,
                              int drafts,
                              int beamWidth,
                              int maxTokens,
                              double evidenceStrictness,
                              boolean requireSections,
                              boolean forbidGeneric) {
            return new Plan(phase, diversity, seed, drafts, beamWidth, maxTokens, evidenceStrictness, requireSections, forbidGeneric);
        }

        /**
         * Embed into generationHint. Retrieval-safe (numbers only).
         */
        public String toHintString() {
            // NOTE: keep it numeric-only; parsers can map ints back to enums if needed.
            int p = phase.ordinal();
            int d = diversity.ordinal();
            return "phase=" + p +
                    ";div=" + d +
                    ";seed=" + seed +
                    ";drafts=" + drafts +
                    ";beam=" + beamWidth +
                    ";maxTok=" + maxTokens +
                    ";evs=" + fmt2(evidenceStrictness) +
                    ";reqSec=" + (requireSections ? 1 : 0) +
                    ";noGen=" + (forbidGeneric ? 1 : 0);
        }
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