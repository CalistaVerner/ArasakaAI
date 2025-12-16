package org.calista.arasaka.ai.think.candidate;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.List;

public interface CandidateEvaluator {

    /**
     * Full evaluation entry point.
     * Implementations may override this for advanced logic.
     */
    default Evaluation evaluate(String userText, String candidateText, List<Statement> context) {
        double s = score(userText, candidateText, context);
        return Evaluation.basic(s);
    }

    /**
     * Minimal scoring function (legacy-compatible).
     */
    double score(String userText, String candidateText, List<Statement> context);

    /**
     * Evaluation result – immutable telemetry object.
     */
    final class Evaluation {

        // --- legacy fields ---
        public final double score;
        public final String critique;
        public final double coverage;
        public final double contextSupport;
        public final double stylePenalty;

        // --- extended reasoning fields ---
        public final double groundedness;        // how much candidate is supported by context (0..1)
        public final double contradictionRisk;   // likelihood of unsupported assertions (0..1)
        public final boolean valid;              // structural + semantic validity
        public final String validationNotes;     // machine-readable diagnostics

        /**
         * Full constructor (enterprise / reasoning engines).
         */
        public Evaluation(
                double score,
                String critique,
                double coverage,
                double contextSupport,
                double stylePenalty,
                double groundedness,
                double contradictionRisk,
                boolean valid,
                String validationNotes
        ) {
            this.score = score;
            this.critique = critique == null ? "" : critique;
            this.coverage = coverage;
            this.contextSupport = contextSupport;
            this.stylePenalty = stylePenalty;
            this.groundedness = groundedness;
            this.contradictionRisk = contradictionRisk;
            this.valid = valid;
            this.validationNotes = validationNotes == null ? "" : validationNotes;
        }

        /**
         * Legacy-compatible minimal evaluation.
         */
        public static Evaluation basic(double score) {
            return new Evaluation(
                    score,
                    "",
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    true,
                    ""
            );
        }

        /**
         * Utility: penalize invalid candidates without destroying ordering.
         */
        public double effectiveScore() {
            if (!valid) {
                return score - 1.0 - contradictionRisk;
            }
            return score;
        }

        @Override
        public String toString() {
            return "Evaluation{" +
                    "score=" + score +
                    ", groundedness=" + groundedness +
                    ", contradictionRisk=" + contradictionRisk +
                    ", valid=" + valid +
                    '}';
        }
    }
}