package org.calista.arasaka.ai.think.candidate;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.ThoughtState;

import java.util.List;

/**
 * CandidateEvaluator — scores drafts for selection and produces critique signals.
 */
public interface CandidateEvaluator {

    Evaluation evaluate(String userText, List<Statement> context, ThoughtState state, String draft);

    /**
     * Evaluation result for a single draft.
     *
     * Important fields:
     * - score: main numeric score used for ranking
     * - valid: structural/grounded validity gate
     * - groundedness / contradictionRisk / structureScore: dimensions for risk-driven verify
     * - critique: short, generator-safe feedback
     * - notes: alias for critique (backward compatibility with engine code)
     * - validationNotes: verbose telemetry (never feed to generator)
     */
    final class Evaluation {

        public double score;
        public boolean valid;

        public double groundedness;
        public double contradictionRisk;
        public double structureScore;

        /** Short generator-safe critique (preferred). */
        public String critique;

        /** Back-compat alias (engine may use e.notes). */
        public String notes;

        /** Verbose telemetry/debug string (NEVER feed to generator). */
        public String validationNotes;

        // optional advanced metrics
        public double coherence;
        public double repetition;
        public double novelty;

        public Evaluation() {
            this.score = 0.0;
            this.valid = false;

            this.groundedness = 0.0;
            this.contradictionRisk = 0.0;
            this.structureScore = 0.0;

            this.critique = "";
            this.notes = "";
            this.validationNotes = "";

            this.coherence = 0.0;
            this.repetition = 0.0;
            this.novelty = 0.0;
        }

        /**
         * Keep critique and notes consistent.
         * Call at the end of evaluation computation.
         */
        public void syncNotes() {
            if (critique != null && !critique.isEmpty()) {
                notes = critique;
            } else if (notes != null && !notes.isEmpty()) {
                critique = notes;
            } else {
                critique = "";
                notes = "";
            }
        }

        /** Hook for future weighting without breaking old callers. */
        public double effectiveScore() {
            return score;
        }
    }
}