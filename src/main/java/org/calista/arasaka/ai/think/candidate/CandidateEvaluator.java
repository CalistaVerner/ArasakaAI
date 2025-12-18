package org.calista.arasaka.ai.think.candidate;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.ThoughtState;

import java.util.List;
import java.util.Locale;

/**
 * CandidateEvaluator — scores drafts for selection and produces critique signals.
 *
 * Enterprise contracts:
 * - Deterministic: for same (userText, context, state, draft) must return same Evaluation.
 * - validationNotes is telemetry only (NEVER feed to generator).
 * - critique/notes are generator-safe.
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

        // ---- core ranking / validity ----
        public double score;
        public boolean valid;

        // ---- risk dimensions ----
        public double groundedness;
        public double contradictionRisk;
        /** Preferred structural quality score (0..1). */
        public double structureScore;

        /**
         * Backward-compat alias.
         * Some legacy code referenced `ev.structure` (older field name).
         * Keep it in sync with {@link #structureScore}.
         */
        @Deprecated
        public double structure;

        // ---- generator-safe feedback ----
        /** Short generator-safe critique (preferred). */
        public String critique;

        /** Back-compat alias (engine/evaluators may still use e.notes). */
        public String notes;

        /** Verbose telemetry/debug string (NEVER feed to generator). */
        public String validationNotes;

        // ---- optional advanced metrics ----
        public double coherence;
        public double repetition;
        public double novelty;

        /** Coverage of query tokens by answer (used by BaseCandidateEvaluator). */
        public double queryCoverage;

        public Evaluation() {
            this.score = 0.0;
            this.valid = false;

            this.groundedness = 0.0;
            this.contradictionRisk = 0.0;
            this.structureScore = 0.0;
            this.structure = 0.0;

            this.critique = "";
            this.notes = "";
            this.validationNotes = "";

            this.coherence = 0.0;
            this.repetition = 0.0;
            this.novelty = 0.0;

            this.queryCoverage = 0.0;
        }

        /**
         * Keep critique and notes consistent.
         * Call at the end of evaluation computation.
         */
        public void syncNotes() {
            // Keep numeric aliases consistent for old callers.
            syncAliases();
            if (critique != null && !critique.isEmpty()) {
                notes = critique;
            } else if (notes != null && !notes.isEmpty()) {
                critique = notes;
            } else {
                critique = "";
                notes = "";
            }
        }

        /**
         * Synchronize legacy numeric aliases.
         * Rule: structureScore is the source of truth; if it is zero but legacy field is set,
         * adopt the legacy value (helps during gradual migrations).
         */
        public void syncAliases() {
            if (structureScore == 0.0 && structure != 0.0) {
                structureScore = structure;
            }
            structure = structureScore;
        }

        /**
         * Backward compatibility with older evaluators/engine code.
         * Previously some code called normalizeNotes(); now we unify with syncNotes().
         */
        public void normalizeNotes() {
            syncNotes();
        }

        /**
         * Mark evaluation as invalid with deterministic score and telemetry note.
         * Keeps notes/critique consistent and generator-safe.
         */
        public Evaluation invalidate(double score, String validationNotes, String critique) {
            this.valid = false;
            this.score = score;
            this.validationNotes = (validationNotes == null ? "" : validationNotes);
            this.critique = (critique == null ? "" : critique);
            // keep aliases deterministic
            syncAliases();
            syncNotes();
            return this;
        }

        /**
         * Convenience factory for invalid evaluations.
         */
        public static Evaluation invalid(double score, String validationNotes, String critique) {
            return new Evaluation().invalidate(score, validationNotes, critique);
        }

        /**
         * Convenience factory for valid evaluations.
         */
        public static Evaluation ok(
                double score,
                double groundedness,
                double contradictionRisk,
                double structureScore,
                double queryCoverage,
                double novelty,
                double repetition,
                double coherence,
                String critique,
                String validationNotes
        ) {
            Evaluation e = new Evaluation();
            e.valid = true;

            e.score = score;
            e.groundedness = groundedness;
            e.contradictionRisk = contradictionRisk;
            e.structureScore = structureScore;
            e.structure = structureScore; // legacy alias

            e.queryCoverage = queryCoverage;
            e.novelty = novelty;
            e.repetition = repetition;
            e.coherence = coherence;

            e.critique = (critique == null ? "" : critique);
            e.validationNotes = (validationNotes == null ? "" : validationNotes);

            e.syncNotes(); // also syncs aliases
            return e;
        }

        /** Hook for future weighting without breaking old callers. */
        public double effectiveScore() {
            return score;
        }

        /** Defensive formatting helper for logs/telemetry (optional). */
        public String brief() {
            return "ok=" + valid
                    + " score=" + String.format(Locale.ROOT, "%.4f", score)
                    + " g=" + String.format(Locale.ROOT, "%.4f", groundedness)
                    + " risk=" + String.format(Locale.ROOT, "%.4f", contradictionRisk)
                    + " qc=" + String.format(Locale.ROOT, "%.4f", queryCoverage)
                    + " nov=" + String.format(Locale.ROOT, "%.4f", novelty)
                    + " rep=" + String.format(Locale.ROOT, "%.4f", repetition)
                    + " st=" + String.format(Locale.ROOT, "%.4f", structureScore);
        }
    }
}