package org.calista.arasaka.ai.think.candidate;

import java.util.Objects;

/**
 * A single candidate answer produced during an iteration.
 *
 * <p>Stores both the raw text and the full {@link CandidateEvaluator.Evaluation} telemetry
 * so the engine and strategies can self-correct deterministically.</p>
 */
public final class Candidate {

    public final String text;

    /**
     * Primary ordering score. Typically use {@link CandidateEvaluator.Evaluation#effectiveScore()}.
     */
    public final double score;

    /**
     * Machine-readable critique/notes (backward compatible field).
     */
    public final String critique;

    /**
     * Full evaluation telemetry (may be {@code null} for legacy callers).
     */
    public final CandidateEvaluator.Evaluation evaluation;

    /**
     * Legacy constructor.
     */
    public Candidate(String text, double score, String critique) {
        this(text, score, critique, null);
    }

    public Candidate(String text, double score, String critique, CandidateEvaluator.Evaluation evaluation) {
        this.text = text == null ? "" : text;
        this.score = score;
        this.critique = critique == null ? "" : critique;
        this.evaluation = evaluation;
    }

    public static Candidate fromEvaluation(String text, CandidateEvaluator.Evaluation e) {
        Objects.requireNonNull(e, "evaluation");
        return new Candidate(text, e.effectiveScore(), e.validationNotes, e);
    }
}