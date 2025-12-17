package org.calista.arasaka.ai.think.candidate;

public final class Candidate {
    public final String text;
    public final double score;

    /** Short machine critique for refinement (NO telemetry). */
    public final String critique;

    /** Full evaluation (may contain telemetry). */
    public final CandidateEvaluator.Evaluation evaluation;

    /** Optional diagnostics for logs; must NOT be used for retrieval/generation. */
    public final String diagnostics;

    public Candidate(String text, double score, String critique, CandidateEvaluator.Evaluation evaluation) {
        this(text, score, critique, evaluation, "");
    }

    public Candidate(String text, double score, String critique, CandidateEvaluator.Evaluation evaluation, String diagnostics) {
        this.text = text == null ? "" : text;
        this.score = score;
        this.critique = critique == null ? "" : critique;
        this.evaluation = evaluation;
        this.diagnostics = diagnostics == null ? "" : diagnostics;
    }

    public static Candidate fromEvaluation(String text, CandidateEvaluator.Evaluation ev) {
        String shortCrit = CandidateControlSignals.compact(ev);         // <<< ВАЖНО
        String diag = (ev == null) ? "" : ev.validationNotes; // telemetry остаётся здесь
        double s = (ev == null) ? Double.NEGATIVE_INFINITY : ev.effectiveScore();
        return new Candidate(text, s, shortCrit, ev, diag);
    }

    @Override
    public String toString() {
        return "Candidate{score=" + score + ", critique='" + critique + "'}";
    }
}