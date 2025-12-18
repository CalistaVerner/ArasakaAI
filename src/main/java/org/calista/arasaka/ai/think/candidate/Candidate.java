package org.calista.arasaka.ai.think.candidate;

/**
 * Candidate — mutable container for one generated draft inside think-loop.
 *
 * Engine expects this object to be mutable:
 * - seed/iteration/draftIndex are filled by engine
 * - evaluation/score/critique are filled after evaluator
 */
public final class Candidate {

    /** Original normalized user query (for debugging/telemetry). */
    public final String query;

    /** Draft text (never null). */
    public String text;

    // engine meta
    public int iteration;
    public long seed;
    public int draftIndex;

    // evaluation result
    public CandidateEvaluator.Evaluation evaluation;
    public double score;
    public String critique;

    public Candidate(String query, String text) {
        this.query = query == null ? "" : query;
        this.text = text == null ? "" : text;

        this.iteration = 0;
        this.seed = 0L;
        this.draftIndex = 0;

        this.evaluation = null;
        this.score = 0.0;
        this.critique = "";
    }

    public static Candidate empty(String query) {
        Candidate c = new Candidate(query, "");
        c.score = Double.NEGATIVE_INFINITY;
        c.critique = "";
        c.evaluation = null;
        return c;
    }

    @Override
    public String toString() {
        return "Candidate{score=" + score + ", it=" + iteration + ", d=" + draftIndex + "}";
    }
}