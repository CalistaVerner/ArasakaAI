package org.calista.arasaka.ai.think;

/**
 * Mutable state passed between iterations of the thought engine.
 * Keep it small and focused: it should enable refinement, not become a dumping ground.
 */
public final class ThoughtState {
    /** Current iteration index (1..N). Engine should set before calling strategies. */
    public int iteration = 0;

    /** Detected intent for the current user input. Engine sets once per think() call. */
    public Intent intent = Intent.UNKNOWN;

    /** Retrieval query for the current iteration. */
    public String query = "";

    /** Hint for generation (e.g., concise/balanced/detailed). */
    public String generationHint = "";

    /** Last evaluator critique for the previously generated candidate. */
    public String lastCritique = "";

    /** Best candidate seen so far across iterations. */
    public Candidate bestSoFar = null;

    /**
     * Optional generation backend (BeamSearch/TensorFlow/etc.).
     * If null, strategies may fall back to deterministic, template-based generation.
     */
    public TextGenerator generator = null;
}
