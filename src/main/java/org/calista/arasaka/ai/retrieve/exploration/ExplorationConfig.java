package org.calista.arasaka.ai.retrieve.exploration;

public final class ExplorationConfig {
    /**
     * Temperature used for softmax-style score smoothing.
     * Lower -> greedier (closer to pure top-k), higher -> flatter.
     */
    public final double temperature;

    /**
     * Upper bound for the candidate pool considered by the exploration strategy.
     * Note: the strategy may internally expand this pool via {@link #candidateMultiplier}.
     */
    public final int topK;

    /**
     * How many reasoning/retrieval iterations to run ("thinking cycles").
     * Each iteration may refine the query and re-rank candidates.
     */
    public final int iterations;

    /**
     * Candidate pool expansion factor. Example: topK=64, multiplier=4 => consider up to 256 candidates.
     * Higher improves recall but increases compute.
     */
    public final int candidateMultiplier;

    /**
     * Diversity strength in [0..1]. 0 -> no diversity penalty (pure score), 1 -> strong novelty preference.
     */
    public final double diversity;

    /**
     * Hard floor for accepting a scored candidate (after scoring, before exploration).
     */
    public final double minScore;

    /**
     * Iteration decay in (0..1]. 1 -> equal weight, lower -> earlier iterations dominate.
     */
    public final double iterationDecay;

    public ExplorationConfig(double temperature, int topK) {
        this(temperature, topK, 2, 4, 0.15, 1e-9, 0.75);
    }

    public ExplorationConfig(
            double temperature,
            int topK,
            int iterations,
            int candidateMultiplier,
            double diversity,
            double minScore,
            double iterationDecay
    ) {
        if (temperature <= 0.0) throw new IllegalArgumentException("temperature must be > 0");
        if (topK < 1) throw new IllegalArgumentException("topK must be >= 1");
        if (iterations < 1) throw new IllegalArgumentException("iterations must be >= 1");
        if (candidateMultiplier < 1) throw new IllegalArgumentException("candidateMultiplier must be >= 1");
        if (!(diversity >= 0.0 && diversity <= 1.0)) throw new IllegalArgumentException("diversity must be in [0..1]");
        if (!Double.isFinite(minScore) || minScore < 0.0) throw new IllegalArgumentException("minScore must be finite and >= 0");
        if (!(iterationDecay > 0.0 && iterationDecay <= 1.0)) throw new IllegalArgumentException("iterationDecay must be in (0..1]");

        this.temperature = temperature;
        this.topK = topK;
        this.iterations = iterations;
        this.candidateMultiplier = candidateMultiplier;
        this.diversity = diversity;
        this.minScore = minScore;
        this.iterationDecay = iterationDecay;
    }
}