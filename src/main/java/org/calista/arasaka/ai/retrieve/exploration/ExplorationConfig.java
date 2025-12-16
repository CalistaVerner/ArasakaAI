package org.calista.arasaka.ai.retrieve.exploration;

/**
 * Retrieval/exploration configuration for the multi-iteration "thinking" pipeline.
 *
 * The config is intentionally data-only (no hidden heuristics).
 * All knobs are explicit and can be learned/tuned from logs.
 */
public final class ExplorationConfig {
    public final double temperature;
    public final int topK;
    public final int iterations;
    public final int candidateMultiplier;
    public final double diversity;
    public final double minScore;
    public final double iterationDecay;

    public final int refineTerms;
    public final int candidateGateMinTokenLen;
    public final int maxCandidatesPerIter;

    public final double qualityFloor;

    /** Early stop if confidence rises above this value (0 disables). */
    public final double earlyStopConfidence;

    /** Enable parallel scoring (deterministic). */
    public final boolean parallel;

    /** If parallel=true and this > 0, retriever uses this parallelism. */
    public final int parallelism;


    public ExplorationConfig(
            double temperature,
            int topK,
            int iterations,
            int candidateMultiplier,
            double diversity,
            double minScore,
            double iterationDecay,
            int refineTerms,
            int candidateGateMinTokenLen,
            int maxCandidatesPerIter,
            double qualityFloor,
            double earlyStopConfidence,
            boolean parallel,
            int parallelism
    ) {
        if (temperature <= 0.0) throw new IllegalArgumentException("temperature must be > 0");
        if (topK < 1) throw new IllegalArgumentException("topK must be >= 1");
        if (iterations < 1) throw new IllegalArgumentException("iterations must be >= 1");
        if (candidateMultiplier < 1) throw new IllegalArgumentException("candidateMultiplier must be >= 1");
        if (!(diversity >= 0.0 && diversity <= 1.0)) throw new IllegalArgumentException("diversity must be in [0..1]");
        if (!Double.isFinite(minScore) || minScore < 0.0) throw new IllegalArgumentException("minScore must be finite and >= 0");
        if (!(iterationDecay > 0.0 && iterationDecay <= 1.0)) throw new IllegalArgumentException("iterationDecay must be in (0..1]");
        if (refineTerms < 0) throw new IllegalArgumentException("refineTerms must be >= 0");
        if (candidateGateMinTokenLen < 1) throw new IllegalArgumentException("candidateGateMinTokenLen must be >= 1");
        if (maxCandidatesPerIter < 1) throw new IllegalArgumentException("maxCandidatesPerIter must be >= 1");
        if (!Double.isFinite(qualityFloor) || qualityFloor < 0.0) throw new IllegalArgumentException("qualityFloor must be finite and >= 0");
        if (!Double.isFinite(earlyStopConfidence) || earlyStopConfidence < 0.0) throw new IllegalArgumentException("earlyStopConfidence must be finite and >= 0");
        if (parallelism < 0) throw new IllegalArgumentException("parallelism must be >= 0");

        this.temperature = temperature;
        this.topK = topK;
        this.iterations = iterations;
        this.candidateMultiplier = candidateMultiplier;
        this.diversity = diversity;
        this.minScore = minScore;
        this.iterationDecay = iterationDecay;

        this.refineTerms = refineTerms;
        this.candidateGateMinTokenLen = candidateGateMinTokenLen;
        this.maxCandidatesPerIter = maxCandidatesPerIter;
        this.qualityFloor = qualityFloor;

        this.earlyStopConfidence = earlyStopConfidence;
        this.parallel = parallel;
        this.parallelism = parallelism;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private double temperature = 0.9;
        private int topK = 64;
        private int iterations = 2;
        private int candidateMultiplier = 4;
        private double diversity = 0.15;
        private double minScore = 0.0;
        private double iterationDecay = 0.8;

        private int refineTerms = 12;
        private int candidateGateMinTokenLen = 3;
        private int maxCandidatesPerIter = 50_000;
        private double qualityFloor = 0.0;

        private double earlyStopConfidence = 0.0;
        private boolean parallel = false;
        private int parallelism = 0;

        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder iterations(int v) { this.iterations = v; return this; }
        public Builder candidateMultiplier(int v) { this.candidateMultiplier = v; return this; }
        public Builder diversity(double v) { this.diversity = v; return this; }
        public Builder minScore(double v) { this.minScore = v; return this; }
        public Builder iterationDecay(double v) { this.iterationDecay = v; return this; }

        public Builder refineTerms(int v) { this.refineTerms = v; return this; }
        public Builder candidateGateMinTokenLen(int v) { this.candidateGateMinTokenLen = v; return this; }
        public Builder maxCandidatesPerIter(int v) { this.maxCandidatesPerIter = v; return this; }
        public Builder qualityFloor(double v) { this.qualityFloor = v; return this; }

        public Builder earlyStopConfidence(double v) { this.earlyStopConfidence = v; return this; }
        public Builder parallel(boolean v) { this.parallel = v; return this; }
        public Builder parallelism(int v) { this.parallelism = v; return this; }

        public ExplorationConfig build() {
            return new ExplorationConfig(
                    temperature, topK, iterations, candidateMultiplier, diversity, minScore, iterationDecay,
                    refineTerms, candidateGateMinTokenLen, maxCandidatesPerIter, qualityFloor,
                    earlyStopConfidence, parallel, parallelism
            );
        }
    }
}