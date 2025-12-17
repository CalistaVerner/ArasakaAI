package org.calista.arasaka.ai.retrieve.exploration;

/**
 * Retrieval/exploration configuration for the multi-iteration "thinking" pipeline.
 *
 * The config is intentionally data-only (no hidden heuristics).
 * All knobs are explicit and can be learned/tuned from logs.
 */
public final class ExplorationConfig {

    // --- sampling / selection ---
    public final double temperature;
    public final int topK;
    public final int iterations;
    public final int candidateMultiplier;
    public final double diversity;
    public final double minScore;
    public final double iterationDecay;

    // --- query refine / candidate gating ---
    public final int refineTerms;
    public final int candidateGateMinTokenLen;
    public final int maxCandidatesPerIter;

    // --- quality gates ---
    public final double qualityFloor;

    /** Compute query token coverage over top-K ranked items. 0 disables coverage calculation. */
    public final int coverageK;

    /** Penalty weight for detected contradictions across selected evidence. 0 disables contradiction checks. */
    public final double contradictionPenalty;

    /** Early stop if confidence rises above this value (0 disables). */
    public final double earlyStopConfidence;

    // --- performance ---
    /** Enable parallel scoring (deterministic). */
    public final boolean parallel;

    /** If parallel=true and this > 0, retriever uses this parallelism. */
    public final int parallelism;

    // --- production RAG triple: retrieve -> rerank -> compress ---
    /** Rerank only top-N (cheap wide retrieval, expensive precision rerank). 0 disables rerank stage. */
    public final int rerankN;

    /** After rerank, keep top-M for selection (recommended 12–20). 0 means "keep as is". */
    public final int rerankM;

    /** How many best sentences to keep per Statement during compression. 0 disables compression. */
    public final int compressSentencesPerStatement;

    /** Hard cap for compressed text per Statement. 0 means "no cap". */
    public final int compressMaxCharsPerStatement;

    /** Drop candidate tokens in refine if they appear in >= this fraction of top-docs (water). */
    public final double refineDfCut;

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
            int coverageK,
            double contradictionPenalty,
            double earlyStopConfidence,
            boolean parallel,
            int parallelism,
            int rerankN,
            int rerankM,
            int compressSentencesPerStatement,
            int compressMaxCharsPerStatement,
            double refineDfCut
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
        if (coverageK < 0) throw new IllegalArgumentException("coverageK must be >= 0");
        if (!Double.isFinite(contradictionPenalty) || contradictionPenalty < 0.0) {
            throw new IllegalArgumentException("contradictionPenalty must be finite and >= 0");
        }
        if (!Double.isFinite(earlyStopConfidence) || earlyStopConfidence < 0.0) {
            throw new IllegalArgumentException("earlyStopConfidence must be finite and >= 0");
        }
        if (parallelism < 0) throw new IllegalArgumentException("parallelism must be >= 0");

        if (rerankN < 0) throw new IllegalArgumentException("rerankN must be >= 0");
        if (rerankM < 0) throw new IllegalArgumentException("rerankM must be >= 0");
        if (compressSentencesPerStatement < 0) throw new IllegalArgumentException("compressSentencesPerStatement must be >= 0");
        if (compressMaxCharsPerStatement < 0) throw new IllegalArgumentException("compressMaxCharsPerStatement must be >= 0");
        if (!Double.isFinite(refineDfCut) || refineDfCut < 0.0 || refineDfCut > 1.0) {
            throw new IllegalArgumentException("refineDfCut must be in [0..1]");
        }

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
        this.coverageK = coverageK;
        this.contradictionPenalty = contradictionPenalty;
        this.earlyStopConfidence = earlyStopConfidence;

        this.parallel = parallel;
        this.parallelism = parallelism;

        this.rerankN = rerankN;
        this.rerankM = rerankM;
        this.compressSentencesPerStatement = compressSentencesPerStatement;
        this.compressMaxCharsPerStatement = compressMaxCharsPerStatement;

        this.refineDfCut = refineDfCut;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        // Existing defaults
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
        private int coverageK = 32;
        private double contradictionPenalty = 0.35;

        private double earlyStopConfidence = 0.0;
        private boolean parallel = false;
        private int parallelism = 0;

        // Triple RAG defaults
        private int rerankN = 80;
        private int rerankM = 16;
        private int compressSentencesPerStatement = 2;
        private int compressMaxCharsPerStatement = 700;

        // Anti-water
        private double refineDfCut = 0.60;

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
        public Builder coverageK(int v) { this.coverageK = v; return this; }
        public Builder contradictionPenalty(double v) { this.contradictionPenalty = v; return this; }

        public Builder earlyStopConfidence(double v) { this.earlyStopConfidence = v; return this; }
        public Builder parallel(boolean v) { this.parallel = v; return this; }
        public Builder parallelism(int v) { this.parallelism = v; return this; }

        public Builder rerankN(int v) { this.rerankN = v; return this; }
        public Builder rerankM(int v) { this.rerankM = v; return this; }
        public Builder compressSentencesPerStatement(int v) { this.compressSentencesPerStatement = v; return this; }
        public Builder compressMaxCharsPerStatement(int v) { this.compressMaxCharsPerStatement = v; return this; }

        public Builder refineDfCut(double v) { this.refineDfCut = v; return this; }

        public ExplorationConfig build() {
            return new ExplorationConfig(
                    temperature, topK, iterations, candidateMultiplier, diversity, minScore, iterationDecay,
                    refineTerms, candidateGateMinTokenLen, maxCandidatesPerIter,
                    qualityFloor, coverageK, contradictionPenalty,
                    earlyStopConfidence, parallel, parallelism,
                    rerankN, rerankM,
                    compressSentencesPerStatement, compressMaxCharsPerStatement,
                    refineDfCut
            );
        }
    }
}