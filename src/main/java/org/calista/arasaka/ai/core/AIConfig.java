package org.calista.arasaka.ai.core;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.calista.arasaka.io.FileIO;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public final class AIConfig {
    public String baseDir = "data";
    public Corpora corpora = new Corpora();
    public Knowledge knowledge = new Knowledge();
    public Events events = new Events();
    public Thinking thinking = new Thinking();
    public Learning learning = new Learning();

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class Corpora {
        public String dir = "corpora";
        public List<String> bootstrap = List.of("base.jsonl");
        public boolean failFast = false;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class Knowledge {
        public String snapshotFile = "knowledge.snapshot.jsonl";
        public int autoSnapshotEveryTurns = 5;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class Events {
        public String logFile = "events.jsonl";
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class Thinking {
        public String engine = "beam";
        public int iterations = 4;
        public int retrieveK = 10;
        public Exploration exploration = new Exploration();
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class Exploration {

        // -------- base retrieval --------
        public double temperature = 0.8;
        public int topK = 40;
        public int iterations = 3;
        public int candidateMultiplier = 4;
        public double diversity = 0.18;
        public double minScore = 1e-9;
        public double iterationDecay = 0.72;

        // -------- query refinement / gating --------
        public int refineTerms = 14;
        public int candidateGateMinTokenLen = 3;
        public int maxCandidatesPerIter = 120_000;

        // -------- quality / validation --------
        /** Minimum acceptable combined quality (confidence ⊗ coverage − penalty). */
        public double qualityFloor = 0.0;

        /** How many top documents are checked for query token coverage (0 disables). */
        public int coverageK = 32;

        /** Penalty weight for contradictory evidence (0 disables). */
        public double contradictionPenalty = 0.35;

        /** Early stop retrieval iterations if confidence reaches this value (0 disables). */
        public double earlyStopConfidence = 0.0;

        // -------- performance --------
        public boolean parallel = false;
        public int parallelism = 0;

        // -------- production RAG: retrieve → rerank → compress --------
        /** Rerank only top-N candidates (precision stage). */
        public int rerankN = 80;

        /** After rerank, keep only top-M (recommended 12–20). */
        public int rerankM = 16;

        /** How many best sentences to keep per Statement during compression. */
        public int compressSentencesPerStatement = 2;

        /** Hard cap for compressed text per Statement (characters). */
        public int compressMaxCharsPerStatement = 700;

        // -------- anti-noise / anti-water --------
        /**
         * Drop candidate tokens in query refinement if they appear
         * in >= this fraction of top documents.
         * Typical range: 0.55–0.70
         */
        public double refineDfCut = 0.60;
    }


    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class Learning {
        public boolean enabled = true;
        public double newStatementWeight = 0.35;
        public double reinforceStep = 0.05;
    }

    public static AIConfig load(FileIO io, Path configFile, ObjectMapper mapper) throws IOException {
        AIConfig cfg = mapper.readValue(io.readString(configFile), AIConfig.class);
        cfg.validate();
        return cfg;
    }

    public void validate() {
        if (baseDir == null || baseDir.isBlank()) throw new IllegalStateException("baseDir required");
        if (corpora == null) corpora = new Corpora();
        if (corpora.dir == null || corpora.dir.isBlank()) throw new IllegalStateException("corpora.dir required");
        if (corpora.bootstrap == null) corpora.bootstrap = List.of();

        if (knowledge == null) knowledge = new Knowledge();
        if (knowledge.snapshotFile == null || knowledge.snapshotFile.isBlank()) throw new IllegalStateException("knowledge.snapshotFile required");
        if (knowledge.autoSnapshotEveryTurns < 1) knowledge.autoSnapshotEveryTurns = 1;

        if (events == null) events = new Events();
        if (events.logFile == null || events.logFile.isBlank()) throw new IllegalStateException("events.logFile required");

        if (thinking == null) thinking = new Thinking();
        if (thinking.iterations < 1) thinking.iterations = 1;
        if (thinking.retrieveK < 1) thinking.retrieveK = 1;
        if (thinking.exploration == null) thinking.exploration = new Exploration();
        if (thinking.exploration.temperature <= 0.0) throw new IllegalStateException("temperature must be > 0");
        if (thinking.exploration.topK < 1) throw new IllegalStateException("topK must be >= 1");

        if (learning == null) learning = new Learning();
        if (learning.newStatementWeight <= 0.0) throw new IllegalStateException("newStatementWeight must be > 0");
        if (learning.reinforceStep <= 0.0) throw new IllegalStateException("reinforceStep must be > 0");
    }
}