package org.calista.arasaka.ai.core;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.calista.arasaka.io.FileIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * AIConfig — простой POJO конфиг:
 * - дефолты в полях
 * - loadOrCreate() создаёт файл, если его нет
 * - validate() нормализует значения
 *
 * Без "поиска отсутствующих полей", без JsonPointer-списков, без магии.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class AIConfig {

    private static final Logger log = LoggerFactory.getLogger(AIConfig.class);

    public String baseDir = "data";
    public Corpora corpora = new Corpora();
    public Knowledge knowledge = new Knowledge();
    public Events events = new Events();
    public Thinking thinking = new Thinking();
    public Learning learning = new Learning();

    // -------------------- Sections --------------------

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
        public String orchestrator = "think";
        public String engine = "iterative";

        // core loop
        public int iterations = 4;
        public int retrieveK = 10;

        // generation loop knobs (Think/Iterative)
        public int draftsPerIteration = 8;
        public int patience = 2;
        public double targetScore = 0.65;

        // refinement knobs (IterativeThoughtEngine)
        public int refineRounds = 1;
        public int refineQueryBudget = 16;

        // owned eval pool knobs (Think owns pool)
        public EvalPool evalPool = new EvalPool();

        // retrieval/exploration knobs (retriever side)
        public Exploration exploration = new Exploration();

        // Think.Config : LTM (ВАЖНО: чтобы оно реально работало)
        public boolean ltmEnabled = true;
        public int ltmCapacity = 50_000;
        public int ltmRecallK = 64;
        public double ltmWriteMinGroundedness = 0.55;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class EvalPool {
        /** Fixed parallelism for evaluation pool. 0 => auto. */
        public int parallelism = 0;

        /** Bounded queue capacity (backpressure via CallerRunsPolicy). */
        public int queueCapacity = 4096;

        /** Thread name prefix for observability. */
        public String threadNamePrefix = "think-eval-";

        /** Shutdown timeout for pool on close. */
        public long shutdownTimeoutMs = 2500;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class Exploration {
        public double temperature = 0.8;
        public int topK = 40;
        public int iterations = 3;
        public int candidateMultiplier = 4;
        public double diversity = 0.18;
        public double minScore = 1e-9;
        public double iterationDecay = 0.72;

        public int refineTerms = 14;
        public int candidateGateMinTokenLen = 3;
        public int maxCandidatesPerIter = 120_000;

        public double qualityFloor = 0.0;
        public int coverageK = 32;
        public double contradictionPenalty = 0.35;
        public double earlyStopConfidence = 0.0;

        public boolean parallel = false;
        public int parallelism = 0;

        public int rerankN = 80;
        public int rerankM = 16;
        public int compressSentencesPerStatement = 2;
        public int compressMaxCharsPerStatement = 700;

        public double refineDfCut = 0.60;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class Learning {
        public boolean enabled = true;
        public double newStatementWeight = 0.35;
        public double reinforceStep = 0.05;
    }

    // -------------------- Load / Create --------------------

    /**
     * Загружает конфиг. Если файла нет (или он пустой) — создаёт дефолтный и пишет на диск.
     */
    public static AIConfig loadOrCreate(FileIO io, Path configFile, ObjectMapper mapper) throws IOException {
        Objects.requireNonNull(io, "io");
        Objects.requireNonNull(configFile, "configFile");
        Objects.requireNonNull(mapper, "mapper");

        String json;
        try {
            json = io.readString(configFile);
        } catch (NoSuchFileException e) {
            AIConfig created = new AIConfig();
            created.validate();
            writePretty(io, configFile, mapper, created);
            log.info("Config file not found. Created default config at {}", configFile);
            return created;
        }

        if (json == null || json.isBlank()) {
            AIConfig created = new AIConfig();
            created.validate();
            writePretty(io, configFile, mapper, created);
            log.warn("Config file {} is empty. Recreated defaults.", configFile);
            return created;
        }

        AIConfig cfg = mapper.readValue(json, AIConfig.class);
        if (cfg == null) cfg = new AIConfig();

        cfg.validate();
        return cfg;
    }

    /**
     * Перезаписывает конфиг на диск (pretty JSON).
     */
    public static void save(FileIO io, Path configFile, ObjectMapper mapper, AIConfig cfg) throws IOException {
        Objects.requireNonNull(io, "io");
        Objects.requireNonNull(configFile, "configFile");
        Objects.requireNonNull(mapper, "mapper");
        Objects.requireNonNull(cfg, "cfg");

        cfg.validate();
        writePretty(io, configFile, mapper, cfg);
    }

    private static void writePretty(FileIO io, Path configFile, ObjectMapper mapper, AIConfig cfg) throws IOException {
        String out = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(cfg);
        // FileIO у тебя enterprise-уровня (atomic writes / fsync опционально) — используем его.
        io.writeString(configFile, out + System.lineSeparator());
    }

    // -------------------- Validation / Normalization --------------------

    public void validate() {
        if (baseDir == null || baseDir.isBlank()) baseDir = "data";

        if (corpora == null) corpora = new Corpora();
        if (corpora.dir == null || corpora.dir.isBlank()) corpora.dir = "corpora";
        if (corpora.bootstrap == null) corpora.bootstrap = List.of();

        if (knowledge == null) knowledge = new Knowledge();
        if (knowledge.snapshotFile == null || knowledge.snapshotFile.isBlank())
            knowledge.snapshotFile = "knowledge.snapshot.jsonl";
        if (knowledge.autoSnapshotEveryTurns < 1) knowledge.autoSnapshotEveryTurns = 1;

        if (events == null) events = new Events();
        if (events.logFile == null || events.logFile.isBlank()) events.logFile = "events.jsonl";

        if (thinking == null) thinking = new Thinking();

        // orchestrator/engine normalization
        if (thinking.orchestrator != null && thinking.orchestrator.isBlank()) thinking.orchestrator = null;
        if (thinking.engine == null || thinking.engine.isBlank()) thinking.engine = "iterative";

        // core loop clamps
        if (thinking.iterations < 1) thinking.iterations = 1;
        if (thinking.retrieveK < 1) thinking.retrieveK = 1;

        if (thinking.draftsPerIteration < 1) thinking.draftsPerIteration = 1;
        if (thinking.patience < 0) thinking.patience = 0;
        if (!Double.isFinite(thinking.targetScore)) thinking.targetScore = 0.65;

        if (thinking.refineRounds < 0) thinking.refineRounds = 0;
        if (thinking.refineQueryBudget < 1) thinking.refineQueryBudget = 1;

        // LTM clamps
        if (thinking.ltmCapacity < 0) thinking.ltmCapacity = 0;
        if (thinking.ltmRecallK < 0) thinking.ltmRecallK = 0;
        if (!Double.isFinite(thinking.ltmWriteMinGroundedness)) thinking.ltmWriteMinGroundedness = 0.55;
        if (thinking.ltmWriteMinGroundedness < 0.0) thinking.ltmWriteMinGroundedness = 0.0;
        if (thinking.ltmWriteMinGroundedness > 1.0) thinking.ltmWriteMinGroundedness = 1.0;

        // evalPool
        if (thinking.evalPool == null) thinking.evalPool = new EvalPool();
        if (thinking.evalPool.parallelism < 0) thinking.evalPool.parallelism = 0;
        if (thinking.evalPool.queueCapacity < 32) thinking.evalPool.queueCapacity = 32;
        if (thinking.evalPool.threadNamePrefix == null || thinking.evalPool.threadNamePrefix.isBlank())
            thinking.evalPool.threadNamePrefix = "think-eval-";
        if (thinking.evalPool.shutdownTimeoutMs < 250) thinking.evalPool.shutdownTimeoutMs = 250;

        // exploration
        if (thinking.exploration == null) thinking.exploration = new Exploration();
        if (!(thinking.exploration.temperature > 0.0)) thinking.exploration.temperature = 0.8;
        if (thinking.exploration.topK < 1) thinking.exploration.topK = 1;

        if (thinking.exploration.iterations < 1) thinking.exploration.iterations = 1;
        if (thinking.exploration.candidateMultiplier < 1) thinking.exploration.candidateMultiplier = 1;
        if (thinking.exploration.diversity < 0.0) thinking.exploration.diversity = 0.0;
        if (!Double.isFinite(thinking.exploration.minScore)) thinking.exploration.minScore = 1e-9;
        if (thinking.exploration.iterationDecay <= 0.0 || thinking.exploration.iterationDecay > 1.0)
            thinking.exploration.iterationDecay = 0.72;

        if (thinking.exploration.refineTerms < 0) thinking.exploration.refineTerms = 0;
        if (thinking.exploration.candidateGateMinTokenLen < 1) thinking.exploration.candidateGateMinTokenLen = 1;
        if (thinking.exploration.maxCandidatesPerIter < 1) thinking.exploration.maxCandidatesPerIter = 1;

        if (!Double.isFinite(thinking.exploration.qualityFloor)) thinking.exploration.qualityFloor = 0.0;
        if (thinking.exploration.coverageK < 0) thinking.exploration.coverageK = 0;
        if (thinking.exploration.contradictionPenalty < 0.0) thinking.exploration.contradictionPenalty = 0.0;
        if (!Double.isFinite(thinking.exploration.earlyStopConfidence)) thinking.exploration.earlyStopConfidence = 0.0;

        if (thinking.exploration.parallelism < 0) thinking.exploration.parallelism = 0;

        if (thinking.exploration.rerankN < 1) thinking.exploration.rerankN = 1;
        if (thinking.exploration.rerankM < 1) thinking.exploration.rerankM = 1;
        if (thinking.exploration.compressSentencesPerStatement < 1) thinking.exploration.compressSentencesPerStatement = 1;
        if (thinking.exploration.compressMaxCharsPerStatement < 120) thinking.exploration.compressMaxCharsPerStatement = 120;

        if (!Double.isFinite(thinking.exploration.refineDfCut) || thinking.exploration.refineDfCut < 0.0 || thinking.exploration.refineDfCut > 1.0)
            thinking.exploration.refineDfCut = 0.60;

        if (learning == null) learning = new Learning();
        if (!(learning.newStatementWeight > 0.0)) learning.newStatementWeight = 0.35;
        if (!(learning.reinforceStep > 0.0)) learning.reinforceStep = 0.05;
    }
}