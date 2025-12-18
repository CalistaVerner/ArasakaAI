// FILE: Think.java
package org.calista.arasaka.ai.think;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.retrieve.scorer.impl.TokenOverlapScorer;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.candidate.impl.BaseCandidateEvaluator;
import org.calista.arasaka.ai.think.engine.impl.IterativeThoughtEngine;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.intent.impl.BaseIntentDetector;
import org.calista.arasaka.ai.think.response.ContextAnswerStrategy;
import org.calista.arasaka.ai.think.response.ResponseStrategy;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.time.Clock;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

/**
 * Think — единый enterprise-оркестратор.
 *
 * Ownership change:
 *  - Think владеет evalExecutor (создание/закрытие).
 *
 * Note:
 *  - Текущий IterativeThoughtEngine (новая версия) не принимает evalPool в конструкторе.
 *    Пул оставлен как owned-resource для будущего параллельного scoring/verify или иных задач.
 */
public final class Think implements AutoCloseable {

    private static final Logger log = LogManager.getLogger(Think.class);

    private final Config config;

    private final Retriever retriever;
    private final Tokenizer tokenizer;
    private final IntentDetector intentDetector;
    private final ResponseStrategy responseStrategy;
    private final CandidateEvaluator evaluator;
    private final TextGenerator generator; // nullable

    private final TokenOverlapScorer overlapScorer;

    private final LongSupplier seedProvider;

    // Owned resources (kept for enterprise future use)
    private final ExecutorService evalPool;
    private final boolean ownsEvalPool;

    // Core engine
    private final IterativeThoughtEngine engine;

    private Think(Builder b) {
        this.retriever = Objects.requireNonNull(b.retriever, "retriever");
        this.tokenizer = Objects.requireNonNull(b.tokenizer, "tokenizer");

        this.config = Objects.requireNonNull(b.config, "config").freezeAndValidate();

        // INTENTS: prod default => autoBootstrap=false
        this.intentDetector = (b.intentDetector != null)
                ? b.intentDetector
                : new BaseIntentDetector(
                BaseIntentDetector.Config.builder()
                        .autoBootstrap(config.devMode) // devMode=true => bootstrap on; prod => off
                        .build()
        );

        this.responseStrategy = (b.responseStrategy != null) ? b.responseStrategy : new ContextAnswerStrategy();
        this.generator = b.generator; // nullable ok

        // Scorer (idf/tokens cache inside)
        this.overlapScorer = (b.overlapScorer != null) ? b.overlapScorer : new TokenOverlapScorer(tokenizer);

        // Evaluator (new ctor)
        this.evaluator = (b.evaluator != null) ? b.evaluator : new BaseCandidateEvaluator(
                tokenizer,
                overlapScorer,
                config.evalMinGroundedness,
                config.evalMaxContradictionRisk,
                config.evalMinQueryCoverage,
                config.evalMaxNovelty,
                config.evalMaxRepetition,
                config.evalMinChars,
                config.evalMaxCharsSoft,
                config.evalMaxCharsHard
        );

        this.seedProvider = (b.seedProvider != null) ? b.seedProvider : defaultSeedProvider(b.clock);

        // Pool ownership (kept; engine currently doesn't consume it)
        this.ownsEvalPool = (b.evalPool == null);
        this.evalPool = ownsEvalPool ? createEvalPool(config) : b.evalPool;

        // ---- Engine (NEW signature) ----
        // IterativeThoughtEngine(retriever, intentDetector, generator, evaluator, strategy,
        //   iterations, retrieveK, draftsPerIteration, patience, targetScore,
        //   exploitEvidenceStrictness, strictVerifyEnabled, strictVerifyMinScore,
        //   ltmEnabled, ltmCapacity, ltmRecallK, ltmWriteMinGroundedness, ltmDecayPerTick, ltmPromotionBoost)
        this.engine = new IterativeThoughtEngine(
                retriever,
                intentDetector,
                generator,
                evaluator,
                responseStrategy,
                evalPool,
                config.evalParallelism,
                config.iterations,
                config.retrieveK,
                config.draftsPerIteration,
                config.patience,
                config.targetScore,
                config.exploitEvidenceStrictness,
                config.strictVerifyEnabled,
                config.strictVerifyMinScore,
                config.ltmEnabled,
                config.ltmCapacity,
                config.ltmRecallK,
                config.ltmWriteMinGroundedness,
                config.ltmDecayPerTick,
                config.ltmPromotionBoost
        );

        logCreation();
    }

    // ---------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------

    public ThoughtResult think(String userText) {
        return engine.think(userText, seedProvider.getAsLong());
    }

    public ThoughtResult think(String userText, long seed) {
        return engine.think(userText, seed);
    }

    // ---------------------------------------------------------------------
    // Getters
    // ---------------------------------------------------------------------

    public Config getConfig() { return config; }

    public IterativeThoughtEngine getEngine() { return engine; }

    public Retriever getRetriever() { return retriever; }

    public Tokenizer getTokenizer() { return tokenizer; }

    public IntentDetector getIntentDetector() { return intentDetector; }

    public ResponseStrategy getResponseStrategy() { return responseStrategy; }

    public CandidateEvaluator getEvaluator() { return evaluator; }

    public TokenOverlapScorer getOverlapScorer() { return overlapScorer; }

    public TextGenerator getGenerator() { return generator; }

    public ExecutorService getEvalPool() { return evalPool; }

    public LongSupplier getSeedProvider() { return seedProvider; }

    // ---------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------

    @Override
    public void close() {
        // IMPORTANT: do not shutdown externally owned executors
        if (ownsEvalPool) {
            shutdownExecutor(evalPool, config.evalShutdownTimeoutMs);
        } else {
            log.debug("Think.close(): evalPool is externally owned; skipping shutdown");
        }
    }

    // ---------------------------------------------------------------------
    // Builder / Config
    // ---------------------------------------------------------------------

    public static Builder builder(Retriever retriever, Tokenizer tokenizer) {
        return new Builder(retriever, tokenizer);
    }

    public static final class Builder {
        private final Retriever retriever;
        private final Tokenizer tokenizer;

        private IntentDetector intentDetector;
        private ResponseStrategy responseStrategy;
        private CandidateEvaluator evaluator;
        private TextGenerator generator;

        private TokenOverlapScorer overlapScorer;

        private LongSupplier seedProvider;
        private Clock clock = Clock.systemUTC();

        private Config config = Config.balanced();

        // Allow external DI to provide its own pool (optional)
        private ExecutorService evalPool;

        private Builder(Retriever retriever, Tokenizer tokenizer) {
            this.retriever = Objects.requireNonNull(retriever, "retriever");
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        }

        public Builder config(Config cfg) {
            this.config = Objects.requireNonNull(cfg, "config");
            return this;
        }

        public Builder intentDetector(IntentDetector detector) {
            this.intentDetector = Objects.requireNonNull(detector, "intentDetector");
            return this;
        }

        public Builder responseStrategy(ResponseStrategy strategy) {
            this.responseStrategy = Objects.requireNonNull(strategy, "responseStrategy");
            return this;
        }

        public Builder evaluator(CandidateEvaluator evaluator) {
            this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
            return this;
        }

        public Builder generator(TextGenerator generator) {
            this.generator = generator; // nullable ok
            return this;
        }

        public Builder overlapScorer(TokenOverlapScorer scorer) {
            this.overlapScorer = Objects.requireNonNull(scorer, "overlapScorer");
            return this;
        }

        public Builder seedProvider(LongSupplier seedProvider) {
            this.seedProvider = Objects.requireNonNull(seedProvider, "seedProvider");
            return this;
        }

        public Builder clock(Clock clock) {
            this.clock = Objects.requireNonNull(clock, "clock");
            return this;
        }

        /**
         * Если хочешь DI-владение пулом — можно передать сюда.
         * Если не задано, Think создаст и закроет сам.
         */
        public Builder evalPool(ExecutorService pool) {
            this.evalPool = Objects.requireNonNull(pool, "evalPool");
            return this;
        }

        public Builder balanced() { this.config = Config.balanced(); return this; }

        public Builder aggressive() { this.config = Config.aggressive(); return this; }

        public Builder dev() { this.config = Config.dev(); return this; }

        public Think build() {
            return new Think(this);
        }
    }

    public static final class Config {
        // profile
        public boolean devMode = false; // prod default

        // thinking loop
        public int iterations = 4;
        public int retrieveK = 24;
        public int draftsPerIteration = 8;
        public int patience = 2;
        public double targetScore = 0.65;

        // strict verify-pass
        public boolean strictVerifyEnabled = true;
        public double strictVerifyMinScore = 0.62;

        // LTM self-regulation
        public double ltmDecayPerTick = 0.0015;
        public double ltmPromotionBoost = 0.06;

        // LTM
        public boolean ltmEnabled = true;
        public int ltmCapacity = 50_000;
        public int ltmRecallK = 64;
        public double ltmWriteMinGroundedness = 0.55;

        // explore/exploit (used by engine as a “strictness knob” now)
        public double exploitEvidenceStrictness = 0.78;

        // Evaluator thresholds (NEW — чтобы не хардкодить в BaseCandidateEvaluator)
        public double evalMinGroundedness = 0.45;
        public double evalMaxContradictionRisk = 0.55;
        public double evalMinQueryCoverage = 0.20;
        public double evalMaxNovelty = 0.72;
        public double evalMaxRepetition = 0.75;

        public int evalMinChars = 16;
        public int evalMaxCharsSoft = 1400;
        public int evalMaxCharsHard = 2400;

        // owned evaluation executor (kept for future use)
        public int evalParallelism = Math.max(1, Math.min(8, Runtime.getRuntime().availableProcessors()));
        public int evalQueueCapacity = 4096;
        public String evalThreadNamePrefix = "think-eval-";
        public long evalShutdownTimeoutMs = 2500;

        private boolean frozen = false;

        public static Config balanced() { return new Config(); }

        public static Config aggressive() {
            Config c = new Config();
            c.iterations = 6;
            c.retrieveK = 32;
            c.draftsPerIteration = 12;
            c.patience = 2;
            c.targetScore = 0.72;

            c.strictVerifyEnabled = true;
            c.strictVerifyMinScore = 0.66;

            c.ltmEnabled = true;
            c.ltmRecallK = 96;
            c.ltmWriteMinGroundedness = 0.60;

            c.exploitEvidenceStrictness = 0.82;

            // evaluator tighter
            c.evalMinGroundedness = 0.50;
            c.evalMaxContradictionRisk = 0.50;
            c.evalMinQueryCoverage = 0.25;
            c.evalMaxNovelty = 0.68;
            c.evalMaxRepetition = 0.70;

            c.evalParallelism = Math.max(2, Math.min(12, Runtime.getRuntime().availableProcessors()));
            c.evalQueueCapacity = 8192;
            return c;
        }

        /** Dev profile: bootstrap intents enabled, slightly looser loops (optional). */
        public static Config dev() {
            Config c = new Config();
            c.devMode = true;
            c.strictVerifyEnabled = true;
            c.strictVerifyMinScore = 0.58;
            c.exploitEvidenceStrictness = 0.74;

            // evaluator looser for iteration speed
            c.evalMinGroundedness = 0.40;
            c.evalMaxContradictionRisk = 0.60;
            return c;
        }

        public Config freezeAndValidate() {
            if (frozen) return this;

            iterations = Math.max(1, iterations);
            retrieveK = Math.max(1, retrieveK);
            draftsPerIteration = Math.max(1, draftsPerIteration);
            patience = Math.max(0, patience);

            exploitEvidenceStrictness = clamp01(exploitEvidenceStrictness);

            strictVerifyMinScore = Math.max(-10, Math.min(10, strictVerifyMinScore));

            ltmCapacity = Math.max(0, ltmCapacity);
            ltmRecallK = Math.max(0, ltmRecallK);
            ltmWriteMinGroundedness = clamp01(ltmWriteMinGroundedness);

            if (ltmDecayPerTick < 0) ltmDecayPerTick = 0.0;
            if (ltmPromotionBoost < 0) ltmPromotionBoost = 0.0;

            evalMinGroundedness = clamp01(evalMinGroundedness);
            evalMaxContradictionRisk = clamp01(evalMaxContradictionRisk);
            evalMinQueryCoverage = clamp01(evalMinQueryCoverage);
            evalMaxNovelty = clamp01(evalMaxNovelty);
            evalMaxRepetition = clamp01(evalMaxRepetition);

            evalMinChars = Math.max(0, evalMinChars);
            evalMaxCharsSoft = Math.max(64, evalMaxCharsSoft);
            evalMaxCharsHard = Math.max(evalMaxCharsSoft, evalMaxCharsHard);

            evalParallelism = Math.max(1, evalParallelism);
            evalQueueCapacity = Math.max(32, evalQueueCapacity);
            if (evalThreadNamePrefix == null || evalThreadNamePrefix.isBlank()) evalThreadNamePrefix = "think-eval-";
            evalShutdownTimeoutMs = Math.max(250, evalShutdownTimeoutMs);

            if (targetScore < -10 || targetScore > 10) targetScore = 0.65;

            frozen = true;
            return this;
        }

        private static double clamp01(double v) {
            if (Double.isNaN(v)) return 0.0;
            if (v < 0.0) return 0.0;
            if (v > 1.0) return 1.0;
            return v;
        }
    }

    // ---------------------------------------------------------------------
    // Internals
    // ---------------------------------------------------------------------

    private static LongSupplier defaultSeedProvider(Clock clock) {
        final Clock c = (clock != null) ? clock : Clock.systemUTC();
        return c::millis;
    }

    private static ExecutorService createEvalPool(Config cfg) {
        final AtomicLong tid = new AtomicLong(1);
        final int par = Math.max(1, cfg.evalParallelism);

        ThreadFactory tf = r -> {
            Thread t = new Thread(r, cfg.evalThreadNamePrefix + tid.getAndIncrement());
            t.setDaemon(true);
            return t;
        };

        // bounded queue + CallerRunsPolicy => backpressure (enterprise latency stability)
        return new ThreadPoolExecutor(
                par,
                par,
                30L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(cfg.evalQueueCapacity),
                tf,
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
    }

    private static void shutdownExecutor(ExecutorService es, long timeoutMs) {
        if (es == null) return;

        es.shutdown();
        try {
            if (!es.awaitTermination(timeoutMs, TimeUnit.MILLISECONDS)) {
                es.shutdownNow();
                es.awaitTermination(Math.max(250, timeoutMs / 2), TimeUnit.MILLISECONDS);
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            es.shutdownNow();
        } catch (Throwable t) {
            log.warn("Failed to shutdown evalPool cleanly", t);
            try { es.shutdownNow(); } catch (Throwable ignored) {}
        }
    }

    private void logCreation() {
        if (!log.isInfoEnabled()) return;

        String msg = ThinkLogFmt.box("Think initialized", b -> {
            b.kv("retriever", retriever.getClass().getName());
            b.kv("tokenizer", tokenizer.getClass().getName());
            b.kv("intentDetector", intentDetector.getClass().getName());
            b.kv("responseStrategy", responseStrategy.getClass().getName());
            b.kv("evaluator", evaluator.getClass().getName());
            b.kv("overlapScorer", overlapScorer.getClass().getName());
            b.kv("generator", (generator == null) ? "<none>" : generator.getClass().getName());
            b.kv("engine", engine.getClass().getName());
            b.sep();
            b.kv("devMode", config.devMode);
            b.kv("iterations", config.iterations);
            b.kv("retrieveK", config.retrieveK);
            b.kv("draftsPerIteration", config.draftsPerIteration);
            b.kv("patience", config.patience);
            b.kv("targetScore", config.targetScore);
            b.sep();
            b.kv("exploitEvidenceStrictness", config.exploitEvidenceStrictness);
            b.kv("strictVerifyEnabled", config.strictVerifyEnabled);
            b.kv("strictVerifyMinScore", config.strictVerifyMinScore);
            b.sep();
            b.kv("ltmEnabled", config.ltmEnabled);
            b.kv("ltmCapacity", config.ltmCapacity);
            b.kv("ltmRecallK", config.ltmRecallK);
            b.kv("ltmWriteMinGroundedness", config.ltmWriteMinGroundedness);
            b.kv("ltmDecayPerTick", config.ltmDecayPerTick);
            b.kv("ltmPromotionBoost", config.ltmPromotionBoost);
            b.sep();
            b.kv("evalParallelism", config.evalParallelism);
            b.kv("evalQueueCapacity", config.evalQueueCapacity);
            b.kv("evalThreadNamePrefix", config.evalThreadNamePrefix);
            b.kv("evalPoolOwnership", ownsEvalPool ? "owned" : "external");
            b.kv("evalPoolUsedByEngine", "false"); // current engine signature
        });

        System.out.println(msg);
    }
}