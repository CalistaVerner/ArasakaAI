package org.calista.arasaka.ai.core;

import org.calista.arasaka.ai.AIApp;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationConfig;
import org.calista.arasaka.ai.retrieve.exploration.impl.SoftmaxSampler;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.retrieve.retriver.impl.KnowledgeRetriever;
import org.calista.arasaka.ai.retrieve.scorer.Scorer;
import org.calista.arasaka.ai.retrieve.scorer.impl.TokenOverlapScorer;
import org.calista.arasaka.ai.think.response.ContextAnswerStrategy;
import org.calista.arasaka.ai.think.response.ResponseStrategy;
import org.calista.arasaka.ai.think.Think;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.candidate.impl.MultiCriteriaCandidateEvaluator;
import org.calista.arasaka.ai.think.intent.impl.IntentDetector;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;
import org.calista.arasaka.ai.think.textGenerator.impl.BigramBeamTextGenerator;
import org.calista.arasaka.ai.tokenizer.Tokenizer;
import org.graalvm.polyglot.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;
import java.util.Objects;

public final class AIComposer {

    private static final Logger log = LoggerFactory.getLogger(AIComposer.class);

    private final Value BUILD_PLAN;
    private final AIApp app;
    private final Context context;

    /**
     * Think instance created by the last buildEngine() call.
     * Keep it to close resources (owned eval pool) on shutdown.
     */
    private volatile Think lastThink;

    public AIComposer(AIApp app) {
        this.app = app;
        this.context = app.getKernel().jsContext();
        try {
            context.eval(Source.newBuilder("js", new File("script/ai_composer.js")).build());
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load script/ai_composer.js", e);
        }
        BUILD_PLAN = context.getBindings("js").getMember("buildPlan");
        if (BUILD_PLAN == null || BUILD_PLAN.isNull()) {
            throw new IllegalStateException("JS buildPlan not found in ai_composer.js");
        }
    }

    /**
     * Backward-compatible method: returns ThoughtCycleEngine (engine),
     * but internally constructs Think orchestrator (owns eval pool).
     */
    public Think buildEngine(AIKernel kernel, Tokenizer tokenizer) {
        Objects.requireNonNull(kernel, "kernel");
        Objects.requireNonNull(tokenizer, "tokenizer");
        AIConfig cfg = kernel.config();

        // ---------- common deps (Java only) ----------
        Scorer scorer = new TokenOverlapScorer(tokenizer);

        ExplorationConfig expl = ExplorationConfig.builder()
                .temperature(cfg.thinking.exploration.temperature)
                .topK(cfg.thinking.exploration.topK)
                .iterations(cfg.thinking.exploration.iterations)
                .candidateMultiplier(cfg.thinking.exploration.candidateMultiplier)
                .diversity(cfg.thinking.exploration.diversity)
                .minScore(cfg.thinking.exploration.minScore)
                .iterationDecay(cfg.thinking.exploration.iterationDecay)

                .refineTerms(cfg.thinking.exploration.refineTerms)
                .candidateGateMinTokenLen(cfg.thinking.exploration.candidateGateMinTokenLen)
                .maxCandidatesPerIter(cfg.thinking.exploration.maxCandidatesPerIter)

                .qualityFloor(cfg.thinking.exploration.qualityFloor)
                .coverageK(cfg.thinking.exploration.coverageK)
                .contradictionPenalty(cfg.thinking.exploration.contradictionPenalty)
                .earlyStopConfidence(cfg.thinking.exploration.earlyStopConfidence)

                .parallel(cfg.thinking.exploration.parallel)
                .parallelism(cfg.thinking.exploration.parallelism)

                // ---- production RAG ----
                .rerankN(cfg.thinking.exploration.rerankN)
                .rerankM(cfg.thinking.exploration.rerankM)
                .compressSentencesPerStatement(cfg.thinking.exploration.compressSentencesPerStatement)
                .compressMaxCharsPerStatement(cfg.thinking.exploration.compressMaxCharsPerStatement)

                // ---- anti-water ----
                .refineDfCut(cfg.thinking.exploration.refineDfCut)

                .build();

        Retriever retriever = new KnowledgeRetriever(
                kernel.knowledge(),
                scorer,
                new SoftmaxSampler(),
                expl,
                12
        );

        org.calista.arasaka.ai.think.intent.IntentDetector intentDetector = new IntentDetector();
        CandidateEvaluator evaluator = new MultiCriteriaCandidateEvaluator(tokenizer);
        ResponseStrategy strategy = new ContextAnswerStrategy();
        TextGenerator generator = new BigramBeamTextGenerator(tokenizer);

        // ---------- JS plan ----------
        @SuppressWarnings("rawtypes")
        Map plan = BUILD_PLAN.execute(Value.asValue(cfg)).as(Map.class);

        Think.Config thinkCfg = new Think.Config();
        thinkCfg.iterations = i(plan, "iterations");
        thinkCfg.retrieveK = i(plan, "retrieveK");
        thinkCfg.draftsPerIteration = i(plan, "draftsPerIteration");
        thinkCfg.patience = i(plan, "patience");
        thinkCfg.targetScore = d(plan, "targetScore");

        thinkCfg.ltmEnabled = b(plan, "ltmEnabled");
        thinkCfg.ltmCapacity = i(plan, "ltmCapacity");
        thinkCfg.ltmRecallK = i(plan, "ltmRecallK");
        thinkCfg.ltmWriteMinGroundedness = d(plan, "ltmWriteMinGroundedness");

        thinkCfg.refineRounds = i(plan, "refineRounds");
        thinkCfg.refineQueryBudget = i(plan, "refineQueryBudget");

// ✅ evalPool: nested
        Object ep = plan.get("evalPool");
        if (ep instanceof Map<?, ?> m) {
            @SuppressWarnings("unchecked")
            Map<String, Object> eval = (Map<String, Object>) m;

            if (eval.containsKey("parallelism")) thinkCfg.evalParallelism = i(eval, "parallelism");
            if (eval.containsKey("queueCapacity")) thinkCfg.evalQueueCapacity = i(eval, "queueCapacity");
            if (eval.containsKey("threadNamePrefix"))
                thinkCfg.evalThreadNamePrefix = s(eval, "threadNamePrefix", thinkCfg.evalThreadNamePrefix);
            if (eval.containsKey("shutdownTimeoutMs"))
                thinkCfg.evalShutdownTimeoutMs = l(eval, "shutdownTimeoutMs");
        }


        // Build Think orchestrator
        log.info("Building Think Orchestrator");
        Think think = Think.builder(retriever, tokenizer)
                .config(thinkCfg)
                .intentDetector(intentDetector)
                .evaluator(evaluator)
                .responseStrategy(strategy)
                .generator(generator)
                .build();

        // keep reference so caller can close it later (important: owned pool)
        this.lastThink = think;

        log.info("Building Iterative Engine via Think");
        return think;
    }

    /**
     * Get the most recently built Think instance (so you can close it on shutdown).
     */
    public Think getLastThink() {
        return lastThink;
    }

    // small helpers
    private static int i(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing plan key: " + k);
        return ((Number) v).intValue();
    }

    private static long l(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing plan key: " + k);
        return ((Number) v).longValue();
    }

    private static double d(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing plan key: " + k);
        return ((Number) v).doubleValue();
    }

    private static boolean b(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) throw new IllegalArgumentException("Missing plan key: " + k);
        return (Boolean) v;
    }

    private static String s(Map<String, Object> m, String k, String def) {
        Object v = m.get(k);
        if (v == null) return def;
        String s = String.valueOf(v);
        return (s == null || s.isBlank()) ? def : s;
    }
}