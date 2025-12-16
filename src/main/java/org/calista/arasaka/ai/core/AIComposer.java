package org.calista.arasaka.ai.core;

import org.calista.arasaka.ai.AIApp;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationConfig;
import org.calista.arasaka.ai.retrieve.exploration.impl.SoftmaxSampler;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.retrieve.retriver.impl.KnowledgeRetriever;
import org.calista.arasaka.ai.retrieve.scorer.Scorer;
import org.calista.arasaka.ai.retrieve.scorer.impl.TokenOverlapScorer;
import org.calista.arasaka.ai.think.*;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.candidate.impl.MultiCriteriaCandidateEvaluator;
import org.calista.arasaka.ai.think.engine.ThoughtCycleEngine;
import org.calista.arasaka.ai.think.engine.impl.IterativeThoughtEngine;
import org.calista.arasaka.ai.think.engine.impl.BeamSearchRagEngine;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.intent.impl.HashedIntentDetector;
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

    public AIComposer(AIApp app){
        this.app = app;
        this.context = app.getKernel().getJS_CONTEXT();
        context.eval(Source.newBuilder("js", new File("script/ai_composer.js")).buildLiteral());
        BUILD_PLAN = context.getBindings("js").getMember("buildPlan");
    }

    public ThoughtCycleEngine buildEngine(AIKernel kernel, Tokenizer tokenizer) {
        Objects.requireNonNull(kernel);
        Objects.requireNonNull(tokenizer);

        AIConfig cfg = kernel.config();

        // ---------- common deps (Java only) ----------
        Scorer scorer = new TokenOverlapScorer(tokenizer);

        ExplorationConfig expl = new ExplorationConfig(
                cfg.thinking.exploration.temperature,
                cfg.thinking.exploration.topK,
                cfg.thinking.exploration.iterations,
                cfg.thinking.exploration.candidateMultiplier,
                cfg.thinking.exploration.diversity,
                cfg.thinking.exploration.minScore,
                cfg.thinking.exploration.iterationDecay,
                cfg.thinking.exploration.refineTerms,
                cfg.thinking.exploration.candidateGateMinTokenLen,
                cfg.thinking.exploration.maxCandidatesPerIter,
                cfg.thinking.exploration.qualityFloor,
                cfg.thinking.exploration.earlyStopConfidence,
                cfg.thinking.exploration.parallel,
                cfg.thinking.exploration.parallelism
        );

        Retriever retriever = new KnowledgeRetriever(
                kernel.knowledge(),
                scorer,
                new SoftmaxSampler(),
                expl,
                12
        );

        IntentDetector intentDetector = new HashedIntentDetector();
        CandidateEvaluator evaluator = new MultiCriteriaCandidateEvaluator(tokenizer);
        ResponseStrategy strategy = new ContextAnswerStrategy();
        TextGenerator generator = new BigramBeamTextGenerator(tokenizer);

        // ---------- JS plan ----------
        Map plan = BUILD_PLAN.execute(Value.asValue(cfg)).as(Map.class);
        String engine = ((String) plan.get("engine")).toLowerCase();
        System.out.println(engine + " Engine");
        if (engine.contains("beam")) {
            log.info("Building Beam Engine");
            return new BeamSearchRagEngine(
                    retriever,
                    intentDetector,
                    strategy,
                    evaluator,
                    generator,
                    i(plan, "iterations"),
                    i(plan, "retrieveK"),
                    i(plan, "beamWidth"),
                    i(plan, "draftsPerBeam"),
                    i(plan, "maxDraftsPerIter"),
                    i(plan, "patience"),
                    d(plan, "targetScore"),
                    d(plan, "diversityPenalty"),
                    d(plan, "minDiversityJaccard"),
                    b(plan, "verifyPassEnabled"),
                    b(plan, "ltmEnabled"),
                    i(plan, "ltmCapacity"),
                    i(plan, "ltmRecallK"),
                    d(plan, "ltmWriteMinGroundedness")
            );
        } else {
            log.info("Building Iterative Engine");
            return new IterativeThoughtEngine(
                    retriever,
                    intentDetector,
                    strategy,
                    evaluator,
                    generator,
                    i(plan, "iterations"),
                    i(plan, "retrieveK"),
                    i(plan, "draftsPerIteration"),
                    i(plan, "patience"),
                    d(plan, "targetScore"),
                    b(plan, "ltmEnabled"),
                    i(plan, "ltmCapacity"),
                    i(plan, "ltmRecallK"),
                    d(plan, "ltmWriteMinGroundedness"),
                    i(plan, "refineRounds"),
                    i(plan, "refineQueryBudget")
            );
        }
    }

    // small helpers
    private static int i(Map<String, Object> m, String k) {
        return ((Number) m.get(k)).intValue();
    }
    private static double d(Map<String, Object> m, String k) {
        return ((Number) m.get(k)).doubleValue();
    }
    private static boolean b(Map<String, Object> m, String k) {
        return (Boolean) m.get(k);
    }
}