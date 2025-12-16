package org.calista.arasaka.ai.core;

import org.calista.arasaka.ai.retrieve.*;
import org.calista.arasaka.ai.think.candidate.impl.AdvancedCandidateEvaluator;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.engine.impl.IterativeThoughtEngine;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.intent.impl.SimpleIntentDetector;
import org.calista.arasaka.ai.tokenizer.*;
import org.calista.arasaka.ai.think.*;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Objects;

public final class AIComposer {
    private AIComposer() {}

    public static IterativeThoughtEngine buildThinking(AIKernel kernel, Tokenizer tokenizer) {
        AIConfig cfg = kernel.config();
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(kernel.knowledge(), "kb");
        Scorer scorer = new TokenOverlapScorer(tokenizer);

        ExplorationConfig expl = new ExplorationConfig(
                cfg.thinking.exploration.temperature,
                cfg.thinking.exploration.topK
        );

        Retriever retriever = new KnowledgeRetriever(kernel.knowledge(), scorer, new SoftmaxSampler(), expl);

        IntentDetector intentDetector = new SimpleIntentDetector();
        CandidateEvaluator evaluator = new AdvancedCandidateEvaluator(tokenizer);

        // Набор стратегий оставляем, но маршрутизацию делаем внутри анонимной стратегии (без новых файлов/классов).
        List<ResponseStrategy> strategies = List.of(
                new GreetingStrategy(),
                new RequestStrategy(),
                new ContextAnswerStrategy()
        );

        ResponseStrategy composite = new ResponseStrategy() {
            @Override
            public boolean supports(Intent intent) {
                return true;
            }

            @Override
            public String generate(String userText, List<org.calista.arasaka.ai.knowledge.Statement> context, ThoughtState state) {
                Intent intent = state != null ? state.intent : Intent.UNKNOWN;
                for (ResponseStrategy s : strategies) {
                    if (s.supports(intent)) return s.generate(userText, context, state);
                }
                // fallback (на случай если supports нигде не сработал)
                return new ContextAnswerStrategy().generate(userText, context, state);
            }
        };

        // Подключение генератора (BeamSearch/TensorFlow) — если появится в cfg.thinking.generator
        // Тип именно org.calista.arasaka.ai.think.TextGenerator (контракт детерминизма и generateN()).
        TextGenerator generator = (TextGenerator) readObject(cfg.thinking, "generator", null);

        int draftsPerIteration =
                readInt(cfg.thinking, "draftsPerIteration",
                        readInt(cfg.thinking, "drafts",
                                Math.max(1, cfg.thinking.exploration.topK)));

        int patience = readInt(cfg.thinking, "patience", 2);
        double targetScore = readDouble(cfg.thinking, "targetScore", 0.75);

        return new IterativeThoughtEngine(
                retriever,
                intentDetector,
                composite,
                evaluator,
                generator,
                cfg.thinking.iterations,
                cfg.thinking.retrieveK,
                draftsPerIteration,
                patience,
                targetScore
        );
    }

    // ---------------- reflection-safe cfg getters ----------------

    private static int readInt(Object obj, String fieldName, int def) {
        Object v = readObject(obj, fieldName, null);
        if (v instanceof Number n) return n.intValue();
        return def;
    }

    private static double readDouble(Object obj, String fieldName, double def) {
        Object v = readObject(obj, fieldName, null);
        if (v instanceof Number n) return n.doubleValue();
        return def;
    }

    private static Object readObject(Object obj, String fieldName, Object def) {
        if (obj == null || fieldName == null || fieldName.isBlank()) return def;
        try {
            Field f = obj.getClass().getDeclaredField(fieldName);
            f.setAccessible(true);
            Object v = f.get(obj);
            return v != null ? v : def;
        } catch (Throwable t) {
            return def;
        }
    }
}