package org.calista.arasaka.ai.core;

import org.calista.arasaka.ai.retrieve.exploration.ExplorationConfig;
import org.calista.arasaka.ai.retrieve.exploration.impl.SoftmaxSampler;
import org.calista.arasaka.ai.retrieve.retriver.impl.KnowledgeRetriever;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.retrieve.scorer.Scorer;
import org.calista.arasaka.ai.retrieve.scorer.impl.TokenOverlapScorer;
import org.calista.arasaka.ai.think.*;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.candidate.impl.AdvancedCandidateEvaluator;
import org.calista.arasaka.ai.think.engine.impl.IterativeThoughtEngine;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.intent.impl.SimpleIntentDetector;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.lang.reflect.Field;
import java.util.Objects;

public final class AIComposer {
    private AIComposer() {}

    public static IterativeThoughtEngine buildThinking(AIKernel kernel, Tokenizer tokenizer) {
        AIConfig cfg = kernel.config();
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(kernel.knowledge(), "kb");
        Objects.requireNonNull(tokenizer, "tokenizer");

        // --- Retriever (RAG) ---
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
                cfg.thinking.exploration.qualityFloor
        );

        Retriever retriever = new KnowledgeRetriever(
                kernel.knowledge(),
                scorer,
                new SoftmaxSampler(),
                expl
        );

        // --- Deterministic intent + advanced evaluation ---
        IntentDetector intentDetector = new SimpleIntentDetector();
        CandidateEvaluator evaluator = new AdvancedCandidateEvaluator(tokenizer);

        // --- Single enterprise strategy (no Greeting/Request hardcode) ---
        ResponseStrategy strategy = new ContextAnswerStrategy();

        // Optional generator (BeamSearch/TensorFlow) from config
        TextGenerator generator = (TextGenerator) readObject(cfg.thinking, "generator", null);

        // IMPORTANT: draftsPerIteration must NOT be derived from exploration.topK.
        // A sane production default is 8..12 (beam width can be separate).
        int draftsPerIteration = readInt(cfg.thinking, "draftsPerIteration",
                readInt(cfg.thinking, "drafts", 8));
        draftsPerIteration = clampInt(draftsPerIteration, 1, 32);

        int iterations = readInt(cfg.thinking, "iterations", cfg.thinking.iterations);
        iterations = clampInt(iterations, 1, 8);

        int retrieveK = readInt(cfg.thinking, "retrieveK", cfg.thinking.retrieveK);
        retrieveK = clampInt(retrieveK, 1, 128);

        int patience = readInt(cfg.thinking, "patience", 2);
        patience = clampInt(patience, 0, 6);

        double targetScore = readDouble(cfg.thinking, "targetScore", 0.75);

        return new IterativeThoughtEngine(
                retriever,
                intentDetector,
                strategy,
                evaluator,
                generator,
                iterations,
                retrieveK,
                draftsPerIteration,
                patience,
                targetScore
        );
    }

    private static int clampInt(int v, int lo, int hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
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