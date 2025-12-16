package org.calista.arasaka.ai.core;

import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.retrieve.*;
import org.calista.arasaka.ai.text.*;
import org.calista.arasaka.ai.think.*;

import java.util.List;

public final class AIComposer {
    private AIComposer() {}

    public static IterativeThoughtEngine buildThinking(AIConfig cfg, KnowledgeBase kb) {
        Tokenizer tokenizer = new SimpleTokenizer();
        Scorer scorer = new TokenOverlapScorer(tokenizer);

        ExplorationConfig expl = new ExplorationConfig(
                cfg.thinking.exploration.temperature,
                cfg.thinking.exploration.topK
        );

        Retriever retriever = new KnowledgeRetriever(kb, scorer, new SoftmaxSampler(), expl);

        IntentDetector intentDetector = new SimpleIntentDetector();
        CandidateEvaluator evaluator = new SimpleCandidateEvaluator(tokenizer);

        List<ResponseStrategy> strategies = List.of(
                new GreetingStrategy(),
                new RequestStrategy(),
                new ContextAnswerStrategy()
        );

        return new IterativeThoughtEngine(
                retriever,
                intentDetector,
                strategies,
                evaluator,
                cfg.thinking.iterations,
                cfg.thinking.retrieveK
        );
    }
}