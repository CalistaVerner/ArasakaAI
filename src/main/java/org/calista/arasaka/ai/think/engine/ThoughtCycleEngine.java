package org.calista.arasaka.ai.think.engine;

import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.think.ContextAnswerStrategy;
import org.calista.arasaka.ai.think.TextGenerator;
import org.calista.arasaka.ai.think.ThoughtResult;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.candidate.impl.AdvancedCandidateEvaluator;
import org.calista.arasaka.ai.think.engine.impl.IterativeThoughtEngine;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.intent.impl.SimpleIntentDetector;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.Objects;

/**
 * ThoughtCycleEngine — единая точка входа в "thinking loop".
 *
 * <p>Enterprise bootstrap: один сильный пайплайн (retrieve -> draft -> eval -> self-correct -> memory),
 * без роутинга по стратегиям и без зашитых сценариев.</p>
 */
public interface ThoughtCycleEngine {

    ThoughtResult think(String userText, long seed);

    // -----------------------------------------------------------------------
    // Enterprise initialization (no extra top-level classes)
    // -----------------------------------------------------------------------

    /**
     * Quick default enterprise pipeline:
     * - Intent: deterministic keyword detector (tunable via ctor later)
     * - Strategy: single smart ContextAnswerStrategy
     * - Evaluator: AdvancedCandidateEvaluator (tokenizer required)
     * - Optional generator: BeamSearch/TensorFlow
     * - Iterative loop: production-grade defaults
     */
    static ThoughtCycleEngine createEnterprise(Retriever retriever, Tokenizer tokenizer, TextGenerator generator) {
        return builder(retriever, tokenizer).generator(generator).build();
    }

    static ThoughtCycleEngine createEnterprise(Retriever retriever, Tokenizer tokenizer) {
        return builder(retriever, tokenizer).build();
    }

    /**
     * Fluent builder with safe defaults.
     * Intentionally kept inside the interface to avoid "million classes".
     */
    static Builder builder(Retriever retriever, Tokenizer tokenizer) {
        return new Builder(retriever, tokenizer);
    }

    final class Builder {
        private final Retriever retriever;
        private final Tokenizer tokenizer;

        private IntentDetector intentDetector = new SimpleIntentDetector();
        private CandidateEvaluator evaluator; // lazily created from tokenizer
        private TextGenerator generator = null;

        // thinking loop defaults (balanced)
        private int iterations = 4;
        private int retrieveK = 24;
        private int draftsPerIteration = 8;
        private int patience = 2;
        private double targetScore = 0.65;

        // LTM defaults (evidence-only)
        private boolean ltmEnabled = true;
        private int ltmCapacity = 50_000;
        private int ltmRecallK = 64;
        private double ltmWriteMinGroundedness = 0.55;

        private Builder(Retriever retriever, Tokenizer tokenizer) {
            this.retriever = Objects.requireNonNull(retriever, "retriever");
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        }

        public Builder intentDetector(IntentDetector detector) {
            this.intentDetector = Objects.requireNonNull(detector, "intentDetector");
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

        public Builder iterations(int v) {
            this.iterations = Math.max(1, v);
            return this;
        }

        public Builder retrieveK(int v) {
            this.retrieveK = Math.max(1, v);
            return this;
        }

        public Builder draftsPerIteration(int v) {
            this.draftsPerIteration = Math.max(1, v);
            return this;
        }

        public Builder patience(int v) {
            this.patience = Math.max(0, v);
            return this;
        }

        public Builder targetScore(double v) {
            this.targetScore = v;
            return this;
        }

        public Builder ltm(boolean enabled) {
            this.ltmEnabled = enabled;
            return this;
        }

        public Builder ltmCapacity(int v) {
            this.ltmCapacity = Math.max(0, v);
            return this;
        }

        public Builder ltmRecallK(int v) {
            this.ltmRecallK = Math.max(0, v);
            return this;
        }

        public Builder ltmWriteMinGroundedness(double v) {
            this.ltmWriteMinGroundedness = v;
            return this;
        }

        /**
         * Builds a single-pipeline engine:
         * Intent -> retrieve -> (generator OR ContextAnswerStrategy) -> Advanced evaluation -> self-correct -> LTM.
         *
         * <p>No GreetingStrategy/RequestStrategy routing.</p>
         */
        public ThoughtCycleEngine build() {
            CandidateEvaluator ev = (this.evaluator != null)
                    ? this.evaluator
                    : new AdvancedCandidateEvaluator(tokenizer);

            // One strategy for all intents (intent only affects tags/format, not routing).
            ContextAnswerStrategy singleStrategy = new ContextAnswerStrategy();

            return new IterativeThoughtEngine(
                    retriever,
                    intentDetector,
                    singleStrategy,
                    ev,
                    generator,
                    iterations,
                    retrieveK,
                    draftsPerIteration,
                    patience,
                    targetScore,
                    ltmEnabled,
                    ltmCapacity,
                    ltmRecallK,
                    ltmWriteMinGroundedness
            );
        }
    }
}