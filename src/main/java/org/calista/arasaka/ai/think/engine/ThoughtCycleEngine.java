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

import java.time.Clock;
import java.util.Objects;
import java.util.function.LongSupplier;

/**
 * ThoughtCycleEngine — единая точка входа в "thinking loop".
 *
 * <p>Enterprise bootstrap: один сильный пайплайн (retrieve -> refine -> draft -> eval -> self-correct -> memory),
 * без роутинга по стратегиям и без зашитых сценариев.</p>
 */
public interface ThoughtCycleEngine {

    ThoughtResult think(String userText, long seed);

    // -----------------------------------------------------------------------
    // Enterprise initialization (no extra top-level classes)
    // -----------------------------------------------------------------------

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

        /**
         * Deterministic seed provider for production runs.
         * By default uses currentTimeMillis (ok for CLI), but in enterprise you can inject stable request-id hash.
         */
        private LongSupplier seedProvider = System::currentTimeMillis;

        // thinking loop defaults (balanced)
        private int iterations = 4;
        private int retrieveK = 24;
        private int draftsPerIteration = 8;
        private int patience = 2;
        private double targetScore = 0.65;

        // refinement (data-driven, no semantic hardcode)
        private int refineRounds = 1;
        private int refineQueryBudget = 16;

        // LTM defaults (evidence-only)
        private boolean ltmEnabled = true;
        private int ltmCapacity = 50_000;
        private int ltmRecallK = 64;
        private double ltmWriteMinGroundedness = 0.55;

        // optional clock (useful for reproducible tests if seedProvider uses it)
        private Clock clock = Clock.systemUTC();

        private Builder(Retriever retriever, Tokenizer tokenizer) {
            this.retriever = Objects.requireNonNull(retriever, "retriever");
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        }

        // ---------------- core wiring ----------------

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

        /**
         * Stable seed for the whole engine (enterprise: use request-id hash).
         * Example: seedProvider(() -> murmur64(requestId)).
         */
        public Builder seedProvider(LongSupplier seedProvider) {
            this.seedProvider = Objects.requireNonNull(seedProvider, "seedProvider");
            return this;
        }

        public Builder clock(Clock clock) {
            this.clock = Objects.requireNonNull(clock, "clock");
            return this;
        }

        // ---------------- behavior knobs ----------------

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

        /**
         * Retrieval refinement rounds (0..2 recommended).
         * Enables: base retrieval -> derive queries from context tags/ids/terms -> refine retrieval.
         * This is what makes "alias" like "что ты такое?" pull the identity core without hardcode.
         */
        public Builder refineRounds(int v) {
            this.refineRounds = Math.max(0, v);
            return this;
        }

        /**
         * Max derived queries per refinement round.
         */
        public Builder refineQueryBudget(int v) {
            this.refineQueryBudget = Math.max(1, v);
            return this;
        }

        // ---------------- LTM knobs ----------------

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

        // ---------------- profiles (no scenario hardcode) ----------------

        /**
         * Balanced profile: good latency / quality trade-off.
         */
        public Builder balanced() {
            this.iterations = 4;
            this.retrieveK = 24;
            this.draftsPerIteration = 8;
            this.patience = 2;
            this.targetScore = 0.65;
            this.refineRounds = 1;
            this.refineQueryBudget = 16;
            this.ltmEnabled = true;
            this.ltmRecallK = 64;
            this.ltmWriteMinGroundedness = 0.55;
            return this;
        }

        /**
         * Aggressive profile: higher quality, more compute.
         */
        public Builder aggressive() {
            this.iterations = 6;
            this.retrieveK = 32;
            this.draftsPerIteration = 12;
            this.patience = 2;
            this.targetScore = 0.72;
            this.refineRounds = 2;
            this.refineQueryBudget = 24;
            this.ltmEnabled = true;
            this.ltmRecallK = 96;
            this.ltmWriteMinGroundedness = 0.60;
            return this;
        }

        /**
         * Builds a single-pipeline engine:
         * Intent -> retrieve -> refine -> (generator OR ContextAnswerStrategy) -> Advanced evaluation -> self-correct -> LTM.
         *
         * <p>No GreetingStrategy/RequestStrategy routing.</p>
         */
        public ThoughtCycleEngine build() {
            CandidateEvaluator ev = (this.evaluator != null)
                    ? this.evaluator
                    : new AdvancedCandidateEvaluator(tokenizer);

            // One strategy for all intents (intent only affects tags/format, not routing).
            ContextAnswerStrategy singleStrategy = new ContextAnswerStrategy();

            // If caller didn't override seedProvider, keep deterministic-ish default (clock-based).
            // Still stable per request if you inject your own.
            LongSupplier seeds = (seedProvider != null) ? seedProvider : (() -> clock.millis());

            // Wrap engine so caller can call think(text) with engine-provided seed if they want.
            // We still expose think(text, seed) as core API.
            IterativeThoughtEngine engine = new IterativeThoughtEngine(
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
                    ltmWriteMinGroundedness,
                    refineRounds,
                    refineQueryBudget
            );

            return new ThoughtCycleEngine() {
                @Override
                public ThoughtResult think(String userText, long seed) {
                    return engine.think(userText, seed);
                }

                /**
                 * Enterprise helper: deterministic seed per request via injected seedProvider.
                 * Keeps public API intact (still an interface method above), but gives you a clean entry point in code.
                 */
                public ThoughtResult think(String userText) {
                    return engine.think(userText, seeds.getAsLong());
                }
            };
        }
    }
}