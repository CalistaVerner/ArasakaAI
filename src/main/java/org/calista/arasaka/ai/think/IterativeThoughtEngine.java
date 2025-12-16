package org.calista.arasaka.ai.think;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.Retriever;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Multi-iteration "think → retrieve → draft → evaluate → refine" pipeline.
 *
 * Key properties:
 *  - No hard-coded critique rules in the engine. All scoring + critique is delegated to CandidateEvaluator.
 *  - Strategies are responsible for assembling the answer (summary → context-backed arguments → steps),
 *    and for self-correction via {@link ThoughtState#bestSoFar} and {@link ThoughtState#lastCritique}.
 *  - Designed to be backend-agnostic: the strategy may call an external generator (BeamSearch/TensorFlow)
 *    via {@link ThoughtState#generator}, or fall back to template-based generation.
 */
public final class IterativeThoughtEngine implements ThoughtCycleEngine {
    private static final Logger log = LogManager.getLogger(IterativeThoughtEngine.class);

    private final Retriever retriever;
    private final IntentDetector intentDetector;
    private final List<ResponseStrategy> strategies;
    private final CandidateEvaluator evaluator;

    private final int iterations;
    private final int retrieveK;

    /**
     * Early stop if we didn't improve for this many iterations.
     * Keep it small so the engine doesn't waste cycles when it's already good.
     */
    private final int patience;

    /**
     * Optional target score for early exit.
     * If evaluator is heuristic, keep it around ~0.9; if evaluator is calibrated, tune as needed.
     */
    private final double targetScore;

    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            List<ResponseStrategy> strategies,
            CandidateEvaluator evaluator,
            int iterations,
            int retrieveK
    ) {
        this(retriever, intentDetector, strategies, evaluator, iterations, retrieveK, 2, 0.92);
    }

    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            List<ResponseStrategy> strategies,
            CandidateEvaluator evaluator,
            int iterations,
            int retrieveK,
            int patience,
            double targetScore
    ) {
        this.retriever = Objects.requireNonNull(retriever, "retriever");
        this.intentDetector = Objects.requireNonNull(intentDetector, "intentDetector");
        this.strategies = List.copyOf(Objects.requireNonNull(strategies, "strategies"));
        if (this.strategies.isEmpty()) throw new IllegalArgumentException("strategies must not be empty");
        this.evaluator = Objects.requireNonNull(evaluator, "evaluator");

        this.iterations = Math.max(1, iterations);
        this.retrieveK = Math.max(1, retrieveK);
        this.patience = Math.max(0, patience);
        this.targetScore = targetScore;
    }

    @Override
    public ThoughtResult think(String userText, long seed) {
        String user = (userText == null) ? "" : userText.trim();
        Intent intent = intentDetector.detect(user);

        ThoughtState state = new ThoughtState();
        state.intent = intent;

        ArrayList<String> trace = new ArrayList<>(iterations * 2);

        Candidate best = new Candidate("", -1e9, "");
        String query = user;
        int noImprove = 0;

        for (int i = 1; i <= iterations; i++) {
            state.iteration = i;
            state.query = query;

            long iterSeed = mix(seed, i);
            List<Statement> ctx = retriever.retrieve(query, retrieveK, iterSeed);

            ResponseStrategy strategy = pickStrategy(intent);
            String candidateText = safeTrim(strategy.generate(user, ctx, state));

            CandidateEvaluator.Evaluation ev = evaluator.evaluate(user, candidateText, ctx);
            state.lastCritique = safeTrim(ev.critique);

            Candidate cand = new Candidate(candidateText, ev.score, ev.critique);
            boolean improved = cand.score > best.score + 1e-9;

            if (improved) {
                best = cand;
                state.bestSoFar = best;
                noImprove = 0;
            } else {
                noImprove++;
            }

            trace.add("iter#" + i +
                    " score=" + fmt(ev.score) +
                    " cov=" + fmt(ev.coverage) +
                    " ctx=" + fmt(ev.contextSupport) +
                    " pen=" + fmt(ev.stylePenalty) +
                    " critique=" + safeTrim(ev.critique));

            log.debug(
                    "think.iter={} intent={} ctx={} score={} best={} cov={} ctxSup={} pen={} critique='{}'",
                    i, intent, ctx.size(), fmt(ev.score), fmt(best.score), fmt(ev.coverage), fmt(ev.contextSupport),
                    fmt(ev.stylePenalty), safeTrim(ev.critique)
            );

            if (best.score >= targetScore) {
                trace.add("earlyStop: targetScore reached");
                break;
            }
            if (patience > 0 && noImprove >= patience) {
                trace.add("earlyStop: noImprove>=" + patience);
                break;
            }

            // Next iteration: steer retrieval with critique + best so far.
            // IMPORTANT: we do not hardcode critique semantics here; we only provide the evaluator critique as signal.
            query = buildNextQuery(user, state);
        }

        log.info("think.done intent={} iters={} bestScore={} answerChars={}", intent, state.iteration, fmt(best.score), best.text.length());
        return new ThoughtResult(best.text, List.copyOf(trace));
    }

    private ResponseStrategy pickStrategy(Intent intent) {
        for (ResponseStrategy s : strategies) {
            if (s.supports(intent)) return s;
        }
        return strategies.get(0);
    }

    private static String buildNextQuery(String userText, ThoughtState state) {
        StringBuilder q = new StringBuilder(256);
        q.append(userText);

        // Feed back the critique (self-correction signal).
        if (!state.lastCritique.isBlank()) q.append(' ').append(state.lastCritique);

        // Feed back a compact slice of the best answer, but do not explode the query.
        if (state.bestSoFar != null && state.bestSoFar.text != null && !state.bestSoFar.text.isBlank()) {
            String best = state.bestSoFar.text;
            int cap = Math.min(220, best.length());
            q.append(" best:").append(best, 0, cap);
        }
        return q.toString();
    }

    private static String safeTrim(String s) {
        return s == null ? "" : s.trim();
    }

    private static String fmt(double v) {
        if (!Double.isFinite(v)) return "NaN";
        return String.format(java.util.Locale.ROOT, "%.3f", v);
    }

    private static long mix(long seed, int iter) {
        long x = seed ^ (0x9E3779B97F4A7C15L * iter);
        x ^= (x >>> 33);
        x *= 0xff51afd7ed558ccdL;
        x ^= (x >>> 33);
        x *= 0xc4ceb9fe1a85ec53L;
        x ^= (x >>> 33);
        return x;
    }
}