package org.calista.arasaka.ai.think.engine.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.CloseableThreadContext;
import org.apache.logging.log4j.ThreadContext;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.retrieve.retriver.impl.KnowledgeRetriever;
import org.calista.arasaka.ai.think.ResponseStrategy;
import org.calista.arasaka.ai.think.ThoughtResult;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.engine.ThoughtCycleEngine;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.text.Normalizer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Enterprise deterministic think-loop:
 * retrieve -> rerank -> compress -> draft -> evaluate -> self-correct -> (optional) LTM.
 *
 * Enterprise rules:
 * - NEVER feed telemetry/critique/internal markers into retriever queries (prevents cache poisoning)
 * - NO domain semantic hardcode (identity/smalltalk lives in corpora)
 * - deterministic ordering and stable keys
 */
public final class IterativeThoughtEngine implements ThoughtCycleEngine {

    private static final Logger log = LogManager.getLogger(IterativeThoughtEngine.class);

    // Dedicated pool to avoid ForkJoin commonPool interference and stabilize tail latency.
    private static final AtomicLong EVAL_THREAD_ID = new AtomicLong(1);
    private static final int EVAL_PAR = Math.max(1, Math.min(8, Runtime.getRuntime().availableProcessors()));
    private static final ExecutorService EVAL_POOL = new ThreadPoolExecutor(
            EVAL_PAR,
            EVAL_PAR,
            30L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(4096),
            r -> {
                Thread t = new Thread(r, "think-eval-" + EVAL_THREAD_ID.getAndIncrement());
                t.setDaemon(true);
                return t;
            },
            new ThreadPoolExecutor.CallerRunsPolicy()
    );

    // tokenization for recall/overlap (>=3 keeps noise low; still configurable via code edits if needed)
    private static final java.util.regex.Pattern WORD = java.util.regex.Pattern.compile("[\\p{L}\\p{Nd}_]{3,}");

    // telemetry signatures + internal markers (generic, non-domain)
    private static final java.util.regex.Pattern TELEMETRY = java.util.regex.Pattern.compile("(?i)"
            + "(^\\s*(noctx;|err=|sec=)\\S*\\s*$)|"
            + "(\\w+=\\S+;)|"
            + "(\\bqc=)|(\\bg=)|(\\bnov=)|(\\brep=)|(\\bmd=)|"
            + "(\\bno_context\\b)|(\\bnoctx\\b)|(\\bmore_evidence\\b)|(\\bfix\\b)|(\\bnotes\\b)");

    // --- Production RAG knobs (engine-level, deterministic defaults) ---
    private static final int RERANK_N = 80;                   // rerank top-N
    private static final int RERANK_M = 16;                   // keep top-M after rerank
    private static final int COMPRESS_SENTENCES = 2;          // best sentences per statement
    private static final int COMPRESS_MAX_CHARS = 700;        // cap per compressed statement
    private static final double DERIVE_DF_CUT = 0.60;         // ban "water" tokens in query-derivation

    private final Retriever retriever;
    private final IntentDetector intentDetector;
    private final ResponseStrategy strategy;
    private final CandidateEvaluator evaluator;
    private final TextGenerator generator;

    private final int iterations;
    private final int retrieveK;
    private final int draftsPerIteration;
    private final int patience;
    private final double targetScore;

    // ---- LTM ----
    private final boolean ltmEnabled;
    private final int ltmCapacity;
    private final int ltmRecallK;
    private final double ltmWriteMinGroundedness;

    private final ConcurrentMap<String, Statement> ltmByKey = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, Double> ltmPriorityByKey = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, Set<String>> ltmTokensByKey = new ConcurrentHashMap<>();

    // refinement inside each iteration
    private final int refineRounds;
    private final int refineQueryBudget;

    // pipelines
    private final List<QueryProvider> queryProviders;
    private final List<Predicate<String>> queryFilters;
    private final List<StopRule> stopRules;

    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            ResponseStrategy strategy,
            CandidateEvaluator evaluator,
            TextGenerator generator,
            int iterations,
            int retrieveK,
            int draftsPerIteration,
            int patience,
            double targetScore,
            boolean ltmEnabled,
            int ltmCapacity,
            int ltmRecallK,
            double ltmWriteMinGroundedness
    ) {
        this(retriever, intentDetector, strategy, evaluator, generator,
                iterations, retrieveK, draftsPerIteration, patience, targetScore,
                ltmEnabled, ltmCapacity, ltmRecallK, ltmWriteMinGroundedness,
                1, 16);
    }

    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            ResponseStrategy strategy,
            CandidateEvaluator evaluator,
            TextGenerator generator,
            int iterations,
            int retrieveK,
            int draftsPerIteration,
            int patience,
            double targetScore,
            boolean ltmEnabled,
            int ltmCapacity,
            int ltmRecallK,
            double ltmWriteMinGroundedness,
            int refineRounds,
            int refineQueryBudget
    ) {

        this.retriever = Objects.requireNonNull(retriever, "retriever");
        this.intentDetector = Objects.requireNonNull(intentDetector, "intentDetector");
        this.strategy = Objects.requireNonNull(strategy, "strategy");
        this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
        this.generator = generator;

        this.iterations = Math.max(1, iterations);
        this.retrieveK = Math.max(1, retrieveK);

        // IMPORTANT: drafts=1 causes deterministic mode collapse (echo-loop) with beam/bigram generators.
        this.draftsPerIteration = Math.max(6, draftsPerIteration);

        this.patience = Math.max(0, patience);
        this.targetScore = targetScore;

        this.ltmEnabled = ltmEnabled;
        this.ltmCapacity = Math.max(0, ltmCapacity);
        this.ltmRecallK = Math.max(0, ltmRecallK);
        this.ltmWriteMinGroundedness = clamp01(ltmWriteMinGroundedness);

        this.refineRounds = Math.max(0, refineRounds);
        this.refineQueryBudget = Math.max(1, refineQueryBudget);

        this.queryProviders = List.of(
                QueryProvider.userText(),
                QueryProvider.intentName(),
                QueryProvider.bestTerms(10, 5) // hygienic terms from bestSoFar
        );

        this.queryFilters = List.of(
                Objects::nonNull,
                s -> !s.trim().isEmpty(),
                s -> !TELEMETRY.matcher(s).find(),
                s -> WORD.matcher(s.toLowerCase(Locale.ROOT)).find()
        );

        this.stopRules = List.of(
                StopRule.targetScore(targetScore),
                StopRule.patience(patience)
        );
    }

    @Override
    public ThoughtResult think(String userText, long seed) {

        // Correlation fields for structured logging (safe: do not put full user text in MDC).
        final String reqId = Long.toUnsignedString(mix(seed, System.nanoTime()), 36);
        try (final CloseableThreadContext.Instance ctc = CloseableThreadContext.put("reqId", reqId)
                .put("seed", Long.toUnsignedString(seed))
                .put("engine", "IterativeThoughtEngine")) {

            final long tStart = System.nanoTime();

            final String q0 = normalizeUserText(userText);
            final Intent intent = intentDetector.detect(q0);
            ThreadContext.put("intent", intent == null ? "UNKNOWN" : intent.name());

            if (log.isInfoEnabled()) {
                log.info("think.start len={} tok={} hasGen={} refineRounds={} ltmEnabled={} iterations={} retrieveK={} draftsPerIter={}",
                        q0.length(), approxQueryTokenCount(q0), generator != null, refineRounds, ltmEnabled, iterations, retrieveK, draftsPerIteration);
            }

            ThoughtState state = new ThoughtState();
            state.intent = intent;
            state.generator = generator;
            state.query = q0;

            if (state.tags == null) state.tags = new HashMap<>(16);
            initResponseSchema(state, q0, intent);

            Candidate globalBest = new Candidate("", Double.NEGATIVE_INFINITY, "", null);
            state.bestSoFar = globalBest;
            state.bestEvaluation = null;

            List<String> trace = new ArrayList<>(Math.max(8, iterations * 3));
            trace.add("reqId=" + reqId + " seed=" + Long.toUnsignedString(seed));
            int stagnation = 0;

            for (int iter = 1; iter <= iterations; iter++) {
                state.iteration = iter;
                state.seed = mix(seed, iter);
                ThreadContext.put("iter", Integer.toString(iter));

                // generator hint only (never used for retrieval)
                state.generationHint = buildGenerationHint(state);

                // ---- queries (no critique, no telemetry) ----
                List<String> baseQueries = buildQueries(q0, state);

                final long tRetr0 = System.nanoTime();
                RetrievalBundle bundle = retrieveRerankCompress(q0, baseQueries, state.seed);
                final long tRetr = System.nanoTime() - tRetr0;

                // ---- LTM recall (optional) ----
                final long tLtm0 = System.nanoTime();
                List<Statement> recalled = ltmEnabled ? recallFromLtm(baseQueries) : List.of();
                state.recalledMemory = recalled;
                final long tLtm = System.nanoTime() - tLtm0;

                // Merge LTM evidence into RAW (for storage/diagnostics), then rerank+compress again (cheap)
                if (!recalled.isEmpty()) {
                    Map<String, Statement> mergedByKey = new ConcurrentHashMap<>();
                    mergeStatements(bundle.rawEvidence, mergedByKey);
                    mergeStatements(recalled, mergedByKey);

                    List<Statement> merged = toDeterministicContext(mergedByKey);
                    List<Statement> reranked = rerankContext(q0, merged, RERANK_N, RERANK_M);
                    List<Statement> compressed = compressContextToStatements(q0, reranked, COMPRESS_SENTENCES, COMPRESS_MAX_CHARS);

                    bundle = bundle.withMerged(merged, reranked, compressed);
                }

                // ---- deterministic context for generator (compressed first) ----
                List<Statement> context = (bundle.compressedContext == null || bundle.compressedContext.isEmpty())
                        ? bundle.rerankedContext
                        : bundle.compressedContext;

                state.lastContext = context;

                // ---- deterministic generation control (tags) ----
                tuneGenerationTags(state, q0, context);

                // ---- drafts ----
                final long tGen0 = System.nanoTime();
                List<String> drafts = Drafts.sanitize(generateDrafts(q0, context, state), draftsPerIteration);
                final long tGen = System.nanoTime() - tGen0;

                // ---- evaluate (stable pool; no commonPool) ----
                final long tEval0 = System.nanoTime();
                List<Candidate> evaluated = evaluateDrafts(q0, context, drafts);
                final long tEval = System.nanoTime() - tEval0;

                Candidate bestIter = pickBest(evaluated).orElse(globalBest);

                state.lastCandidate = bestIter;
                state.lastCritique = bestIter.critique; // internal only (numeric-only)
                state.lastEvaluation = bestIter.evaluation;

                boolean improved = bestIter.score > globalBest.score + 1e-9;
                if (improved) {
                    globalBest = bestIter;
                    state.bestSoFar = bestIter;
                    state.bestEvaluation = bestIter.evaluation;
                    stagnation = 0;
                } else {
                    stagnation++;
                }

                // ---- LTM writeback (evidence-only) ----
                if (ltmEnabled && bestIter.evaluation != null && bestIter.evaluation.valid) {
                    // store evidence, not answer; use reranked/compressed evidence
                    maybeWritebackToLtm(bestIter, context);
                }

                trace.add("iter=" + iter
                        + " raw=" + safeSize(bundle.rawEvidence)
                        + " rerank=" + safeSize(bundle.rerankedContext)
                        + " compress=" + safeSize(bundle.compressedContext)
                        + " ltm=" + (state.recalledMemory == null ? 0 : state.recalledMemory.size())
                        + " drafts=" + drafts.size()
                        + " bestIter=" + fmt(bestIter.score)
                        + " global=" + fmt(globalBest.score)
                        + " tRetrMs=" + (tRetr / 1_000_000)
                        + " tLtmMs=" + (tLtm / 1_000_000)
                        + " tGenMs=" + (tGen / 1_000_000)
                        + " tEvalMs=" + (tEval / 1_000_000)
                        + (bundle.traceQuality != null ? (" rq=" + fmt(bundle.traceQuality)) : ""));

                if (log.isDebugEnabled()) {
                    log.debug("think.iter raw={} rerank={} compress={} ltm={} drafts={} bestIter={} global={} tRetrMs={} tLtmMs={} tGenMs={} tEvalMs={}{}",
                            safeSize(bundle.rawEvidence), safeSize(bundle.rerankedContext), safeSize(bundle.compressedContext),
                            state.recalledMemory == null ? 0 : state.recalledMemory.size(), drafts.size(),
                            fmt(bestIter.score), fmt(globalBest.score),
                            (tRetr / 1_000_000), (tLtm / 1_000_000), (tGen / 1_000_000), (tEval / 1_000_000),
                            bundle.traceQuality != null ? (" rq=" + fmt(bundle.traceQuality)) : "");
                }

                if (shouldStop(state, globalBest, stagnation)) break;
            }

            CandidateEvaluator.Evaluation ev = (globalBest == null) ? null : globalBest.evaluation;
            long tTotal = System.nanoTime() - tStart;
            if (log.isInfoEnabled()) {
                log.info("think.end answerLen={} score={} valid={} totalMs={} traceN={} ltmSize={}",
                        globalBest == null ? 0 : globalBest.text.length(),
                        fmt(globalBest == null ? 0.0 : globalBest.score),
                        ev != null && ev.valid,
                        (tTotal / 1_000_000),
                        trace.size(),
                        ltmByKey.size());
            }

            // avoid MDC leakage across requests (esp. in thread pools)
            ThreadContext.remove("intent");
            ThreadContext.remove("iter");

            return new ThoughtResult(globalBest == null ? "" : globalBest.text, trace, intent, globalBest, ev);
        }
    }

    // -------------------- evaluation (deterministic + stable parallelism) --------------------

    private List<Candidate> evaluateDrafts(String userText, List<Statement> context, List<String> drafts) {
        if (drafts == null || drafts.isEmpty()) return List.of();

        // Small lists are cheaper sequentially and avoid context switching.
        if (drafts.size() <= 2 || EVAL_PAR <= 1) {
            ArrayList<Candidate> out = new ArrayList<>(drafts.size());
            for (String d : drafts) {
                CandidateEvaluator.Evaluation ev = evaluator.evaluate(userText, d, context);
                out.add(Candidate.fromEvaluation(d, ev));
            }
            return out;
        }

        // Snapshot MDC to propagate correlation fields into worker threads (if evaluator logs).
        final Map<String, String> mdc = ThreadContext.getImmutableContext();

        ArrayList<CompletableFuture<Candidate>> fs = new ArrayList<>(drafts.size());
        for (String d : drafts) {
            fs.add(CompletableFuture.supplyAsync(() -> {
                if (mdc != null && !mdc.isEmpty()) ThreadContext.putAll(mdc);
                try {
                    CandidateEvaluator.Evaluation ev = evaluator.evaluate(userText, d, context);
                    return Candidate.fromEvaluation(d, ev);
                } finally {
                    if (mdc != null && !mdc.isEmpty()) ThreadContext.clearMap();
                }
            }, EVAL_POOL));
        }

        CompletableFuture.allOf(fs.toArray(new CompletableFuture[0])).join();

        ArrayList<Candidate> out = new ArrayList<>(drafts.size());
        for (CompletableFuture<Candidate> f : fs) {
            Candidate c;
            try {
                c = f.getNow(null);
            } catch (Throwable ignored) {
                c = null;
            }
            if (c != null) out.add(c);
        }
        return out;
    }

    // -------------------- retrieve -> rerank -> compress --------------------

    private RetrievalBundle retrieveRerankCompress(String userQuery, List<String> baseQueries, long seed) {
        // Fast path: if retriever is KnowledgeRetriever, we get a full Trace (already has rerank/compress internally).
        if (retriever instanceof KnowledgeRetriever kr) {
            // IMPORTANT: never pass critique/telemetry. baseQueries are already hygienic.
            String joined = String.join("\n", baseQueries);
            KnowledgeRetriever.Trace tr = kr.retrieveTrace(joined, retrieveK, seed);

            List<Statement> selected = (tr.selected == null) ? List.of() : tr.selected;

            // Prefer compressedContext from retriever if available; else compress here deterministically.
            List<Statement> compressed;
            if (tr.compressedContext != null && !tr.compressedContext.isEmpty()) {
                compressed = tr.compressedContext.stream()
                        .map(IterativeThoughtEngine::syntheticStatementOrNull)
                        .filter(Objects::nonNull)
                        .toList();
                if (compressed.isEmpty())
                    compressed = compressContextToStatements(userQuery, selected, COMPRESS_SENTENCES, COMPRESS_MAX_CHARS);
            } else {
                compressed = compressContextToStatements(userQuery, selected, COMPRESS_SENTENCES, COMPRESS_MAX_CHARS);
            }

            // rerankedTop is already post-rerank (topM). Use it if present; else rerank locally.
            List<Statement> reranked;
            if (tr.rerankedTop != null && !tr.rerankedTop.isEmpty()) {
                reranked = tr.rerankedTop.stream()
                        .map(s -> s == null ? null : s.item)
                        .filter(Objects::nonNull)
                        .toList();
            } else {
                reranked = rerankContext(userQuery, selected, RERANK_N, RERANK_M);
            }

            return new RetrievalBundle(selected, reranked, compressed, tr.quality);
        }

        // Fallback: use existing engine pipeline, then do rerank+compress here (production RAG).
        Map<String, Statement> mergedByKey = new ConcurrentHashMap<>();
        retrieveAndMerge(baseQueries, mergedByKey, seed);

        for (int r = 0; r < refineRounds; r++) {
            List<Statement> ctxNow = toDeterministicContext(mergedByKey);

            // NOTE: anti-water IDF derive
            List<String> derived = deriveQueriesFromContext(ctxNow, refineQueryBudget);
            if (derived.isEmpty()) break;
            retrieveAndMerge(derived, mergedByKey, mix(seed, 0x9E3779B97F4A7C15L + r));
        }

        List<Statement> raw = toDeterministicContext(mergedByKey);

        List<Statement> reranked = rerankContext(userQuery, raw, RERANK_N, RERANK_M);
        List<Statement> compressed = compressContextToStatements(userQuery, reranked, COMPRESS_SENTENCES, COMPRESS_MAX_CHARS);

        return new RetrievalBundle(raw, reranked, compressed, null);
    }

    private static int safeSize(List<?> xs) {
        return xs == null ? 0 : xs.size();
    }

    // -------------------- rerank --------------------

    /**
     * Deterministic rerank: scores each Statement by token overlap with the userQuery.
     * Keep topM from topN candidates to reduce noise before generation.
     */
    private static List<Statement> rerankContext(String userQuery, List<Statement> raw, int topN, int topM) {
        if (raw == null || raw.isEmpty()) return List.of();

        Set<String> qTok = tokens(userQuery);
        if (qTok.isEmpty()) {
            // No tokens: fallback to stable ordering
            return raw.stream()
                    .sorted(Comparator.comparing(IterativeThoughtEngine::stableStmtKey, Comparator.nullsLast(String::compareTo)))
                    .limit(Math.max(1, topM))
                    .toList();
        }

        int n = Math.min(Math.max(1, topN), raw.size());
        ArrayList<ScoredStmt> scored = new ArrayList<>(n);

        for (int i = 0; i < n; i++) {
            Statement s = raw.get(i);
            if (s == null || s.text == null || s.text.isBlank()) continue;
            double ov = overlapScore(qTok, s.text);
            scored.add(new ScoredStmt(s, ov));
        }

        scored.sort((a, b) -> {
            int cmp = Double.compare(b.score, a.score);
            if (cmp != 0) return cmp;
            String ka = stableStmtKey(a.stmt);
            String kb = stableStmtKey(b.stmt);
            if (ka == null && kb == null) return 0;
            if (ka == null) return 1;
            if (kb == null) return -1;
            return ka.compareTo(kb);
        });

        int m = Math.min(Math.max(1, topM), scored.size());
        ArrayList<Statement> out = new ArrayList<>(m);
        for (int i = 0; i < m; i++) out.add(scored.get(i).stmt);
        return out;
    }

    private static double overlapScore(Set<String> qTok, String text) {
        if (qTok == null || qTok.isEmpty() || text == null || text.isBlank()) return 0.0;
        Set<String> tTok = tokens(text);
        if (tTok.isEmpty()) return 0.0;

        int inter = 0;
        for (String t : qTok) if (tTok.contains(t)) inter++;

        // precision-like: how much of query is covered
        return (double) inter / (double) Math.max(1, qTok.size());
    }

    private static final class ScoredStmt {
        final Statement stmt;
        final double score;

        ScoredStmt(Statement stmt, double score) {
            this.stmt = stmt;
            this.score = score;
        }
    }

    // -------------------- generation --------------------

    private List<String> generateDrafts(String userText, List<Statement> context, ThoughtState state) {
        if (generator != null) return generator.generateN(userText, context, state, draftsPerIteration);

        ArrayList<String> out = new ArrayList<>(draftsPerIteration);
        for (int d = 0; d < draftsPerIteration; d++) {
            state.draftIndex = d;
            out.add(strategy.generate(userText, context, state));
        }
        return out;
    }

    // -------------------- stop rules --------------------

    private boolean shouldStop(ThoughtState state, Candidate globalBest, int stagnation) {
        for (StopRule r : stopRules) {
            if (r.shouldStop(state, globalBest, stagnation)) return true;
        }
        return false;
    }

    // -------------------- schema/hints --------------------

    private static String buildGenerationHint(ThoughtState state) {
        String sections = Optional.ofNullable(state)
                .map(s -> s.tags)
                .map(t -> t.get("response.sections"))
                .filter(v -> v != null && !v.isBlank())
                .orElse("summary,evidence,actions");

        String style = Optional.ofNullable(state)
                .map(s -> s.tags)
                .map(t -> t.get("response.style"))
                .filter(v -> v != null && !v.isBlank())
                .orElse("md");

        StringBuilder sb = new StringBuilder(256);
        sb.append("format=").append(style);
        sb.append(";sections=").append(sections);
        sb.append(";intent=").append(Optional.ofNullable(state.intent).orElse(Intent.UNKNOWN).name());
        sb.append(";iter=").append(state.iteration);

        Optional.ofNullable(state.lastEvaluation).ifPresent(last -> {
            sb.append(";last_g=").append(fmt(last.groundedness));
            sb.append(";last_r=").append(fmt(last.contradictionRisk));
            sb.append(";last_v=").append(last.valid ? 1 : 0);
        });

        Optional.ofNullable(state.bestEvaluation).ifPresent(best -> {
            sb.append(";best_g=").append(fmt(best.groundedness));
            sb.append(";best_r=").append(fmt(best.contradictionRisk));
            sb.append(";best_v=").append(best.valid ? 1 : 0);
        });

        // Numeric-only refinement signals (safe for generation, blocked from retrieval by TELEMETRY filter).
        Optional.ofNullable(state.lastCandidate).ifPresent(c -> {
            if (c.critique != null && !c.critique.isBlank()) sb.append(";last_ctl=").append(c.critique);
        });
        Optional.ofNullable(state.bestSoFar).ifPresent(c -> {
            if (c.critique != null && !c.critique.isBlank()) sb.append(";best_ctl=").append(c.critique);
        });

        appendRepairHint(state, sb);
        return sb.toString();
    }

    private static void initResponseSchema(ThoughtState state, String userText, Intent intent) {
        if (state == null) return;
        if (state.tags == null) state.tags = new HashMap<>(16);

        boolean ru = isMostlyCyrillic(userText);

        state.tags.putIfAbsent("response.style", "md");
        state.tags.putIfAbsent("response.sections", "summary,evidence,actions");
        state.tags.putIfAbsent("response.label.summary", ru ? "Вывод" : "Conclusion");
        state.tags.putIfAbsent("response.label.evidence", ru ? "Опора на контекст" : "Evidence from context");
        state.tags.putIfAbsent("response.label.actions", ru ? "Следующие шаги" : "Next steps");
    }

    private static void tuneGenerationTags(ThoughtState state, String userText, List<Statement> context) {
        if (state == null) return;
        if (state.tags == null) state.tags = new HashMap<>(16);

        int qTok = approxQueryTokenCount(userText);

        int minTok = safeInt(state.tags.get("gen.minTok"), -1);
        int maxTok = safeInt(state.tags.get("gen.maxTok"), -1);

        if (qTok > 0 && qTok <= 5) {
            if (minTok < 0) minTok = 18;
            if (maxTok < 0) maxTok = 48;
        }

        CandidateEvaluator.Evaluation last = state.lastEvaluation;
        if (last != null) {
            if (!last.valid) {
                if (minTok < 0) minTok = 20;
                else minTok = Math.min(96, minTok + 4);

                if (maxTok < 0) maxTok = 72;
                else maxTok = Math.min(160, maxTok + 8);

                applyRepairTags(state, last, userText);
            } else {
                clearRepairTags(state.tags);
            }
        }

        if (minTok > 0) state.tags.put("gen.minTok", Integer.toString(minTok));
        if (maxTok > 0) state.tags.put("gen.maxTok", Integer.toString(maxTok));
    }

    private static int safeInt(String s, int def) {
        if (s == null || s.isBlank()) return def;
        try {
            return Integer.parseInt(s.trim());
        } catch (Exception e) {
            return def;
        }
    }

    private static void applyRepairTags(ThoughtState state, CandidateEvaluator.Evaluation ev, String userText) {
        if (state == null || state.tags == null || ev == null) return;

        boolean addEvidence = ev.groundedness < 0.35;
        boolean reduceNovelty = ev.contextSupport < 0.35;
        boolean fixStructure = ev.structureScore < 0.35;
        boolean avoidEcho = isEchoRisk(userText, state.lastCandidate == null ? "" : state.lastCandidate.text);

        if (addEvidence) state.tags.put("repair.addEvidence", "1");
        if (reduceNovelty) state.tags.put("repair.reduceNovelty", "1");
        if (fixStructure) state.tags.put("repair.fixStructure", "1");
        if (avoidEcho) state.tags.put("repair.avoidEcho", "1");
    }

    private static void clearRepairTags(Map<String, String> tags) {
        if (tags == null || tags.isEmpty()) return;
        tags.remove("repair.addEvidence");
        tags.remove("repair.reduceNovelty");
        tags.remove("repair.fixStructure");
        tags.remove("repair.avoidEcho");
    }

    private static void appendRepairHint(ThoughtState state, StringBuilder sb) {
        if (state == null || state.tags == null || sb == null) return;

        // Compact, deterministic order
        String a = state.tags.get("repair.addEvidence");
        String n = state.tags.get("repair.reduceNovelty");
        String f = state.tags.get("repair.fixStructure");
        String e = state.tags.get("repair.avoidEcho");

        if ("1".equals(a) || "1".equals(n) || "1".equals(f) || "1".equals(e)) {
            sb.append(";repair=");
            boolean first = true;

            if ("1".equals(a)) {
                sb.append("addEvidence");
                first = false;
            }
            if ("1".equals(n)) {
                sb.append(first ? "" : ",").append("reduceNovelty");
                first = false;
            }
            if ("1".equals(f)) {
                sb.append(first ? "" : ",").append("fixStructure");
                first = false;
            }
            if ("1".equals(e)) {
                sb.append(first ? "" : ",").append("avoidEcho");
            }
        }
    }

    // Echo-risk: generic overlap heuristic, no domain rules.
    private static boolean isEchoRisk(String userText, String lastText) {
        if (lastText == null || lastText.isBlank()) return false;

        // Too short responses are often low-signal echoes
        if (lastText.trim().length() < 40) return true;

        Set<String> q = tokens(userText);
        Set<String> a = tokens(lastText);
        if (q.isEmpty() || a.isEmpty()) return false;

        int inter = 0;
        for (String t : q) if (a.contains(t)) inter++;

        double jacc = (double) inter / (double) Math.max(1, (q.size() + a.size() - inter));
        // High token overlap => likely echo/paraphrase loop
        return jacc >= 0.72;
    }

    // -------------------- pipeline: query building --------------------

    private List<String> buildQueries(String userText, ThoughtState state) {
        LinkedHashSet<String> out = new LinkedHashSet<>();
        for (QueryProvider qp : queryProviders) {
            List<String> qs = qp.queries(userText, state);
            if (qs == null) continue;
            for (String q : qs) {
                if (q == null) continue;
                String s = q.trim();
                boolean ok = true;
                for (Predicate<String> f : queryFilters) {
                    if (!f.test(s)) { ok = false; break; }
                }
                if (ok) out.add(s);
            }
        }
        if (out.isEmpty()) out.add(userText == null ? "" : userText.trim());
        return new ArrayList<>(out);
    }

    // -------------------- retrieval fallback helpers --------------------

    private void retrieveAndMerge(List<String> queries, Map<String, Statement> mergedByKey, long seed) {
        if (queries == null || queries.isEmpty()) return;
        for (int i = 0; i < queries.size(); i++) {
            String q = queries.get(i);
            if (q == null || q.isBlank()) continue;
            List<Statement> got = retriever.retrieve(q, retrieveK, mix(seed, i));
            mergeStatements(got, mergedByKey);
        }
    }

    private static void mergeStatements(List<Statement> stmts, Map<String, Statement> mergedByKey) {
        if (stmts == null || stmts.isEmpty()) return;
        for (Statement s : stmts) {
            if (s == null) continue;
            String k = stableStmtKey(s);
            if (k == null) continue;
            mergedByKey.putIfAbsent(k, s);
        }
    }

    private static List<Statement> toDeterministicContext(Map<String, Statement> mergedByKey) {
        if (mergedByKey == null || mergedByKey.isEmpty()) return List.of();
        ArrayList<Map.Entry<String, Statement>> xs = new ArrayList<>(mergedByKey.entrySet());
        xs.sort(Map.Entry.comparingByKey());
        ArrayList<Statement> out = new ArrayList<>(xs.size());
        for (var e : xs) out.add(e.getValue());
        return out;
    }

    // -------------------- query derivation (anti-water) --------------------

    private static List<String> deriveQueriesFromContext(List<Statement> ctx, int budget) {
        if (ctx == null || ctx.isEmpty() || budget <= 0) return List.of();

        // Token DF across statements
        HashMap<String, Integer> df = new HashMap<>(512);
        int n = 0;
        for (Statement s : ctx) {
            if (s == null || s.text == null || s.text.isBlank()) continue;
            n++;
            Set<String> st = tokens(s.text);
            for (String t : st) df.merge(t, 1, Integer::sum);
        }
        if (n <= 0) return List.of();

        // take informative tokens (low DF)
        ArrayList<String> cands = new ArrayList<>(df.size());
        for (var e : df.entrySet()) {
            double frac = (double) e.getValue() / (double) n;
            if (frac <= DERIVE_DF_CUT && e.getKey().length() >= 4) cands.add(e.getKey());
        }

        cands.sort(String::compareTo);
        if (cands.isEmpty()) return List.of();

        int k = Math.min(budget, cands.size());
        return cands.subList(0, k);
    }

    // -------------------- compression --------------------

    private static List<Statement> compressContextToStatements(String userQuery, List<Statement> ctx, int sentencesPerStmt, int maxChars) {
        if (ctx == null || ctx.isEmpty()) return List.of();
        int sps = Math.max(1, sentencesPerStmt);
        int cap = Math.max(120, maxChars);

        ArrayList<Statement> out = new ArrayList<>(ctx.size());
        for (Statement st : ctx) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            String txt = st.text.trim();

            // crude sentence split (unicode-safe enough)
            String[] parts = txt.split("(?<=[.!?…。！？])\\s+");
            StringBuilder b = new StringBuilder(Math.min(txt.length(), cap));
            int taken = 0;
            for (String p : parts) {
                String x = p == null ? "" : p.trim();
                if (x.isEmpty()) continue;
                if (b.length() > 0) b.append(' ');
                if (b.length() + x.length() > cap) {
                    int rem = cap - b.length();
                    if (rem > 12) b.append(x, 0, rem - 1).append('…');
                    break;
                }
                b.append(x);
                taken++;
                if (taken >= sps) break;
            }

            String compressed = b.toString().trim();
            if (compressed.isEmpty()) continue;

            Statement syn = new Statement();
            syn.id = st.id;
            syn.text = compressed;
            syn.tags = st.tags;

            out.add(syn);
        }

        // deterministic ordering
        out.sort(Comparator.comparing(IterativeThoughtEngine::stableStmtKey, Comparator.nullsLast(String::compareTo)));
        return out;
    }

    private static Statement syntheticStatementOrNull(Object o) {
        if (o == null) return null;
        if (o instanceof Statement s) return s;
        // If retriever returns compressed context as String, wrap it
        if (o instanceof String str) {
            if (str.isBlank()) return null;
            Statement st = new Statement();
            st.text = str;
            return st;
        }
        return null;
    }

    // -------------------- LTM --------------------

    private List<Statement> recallFromLtm(List<String> queries) {
        if (ltmByKey.isEmpty() || ltmRecallK <= 0) return List.of();

        Set<String> qTokens = queries.stream()
                .flatMap(q -> tokens(q).stream())
                .collect(Collectors.toCollection(LinkedHashSet::new));

        var all = new ArrayList<>(ltmByKey.entrySet());
        all.sort((a, b) -> {
            double sa = recallScore(a.getKey(), a.getValue(), qTokens);
            double sb = recallScore(b.getKey(), b.getValue(), qTokens);
            int cmp = Double.compare(sb, sa);
            return (cmp != 0) ? cmp : a.getKey().compareTo(b.getKey());
        });

        int k = Math.min(ltmRecallK, all.size());
        ArrayList<Statement> out = new ArrayList<>(k);
        for (int i = 0; i < k; i++) out.add(all.get(i).getValue());
        return out;
    }

    private double recallScore(String key, Statement s, Set<String> qTokens) {
        if (s == null || s.text == null || s.text.isBlank()) return 0.0;

        Set<String> st = ltmTokensByKey.computeIfAbsent(key, k -> tokens(s.text));

        int inter = 0;
        for (String t : qTokens) if (st.contains(t)) inter++;

        double overlap = (st.isEmpty() || qTokens.isEmpty()) ? 0.0 : (double) inter / (double) Math.max(1, qTokens.size());

        double pr = ltmPriorityByKey.getOrDefault(key, 0.0);
        return overlap * 0.80 + clamp01(pr) * 0.20;
    }

    private void maybeWritebackToLtm(Candidate best, List<Statement> ctx) {
        if (best == null || best.evaluation == null) return;
        if (!best.evaluation.valid) return;
        if (best.evaluation.groundedness < ltmWriteMinGroundedness) return;
        if (ltmCapacity <= 0) return;

        int stored = 0;
        for (Statement s : ctx) {
            if (s == null) continue;
            String k = stableStmtKey(s);
            if (k == null) continue;

            ltmByKey.putIfAbsent(k, s);
            ltmPriorityByKey.merge(k, 0.02, Double::sum);

            if (++stored >= Math.min(12, ctx.size())) break;
        }

        if (ltmByKey.size() > ltmCapacity) shrinkLtmToCapacity();
    }

    private void shrinkLtmToCapacity() {
        if (ltmByKey.size() <= ltmCapacity) return;

        ArrayList<String> keys = new ArrayList<>(ltmByKey.keySet());
        keys.sort((a, b) -> {
            double pa = ltmPriorityByKey.getOrDefault(a, 0.0);
            double pb = ltmPriorityByKey.getOrDefault(b, 0.0);
            int cmp = Double.compare(pa, pb);
            return (cmp != 0) ? cmp : a.compareTo(b);
        });

        int drop = Math.max(0, ltmByKey.size() - ltmCapacity);
        for (int i = 0; i < drop; i++) {
            String k = keys.get(i);
            ltmByKey.remove(k);
            ltmPriorityByKey.remove(k);
            ltmTokensByKey.remove(k);
        }
    }

    // -------------------- candidate picking --------------------

    private static Optional<Candidate> pickBest(List<Candidate> evaluated) {
        if (evaluated == null || evaluated.isEmpty()) return Optional.empty();
        Candidate best = null;
        for (Candidate c : evaluated) {
            if (c == null) continue;
            if (best == null || c.score > best.score) best = c;
        }
        return Optional.ofNullable(best);
    }

    // -------------------- normalization & tokens --------------------

    private String normalizeUserText(String s) {
        if (s == null) return "";
        String x = s.trim();
        if (x.isEmpty()) return "";
        x = Normalizer.normalize(x, Normalizer.Form.NFKC);
        x = x.replace('\u00A0', ' ');
        x = x.replaceAll("\\s+", " ").trim();
        return x;
    }

    private static boolean isMostlyCyrillic(String s) {
        if (s == null || s.isBlank()) return true;
        int cyr = 0, lat = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.UnicodeBlock.of(c) == Character.UnicodeBlock.CYRILLIC) cyr++;
            else if (Character.UnicodeBlock.of(c) == Character.UnicodeBlock.BASIC_LATIN) lat++;
        }
        return cyr >= lat;
    }

    private static Set<String> tokens(String text) {
        if (text == null || text.isBlank()) return Set.of();
        String s = text.toLowerCase(Locale.ROOT);
        LinkedHashSet<String> out = new LinkedHashSet<>();
        java.util.regex.Matcher m = WORD.matcher(s);
        while (m.find()) {
            String w = m.group();
            if (w == null) continue;
            out.add(w);
            if (out.size() >= 128) break;
        }
        return out;
    }

    private static int approxQueryTokenCount(String s) {
        if (s == null || s.isBlank()) return 0;
        int c = 0;
        java.util.regex.Matcher m = WORD.matcher(s.toLowerCase(Locale.ROOT));
        while (m.find()) {
            c++;
            if (c >= 256) break;
        }
        return c;
    }

    private static String stableStmtKey(Statement s) {
        if (s == null) return null;
        if (s.id != null && !s.id.isBlank()) return "id:" + s.id.trim();
        if (s.text == null || s.text.isBlank()) return null;
        String t = s.text.trim();
        if (t.length() > 160) t = t.substring(0, 160);
        return "tx:" + t.toLowerCase(Locale.ROOT);
    }

    // -------------------- math/format --------------------

    private static double clamp01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    private static long mix(long a, long b) {
        long x = a ^ (b * 0x9E3779B97F4A7C15L);
        x ^= (x >>> 30);
        x *= 0xBF58476D1CE4E5B9L;
        x ^= (x >>> 27);
        x *= 0x94D049BB133111EBL;
        x ^= (x >>> 31);
        return x;
    }

    private static String fmt(double v) {
        if (!Double.isFinite(v)) return "0";
        return String.format(Locale.ROOT, "%.4f", v);
    }

    // -------------------- nested pipeline contracts --------------------

    private interface QueryProvider {
        List<String> queries(String userText, ThoughtState state);

        static QueryProvider userText() {
            return (u, s) -> u == null || u.isBlank() ? List.of() : List.of(u);
        }

        static QueryProvider intentName() {
            return (u, s) -> (s == null || s.intent == null) ? List.of() : List.of(s.intent.name());
        }

        static QueryProvider bestTerms(int maxTerms, int minLen) {
            int mt = Math.max(1, maxTerms);
            int ml = Math.max(2, minLen);
            return (u, s) -> {
                Candidate best = (s == null) ? null : s.bestSoFar;
                if (best == null || best.text == null || best.text.isBlank()) return List.of();
                return tokens(best.text).stream().filter(t -> t.length() >= ml).limit(mt).toList();
            };
        }
    }

    private static final class Drafts {
        static List<String> sanitize(List<String> drafts, int limit) {
            int lim = Math.max(1, limit);
            if (drafts == null || drafts.isEmpty()) return List.of("");

            LinkedHashSet<String> uniq = new LinkedHashSet<>();
            for (String d : drafts) {
                if (d == null) continue;
                String s = d.trim();
                if (s.isEmpty()) continue;
                uniq.add(s);
                if (uniq.size() >= lim) break;
            }
            return uniq.isEmpty() ? List.of("") : new ArrayList<>(uniq);
        }
    }

    private interface StopRule {
        boolean shouldStop(ThoughtState state, Candidate best, int stagnation);

        static StopRule targetScore(double target) {
            return (s, b, stg) -> b != null && b.score >= target;
        }

        static StopRule patience(int patience) {
            return (s, b, stg) -> stg >= patience;
        }
    }

    private record RetrievalBundle(
            List<Statement> rawEvidence,
            List<Statement> rerankedContext,
            List<Statement> compressedContext,
            Double traceQuality
    ) {
        RetrievalBundle withMerged(List<Statement> raw, List<Statement> reranked, List<Statement> compressed) {
            return new RetrievalBundle(raw, reranked, compressed, traceQuality);
        }
    }
}