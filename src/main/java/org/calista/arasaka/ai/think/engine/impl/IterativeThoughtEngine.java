// FILE: IterativeThoughtEngine.java
package org.calista.arasaka.ai.think.engine.impl;

import org.apache.logging.log4j.CloseableThreadContext;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.ThreadContext;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.retrieve.retriver.impl.KnowledgeRetriever;
import org.calista.arasaka.ai.think.ThoughtResult;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateControlSignals;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.candidate.impl.MultiCriteriaCandidateEvaluator;
import org.calista.arasaka.ai.think.engine.ThoughtCycleEngine;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.response.ResponseStrategy;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;

import java.text.Normalizer;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Enterprise deterministic think-loop:
 * retrieve -> rerank -> compress -> draft -> evaluate -> self-correct -> (optional) LTM.
 *
 * Key enterprise upgrades in this version:
 *  3.1 Two-phase pipeline Explore -> Exploit (state.tags + few config knobs).
 *  3.2 Mandatory strict verify-pass for best candidate (repair tags + force continue).
 *  3.3 Hybrid rerank: overlap + source priority + LTM penalty + DF penalty (no new entities).
 *  3.5 Anti-collapse deterministic diversity: per-draft seed mixing + gen.diversity tags.
 *  3.6 LTM self-regulation: decay + promotion (fresh retrieval confirms -> priority boost).
 *
 * Previous fixes retained:
 *  - DO NOT poison retrieval with "UNKNOWN" intent token.
 *  - DO NOT derive refine queries from a bad/invalid bestSoFar.
 *  - Keep multiple drafts (avoid mode collapse).
 *  - Very short inputs (<=2 tokens) -> summary-only.
 */
public final class IterativeThoughtEngine implements ThoughtCycleEngine {

    private static final Logger log = LogManager.getLogger(IterativeThoughtEngine.class);

    // tokenization for recall/overlap
    private static final java.util.regex.Pattern WORD = java.util.regex.Pattern.compile("[\\p{L}\\p{Nd}_]{3,}");

    // telemetry signatures + internal markers (generic, non-domain)
    private static final java.util.regex.Pattern TELEMETRY = java.util.regex.Pattern.compile("(?i)"
            + "(^\\s*(noctx;|err=|sec=)\\S*\\s*$)|"
            + "(\\w+=\\S+;)|"
            + "(\\bqc=)|(\\bg=)|(\\bnov=)|(\\brep=)|(\\bmd=)|"
            + "(\\bno_context\\b)|(\\bnoctx\\b)|(\\bmore_evidence\\b)|(\\bfix\\b)|(\\bnotes\\b)");

    // --- RAG knobs ---
    private static final int RERANK_N = 120;
    private static final int RERANK_M = 18;
    private static final int COMPRESS_SENTENCES = 2;
    private static final int COMPRESS_MAX_CHARS = 700;
    private static final double DERIVE_DF_CUT = 0.60;

    // --- Explore/Exploit defaults (can be overridden by Think.Config) ---
    private final int exploreIters;
    private final double exploreRetrieveKMult;
    private final double exploitEvidenceStrictness;

    // --- strict verify-pass policy ---
    private final boolean strictVerifyEnabled;
    private final double strictVerifyMinScore;

    private final Retriever retriever;
    private final IntentDetector intentDetector;
    private final ResponseStrategy strategy;
    private final CandidateEvaluator evaluator;
    private final TextGenerator generator;

    // injected eval resources (owned by Think)
    private final ExecutorService evalPool;
    private final int evalParallelism;

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

    // LTM self-regulation
    private final AtomicLong ltmTick = new AtomicLong(0);
    private final ConcurrentMap<String, Long> ltmLastTouchTick = new ConcurrentHashMap<>();
    private final double ltmDecayPerTick;     // e.g. 0.0015
    private final double ltmPromotionBoost;   // e.g. +0.06

    // refinement
    private final int refineRounds;
    private final int refineQueryBudget;

    private final List<QueryProvider> queryProviders;
    private final List<Predicate<String>> queryFilters;
    private final List<StopRule> stopRules;

    /**
     * Backward-compatible constructor (old signature): uses built-in defaults for new knobs.
     */
    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            ResponseStrategy strategy,
            CandidateEvaluator evaluator,
            TextGenerator generator,
            ExecutorService evalPool,
            int evalParallelism,
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
                evalPool, evalParallelism,
                iterations, retrieveK, draftsPerIteration, patience, targetScore,
                ltmEnabled, ltmCapacity, ltmRecallK, ltmWriteMinGroundedness,
                1, 16,
                2, 1.5, 0.78,
                true, 0.62,
                0.0015, 0.06
        );
    }

    /**
     * Backward-compatible constructor (old + refine): uses built-in defaults for new knobs.
     */
    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            ResponseStrategy strategy,
            CandidateEvaluator evaluator,
            TextGenerator generator,
            ExecutorService evalPool,
            int evalParallelism,
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
        this(retriever, intentDetector, strategy, evaluator, generator,
                evalPool, evalParallelism,
                iterations, retrieveK, draftsPerIteration, patience, targetScore,
                ltmEnabled, ltmCapacity, ltmRecallK, ltmWriteMinGroundedness,
                refineRounds, refineQueryBudget,
                2, 1.5, 0.78,
                true, 0.62,
                0.0015, 0.06
        );
    }

    /**
     * Full constructor (new knobs injected from Think.Config).
     */
    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            ResponseStrategy strategy,
            CandidateEvaluator evaluator,
            TextGenerator generator,
            ExecutorService evalPool,
            int evalParallelism,
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
            int refineQueryBudget,
            int exploreIters,
            double exploreRetrieveKMult,
            double exploitEvidenceStrictness,
            boolean strictVerifyEnabled,
            double strictVerifyMinScore,
            double ltmDecayPerTick,
            double ltmPromotionBoost
    ) {
        this.retriever = Objects.requireNonNull(retriever, "retriever");
        this.intentDetector = Objects.requireNonNull(intentDetector, "intentDetector");
        this.strategy = Objects.requireNonNull(strategy, "strategy");
        this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
        this.generator = generator;

        this.evalPool = Objects.requireNonNull(evalPool, "evalPool");
        this.evalParallelism = Math.max(1, evalParallelism);

        this.iterations = Math.max(1, iterations);
        this.retrieveK = Math.max(1, retrieveK);

        // IMPORTANT: drafts=1 causes deterministic mode collapse with beam/bigram generators.
        this.draftsPerIteration = Math.max(6, draftsPerIteration);

        this.patience = Math.max(0, patience);
        this.targetScore = targetScore;

        this.ltmEnabled = ltmEnabled;
        this.ltmCapacity = Math.max(0, ltmCapacity);
        this.ltmRecallK = Math.max(0, ltmRecallK);
        this.ltmWriteMinGroundedness = clamp01(ltmWriteMinGroundedness);

        this.refineRounds = Math.max(0, refineRounds);
        this.refineQueryBudget = Math.max(1, refineQueryBudget);

        this.exploreIters = Math.max(0, exploreIters);
        this.exploreRetrieveKMult = Math.max(1.0, exploreRetrieveKMult);
        this.exploitEvidenceStrictness = clamp01(exploitEvidenceStrictness);

        this.strictVerifyEnabled = strictVerifyEnabled;
        this.strictVerifyMinScore = strictVerifyMinScore;

        this.ltmDecayPerTick = Math.max(0.0, ltmDecayPerTick);
        this.ltmPromotionBoost = Math.max(0.0, ltmPromotionBoost);

        this.queryProviders = List.of(
                QueryProvider.userText(),
                QueryProvider.intentName(),       // FIX: skips UNKNOWN
                QueryProvider.bestTerms(10, 5)    // FIX: gated on bestSoFar quality
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

        if (log.isInfoEnabled()) {
            log.info("IterativeThoughtEngine created: iterations={} retrieveK={} draftsPerIter={} patience={} targetScore={} refineRounds={} refineQueryBudget={} exploreIters={} exploreKMult={} exploitStrict={} strictVerify={} strictMinScore={} ltmEnabled={} ltmCap={} ltmRecallK={} evalPar={}",
                    this.iterations, this.retrieveK, this.draftsPerIteration, this.patience, fmt(this.targetScore),
                    this.refineRounds, this.refineQueryBudget,
                    this.exploreIters, fmt(this.exploreRetrieveKMult), fmt(this.exploitEvidenceStrictness),
                    this.strictVerifyEnabled, fmt(this.strictVerifyMinScore),
                    this.ltmEnabled, this.ltmCapacity, this.ltmRecallK,
                    this.evalParallelism);
        }
    }

    @Override
    public ThoughtResult think(String userText, long seed) {

        final String reqId = Long.toUnsignedString(mix(seed, System.nanoTime()), 36);
        try (final CloseableThreadContext.Instance ctc = CloseableThreadContext.put("reqId", reqId)
                .put("seed", Long.toUnsignedString(seed))
                .put("engine", "IterativeThoughtEngine")) {

            final long tStart = System.nanoTime();

            final String q0 = normalizeUserText(userText);
            final Intent intent = intentDetector.detect(q0);
            ThreadContext.put("intent", intent == null ? "UNKNOWN" : intent.name());

            if (log.isInfoEnabled()) {
                log.info("think.start len={} tok={} intent={} hasGen={} refineRounds={} ltmEnabled={} iterations={} retrieveK={} draftsPerIter={} evalPar={}",
                        q0.length(), approxQueryTokenCount(q0),
                        (intent == null ? "UNKNOWN" : intent.name()),
                        generator != null, refineRounds, ltmEnabled, iterations, retrieveK, draftsPerIteration, evalParallelism);
            }
            if (log.isDebugEnabled()) {
                log.debug("think.q0='{}'", preview(q0, 240));
            }

            ThoughtState state = new ThoughtState();
            state.intent = intent;
            state.generator = generator;
            state.query = q0;

            if (state.tags == null) state.tags = new HashMap<>(24);
            initResponseSchema(state, q0, intent); // FIX: short queries => summary only

            Candidate globalBest = new Candidate("", Double.NEGATIVE_INFINITY, "", null);
            state.bestSoFar = globalBest;
            state.bestEvaluation = null;

            List<String> trace = new ArrayList<>(Math.max(10, iterations * 4));
            trace.add("reqId=" + reqId + " seed=" + Long.toUnsignedString(seed));

            int stagnation = 0;
            StopDecision stopDecision = null;

            // if strict verify fails, we prevent early stop this episode
            int forcedExtraIters = 0;

            for (int iter = 1; iter <= iterations + forcedExtraIters; iter++) {
                state.iteration = iter;
                long iterSeed = mix(seed, iter);
                state.seed = iterSeed;
                ThreadContext.put("iter", Integer.toString(iter));

                // 3.1 Explore -> Exploit
                boolean explore = (iter <= exploreIters);
                setPhaseControls(state, explore);

                // generator hint only (never used for retrieval)
                state.generationHint = buildGenerationHint(state);

                // ---- queries ----
                List<String> baseQueries = buildQueries(q0, state);
                if (log.isDebugEnabled()) {
                    log.debug("iter={} phase={} queries.n={} queries={}", iter, explore ? "EXPLORE" : "EXPLOIT", baseQueries.size(), baseQueries);
                }

                // ---- retrieve/rerank/compress ----
                final long tRetr0 = System.nanoTime();

                int effK = explore ? (int) Math.ceil(retrieveK * exploreRetrieveKMult) : retrieveK;

                RetrievalBundle bundle = retrieveRerankCompress(q0, baseQueries, iterSeed, effK);
                final long tRetr = System.nanoTime() - tRetr0;

                // 3.6 Promotion: if fresh retrieval matches LTM => boost
                if (ltmEnabled) {
                    promoteLtmOnFreshEvidence(bundle.rawEvidence);
                }

                // ---- LTM recall ----
                final long tLtm0 = System.nanoTime();
                List<Statement> recalled = ltmEnabled ? recallFromLtm(baseQueries) : List.of();
                state.recalledMemory = recalled;
                final long tLtm = System.nanoTime() - tLtm0;

                // ---- merge + hybrid rerank ----
                final Map<String, Byte> origin = new HashMap<>(256); // 1=retrieval, 2=ltm, 3=both
                if (bundle.rawEvidence != null) {
                    for (Statement s : bundle.rawEvidence) {
                        String k = stableStmtKey(s);
                        if (k != null) origin.put(k, (byte) 1);
                    }
                }
                if (recalled != null && !recalled.isEmpty()) {
                    for (Statement s : recalled) {
                        String k = stableStmtKey(s);
                        if (k == null) continue;
                        origin.merge(k, (byte) 2, (a, b) -> (byte) (a | b));
                    }
                }

                if (!recalled.isEmpty()) {
                    Map<String, Statement> mergedByKey = new ConcurrentHashMap<>();
                    mergeStatements(bundle.rawEvidence, mergedByKey);
                    mergeStatements(recalled, mergedByKey);

                    List<Statement> merged = toDeterministicContext(mergedByKey);

                    DfStats df = buildDfStats(merged, 96);
                    List<Statement> reranked = rerankContextHybrid(q0, merged, RERANK_N, RERANK_M, origin, df.df, df.docs);
                    List<Statement> compressed = compressContextToStatements(q0, reranked, COMPRESS_SENTENCES, COMPRESS_MAX_CHARS);

                    bundle = bundle.withMerged(merged, reranked, compressed);
                } else {
                    // even without LTM: rerank hybrid with source priority + DF penalty
                    DfStats df = buildDfStats(bundle.rawEvidence, 96);
                    List<Statement> reranked = rerankContextHybrid(q0, bundle.rawEvidence, RERANK_N, RERANK_M, origin, df.df, df.docs);
                    List<Statement> compressed = compressContextToStatements(q0, reranked, COMPRESS_SENTENCES, COMPRESS_MAX_CHARS);
                    bundle = bundle.withMerged(bundle.rawEvidence, reranked, compressed);
                }

                // ---- context for generator (compressed first) ----
                List<Statement> context = (bundle.compressedContext == null || bundle.compressedContext.isEmpty())
                        ? bundle.rerankedContext
                        : bundle.compressedContext;

                state.lastContext = context;

                if (log.isTraceEnabled()) {
                    log.trace("iter={} phase={} ctx.raw.n={} ctx.rerank.n={} ctx.compress.n={} ctx.preview={}",
                            iter, explore ? "EXPLORE" : "EXPLOIT",
                            safeSize(bundle.rawEvidence), safeSize(bundle.rerankedContext), safeSize(bundle.compressedContext),
                            previewContext(context, 3, 260));
                }

                // ---- deterministic generation control (tags) ----
                tuneGenerationTags(state, q0, context);

                // 3.5: explicit diversity tag for generator
                state.tags.put("gen.diversity", explore ? "high" : "low");
                state.tags.put("think.phase", explore ? "explore" : "exploit");
                // expose stricter constraints to generator in exploit
                if (!explore) {
                    state.tags.put("eval.grounded.min", fmt(this.exploitEvidenceStrictness));
                    state.tags.put("repair.require_evidence", "true");
                    state.tags.put("repair.forbid_generic", "true");
                } else {
                    state.tags.put("eval.grounded.min", fmt(0.60));
                }

                if (log.isTraceEnabled() && state.tags != null && !state.tags.isEmpty()) {
                    log.trace("iter={} tags={}", iter, stableSmallMap(state.tags, 40));
                }

                // ---- drafts (deterministic diversity) ----
                final long tGen0 = System.nanoTime();
                List<String> drafts = Drafts.sanitize(generateDraftsDeterministic(q0, context, state, iterSeed), draftsPerIteration);
                final long tGen = System.nanoTime() - tGen0;

                if (log.isTraceEnabled()) {
                    log.trace("iter={} drafts.n={} drafts.preview={}", iter, drafts.size(), previewList(drafts, 3, 220));
                }

                // ---- evaluate ----
                final long tEval0 = System.nanoTime();
                List<Candidate> evaluated = evaluateDrafts(q0, context, drafts);
                final long tEval = System.nanoTime() - tEval0;

                Candidate bestIter = pickBest(evaluated).orElse(globalBest);

                state.lastCandidate = bestIter;
                state.lastCritique = bestIter.critique;
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

                // 3.2 Strict verify-pass (optionally gated by score)
                boolean strictOk = true;
                boolean strictTried = false;

                if (strictVerifyEnabled && bestIter != null
                        && (bestIter.score >= strictVerifyMinScore || improved)) {

                    Candidate strictChecked = strictVerifyCandidate(q0, context, bestIter);
                    strictTried = true;
                    strictOk = (strictChecked.evaluation != null && strictChecked.evaluation.valid);

                    if (!strictOk) {
                        // force repair mode; ensure loop continues (even if stop rules would fire)
                        applyStrictRepairTags(state);
                        state.phase = CandidateControlSignals.Phase.REPAIR.ordinal();
                        state.diversity = CandidateControlSignals.Diversity.MEDIUM.ordinal();

                        // Prevent premature stop: add exactly +1 extra iter (at most 2 total)
                        if (forcedExtraIters < 2) forcedExtraIters++;

                        // Also treat as "not improved" for stagnation logic (avoid locking-in bad best)
                        stagnation = Math.max(0, stagnation - 1);

                        // Keep globalBest as-is (do not promote failed strict candidate)
                        bestIter = strictChecked;
                    } else {
                        clearStrictRepairTags(state.tags);
                    }
                }

                // ---- LTM writeback (evidence-only) ----
                if (ltmEnabled && bestIter.evaluation != null && bestIter.evaluation.valid) {
                    int before = ltmByKey.size();
                    maybeWritebackToLtm(bestIter, context);
                    int after = ltmByKey.size();
                    if (log.isDebugEnabled() && after != before) {
                        log.debug("iter={} ltm.write added={} ltm.size={}", iter, (after - before), after);
                    }
                }

                trace.add("iter=" + iter
                        + " phase=" + (explore ? "explore" : "exploit")
                        + " effK=" + effK
                        + " raw=" + safeSize(bundle.rawEvidence)
                        + " rerank=" + safeSize(bundle.rerankedContext)
                        + " compress=" + safeSize(bundle.compressedContext)
                        + " ltm=" + (state.recalledMemory == null ? 0 : state.recalledMemory.size())
                        + " drafts=" + drafts.size()
                        + " bestIter=" + fmt(bestIter.score)
                        + " global=" + fmt(globalBest.score)
                        + " stg=" + stagnation
                        + " strict=" + (strictTried ? (strictOk ? "ok" : "fail") : "skip")
                        + " tRetrMs=" + (tRetr / 1_000_000)
                        + " tLtmMs=" + (tLtm / 1_000_000)
                        + " tGenMs=" + (tGen / 1_000_000)
                        + " tEvalMs=" + (tEval / 1_000_000)
                        + (bundle.traceQuality != null ? (" rq=" + fmt(bundle.traceQuality)) : ""));

                // ---- per-iteration compact summary (INFO) ----
                if (log.isInfoEnabled()) {
                    CandidateEvaluator.Evaluation ev = bestIter.evaluation;
                    log.info("think.iter iter={} phase={} improved={} stg={} bestIter={} global={} eff={} valid={} g={} cs={} cov={} st={} risk={} ctx={} ltm={} drafts={} strict={} tRetrMs={} tLtmMs={} tGenMs={} tEvalMs={}{}",
                            iter, (explore ? "EXPLORE" : "EXPLOIT"),
                            improved, stagnation,
                            fmt(bestIter.score), fmt(globalBest.score),
                            fmt(ev == null ? 0.0 : ev.effectiveScore()),
                            (ev != null && ev.valid),
                            (ev == null ? "0" : fmt(ev.groundedness)),
                            (ev == null ? "0" : fmt(ev.contextSupport)),
                            (ev == null ? "0" : fmt(ev.coverage)),
                            (ev == null ? "0" : fmt(ev.structureScore)),
                            (ev == null ? "0" : fmt(ev.contradictionRisk)),
                            safeSize(context),
                            (state.recalledMemory == null ? 0 : state.recalledMemory.size()),
                            drafts.size(),
                            (strictTried ? (strictOk ? "ok" : "fail") : "skip"),
                            (tRetr / 1_000_000), (tLtm / 1_000_000), (tGen / 1_000_000), (tEval / 1_000_000),
                            bundle.traceQuality != null ? (" rq=" + fmt(bundle.traceQuality)) : "");
                }

                // ---- DEBUG: top candidates ----
                if (log.isDebugEnabled()) {
                    log.debug("iter={} top={}", iter, topCandidatesSummary(evaluated, 4));
                }

                // ---- stop ----
                stopDecision = shouldStopWithReason(state, globalBest, stagnation);
                // if strict verify failed in this iter, do not stop
                if (state.tags != null && "1".equals(state.tags.get("repair.verifyFail"))) {
                    stopDecision = new StopDecision(false, "strict_repair_forced");
                }

                if (stopDecision.stop) {
                    if (log.isDebugEnabled()) {
                        log.debug("think.stop iter={} reason={} bestScore={} target={} stg={} patience={}",
                                iter, stopDecision.reason, fmt(globalBest.score), fmt(targetScore), stagnation, patience);
                    }
                    break;
                }
            }

            CandidateEvaluator.Evaluation ev = (globalBest == null) ? null : globalBest.evaluation;
            long tTotal = System.nanoTime() - tStart;

            if (log.isInfoEnabled()) {
                log.info("think.end answerLen={} score={} eff={} valid={} g={} cs={} cov={} st={} risk={} stylePen={} totalMs={} traceN={} ltmSize={} stopReason={}",
                        globalBest == null ? 0 : globalBest.text.length(),
                        fmt(globalBest == null ? 0.0 : globalBest.score),
                        fmt(ev == null ? 0.0 : ev.effectiveScore()),
                        ev != null && ev.valid,
                        ev == null ? "0" : fmt(ev.groundedness),
                        ev == null ? "0" : fmt(ev.contextSupport),
                        ev == null ? "0" : fmt(ev.coverage),
                        ev == null ? "0" : fmt(ev.structureScore),
                        ev == null ? "0" : fmt(ev.contradictionRisk),
                        ev == null ? "0" : fmt(ev.stylePenalty),
                        (tTotal / 1_000_000),
                        trace.size(),
                        ltmByKey.size(),
                        stopDecision == null ? "none" : stopDecision.reason);
            }

            ThreadContext.remove("intent");
            ThreadContext.remove("iter");

            return new ThoughtResult(globalBest == null ? "" : globalBest.text, trace, intent, globalBest, ev);
        }
    }

    // -------------------- explore/exploit controls --------------------

    private void setPhaseControls(ThoughtState state, boolean explore) {
        if (state == null) return;

        state.phase = explore
                ? CandidateControlSignals.Phase.EXPLORE.ordinal()
                : CandidateControlSignals.Phase.EXPLOIT.ordinal();

        state.diversity = explore
                ? CandidateControlSignals.Diversity.HIGH.ordinal()
                : CandidateControlSignals.Diversity.LOW.ordinal();

        state.evidenceStrictness = explore ? 0.62 : exploitEvidenceStrictness;
    }

    // -------------------- strict verify-pass --------------------

    private Candidate strictVerifyCandidate(String userText, List<Statement> context, Candidate best) {
        if (best == null) return new Candidate("", Double.NEGATIVE_INFINITY, "", null);

        CandidateEvaluator.Evaluation sev = null;

        if (evaluator instanceof MultiCriteriaCandidateEvaluator qev) {
            sev = qev.evaluateStrict(userText, best.text, context);
        } else {
            // fallback: run normal evaluator again (still deterministic)
            sev = evaluator.evaluate(userText, best.text, context);
        }

        return Candidate.fromEvaluation(best.text, sev);
    }

    private static void applyStrictRepairTags(ThoughtState state) {
        if (state == null) return;
        if (state.tags == null) state.tags = new HashMap<>(16);

        state.tags.put("repair.verifyFail", "1");
        state.tags.put("repair.addEvidence", "1");
        state.tags.put("repair.reduceNovelty", "1");
        state.tags.put("repair.require_evidence", "true");
        state.tags.put("repair.forbid_generic", "true");
        state.tags.put("repair.cite_ids", "true");
    }

    private static void clearStrictRepairTags(Map<String, String> tags) {
        if (tags == null) return;
        tags.remove("repair.verifyFail");
    }

    // -------------------- evaluation --------------------

    private List<Candidate> evaluateDrafts(String userText, List<Statement> context, List<String> drafts) {
        if (drafts == null || drafts.isEmpty()) return List.of();

        if (drafts.size() <= 2 || evalParallelism <= 1) {
            ArrayList<Candidate> out = new ArrayList<>(drafts.size());
            for (String d : drafts) {
                CandidateEvaluator.Evaluation ev = evaluator.evaluate(userText, d, context);
                out.add(Candidate.fromEvaluation(d, ev));
            }
            return out;
        }

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
            }, evalPool));
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

    private RetrievalBundle retrieveRerankCompress(String userQuery, List<String> baseQueries, long seed, int effectiveRetrieveK) {

        if (log.isTraceEnabled()) {
            log.trace("retrieve.start seed={} k={} queries={}", Long.toUnsignedString(seed), effectiveRetrieveK, baseQueries);
        }

        if (retriever instanceof KnowledgeRetriever kr) {
            String joined = String.join("\n", baseQueries);
            KnowledgeRetriever.Trace tr = kr.retrieveTrace(joined, effectiveRetrieveK, seed);

            List<Statement> selected = (tr.selected == null) ? List.of() : tr.selected;

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

            // Note: we still compute our rerank later (hybrid). Here keep trace-provided rerank if present.
            List<Statement> reranked;
            if (tr.rerankedTop != null && !tr.rerankedTop.isEmpty()) {
                reranked = tr.rerankedTop.stream()
                        .map(s -> s == null ? null : s.item)
                        .filter(Objects::nonNull)
                        .toList();
            } else {
                reranked = rerankContextOverlap(userQuery, selected, RERANK_N, RERANK_M);
            }

            if (log.isDebugEnabled()) {
                log.debug("retrieve.done type=KnowledgeRetriever raw={} rerank={} compress={} rq={}",
                        safeSize(selected), safeSize(reranked), safeSize(compressed), (tr.quality == 0 ? "null" : fmt(tr.quality)));
            }

            return new RetrievalBundle(selected, reranked, compressed, tr.quality);
        }

        Map<String, Statement> mergedByKey = new ConcurrentHashMap<>();
        retrieveAndMerge(baseQueries, mergedByKey, seed, effectiveRetrieveK);

        for (int r = 0; r < refineRounds; r++) {
            List<Statement> ctxNow = toDeterministicContext(mergedByKey);

            List<String> derived = deriveQueriesFromContext(ctxNow, refineQueryBudget);
            if (log.isTraceEnabled()) {
                log.trace("retrieve.refine round={} derived.n={} derived={}", r, derived.size(), derived);
            }
            if (derived.isEmpty()) break;

            retrieveAndMerge(derived, mergedByKey, mix(seed, 0x9E3779B97F4A7C15L + r), effectiveRetrieveK);
        }

        List<Statement> raw = toDeterministicContext(mergedByKey);
        List<Statement> reranked = rerankContextOverlap(userQuery, raw, RERANK_N, RERANK_M);
        List<Statement> compressed = compressContextToStatements(userQuery, reranked, COMPRESS_SENTENCES, COMPRESS_MAX_CHARS);

        if (log.isDebugEnabled()) {
            log.debug("retrieve.done type=GenericRetriever raw={} rerank={} compress={}",
                    safeSize(raw), safeSize(reranked), safeSize(compressed));
        }

        return new RetrievalBundle(raw, reranked, compressed, null);
    }

    private static int safeSize(List<?> xs) {
        return xs == null ? 0 : xs.size();
    }

    // -------------------- rerank (hybrid) --------------------

    private static List<Statement> rerankContextHybrid(
            String userQuery,
            List<Statement> raw,
            int topN,
            int topM,
            Map<String, Byte> origin,
            Map<String, Integer> df,
            int dfDocs
    ) {
        if (raw == null || raw.isEmpty()) return List.of();

        Set<String> qTok = tokens(userQuery);
        if (qTok.isEmpty()) {
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

            String key = stableStmtKey(s);

            double ov = overlapScore(qTok, s.text);

            byte o = (origin == null || key == null) ? 0 : origin.getOrDefault(key, (byte) 0);

            // source prior: retrieval bonus, LTM-only slight penalty
            double sourceBonus = ((o & 1) != 0) ? 0.08 : 0.0;
            double ltmPenalty = (o == 2) ? 0.05 : 0.0;

            // DF penalty: too-common tokens (generic statements) sink a bit
            double dfPenalty = 0.0;
            if (df != null && dfDocs > 0) {
                Set<String> stTok = tokens(s.text);
                int seen = 0;
                double acc = 0.0;
                for (String t : stTok) {
                    Integer c = df.get(t);
                    if (c != null) {
                        acc += (double) c / (double) dfDocs;
                        seen++;
                    }
                    if (seen >= 12) break;
                }
                if (seen > 0) dfPenalty = acc / seen; // 0..1
            }

            double score = ov + sourceBonus - ltmPenalty - 0.10 * dfPenalty;
            scored.add(new ScoredStmt(s, score));
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

    // legacy overlap-only rerank kept (used as fallback before hybrid merge is applied)
    private static List<Statement> rerankContextOverlap(String userQuery, List<Statement> raw, int topN, int topM) {
        if (raw == null || raw.isEmpty()) return List.of();

        Set<String> qTok = tokens(userQuery);
        if (qTok.isEmpty()) {
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

    private record DfStats(Map<String, Integer> df, int docs) {}

    private static DfStats buildDfStats(List<Statement> xs, int maxDocs) {
        if (xs == null || xs.isEmpty()) return new DfStats(Map.of(), 0);

        int docs = 0;
        HashMap<String, Integer> df = new HashMap<>(512);

        for (int i = 0; i < xs.size() && docs < maxDocs; i++) {
            Statement s = xs.get(i);
            if (s == null || s.text == null || s.text.isBlank()) continue;
            docs++;
            Set<String> st = tokens(s.text);
            for (String t : st) df.merge(t, 1, Integer::sum);
        }

        return new DfStats(df, docs);
    }

    // -------------------- generation (deterministic diversity) --------------------

    private List<String> generateDraftsDeterministic(String userText, List<Statement> context, ThoughtState state, long iterSeed) {
        int need = Math.max(1, draftsPerIteration);
        ArrayList<String> out = new ArrayList<>(need);

        long savedSeed = state.seed;
        int savedIdx = state.draftIndex;

        try {
            state.draftsRequested = need;

            for (int d = 0; d < need; d++) {
                state.draftIndex = d;

                long draftSeed = mix(iterSeed, 0xD1B54A32D192ED03L ^ (long) d);
                state.tags.put("gen.draftSeed", Long.toUnsignedString(draftSeed));
                state.seed = draftSeed;

                if (generator != null) {
                    // generator will also see state.draftIndex and state.seed
                    String s = generator.generate(generator.prepareUserText(userText, context, state), context, state);
                    if (s == null) s = "";
                    out.add(s);
                } else {
                    out.add(strategy.generate(userText, context, state));
                }
            }
        } finally {
            state.seed = savedSeed;
            state.draftIndex = savedIdx;
        }

        if (out.isEmpty()) out.add("");
        return out;
    }

    // -------------------- stop rules --------------------

    private StopDecision shouldStopWithReason(ThoughtState state, Candidate globalBest, int stagnation) {
        for (StopRule r : stopRules) {
            if (r.shouldStop(state, globalBest, stagnation)) {
                return new StopDecision(true, r.name());
            }
        }
        return new StopDecision(false, "none");
    }

    private record StopDecision(boolean stop, String reason) {}

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

        StringBuilder sb = new StringBuilder(320);
        sb.append("format=").append(style);
        sb.append(";sections=").append(sections);
        sb.append(";intent=").append(Optional.ofNullable(state.intent).orElse(Intent.UNKNOWN).name());
        sb.append(";iter=").append(state.iteration);

        sb.append(";phase=").append(state.phase);
        sb.append(";div=").append(state.diversity);
        sb.append(";evs=").append(fmt(state.evidenceStrictness));

        Optional.ofNullable(state.lastEvaluation).ifPresent(last -> {
            sb.append(";last_g=").append(fmt(last.groundedness));
            sb.append(";last_cs=").append(fmt(last.contextSupport));
            sb.append(";last_cov=").append(fmt(last.coverage));
            sb.append(";last_st=").append(fmt(last.structureScore));
            sb.append(";last_r=").append(fmt(last.contradictionRisk));
            sb.append(";last_v=").append(last.valid ? 1 : 0);
        });

        Optional.ofNullable(state.bestEvaluation).ifPresent(best -> {
            sb.append(";best_g=").append(fmt(best.groundedness));
            sb.append(";best_cs=").append(fmt(best.contextSupport));
            sb.append(";best_cov=").append(fmt(best.coverage));
            sb.append(";best_st=").append(fmt(best.structureScore));
            sb.append(";best_r=").append(fmt(best.contradictionRisk));
            sb.append(";best_v=").append(best.valid ? 1 : 0);
        });

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

        int qTok = approxQueryTokenCount(userText);
        if (qTok > 0 && qTok <= 2) {
            state.tags.put("response.sections", "summary");
        } else {
            state.tags.putIfAbsent("response.sections", "summary,evidence,actions");
        }

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
        // keep verifyFail until strict pass clears it explicitly
    }

    private static void appendRepairHint(ThoughtState state, StringBuilder sb) {
        if (state == null || state.tags == null || sb == null) return;

        String a = state.tags.get("repair.addEvidence");
        String n = state.tags.get("repair.reduceNovelty");
        String f = state.tags.get("repair.fixStructure");
        String e = state.tags.get("repair.avoidEcho");

        if ("1".equals(a) || "1".equals(n) || "1".equals(f) || "1".equals(e)) {
            sb.append(";repair=");
            boolean first = true;

            if ("1".equals(a)) { sb.append("addEvidence"); first = false; }
            if ("1".equals(n)) { sb.append(first ? "" : ",").append("reduceNovelty"); first = false; }
            if ("1".equals(f)) { sb.append(first ? "" : ",").append("fixStructure"); first = false; }
            if ("1".equals(e)) { sb.append(first ? "" : ",").append("avoidEcho"); }
        }
    }

    private static boolean isEchoRisk(String userText, String lastText) {
        if (lastText == null || lastText.isBlank()) return false;

        if (lastText.trim().length() < 40) return true;

        Set<String> q = tokens(userText);
        Set<String> a = tokens(lastText);
        if (q.isEmpty() || a.isEmpty()) return false;

        int inter = 0;
        for (String t : q) if (a.contains(t)) inter++;

        double jacc = (double) inter / (double) Math.max(1, (q.size() + a.size() - inter));
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

    private void retrieveAndMerge(List<String> queries, Map<String, Statement> mergedByKey, long seed, int effectiveRetrieveK) {
        if (queries == null || queries.isEmpty()) return;
        for (int i = 0; i < queries.size(); i++) {
            String q = queries.get(i);
            if (q == null || q.isBlank()) continue;
            List<Statement> got = retriever.retrieve(q, effectiveRetrieveK, mix(seed, i));
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

        HashMap<String, Integer> df = new HashMap<>(512);
        int n = 0;
        for (Statement s : ctx) {
            if (s == null || s.text == null || s.text.isBlank()) continue;
            n++;
            Set<String> st = tokens(s.text);
            for (String t : st) df.merge(t, 1, Integer::sum);
        }
        if (n <= 0) return List.of();

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

        out.sort(Comparator.comparing(IterativeThoughtEngine::stableStmtKey, Comparator.nullsLast(String::compareTo)));
        return out;
    }

    private static Statement syntheticStatementOrNull(Object o) {
        if (o == null) return null;
        if (o instanceof Statement s) return s;
        String s = o.toString();
        if (s == null || s.isBlank()) return null;
        Statement st = new Statement();
        st.id = "syn";
        st.text = s.trim();
        return st;
    }

    // -------------------- LTM recall / write / decay / promotion --------------------

    private List<Statement> recallFromLtm(List<String> queries) {
        if (!ltmEnabled || ltmByKey.isEmpty() || ltmRecallK <= 0) return List.of();

        LinkedHashSet<String> qTokAll = new LinkedHashSet<>();
        if (queries != null) {
            for (String q : queries) {
                Set<String> t = tokens(q);
                qTokAll.addAll(t);
                if (qTokAll.size() > 128) break;
            }
        }
        if (qTokAll.isEmpty()) return List.of();

        ArrayList<ScoredKey> scored = new ArrayList<>(Math.min(2048, ltmByKey.size()));
        long now = ltmTick.get();

        for (var e : ltmByKey.entrySet()) {
            String key = e.getKey();
            Statement st = e.getValue();
            if (key == null || st == null || st.text == null || st.text.isBlank()) continue;

            Set<String> tok = ltmTokensByKey.computeIfAbsent(key, k -> tokens(st.text));

            double ov = overlapScore(qTokAll, tok);
            if (ov <= 0.0) continue;

            // decay priority over ticks since last touch
            double pr = ltmPriorityByKey.getOrDefault(key, 0.0);
            long last = ltmLastTouchTick.getOrDefault(key, 0L);
            double prEff = clamp01(pr - Math.max(0L, now - last) * ltmDecayPerTick);

            double score = 0.80 * ov + 0.20 * prEff;
            scored.add(new ScoredKey(key, score));
        }

        scored.sort((a, b) -> {
            int c = Double.compare(b.score, a.score);
            if (c != 0) return c;
            return a.key.compareTo(b.key);
        });

        int k = Math.min(ltmRecallK, scored.size());
        ArrayList<Statement> out = new ArrayList<>(k);

        long touch = ltmTick.incrementAndGet();
        for (int i = 0; i < k; i++) {
            String key = scored.get(i).key;
            Statement st = ltmByKey.get(key);
            if (st != null) out.add(st);

            // mild touch (recall alone shouldn't promote much)
            ltmLastTouchTick.put(key, touch);
            ltmPriorityByKey.merge(key, 0.005, Double::sum);
        }

        return out;
    }

    private void promoteLtmOnFreshEvidence(List<Statement> fresh) {
        if (!ltmEnabled || fresh == null || fresh.isEmpty() || ltmByKey.isEmpty()) return;

        long touch = ltmTick.incrementAndGet();

        for (Statement s : fresh) {
            String k = stableStmtKey(s);
            if (k == null) continue;
            if (ltmByKey.containsKey(k)) {
                ltmPriorityByKey.merge(k, ltmPromotionBoost, Double::sum);
                ltmLastTouchTick.put(k, touch);
            }
        }
    }

    private void maybeWritebackToLtm(Candidate best, List<Statement> context) {
        if (!ltmEnabled || best == null || best.evaluation == null) return;
        if (!best.evaluation.valid) return;
        if (best.evaluation.groundedness < ltmWriteMinGroundedness) return;
        if (context == null || context.isEmpty()) return;

        // Evidence-only write policy: store only statements (not generated text)
        // and only those that are "useful": we store top context as LTM cache.
        int add = 0;
        long touch = ltmTick.incrementAndGet();

        for (Statement st : context) {
            if (st == null || st.text == null || st.text.isBlank()) continue;

            String key = stableStmtKey(st);
            if (key == null) continue;

            if (ltmByKey.putIfAbsent(key, st) == null) {
                add++;
                ltmTokensByKey.putIfAbsent(key, tokens(st.text));
            }

            ltmPriorityByKey.merge(key, 0.02, Double::sum);
            ltmLastTouchTick.put(key, touch);

            // capacity enforcement (best-effort deterministic trim)
            if (ltmCapacity > 0 && ltmByKey.size() > ltmCapacity) {
                trimLtmToCapacity();
            }
        }

        if (log.isTraceEnabled() && add > 0) {
            log.trace("ltm.write add={} size={}", add, ltmByKey.size());
        }
    }

    private void trimLtmToCapacity() {
        if (ltmCapacity <= 0) return;
        int over = ltmByKey.size() - ltmCapacity;
        if (over <= 0) return;

        // deterministic trim: drop lowest effective priority (priority - decay)
        long now = ltmTick.get();

        ArrayList<ScoredKey> xs = new ArrayList<>(Math.min(4096, ltmByKey.size()));
        for (String k : ltmByKey.keySet()) {
            double pr = ltmPriorityByKey.getOrDefault(k, 0.0);
            long last = ltmLastTouchTick.getOrDefault(k, 0L);
            double eff = pr - Math.max(0L, now - last) * ltmDecayPerTick;
            xs.add(new ScoredKey(k, eff));
        }

        xs.sort((a, b) -> {
            int c = Double.compare(a.score, b.score); // ascending (drop lowest)
            if (c != 0) return c;
            return a.key.compareTo(b.key);
        });

        int drop = Math.min(over, xs.size());
        for (int i = 0; i < drop; i++) {
            String k = xs.get(i).key;
            ltmByKey.remove(k);
            ltmPriorityByKey.remove(k);
            ltmTokensByKey.remove(k);
            ltmLastTouchTick.remove(k);
        }
    }

    private record ScoredKey(String key, double score) {}

    private static double overlapScore(Set<String> qTok, Set<String> tTok) {
        if (qTok == null || qTok.isEmpty() || tTok == null || tTok.isEmpty()) return 0.0;
        int inter = 0;
        for (String t : qTok) if (tTok.contains(t)) inter++;
        return (double) inter / (double) Math.max(1, qTok.size());
    }

    // -------------------- utilities / helpers --------------------

    private static String normalizeUserText(String s) {
        if (s == null) return "";
        String x = s.trim();
        if (x.isEmpty()) return "";
        x = Normalizer.normalize(x, Normalizer.Form.NFKC);
        x = x.replace('\u00A0', ' ');
        x = x.replaceAll("\\s+", " ").trim();
        return x;
    }

    private static boolean isMostlyCyrillic(String s) {
        if (s == null || s.isBlank()) return false;
        int cyr = 0, total = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (!Character.isLetter(ch)) continue;
            total++;
            if ((ch >= 'А' && ch <= 'я') || ch == 'ё' || ch == 'Ё') cyr++;
        }
        if (total <= 0) return false;
        return ((double) cyr / (double) total) >= 0.55;
    }

    private static int approxQueryTokenCount(String s) {
        if (s == null || s.isBlank()) return 0;
        int n = 0;
        java.util.regex.Matcher m = WORD.matcher(s.toLowerCase(Locale.ROOT));
        while (m.find()) {
            n++;
            if (n >= 64) break;
        }
        return n;
    }

    private static Set<String> tokens(String s) {
        if (s == null || s.isBlank()) return Set.of();
        HashSet<String> out = new HashSet<>(64);
        java.util.regex.Matcher m = WORD.matcher(s.toLowerCase(Locale.ROOT));
        while (m.find()) {
            String t = m.group();
            if (t == null || t.isBlank()) continue;
            out.add(t);
            if (out.size() >= 256) break;
        }
        return out;
    }

    private static String stableStmtKey(Statement s) {
        if (s == null) return null;
        String id = (s.id == null) ? "" : s.id.trim();
        String txt = (s.text == null) ? "" : s.text.trim();
        if (id.isEmpty() && txt.isEmpty()) return null;
        // deterministic key: id + hash(text)
        long h = fnv1a64(txt);
        return id + "#" + Long.toUnsignedString(h, 36);
    }

    private static long fnv1a64(String s) {
        if (s == null) return 0L;
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= (byte) s.charAt(i);
            h *= 0x100000001b3L;
        }
        return h;
    }

    private static long mix(long a, long b) {
        long x = a + 0x9E3779B97F4A7C15L;
        x ^= b + 0xBF58476D1CE4E5B9L;
        x ^= (x >>> 30);
        x *= 0xBF58476D1CE4E5B9L;
        x ^= (x >>> 27);
        x *= 0x94D049BB133111EBL;
        x ^= (x >>> 31);
        return x;
    }

    private static double clamp01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.3f", v);
    }

    private static String preview(String s, int max) {
        if (s == null) return "";
        String x = s.replace('\n', ' ').replace('\r', ' ').trim();
        if (x.length() <= max) return x;
        return x.substring(0, Math.max(0, max - 1)) + "…";
    }

    private static String previewList(List<String> xs, int n, int maxEach) {
        if (xs == null || xs.isEmpty()) return "[]";
        int k = Math.min(n, xs.size());
        ArrayList<String> out = new ArrayList<>(k);
        for (int i = 0; i < k; i++) out.add("'" + preview(xs.get(i), maxEach) + "'");
        return out.toString();
    }

    private static String previewContext(List<Statement> ctx, int n, int maxEach) {
        if (ctx == null || ctx.isEmpty()) return "[]";
        int k = Math.min(n, ctx.size());
        ArrayList<String> out = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            Statement s = ctx.get(i);
            out.add("#" + i + ":" + preview(s == null ? "" : s.text, maxEach));
        }
        return out.toString();
    }

    private static Map<String, String> stableSmallMap(Map<String, String> m, int maxKeys) {
        if (m == null || m.isEmpty()) return Map.of();
        ArrayList<String> keys = new ArrayList<>(m.keySet());
        keys.sort(String::compareTo);
        LinkedHashMap<String, String> out = new LinkedHashMap<>();
        int n = 0;
        for (String k : keys) {
            out.put(k, m.get(k));
            n++;
            if (n >= maxKeys) break;
        }
        return out;
    }

    private static Optional<Candidate> pickBest(List<Candidate> xs) {
        if (xs == null || xs.isEmpty()) return Optional.empty();
        Candidate best = null;
        for (Candidate c : xs) {
            if (c == null) continue;
            if (best == null || c.score > best.score) best = c;
        }
        return Optional.ofNullable(best);
    }

    private static String topCandidatesSummary(List<Candidate> xs, int k) {
        if (xs == null || xs.isEmpty()) return "[]";
        ArrayList<Candidate> copy = new ArrayList<>(xs);
        copy.sort((a, b) -> Double.compare(b.score, a.score));
        int n = Math.min(Math.max(1, k), copy.size());
        ArrayList<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Candidate c = copy.get(i);
            CandidateEvaluator.Evaluation ev = c == null ? null : c.evaluation;
            out.add("{" + i + " s=" + fmt(c.score)
                    + " v=" + (ev != null && ev.valid ? 1 : 0)
                    + " g=" + (ev == null ? "0" : fmt(ev.groundedness))
                    + " cs=" + (ev == null ? "0" : fmt(ev.contextSupport))
                    + " st=" + (ev == null ? "0" : fmt(ev.structureScore))
                    + " r=" + (ev == null ? "0" : fmt(ev.contradictionRisk))
                    + "}");
        }
        return out.toString();
    }

    // -------------------- QueryProvider / StopRule / RetrievalBundle --------------------

    private interface QueryProvider {
        List<String> queries(String userText, ThoughtState state);

        static QueryProvider userText() {
            return (u, s) -> List.of(u == null ? "" : u.trim());
        }

        static QueryProvider intentName() {
            return (u, s) -> {
                Intent it = (s == null) ? Intent.UNKNOWN : s.intent;
                if (it == null || it == Intent.UNKNOWN) return List.of();
                return List.of(it.name().toLowerCase(Locale.ROOT));
            };
        }

        static QueryProvider bestTerms(int maxTerms, int minLen) {
            return (u, s) -> {
                if (s == null || s.bestEvaluation == null || s.bestSoFar == null) return List.of();
                if (!s.bestEvaluation.valid) return List.of();
                if (s.bestEvaluation.groundedness < 0.45) return List.of();

                Set<String> t = tokens(s.bestSoFar.text);
                ArrayList<String> out = new ArrayList<>(Math.min(maxTerms, t.size()));
                for (String x : t) {
                    if (x.length() >= minLen) out.add(x);
                }
                out.sort(String::compareTo);
                if (out.size() > maxTerms) return out.subList(0, maxTerms);
                return out;
            };
        }
    }

    private interface StopRule {
        boolean shouldStop(ThoughtState state, Candidate best, int stagnation);

        String name();

        static StopRule targetScore(double target) {
            return new StopRule() {
                @Override public boolean shouldStop(ThoughtState state, Candidate best, int stagnation) {
                    return best != null && best.score >= target;
                }
                @Override public String name() { return "targetScore"; }
            };
        }

        static StopRule patience(int patience) {
            return new StopRule() {
                @Override public boolean shouldStop(ThoughtState state, Candidate best, int stagnation) {
                    return patience > 0 && stagnation >= patience;
                }
                @Override public String name() { return "patience"; }
            };
        }
    }

    private record RetrievalBundle(
            List<Statement> rawEvidence,
            List<Statement> rerankedContext,
            List<Statement> compressedContext,
            Double traceQuality
    ) {
        RetrievalBundle withMerged(List<Statement> merged, List<Statement> reranked, List<Statement> compressed) {
            return new RetrievalBundle(merged, reranked, compressed, traceQuality);
        }
    }

    // -------------------- Drafts helper --------------------

    private static final class Drafts {
        private Drafts() {}

        static List<String> sanitize(List<String> drafts, int want) {
            if (drafts == null || drafts.isEmpty()) return List.of("");

            LinkedHashSet<String> uniq = new LinkedHashSet<>();
            for (String s : drafts) {
                String x = (s == null) ? "" : s.trim();
                if (x.isEmpty()) continue;
                uniq.add(x);
            }

            ArrayList<String> out = new ArrayList<>(uniq);
            if (out.isEmpty()) out.add("");

            // keep at least 2..4 (avoid collapse), but respect want as upper bound
            int cap = Math.max(2, Math.min(want, Math.max(4, out.size())));
            if (out.size() > cap) out = new ArrayList<>(out.subList(0, cap));
            return out;
        }
    }
}