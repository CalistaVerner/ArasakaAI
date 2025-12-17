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

    private final Knobs knobs;

    /**
     * Engine-local knobs and patterns. Package-private on purpose:
     * Think.Config can build/tune this without exposing a new public API surface.
     *
     * All values must remain deterministic and side-effect free.
     */
    static final class Knobs {
        // Token-ish word extraction for query derivation / near-dedup signatures
        final java.util.regex.Pattern WORD;

        // Telemetry signatures + internal markers (generic, non-domain)
        final java.util.regex.Pattern TELEMETRY;

        // --- RAG knobs ---
        final int rerankN;
        final int rerankM;
        final int compressSentences;
        final int compressMaxChars;
        final double deriveDfCut;

        // --- Strict verify risk gating ---
        final double strictForceRisk;          // if contradictionRisk >= this -> force strict verify
        final double strictForceUngrounded;    // if groundedness <= this -> force strict verify
        final double strictForceWeakStructure; // if structureScore <= this -> force strict verify

        Knobs(
                java.util.regex.Pattern word,
                java.util.regex.Pattern telemetry,
                int rerankN,
                int rerankM,
                int compressSentences,
                int compressMaxChars,
                double deriveDfCut,
                double strictForceRisk,
                double strictForceUngrounded,
                double strictForceWeakStructure
        ) {
            this.WORD = Objects.requireNonNull(word, "word");
            this.TELEMETRY = Objects.requireNonNull(telemetry, "telemetry");
            this.rerankN = rerankN;
            this.rerankM = rerankM;
            this.compressSentences = compressSentences;
            this.compressMaxChars = compressMaxChars;
            this.deriveDfCut = deriveDfCut;
            this.strictForceRisk = strictForceRisk;
            this.strictForceUngrounded = strictForceUngrounded;
            this.strictForceWeakStructure = strictForceWeakStructure;
        }

        static Knobs defaults() {
            return new Knobs(
                    java.util.regex.Pattern.compile("[\\p{L}\\p{Nd}_]{3,}"),
                    java.util.regex.Pattern.compile("(?i)"
                            + "(^\\s*(noctx;|err=|sec=)\\S*\\s*$)|"
                            + "(\\w+=\\S+;)|"
                            + "(\\bqc=)|(\\bg=)|(\\bnov=)|(\\brep=)|(\\bmd=)|"
                            + "(\\bno_context\\b)|(\\bnoctx\\b)|(\\bmore_evidence\\b)|(\\bfix\\b)|(\\bnotes\\b)"),
                    120,
                    18,
                    2,
                    700,
                    0.60,
                    0.45, // contradictionRisk force threshold
                    0.35, // groundedness force threshold (low -> force)
                    0.35  // structureScore force threshold (low -> force)
            );
        }
    }

    /**
     * Canonical tag keys used by this engine (avoid typos / drift between engine <-> generator).
     */
    static final class Tags {
        private Tags() {}

        // Phase / diversity hints
        static final String GEN_DIVERSITY = "gen.diversity";
        static final String EVIDENCE_STRICTNESS = "evidence.strictness";

        // Repair mode signals
        static final String REPAIR_ADD_EVIDENCE = "repair.addEvidence";
        static final String REPAIR_REDUCE_NOVELTY = "repair.reduceNovelty";
        static final String REPAIR_REQUIRE_EVIDENCE = "repair.require_evidence";
        static final String REPAIR_FORBID_GENERIC = "repair.forbid_generic";
        static final String REPAIR_CITE_IDS = "repair.cite_ids";
        static final String REPAIR_VERIFY_FAIL = "repair.verifyFail";
        static final String REPAIR_FIX_STRUCTURE = "repair.fixStructure";
        static final String REPAIR_AVOID_ECHO = "repair.avoidEcho";
    }

    // --- Explore/Exploit defaults (can be overridden by Think.Config) ---
    private final int exploreIters;
    private final double exploreRetrieveKMult;
    private final double exploitEvidenceStrictness;

    // --- strict verify-pass policy ---
    private final boolean strictVerifyEnabled;
    private final double strictVerifyMinScore;

    // --- LTM knobs ---
    private final boolean ltmEnabled;
    private final int ltmCapacity;
    private final int ltmRecallK;
    private final double ltmWriteMinGroundedness;
    private final double ltmDecayPerTick;
    private final double ltmPromotionBoost;

    // --- refinement knobs ---
    private final int refineRounds;
    private final int refineQueryBudget;

    // Dependencies
    private final Retriever retriever;
    private final IntentDetector intentDetector;
    private final ResponseStrategy strategy;
    private final CandidateEvaluator evaluator;
    private final TextGenerator generator;
    private final ExecutorService evalPool;
    private final int evalParallelism;

    // core loop knobs
    private final int iterations;
    private final int retrieveK;
    private final int draftsPerIteration;
    private final int patience;
    private final double targetScore;

    // LTM store (simple local cache)
    private final LinkedHashMap<String, LtmCell> ltm;
    private final AtomicLong ltmTick = new AtomicLong(0);

    // --------- Constructors (API preserved) ---------

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
        this(retriever, intentDetector, strategy, evaluator, generator, evalPool, evalParallelism,
                iterations, retrieveK, draftsPerIteration, patience, targetScore,
                ltmEnabled, ltmCapacity, ltmRecallK, ltmWriteMinGroundedness,
                1, 3
        );
    }

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
        this(retriever, intentDetector, strategy, evaluator, generator, evalPool, evalParallelism,
                iterations, retrieveK, draftsPerIteration, patience, targetScore,
                ltmEnabled, ltmCapacity, ltmRecallK, ltmWriteMinGroundedness,
                refineRounds, refineQueryBudget,
                1, 1.15, 0.62,
                true, 0.78,
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
        this(
                retriever,
                intentDetector,
                strategy,
                evaluator,
                generator,
                evalPool,
                evalParallelism,
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
                refineQueryBudget,
                exploreIters,
                exploreRetrieveKMult,
                exploitEvidenceStrictness,
                strictVerifyEnabled,
                strictVerifyMinScore,
                ltmDecayPerTick,
                ltmPromotionBoost,
                Knobs.defaults()
        );
    }

    /**
     * Full constructor with engine-local knobs.
     * This keeps the public API stable and lets Think.Config tune internals without adding new public classes.
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
            double ltmPromotionBoost,
            Knobs knobs
    ) {
        this.retriever = Objects.requireNonNull(retriever, "retriever");
        this.intentDetector = Objects.requireNonNull(intentDetector, "intentDetector");
        this.strategy = Objects.requireNonNull(strategy, "strategy");
        this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
        this.generator = generator;

        this.knobs = (knobs != null ? knobs : Knobs.defaults());

        this.evalPool = Objects.requireNonNull(evalPool, "evalPool");
        this.evalParallelism = Math.max(1, evalParallelism);

        this.iterations = Math.max(1, iterations);
        this.retrieveK = Math.max(1, retrieveK);

        // IMPORTANT: drafts=1 causes deterministic mode collapse with beam/bigram generators.
        this.draftsPerIteration = Math.max(2, draftsPerIteration);

        this.patience = Math.max(0, patience);
        this.targetScore = targetScore;

        this.ltmEnabled = ltmEnabled;
        this.ltmCapacity = Math.max(0, ltmCapacity);
        this.ltmRecallK = Math.max(0, ltmRecallK);
        this.ltmWriteMinGroundedness = ltmWriteMinGroundedness;

        this.refineRounds = Math.max(0, refineRounds);
        this.refineQueryBudget = Math.max(1, refineQueryBudget);

        this.exploreIters = Math.max(0, exploreIters);
        this.exploreRetrieveKMult = Math.max(1.0, exploreRetrieveKMult);
        this.exploitEvidenceStrictness = clamp01(exploitEvidenceStrictness);

        this.strictVerifyEnabled = strictVerifyEnabled;
        this.strictVerifyMinScore = strictVerifyMinScore;

        this.ltmDecayPerTick = Math.max(0.0, ltmDecayPerTick);
        this.ltmPromotionBoost = Math.max(0.0, ltmPromotionBoost);

        this.ltm = new LinkedHashMap<>(Math.max(16, this.ltmCapacity), 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, LtmCell> eldest) {
                return ltmCapacity > 0 && size() > ltmCapacity;
            }
        };
    }

    // --------- Main loop ---------

    @Override
    public ThoughtResult think(String userText, long seed) {

        final String reqId = Long.toUnsignedString(mix(seed, System.nanoTime()), 36);
        try (final CloseableThreadContext.Instance ctc =
                     CloseableThreadContext.put("rid", reqId).put("engine", "IterativeThoughtEngine")) {

            final long tStart = System.nanoTime();

            final String q0 = normalizeUserText(userText);
            final Intent intent = intentDetector.detect(q0);
            final int qTok = approxQueryTokenCount(q0);

            ThreadContext.put("intent", intent == null ? "UNKNOWN" : intent.name());

            ThoughtState state = new ThoughtState();
            state.seed = seed;
            state.intent = intent;

            // default schema tags for generator (summary-only for very short prompts)
            applyDefaultResponseSchema(state, q0, intent);
            applyDefaultLengthHints(state, q0);

            log.info("think.start len={} tok={} intent={} hasGen={} refineRounds={} ltmEnabled={} iterations={} retrieveK={} draftsPerIter={} evalPar={}",
                    q0.length(), approxQueryTokenCount(q0),
                    (intent == null ? "UNKNOWN" : intent.name()),
                    generator != null, refineRounds, ltmEnabled, iterations, retrieveK, draftsPerIteration, evalParallelism);
            log.debug("think.q0='{}'", q0);

            Candidate globalBest = Candidate.empty(q0);
            int stagnation = 0;
            boolean strictTried = false;

            for (int iter = 1; iter <= iterations; iter++) {

                state.iteration = iter;
                state.draftIndex = 0;

                boolean explore = (iter <= exploreIters);
                state.phase = explore
                        ? CandidateControlSignals.Phase.EXPLORE.ordinal()
                        : CandidateControlSignals.Phase.EXPLOIT.ordinal();

                // diversity schedule
                state.diversity = explore
                        ? CandidateControlSignals.Diversity.HIGH.ordinal()
                        : CandidateControlSignals.Diversity.MEDIUM.ordinal();

                // evidence strictness schedule
                if (state.tags == null) state.tags = new HashMap<>(16);
                state.tags.put(Tags.GEN_DIVERSITY, explore ? "high" : "medium");
                state.tags.put(Tags.EVIDENCE_STRICTNESS, explore ? "0.55" : fmt(exploitEvidenceStrictness));

                // raise eval constraint in exploit phase
                if (!explore) {
                    state.tags.put("eval.grounded.min", fmt(this.exploitEvidenceStrictness));
                    state.tags.put(Tags.REPAIR_REQUIRE_EVIDENCE, "true");
                    state.tags.put(Tags.REPAIR_FORBID_GENERIC, "true");
                } else {
                    state.tags.put("eval.grounded.min", fmt(0.60));
                }

                List<String> refineQueries = deriveRefineQueries(q0, state, refineRounds, refineQueryBudget);

                // expose deterministic retrieval queries for telemetry and final trace
                state.lastQueries = refineQueries;

                // retrieve
                int k = explore ? (int) Math.ceil(retrieveK * exploreRetrieveKMult) : retrieveK;
                k = Math.max(1, k);

                log.trace("{} iter={} phase={} queries.n={} queries={}", reqId, iter,
                        (explore ? "EXPLORE" : "EXPLOIT"),
                        refineQueries.size(), refineQueries);

                List<Statement> retrieved = doRetrieve(seed, k, refineQueries);

                // rerank + compress
                List<Statement> context = rerankAndCompress(q0, retrieved, k);

                // expose explicit context snapshot to final ResponseStrategy
                state.lastContext = context;

                // generate drafts
                List<String> drafts = generateDraftsDeterministic(q0, context, state, draftsPerIteration);

                // sanitize drafts (stable exact + near dedup)
                drafts = Drafts.sanitize(drafts, draftsPerIteration);

                // evaluate
                List<Candidate> evaluated = evaluateDrafts(q0, context, state, drafts);

                Candidate bestIter = pickBestCandidate(evaluated);

                if (bestIter == null) bestIter = Candidate.empty(q0);

                // keep debug pointers
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

                // 3.2 Strict verify-pass (risk-driven)
                boolean strictOk = true;

                CandidateEvaluator.Evaluation ev0 = (bestIter == null ? null : bestIter.evaluation);
                boolean veryShortQ = (qTok > 0 && qTok <= 2);

                // Risk-driven strict verify:
                // - force when risky/ungrounded/invalid even if score looks ok
                // - avoid over-structuring for very short prompts (<=2 tokens)
                boolean riskyAny = ev0 != null && (!ev0.valid
                        || ev0.contradictionRisk >= knobs.strictForceRisk
                        || ev0.groundedness <= knobs.strictForceUngrounded
                        || ev0.structureScore <= knobs.strictForceWeakStructure);

                boolean riskyShort = ev0 != null && (!ev0.valid
                        || ev0.contradictionRisk >= knobs.strictForceRisk
                        || ev0.groundedness <= knobs.strictForceUngrounded);

                boolean doStrictVerify =
                        strictVerifyEnabled
                                && bestIter != null
                                && ((!veryShortQ && (bestIter.score >= strictVerifyMinScore || improved || riskyAny))
                                || (veryShortQ && (riskyShort || improved)));

                if (doStrictVerify) {

                    Candidate strictChecked = strictVerifyCandidate(q0, context, bestIter);
                    strictTried = true;
                    strictOk = (strictChecked.evaluation != null && strictChecked.evaluation.valid);

                    if (!strictOk) {
                        // force repair mode; ensure loop continues (even if stop rules would fire)
                        applyStrictRepairTags(state, veryShortQ);
                        state.phase = CandidateControlSignals.Phase.REPAIR.ordinal();
                        state.diversity = CandidateControlSignals.Diversity.MEDIUM.ordinal();

                        // Prevent premature stopping on a "high score but invalid" answer
                        stagnation = 0;
                    } else {
                        // strict verify passed -> take stricter candidate as global best if better
                        if (strictChecked.score > globalBest.score + 1e-9) {
                            globalBest = strictChecked;
                            state.bestSoFar = strictChecked;
                            state.bestEvaluation = strictChecked.evaluation;
                        }
                    }
                }

                // stop rules
                if (shouldStop(iter, globalBest, stagnation, strictOk)) {
                    break;
                }

                // optional LTM: decay/promotion + store if valid
                if (ltmEnabled && globalBest != null && globalBest.evaluation != null && globalBest.evaluation.valid) {
                    ltmTick.incrementAndGet();
                    ltmDecay();
                    maybeWriteLtm(q0, context, globalBest);
                }
            }

            // finalize response with strategy
            ThoughtResult result = strategy.build(q0, state, globalBest);

            long dtMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - tStart);
            log.info("think.done dtMs={} score={} strictTried={} bestValid={} outLen={}",
                    dtMs,
                    (globalBest == null ? 0.0 : globalBest.score),
                    strictTried,
                    (globalBest != null && globalBest.evaluation != null && globalBest.evaluation.valid),
                    (result == null || result.text == null ? 0 : result.text.length())
            );

            return result;
        }
    }

    // --------- Strict verify + repair tags ---------

    Candidate strictVerifyCandidate(String q0, List<Statement> context, Candidate best) {
        if (best == null) return Candidate.empty(q0);

        // strict mode: reduce novelty, require evidence, avoid echo, fix structure if needed
        ThoughtState s2 = new ThoughtState();
        s2.seed = mix(best.seed, 0x9E3779B97F4A7C15L);
        s2.iteration = best.iteration;
        s2.draftIndex = 999;

        s2.tags = new HashMap<>();
        s2.tags.put(Tags.REPAIR_ADD_EVIDENCE, "1");
        s2.tags.put(Tags.REPAIR_REDUCE_NOVELTY, "1");
        s2.tags.put(Tags.REPAIR_REQUIRE_EVIDENCE, "true");
        s2.tags.put(Tags.REPAIR_FORBID_GENERIC, "true");
        s2.tags.put(Tags.REPAIR_FIX_STRUCTURE, "true");
        s2.tags.put(Tags.REPAIR_AVOID_ECHO, "true");
        s2.tags.put(Tags.REPAIR_CITE_IDS, "true");

        // Provide previous candidate as hint (generator may use it)
        s2.generationHint = best.text;

        List<String> drafts = generateDraftsDeterministic(q0, context, s2, 2);
        drafts = Drafts.sanitize(drafts, 2);

        List<Candidate> evaluated = evaluateDrafts(q0, context, s2, drafts);
        Candidate strictBest = pickBestCandidate(evaluated);

        // If strict best is still telemetry/noctx -> reject
        if (strictBest != null && strictBest.text != null && knobs.TELEMETRY.matcher(strictBest.text).find()) {
            strictBest = Candidate.empty(q0);
        }

        return strictBest == null ? Candidate.empty(q0) : strictBest;
    }

    void applyStrictRepairTags(ThoughtState state, boolean veryShortQ) {
        if (state == null) return;
        if (state.tags == null) state.tags = new HashMap<>(16);

        state.tags.put(Tags.REPAIR_VERIFY_FAIL, "1");
        state.tags.put(Tags.REPAIR_ADD_EVIDENCE, "1");
        state.tags.put(Tags.REPAIR_REDUCE_NOVELTY, "1");
        state.tags.put(Tags.REPAIR_REQUIRE_EVIDENCE, "true");
        state.tags.put(Tags.REPAIR_CITE_IDS, "true");

        // For very short prompts avoid over-constraining style; focus on grounding.
        if (!veryShortQ) {
            state.tags.put(Tags.REPAIR_FORBID_GENERIC, "true");
        }
    }

    // --------- Retrieval / rerank / compress ---------

    private List<Statement> doRetrieve(long seed, int k, List<String> queries) {
        if (retriever == null) return List.of();
        if (queries == null || queries.isEmpty()) return List.of();

        long s = mix(seed, 0xD1B54A32D192ED03L);
        try {
            return retriever.retrieve(queries, k, s);
        } catch (Exception e) {
            log.warn("retrieve.fail k={} q={}", k, queries, e);
            return List.of();
        }
    }

    private List<Statement> rerankAndCompress(String q0, List<Statement> retrieved, int k) {
        if (retrieved == null || retrieved.isEmpty()) return List.of();

        List<ScoredStmt> scored = new ArrayList<>(Math.min(retrieved.size(), knobs.rerankN));

        int n = Math.min(retrieved.size(), knobs.rerankN);
        for (int i = 0; i < n; i++) {
            Statement st = retrieved.get(i);
            if (st == null || st.text == null || st.text.isBlank()) continue;
            double score = overlapScore(q0, st.text);

            // mild preference to higher-priority sources
            score += 0.01 * clamp01(st.priority);

            // LTM penalty if it looks stale/unconfirmed (keeps RAG fresh)
            score -= ltmPenalty(st);

            // DF penalty: generic terms often rank high but are useless
            score -= dfPenalty(st);

            scored.add(new ScoredStmt(st, score));
        }

        scored.sort((a, b) -> Double.compare(b.score, a.score));

        int m = Math.min(knobs.rerankM, scored.size());
        List<Statement> top = new ArrayList<>(m);
        for (int i = 0; i < m; i++) top.add(scored.get(i).stmt);

        // compress
        return compress(top, knobs.compressSentences, knobs.compressMaxChars);
    }

    private List<Statement> compress(List<Statement> top, int sentences, int maxChars) {
        if (top == null || top.isEmpty()) return List.of();

        ArrayList<Statement> out = new ArrayList<>(top.size());
        int budget = Math.max(120, maxChars);

        for (Statement st : top) {
            if (st == null || st.text == null) continue;
            String t = st.text.trim();
            if (t.isEmpty()) continue;

            String shortT = takeSentences(t, sentences);
            if (shortT.length() > budget) {
                shortT = shortT.substring(0, Math.max(0, budget));
            }
            budget -= shortT.length();
            out.add(st.withText(shortT));

            if (budget <= 0) break;
        }
        return out;
    }

    // --------- Draft generation / evaluation ---------

    private List<String> generateDraftsDeterministic(String q0, List<Statement> ctx, ThoughtState state, int drafts) {
        int n = Math.max(1, drafts);

        // Generator path (preferred: BeamSearch/TensorFlow).
        if (generator != null) {
            ArrayList<String> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                ThoughtState s2 = state.copyForDraft(i);
                s2.draftIndex = i;
                s2.seed = mix(state.seed, i * 0x9E3779B97F4A7C15L);
                try {
                    String d = generator.generate(q0, ctx, s2);
                    out.add(d == null ? "" : d);
                } catch (Exception e) {
                    log.warn("gen.fail i={}", i, e);
                    out.add("");
                }
            }
            return out;
        }

        // Fallback path (no generator): use ResponseStrategy deterministically.
        ArrayList<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            try {
                ThoughtState s2 = state.copyForDraft(i);
                s2.draftIndex = i;
                String d = strategy.generate(q0, ctx, s2);
                out.add(d == null ? "" : d);
            } catch (Exception e) {
                log.warn("strategy.gen.fail i={}", i, e);
                out.add("");
            }
        }
        return out;
    }

    private List<Candidate> evaluateDrafts(String q0, List<Statement> ctx, ThoughtState state, List<String> drafts) {
        if (drafts == null || drafts.isEmpty()) return List.of(Candidate.empty(q0));

        // evaluation parallelism

        int par = Math.min(evalParallelism, drafts.size());
        if (par <= 1) {
            ArrayList<Candidate> out = new ArrayList<>(drafts.size());
            for (int i = 0; i < drafts.size(); i++) {
                out.add(evaluateOne(q0, ctx, state, drafts.get(i), i));
            }
            return out;
        }

        List<Callable<Candidate>> tasks = new ArrayList<>(drafts.size());
        for (int i = 0; i < drafts.size(); i++) {
            final int idx = i;
            tasks.add(() -> evaluateOne(q0, ctx, state, drafts.get(idx), idx));
        }

        try {
            List<Future<Candidate>> futures = evalPool.invokeAll(tasks);
            ArrayList<Candidate> out = new ArrayList<>(futures.size());
            for (Future<Candidate> f : futures) {
                try {
                    out.add(f.get());
                } catch (Exception e) {
                    log.warn("eval.future.fail", e);
                    out.add(Candidate.empty(q0));
                }
            }
            return out;
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            ArrayList<Candidate> out = new ArrayList<>(drafts.size());
            for (int i = 0; i < drafts.size(); i++) out.add(Candidate.empty(q0));
            return out;
        }
    }

    private Candidate evaluateOne(String q0, List<Statement> ctx, ThoughtState state, String draft, int draftIdx) {
        ThoughtState s2 = state.copyForDraft(draftIdx);
        s2.draftIndex = draftIdx;

        Candidate c = new Candidate(q0, draft);
        c.iteration = state.iteration;
        c.seed = state.seed;
        c.draftIndex = draftIdx;

        CandidateEvaluator.Evaluation e = evaluator.evaluate(q0, ctx, s2, c.text);
        c.evaluation = e;
        c.score = (e == null ? 0.0 : e.score);
        c.critique = (e == null ? "" : e.notes);

        return c;
    }

    private Candidate pickBestCandidate(List<Candidate> evaluated) {
        if (evaluated == null || evaluated.isEmpty()) return null;

        Candidate best = null;
        double bestScore = -1e9;

        for (Candidate c : evaluated) {
            if (c == null) continue;
            double s = c.score;
            if (best == null || s > bestScore) {
                best = c;
                bestScore = s;
            }
        }
        return best;
    }

    // --------- Stop rules ---------

    private boolean shouldStop(int iter, Candidate best, int stagnation, boolean strictOk) {
        if (best == null) return false;
        if (!strictOk) return false; // strict failed -> must continue repair
        if (targetScore > 0 && best.score >= targetScore) return true;
        if (patience > 0 && stagnation >= patience) return true;
        return iter >= iterations;
    }

    // --------- Refinement queries ---------

    private List<String> deriveRefineQueries(String q0, ThoughtState state, int rounds, int budget) {
        // baseline: always include raw q0
        ArrayList<String> out = new ArrayList<>();
        out.add(q0);

        if (rounds <= 0) return out;

        Candidate best = state.bestSoFar;
        CandidateEvaluator.Evaluation ev = state.bestEvaluation;

        // DO NOT derive from bad/invalid bestSoFar
        if (best == null || ev == null || !ev.valid) {
            return out;
        }

        String text = best.text;
        if (text == null || text.isBlank()) return out;

        // derive top terms from best answer but avoid telemetry/internal markers
        Map<String, Integer> df = approxDocumentFrequency(text);
        if (df.isEmpty()) return out;

        // pick terms with high DF ratio (simplified)
        List<String> terms = df.entrySet().stream()
                .filter(e -> e.getValue() >= 2)
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .map(Map.Entry::getKey)
                .limit(budget)
                .collect(Collectors.toList());

        for (String t : terms) {
            if (t == null || t.isBlank()) continue;
            if (knobs.TELEMETRY.matcher(t).find()) continue;
            out.add(t);
            if (out.size() >= budget) break;
        }

        return out;
    }

    private Map<String, Integer> approxDocumentFrequency(String text) {
        if (text == null || text.isBlank()) return Map.of();

        String s = text.toLowerCase(Locale.ROOT);
        Map<String, Integer> df = new HashMap<>();

        java.util.regex.Matcher m = knobs.WORD.matcher(s);
        while (m.find()) {
            String w = m.group();
            if (w == null || w.isBlank()) continue;
            df.merge(w, 1, Integer::sum);
        }

        // drop very generic / low DF cut
        int max = df.values().stream().max(Integer::compareTo).orElse(1);
        double cut = knobs.deriveDfCut;

        df.entrySet().removeIf(e -> (e.getValue() / (double) max) < cut);
        return df;
    }

    // --------- LTM ---------

    private double ltmPenalty(Statement st) {
        if (!ltmEnabled || st == null) return 0.0;
        if (ltm.isEmpty()) return 0.0;
        if (st.id == null || st.id.isBlank()) return 0.0;

        LtmCell cell = ltm.get(st.id);
        if (cell == null) return 0.0;

        long age = Math.max(0, ltmTick.get() - cell.lastSeenTick);
        return 0.02 * Math.min(5, age);
    }


    private double dfPenalty(Statement st) {
        // minimal heuristic: penalize very short/generic statements a bit
        if (st == null || st.text == null) return 0.0;
        String t = st.text.trim();
        if (t.length() < 40) return 0.03;
        return 0.0;
    }

    private void ltmDecay() {
        if (!ltmEnabled || ltm.isEmpty()) return;
        long now = ltmTick.get();

        for (LtmCell cell : ltm.values()) {
            if (cell == null) continue;
            long age = Math.max(0, now - cell.lastSeenTick);
            if (age <= 0) continue;
            cell.score = Math.max(0, cell.score - ltmDecayPerTick * age);
        }
    }

    private void maybeWriteLtm(String q0, List<Statement> ctx, Candidate best) {
        if (!ltmEnabled) return;
        if (best == null || best.evaluation == null) return;
        if (!best.evaluation.valid) return;

        // groundedness gate
        if (best.evaluation.groundedness < ltmWriteMinGroundedness) return;
        if (ctx == null || ctx.isEmpty()) return;

        // ВАЖНО: tick уже увеличен выше (ltmTick.incrementAndGet(); ltmDecay(); maybeWriteLtm(...))
        // Тут не надо увеличивать второй раз — берём текущее значение.
        final long tick = ltmTick.get();

        // Пока отдельного ltmWriteK нет — используем ltmRecallK как fallback.
        final int k = Math.min(Math.max(1, ltmRecallK), ctx.size());

        for (int i = 0; i < k; i++) {
            Statement st = ctx.get(i);
            if (st == null) continue;

            String id = st.id;
            if (id == null || id.isBlank()) continue;

            // top evidence важнее (детерминированно)
            double posW = 1.0 / (1.0 + i); // 1.0, 0.5, 0.33...
            double boost = ltmPromotionBoost * posW;

            // детерминированные токены (без магии)
            final Set<String> toks = ltmTokens(st, q0);

            ltm.compute(id, (key, old) -> {
                if (old == null) {
                    return new LtmCell(st, toks, tick, boost);
                }

                // merge tokens безопасно (old.tokens может быть Set.of())
                HashSet<String> merged = new HashSet<>(Math.max(16, (old.tokens == null ? 0 : old.tokens.size()) + toks.size()));
                if (old.tokens != null && !old.tokens.isEmpty()) merged.addAll(old.tokens);
                if (!toks.isEmpty()) merged.addAll(toks);

                // создаём новый cell (stmt/tokens у вас final)
                return new LtmCell(
                        st,                    // обновляем на актуальный Statement
                        merged,
                        tick,                  // lastSeenTick
                        old.score + boost      // накапливаем score
                );
            });
        }
    }

    private Set<String> ltmTokens(Statement st, String q0) {
        HashSet<String> out = new HashSet<>(32);

        // 1) токены из факта
        if (st != null && st.text != null && !st.text.isBlank()) {
            addTokens(out, st.text);
        }

        // 2) опционально токены из вопроса (связка "вопрос -> факт")
        if (q0 != null && !q0.isBlank()) {
            addTokens(out, q0);
        }

        return out.isEmpty() ? Set.of() : out;
    }

    private void addTokens(Set<String> out, String text) {
        if (text == null || text.isBlank()) return;
        var m = knobs.WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (m.find()) out.add(m.group());
    }



    // --------- Utilities ---------

    private static String normalizeUserText(String s) {
        if (s == null) return "";
        String x = Normalizer.normalize(s, Normalizer.Form.NFKC).trim();
        // clamp extreme length
        if (x.length() > 4096) x = x.substring(0, 4096);
        return x;
    }

    private int approxQueryTokenCount(String s) {
        if (s == null || s.isBlank()) return 0;
        int cnt = 0;
        java.util.regex.Matcher m = knobs.WORD.matcher(s.toLowerCase(Locale.ROOT));
        while (m.find()) cnt++;
        return cnt;
    }

    private static boolean isMostlyCyrillic(String text) {
        if (text == null || text.isBlank()) return false;
        int cyr = 0, lat = 0;
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            Character.UnicodeBlock b = Character.UnicodeBlock.of(ch);
            if (b == Character.UnicodeBlock.CYRILLIC) cyr++;
            else if (b == Character.UnicodeBlock.BASIC_LATIN || b == Character.UnicodeBlock.LATIN_1_SUPPLEMENT) lat++;
        }
        return cyr >= lat;
    }

    private void applyDefaultResponseSchema(ThoughtState state, String userText, Intent intent) {
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

        // language hint for generator
        state.tags.putIfAbsent("lang", ru ? "RU" : "EN");
    }

    private void applyDefaultLengthHints(ThoughtState state, String userText) {
        if (state == null) return;
        if (state.tags == null) state.tags = new HashMap<>(16);

        int qTok = approxQueryTokenCount(userText);

        int minTok = safeInt(state.tags.get("gen.minTok"), -1);
        int maxTok = safeInt(state.tags.get("gen.maxTok"), -1);

        if (qTok > 0 && qTok <= 5) {
            if (minTok < 0) minTok = 8;
            if (maxTok < 0) maxTok = 40;
        } else if (qTok > 0 && qTok <= 20) {
            if (minTok < 0) minTok = 20;
            if (maxTok < 0) maxTok = 120;
        } else {
            if (minTok < 0) minTok = 40;
            if (maxTok < 0) maxTok = 220;
        }

        state.tags.put("gen.minTok", Integer.toString(minTok));
        state.tags.put("gen.maxTok", Integer.toString(maxTok));
    }

    private static int safeInt(Object v, int def) {
        if (v == null) return def;
        try {
            return Integer.parseInt(String.valueOf(v));
        } catch (Exception e) {
            return def;
        }
    }

    private static String fmt(double x) {
        return String.format(Locale.ROOT, "%.4f", x);
    }

    private static double clamp01(double x) {
        if (x < 0) return 0;
        if (x > 1) return 1;
        return x;
    }

    private static String takeSentences(String text, int n) {
        if (text == null) return "";
        if (n <= 0) return "";

        String t = text.trim();
        if (t.isEmpty()) return "";

        int count = 0;
        int end = t.length();
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (c == '.' || c == '!' || c == '?') {
                count++;
                if (count >= n) {
                    end = i + 1;
                    break;
                }
            }
        }
        return t.substring(0, end).trim();
    }

    private double overlapScore(String q0, String text) {
        if (q0 == null || text == null) return 0;
        String q = q0.toLowerCase(Locale.ROOT);
        String t = text.toLowerCase(Locale.ROOT);

        // very cheap token overlap (WORD)
        Set<String> qs = new HashSet<>();
        java.util.regex.Matcher m1 = knobs.WORD.matcher(q);
        while (m1.find()) qs.add(m1.group());

        if (qs.isEmpty()) return 0;

        int hit = 0;
        java.util.regex.Matcher m2 = knobs.WORD.matcher(t);
        while (m2.find()) {
            if (qs.contains(m2.group())) hit++;
        }

        return hit / (double) Math.max(1, qs.size());
    }

    private static long mix(long a, long b) {
        long x = a ^ (b + 0x9E3779B97F4A7C15L);
        x ^= (x >>> 30);
        x *= 0xBF58476D1CE4E5B9L;
        x ^= (x >>> 27);
        x *= 0x94D049BB133111EBL;
        x ^= (x >>> 31);
        return x;
    }

    // --------- Small internal structs ---------

    private static final class ScoredStmt {
        final Statement stmt;
        final double score;

        ScoredStmt(Statement stmt, double score) {
            this.stmt = stmt;
            this.score = score;
        }
    }

    private static final class LtmCell {
        final String id;
        final Statement stmt;
        final Set<String> tokens;
        long lastSeenTick;
        double score;

        LtmCell(Statement stmt, Set<String> tokens, long lastSeenTick, double score) {
            this.stmt = stmt;
            this.id = (stmt == null || stmt.id == null) ? "" : stmt.id;
            this.tokens = (tokens == null) ? Set.of() : tokens;
            this.lastSeenTick = lastSeenTick;
            this.score = score;
        }
    }

    // --------- Draft sanitization (stable + near dedup) ---------

    private static final class Drafts {
        private Drafts() {}

        static List<String> sanitize(List<String> drafts, int want) {
            if (drafts == null || drafts.isEmpty()) return List.of("");

            // Stable (deterministic) sanitization pipeline:
            //  1) trim + normalize whitespace
            //  2) exact dedup (normalized text)
            //  3) near dedup (token signature)
            //  4) stable cap (keep first N in input order)
            LinkedHashMap<String, String> byExact = new LinkedHashMap<>();
            LinkedHashMap<String, String> bySig = new LinkedHashMap<>();

            for (String s : drafts) {
                String x = (s == null) ? "" : normalizeWhitespace(s);
                if (x.isEmpty()) continue;

                // exact dedup (normalized)
                if (!byExact.containsKey(x)) byExact.put(x, x);

                // near dedup signature
                String sig = nearSignature(x);
                if (!sig.isEmpty() && !bySig.containsKey(sig)) {
                    bySig.put(sig, x);
                }
            }

            ArrayList<String> out = new ArrayList<>(byExact.values());

            // If near-dedup reduced too much, fall back to exact set.
            // Otherwise, keep near-dedup list (it fights mode collapse better).
            if (!bySig.isEmpty() && bySig.size() < out.size()) {
                out = new ArrayList<>(bySig.values());
            }

            if (out.isEmpty()) out.add("");

            // keep at least 2..4 (avoid collapse), but respect want as upper bound
            int cap = Math.max(2, Math.min(want, Math.max(4, out.size())));
            if (out.size() > cap) out = new ArrayList<>(out.subList(0, cap));
            return out;
        }

        private static String normalizeWhitespace(String s) {
            if (s == null) return "";
            // NFKC to normalize weird spaces/punctuation; then collapse whitespace.
            String x = Normalizer.normalize(s, Normalizer.Form.NFKC).trim();
            if (x.isEmpty()) return "";
            StringBuilder b = new StringBuilder(x.length());
            boolean ws = false;
            for (int i = 0; i < x.length(); i++) {
                char c = x.charAt(i);
                if (Character.isWhitespace(c) || c == '\u00A0') {
                    if (!ws) {
                        b.append(' ');
                        ws = true;
                    }
                } else if (!Character.isISOControl(c)) {
                    b.append(c);
                    ws = false;
                }
            }
            return b.toString().trim();
        }

        /**
         * Cheap near-dup signature: take normalized alnum "words" (len>=3), cap to 24 tokens.
         * Deterministic, allocation-light, good enough to kill generator mode-collapse.
         */
        private static String nearSignature(String x) {
            if (x == null || x.isEmpty()) return "";
            String s = x.toLowerCase(Locale.ROOT);

            StringBuilder sig = new StringBuilder(160);
            StringBuilder tok = new StringBuilder(24);

            int tokens = 0;
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (Character.isLetterOrDigit(c) || c == '_') {
                    tok.append(c);
                } else {
                    if (tok.length() >= 3) {
                        if (sig.length() > 0) sig.append(' ');
                        sig.append(tok);
                        tokens++;
                        if (tokens >= 24) break;
                    }
                    tok.setLength(0);
                }
            }
            if (tokens < 24 && tok.length() >= 3) {
                if (sig.length() > 0) sig.append(' ');
                sig.append(tok);
            }
            return sig.toString();
        }
    }
}