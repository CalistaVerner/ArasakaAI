// FILE: IterativeThoughtEngine.java
package org.calista.arasaka.ai.think.engine.impl;

import org.apache.logging.log4j.CloseableThreadContext;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.think.ThoughtResult;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateControlSignals;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.engine.ThoughtCycleEngine;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.response.ResponseStrategy;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;
import org.calista.arasaka.ai.think.utils.Drafts;
import org.calista.arasaka.ai.think.utils.Tags;

import java.text.Normalizer;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * IterativeThoughtEngine — deterministic multi-iteration reasoning pipeline:
 * retrieve -> (LTM recall) -> rerank -> compress -> generate drafts -> evaluate -> (strict verify) -> repair -> stop.
 */
public final class IterativeThoughtEngine implements ThoughtCycleEngine {

    private static final Logger log = LogManager.getLogger(IterativeThoughtEngine.class);

    // --------- Dependencies ---------

    private final Retriever retriever;
    private final IntentDetector intentDetector;
    private final TextGenerator generator;
    private final CandidateEvaluator evaluator;
    private final ResponseStrategy strategy;

    // Evaluation executor (injected; ownership is handled by Think)
    private final ExecutorService evalPool;
    private final int evalParallelism;

    // --------- Knobs ---------

    static final class Knobs {
        final java.util.regex.Pattern WORD;
        final java.util.regex.Pattern TELEMETRY;

        final int maxUserChars;
        final int retrieveKMin;
        final int deriveMaxQueries;
        final int compressMaxChars;

        final int rerankN;
        final int rerankM;
        final int compressSentences;

        final double deriveDfCut;

        final double strictForceRisk;
        final double strictForceUngrounded;
        final double strictForceWeakStructure;

        Knobs(
                java.util.regex.Pattern word,
                java.util.regex.Pattern telemetry,
                int maxUserChars,
                int retrieveKMin,
                int deriveMaxQueries,
                int compressMaxChars,
                int rerankN,
                int rerankM,
                int compressSentences,
                double deriveDfCut,
                double strictForceRisk,
                double strictForceUngrounded,
                double strictForceWeakStructure
        ) {
            this.WORD = Objects.requireNonNull(word, "word");
            this.TELEMETRY = Objects.requireNonNull(telemetry, "telemetry");

            this.maxUserChars = maxUserChars;
            this.retrieveKMin = retrieveKMin;
            this.deriveMaxQueries = deriveMaxQueries;
            this.compressMaxChars = compressMaxChars;

            this.rerankN = rerankN;
            this.rerankM = rerankM;
            this.compressSentences = compressSentences;

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
                    4096,
                    2,
                    4,
                    700,
                    120,
                    18,
                    2,
                    0.16,
                    0.52,
                    0.52,
                    0.45
            );
        }
    }

    private final Knobs knobs;

    // --------- Config ---------

    private final int iterations;
    private final int retrieveK;
    private final int draftsPerIteration;
    private final int patience;
    private final double targetScore;

    private final double exploitEvidenceStrictness;

    // strict verify-pass policy
    private final boolean strictVerifyEnabled;
    private final double strictVerifyMinScore;

    // LTM knobs
    private final boolean ltmEnabled;
    private final int ltmCapacity;
    private final int ltmRecallK;
    private final double ltmWriteMinGroundedness;
    private final double ltmDecayPerTick;
    private final double ltmPromotionBoost;

    // LTM store
    private final LinkedHashMap<String, LtmCell> ltm;

    /**
     * Global monotonic tick of the thought-cycle.
     * IMPORTANT: incremented once per iteration, regardless of LTM enabled/disabled.
     */
    private final AtomicLong thoughtTick = new AtomicLong(0);

    /**
     * Tracks last tick when LTM decay was applied.
     * Ensures decay is applied at most once per tick (even if called multiple times).
     */
    private long ltmLastDecayTick = 0L;

    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            TextGenerator generator,
            CandidateEvaluator evaluator,
            ResponseStrategy strategy,
            ExecutorService evalPool,
            int evalParallelism,
            int iterations,
            int retrieveK,
            int draftsPerIteration,
            int patience,
            double targetScore,
            double exploitEvidenceStrictness,
            boolean strictVerifyEnabled,
            double strictVerifyMinScore,
            boolean ltmEnabled,
            int ltmCapacity,
            int ltmRecallK,
            double ltmWriteMinGroundedness,
            double ltmDecayPerTick,
            double ltmPromotionBoost
    ) {
        this(retriever, intentDetector, generator, evaluator, strategy,
                Knobs.defaults(),
                evalPool, evalParallelism,
                iterations, retrieveK, draftsPerIteration, patience, targetScore,
                exploitEvidenceStrictness,
                strictVerifyEnabled, strictVerifyMinScore,
                ltmEnabled, ltmCapacity, ltmRecallK, ltmWriteMinGroundedness, ltmDecayPerTick, ltmPromotionBoost
        );
    }

    public IterativeThoughtEngine(
            Retriever retriever,
            IntentDetector intentDetector,
            TextGenerator generator,
            CandidateEvaluator evaluator,
            ResponseStrategy strategy,
            Knobs knobs,
            ExecutorService evalPool,
            int evalParallelism,
            int iterations,
            int retrieveK,
            int draftsPerIteration,
            int patience,
            double targetScore,
            double exploitEvidenceStrictness,
            boolean strictVerifyEnabled,
            double strictVerifyMinScore,
            boolean ltmEnabled,
            int ltmCapacity,
            int ltmRecallK,
            double ltmWriteMinGroundedness,
            double ltmDecayPerTick,
            double ltmPromotionBoost
    ) {
        this.retriever = retriever;
        this.intentDetector = intentDetector;
        this.generator = generator;
        this.evaluator = evaluator;
        this.strategy = strategy;
        this.knobs = (knobs == null ? Knobs.defaults() : knobs);

        this.evalPool = evalPool;
        this.evalParallelism = Math.max(1, evalParallelism);

        this.iterations = Math.max(1, iterations);
        this.retrieveK = Math.max(1, retrieveK);
        this.draftsPerIteration = Math.max(1, draftsPerIteration);
        this.patience = Math.max(0, patience);
        this.targetScore = targetScore;

        this.exploitEvidenceStrictness = clamp01(exploitEvidenceStrictness);

        this.strictVerifyEnabled = strictVerifyEnabled;
        this.strictVerifyMinScore = strictVerifyMinScore;

        this.ltmEnabled = ltmEnabled;
        this.ltmCapacity = Math.max(0, ltmCapacity);
        this.ltmRecallK = Math.max(0, ltmRecallK);
        this.ltmWriteMinGroundedness = clamp01(ltmWriteMinGroundedness);

        this.ltmDecayPerTick = Math.max(0.0, ltmDecayPerTick);
        this.ltmPromotionBoost = Math.max(0.0, ltmPromotionBoost);

        this.ltm = new LinkedHashMap<>(Math.max(16, this.ltmCapacity), 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, LtmCell> eldest) {
                return ltmCapacity > 0 && size() > ltmCapacity;
            }
        };
    }

    @Override
    public ThoughtResult think(String userText, long seed) {
        final String reqId = Long.toUnsignedString(mix(seed, System.nanoTime()), 36);
        try (final CloseableThreadContext.Instance ctc = CloseableThreadContext.put("req", reqId)) {

            final String q0 = normalizeUserText(userText);
            final Set<String> qTokensLower = buildQueryTokenSet(q0);

            final Intent intent = intentDetector.detect(q0);

            ThoughtState state = new ThoughtState();
            state.seed = seed;
            state.iteration = 0;
            state.draftIndex = -1;
            state.phase = CandidateControlSignals.Phase.EXPLORE.ordinal();
            state.diversity = CandidateControlSignals.Diversity.MEDIUM.ordinal();
            state.intent = intent;

            Candidate globalBest = Candidate.empty(q0);
            double prevBestScore = globalBest.score;
            int stagnation = 0;
            boolean strictTried = false;
            boolean strictOk = false;

            log.info("{} {} think.start len={} tok={} intent={} hasGen={} refineRounds={} ltmEnabled={} iterations={} retrieveK={} draftsPerIter={} evalPar={}",
                    reqId,
                    (intent == null ? "UNKNOWN" : intent.name()),
                    (q0 == null ? 0 : q0.length()),
                    approxQueryTokenCount(q0),
                    (intent == null ? "UNKNOWN" : intent.name()),
                    (generator != null),
                    1,
                    ltmEnabled,
                    iterations,
                    retrieveK,
                    draftsPerIteration,
                    evalParallelism
            );
            log.debug("{} {} think.q0='{}'", reqId, (intent == null ? "UNKNOWN" : intent.name()), q0);

            for (int iter = 1; iter <= iterations; iter++) {

                // ✅ ONE monotonic tick per iteration (independent of LTM flag)
                final long tick = thoughtTick.incrementAndGet();

                state.iteration = iter;

                // Per-iteration runtime cache (shared across drafts via ThoughtState.copyForDraft)
                if (state.cache == null) state.cache = new HashMap<>(8);
                else state.cache.clear();

                boolean explore = (iter == 1) || (state.phase == CandidateControlSignals.Phase.EXPLORE.ordinal());
                List<String> refineQueries = deriveQueries(q0, intent, state, explore);
                state.lastQueries = refineQueries;

                int k = Math.max(1, (explore ? Math.max(knobs.retrieveKMin, retrieveK) : retrieveK));

                log.debug("{} {} iter={} tick={} phase={} queries.n={} queries={}",
                        reqId, (intent == null ? "UNKNOWN" : intent.name()),
                        iter, tick,
                        (explore ? "EXPLORE" : "EXPLOIT"),
                        refineQueries.size(), refineQueries);

                List<Statement> retrieved = doRetrieve(seed, k, refineQueries);

                // ---- LTM recall ----
                List<Statement> recalled = recallFromLtm(refineQueries, qTokensLower, ltmRecallK, tick);
                state.recalledMemory = recalled;
                state.recallQueries = refineQueries;

                List<Statement> mergedEvidence = mergeById(retrieved, recalled);

                List<Statement> context = rerankAndCompress(qTokensLower, mergedEvidence, k);
                state.lastContext = context;

                // Deterministic, non-empty drafts. If generator collapses into duplicates/empties,
                // we re-generate (deterministically) instead of wasting eval budget on "".
                List<String> drafts = generateDraftsDeterministic(q0, context, state, draftsPerIteration);

                List<Candidate> evaluated = evaluateDrafts(q0, context, state, drafts);
                Candidate bestIter = pickBestCandidate(evaluated);

                if (bestIter != null && bestIter.text != null && knobs.TELEMETRY.matcher(bestIter.text).find()) {
                    bestIter = Candidate.empty(q0);
                }

                boolean improved = (bestIter != null && bestIter.score > globalBest.score + 1e-9);
                boolean riskyAny = (bestIter != null && bestIter.evaluation != null && bestIter.evaluation.contradictionRisk >= 0.55);

                if (improved) {
                    globalBest = bestIter;
                    stagnation = 0;
                } else {
                    stagnation++;
                }

                // Keep ThoughtState as the "working memory" of the loop.
                // Strategies/generators can anchor on bestSoFar + bestEvaluation for real iterative improvement.
                updateStateAfterIteration(state, refineQueries, context, evaluated, bestIter, globalBest, prevBestScore, stagnation, tick);
                prevBestScore = (globalBest == null ? prevBestScore : globalBest.score);

                final boolean veryShortQ = (q0 == null || q0.length() <= 12);
                final boolean riskyShort = veryShortQ && riskyAny;

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
                        applyStrictRepairTags(state, veryShortQ);
                        // Strengthen repair with metric-driven hints (no domain hardcode):
                        // if strict pass says "structure/grounding" is weak, tighten sections.
                        tightenRepairFromEvaluation(state, strictChecked == null ? null : strictChecked.evaluation);
                        state.phase = CandidateControlSignals.Phase.REPAIR.ordinal();
                        state.diversity = CandidateControlSignals.Diversity.HIGH.ordinal();
                    } else if (strictChecked.score > globalBest.score) {
                        globalBest = strictChecked;
                        stagnation = 0;
                        updateStateAfterIteration(state, refineQueries, context, evaluated, bestIter, globalBest, prevBestScore, stagnation, tick);
                        prevBestScore = globalBest.score;
                    }
                }

                // ✅ LTM side-effects happen AFTER stop-check decision inputs are ready,
                // but tick remains global and already incremented.
                if (ltmEnabled) {
                    ltmDecay(tick);
                    if (globalBest != null && globalBest.evaluation != null && globalBest.evaluation.valid) {
                        maybeWriteLtm(tick, q0, context, globalBest);
                    }
                }

                if (shouldStop(iter, globalBest, stagnation, strictOk)) break;
            }

            ThoughtResult result = strategy.build(q0, state, globalBest);
            if (result == null) result = ThoughtResult.empty(q0);

            if (result.answer != null && knobs.TELEMETRY.matcher(result.answer).find()) {
                result = ThoughtResult.empty(q0);
            }

            log.info("{} {} think.done score={} strictTried={} strictOk={} stagnation={}",
                    reqId,
                    (intent == null ? "UNKNOWN" : intent.name()),
                    (globalBest == null ? -1.0 : globalBest.score),
                    strictTried,
                    strictOk,
                    stagnation
            );

            return result;
        }
    }

    // --------- Core stages ---------

    private List<String> deriveQueries(String q0, Intent intent, ThoughtState state, boolean explore) {
        // Deterministic query derivation (no domain hard-code):
        // q0 -> intent label (if known) -> exploit (bestSoFar terms) -> explore (context terms).
        final int maxQ = Math.max(1, knobs.deriveMaxQueries);

        final LinkedHashSet<String> uniq = new LinkedHashSet<>(maxQ * 2);
        final String base = (q0 == null ? "" : q0.trim());
        if (!base.isBlank()) uniq.add(base);

        if (intent != null) {
            String name = intent.name();
            if (name != null && !name.isBlank() && !"UNKNOWN".equalsIgnoreCase(name)) {
                uniq.add(name.toLowerCase(Locale.ROOT));
            }
        }

        // Build DF map from current context (if any) to filter overly common tokens.
        final List<Statement> ctx = (state == null ? null : state.lastContext);
        final Df df = buildDf(ctx);

        // Exploit: if we have a valid best-so-far, mine its discriminative terms.
        if (!explore && state != null && state.bestSoFar != null && state.bestEvaluation != null && state.bestEvaluation.valid) {
            List<String> terms = topTerms(state.bestSoFar.text, df, 8);
            for (String t : terms) {
                if (uniq.size() >= maxQ) break;
                uniq.add(t);
            }
        }

        // Explore: mine context terms if we still have budget.
        if (uniq.size() < maxQ) {
            List<String> terms = topTermsFromContext(ctx, df, 8);
            for (String t : terms) {
                if (uniq.size() >= maxQ) break;
                uniq.add(t);
            }
        }

        // If still short, fall back to query tokens (stable order by first appearance).
        if (uniq.size() < maxQ && base != null && !base.isBlank()) {
            for (String t : tokensInOrder(base, 12)) {
                if (uniq.size() >= maxQ) break;
                uniq.add(t);
            }
        }

        if (uniq.isEmpty()) return List.of();
        ArrayList<String> out = new ArrayList<>(Math.min(maxQ, uniq.size()));
        for (String s : uniq) {
            if (out.size() >= maxQ) break;
            if (s == null) continue;
            String x = s.trim();
            if (!x.isBlank()) out.add(x);
        }
        return out;
    }

    // --------- State as working memory (enterprise deterministic) ---------

    private void updateStateAfterIteration(
            ThoughtState state,
            List<String> refineQueries,
            List<Statement> context,
            List<Candidate> evaluated,
            Candidate bestIter,
            Candidate globalBest,
            double prevBestScore,
            int stagnation,
            long tick
    ) {
        if (state == null) return;

        state.lastQueries = refineQueries;
        state.lastContext = context;

        state.lastCandidate = bestIter;
        state.lastEvaluation = (bestIter == null ? null : bestIter.evaluation);
        state.lastCritique = (bestIter == null ? null : bestIter.critique);

        state.bestSoFar = globalBest;
        state.bestEvaluation = (globalBest == null ? null : globalBest.evaluation);

        state.stagnation = stagnation;
        state.scoreDelta = (globalBest == null ? 0.0 : (globalBest.score - prevBestScore));

        // Use best as generation anchor starting from iter>=2 (unless we are in repair, then it may be overridden).
        if (state.iteration >= 2 && state.phase != CandidateControlSignals.Phase.REPAIR.ordinal()) {
            if (state.bestSoFar != null && state.bestSoFar.text != null && !state.bestSoFar.text.isBlank()) {
                state.generationHint = state.bestSoFar.text;
            }
        }

        // Lightweight, non-telemetry trace (never feed to generator):
        CandidateEvaluator.Evaluation bev = state.bestEvaluation;
        String line = "tick=" + tick
                + " iter=" + state.iteration
                + " phase=" + phaseName(state.phase)
                + " ctx=" + (context == null ? 0 : context.size())
                + " drafts=" + (evaluated == null ? 0 : evaluated.size())
                + " best=" + fmt2(state.bestSoFar == null ? Double.NEGATIVE_INFINITY : state.bestSoFar.score)
                + " d=" + fmt2(state.scoreDelta)
                + (bev == null ? "" : (" g=" + fmt2(bev.groundedness) + " qc=" + fmt2(bev.queryCoverage) + " risk=" + fmt2(bev.contradictionRisk)));
        state.trace(line);
    }

    private void tightenRepairFromEvaluation(ThoughtState state, CandidateEvaluator.Evaluation ev) {
        if (state == null) return;
        if (ev == null) return;
        Map<String, String> t = state.ensureTags();

        // No domain rules: tighten based on metric deficits.
        if (ev.groundedness < knobs.strictForceUngrounded || ev.queryCoverage < 0.40) {
            t.put("response.sections", "summary,evidence,actions");
            t.put(Tags.REPAIR_REQUIRE_EVIDENCE, "true");
        }
        if (ev.contradictionRisk >= knobs.strictForceRisk) {
            t.put(Tags.REPAIR_ADD_EVIDENCE, "1");
            t.put(Tags.REPAIR_CITE_IDS, "true");
        }
        if (ev.structure < knobs.strictForceWeakStructure) {
            t.put(Tags.REPAIR_FIX_STRUCTURE, "true");
            t.put("response.sections", "summary,evidence,actions");
        }
        if (ev.repetition >= 0.40) {
            t.put(Tags.REPAIR_AVOID_ECHO, "true");
        }
        if (ev.novelty >= 0.55) {
            t.put(Tags.REPAIR_REDUCE_NOVELTY, "1");
        }
    }

    private static String fmt2(double v) {
        if (!Double.isFinite(v)) return "-";
        return String.format(Locale.ROOT, "%.2f", v);
    }

    // --------- Query derivation helpers (no magic) ---------

    private static final class Df {
        final Map<String, Integer> df;
        final int docs;
        Df(Map<String, Integer> df, int docs) {
            this.df = (df == null ? Map.of() : df);
            this.docs = Math.max(0, docs);
        }
        int get(String tokLower) {
            if (tokLower == null) return 0;
            Integer v = df.get(tokLower);
            return v == null ? 0 : v;
        }
        double ratio(String tokLower) {
            if (docs <= 0) return 0.0;
            return get(tokLower) / (double) docs;
        }
    }

    private Df buildDf(List<Statement> ctx) {
        if (ctx == null || ctx.isEmpty()) return new Df(Map.of(), 0);
        HashMap<String, Integer> df = new HashMap<>(128);
        int docs = 0;
        for (Statement st : ctx) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            docs++;
            HashSet<String> seen = new HashSet<>(64);
            java.util.regex.Matcher m = knobs.WORD.matcher(st.text.toLowerCase(Locale.ROOT));
            while (m.find()) {
                String tok = m.group();
                if (tok == null || tok.isBlank()) continue;
                seen.add(tok);
            }
            for (String tok : seen) {
                df.merge(tok, 1, Integer::sum);
            }
        }
        return new Df(df, docs);
    }

    private List<String> topTerms(String text, Df df, int limit) {
        if (text == null || text.isBlank() || limit <= 0) return List.of();
        LinkedHashSet<String> out = new LinkedHashSet<>(limit * 2);
        java.util.regex.Matcher m = knobs.WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (m.find() && out.size() < limit) {
            String tok = m.group();
            if (tok == null || tok.isBlank()) continue;
            if (df != null && df.docs > 0 && df.ratio(tok) >= knobs.deriveDfCut) continue; // too common
            out.add(tok);
        }
        return out.isEmpty() ? List.of() : new ArrayList<>(out);
    }

    private List<String> topTermsFromContext(List<Statement> ctx, Df df, int limit) {
        if (ctx == null || ctx.isEmpty() || limit <= 0) return List.of();

        // Rank tokens by (1 - dfRatio) and stable first-seen order.
        LinkedHashMap<String, Double> score = new LinkedHashMap<>(128);
        for (Statement st : ctx) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            java.util.regex.Matcher m = knobs.WORD.matcher(st.text.toLowerCase(Locale.ROOT));
            while (m.find()) {
                String tok = m.group();
                if (tok == null || tok.isBlank()) continue;
                if (df != null && df.docs > 0 && df.ratio(tok) >= knobs.deriveDfCut) continue;
                // prefer rarer tokens, but keep stable insertion order
                score.putIfAbsent(tok, 1.0 - (df == null ? 0.0 : df.ratio(tok)));
            }
        }
        if (score.isEmpty()) return List.of();

        ArrayList<Map.Entry<String, Double>> entries = new ArrayList<>(score.entrySet());
        entries.sort((a, b) -> {
            int c = Double.compare(b.getValue(), a.getValue());
            if (c != 0) return c;
            return a.getKey().compareTo(b.getKey());
        });

        ArrayList<String> out = new ArrayList<>(Math.min(limit, entries.size()));
        for (Map.Entry<String, Double> e : entries) {
            if (out.size() >= limit) break;
            out.add(e.getKey());
        }
        return out;
    }

    private List<String> tokensInOrder(String text, int limit) {
        if (text == null || text.isBlank() || limit <= 0) return List.of();
        LinkedHashSet<String> out = new LinkedHashSet<>(limit * 2);
        java.util.regex.Matcher m = knobs.WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (m.find() && out.size() < limit) {
            String tok = m.group();
            if (tok == null || tok.isBlank()) continue;
            out.add(tok);
        }
        return out.isEmpty() ? List.of() : new ArrayList<>(out);
    }

    private List<String> generateDraftsDeterministic(String q0, List<Statement> context, ThoughtState baseState, int n) {
        if (generator == null) return List.of();
        n = Math.max(1, n);

        final boolean noctx = (context == null || context.isEmpty());
        final String q = (q0 == null ? "" : q0);

        // Deterministic re-generation loop:
        // - generate -> normalize -> keep non-empty -> stable de-dup
        // - if not enough drafts, generate more with new deterministic variant ids.
        final int maxAttempts = Math.max(n, n * 3);
        final LinkedHashSet<String> uniq = new LinkedHashSet<>(n * 2);
        int attempt = 0;

        while (uniq.size() < n && attempt < maxAttempts) {
            final int variant = attempt;
            ThoughtState s = (baseState == null) ? new ThoughtState() : baseState.copyForDraft(variant);
            s.draftIndex = variant;

            // deterministic seed split per-variant
            s.seed = mix(s.seed, 0xD6E8FEB86659FD93L + variant);

            applyDraftDiversityTags(s, variant, n, noctx, q);

            try {
                String draft = generator.generate(q, context, s);
                String norm = Drafts.normalize(draft);
                if (!norm.isBlank()) uniq.add(norm);
            } catch (Exception e) {
                log.warn("generate.fail i={} q='{}'", variant, q, e);
            }
            attempt++;
        }

        if (uniq.isEmpty()) return List.of();
        ArrayList<String> out = new ArrayList<>(Math.min(n, uniq.size()));
        for (String x : uniq) {
            if (out.size() >= n) break;
            out.add(x);
        }
        return out;
    }

    private void applyDraftDiversityTags(ThoughtState s, int i, int n, boolean noctx, String q0) {
        if (s == null) return;

        if (s.tags == null) s.tags = new HashMap<>(16);

        // Engine emits only control signals; generator decides wording.
        s.tags.put("gen.variant", Integer.toString(i));
        s.tags.put("gen.total", Integer.toString(Math.max(1, n)));
        s.tags.put("gen.noTelemetry", "true");

        s.tags.put("gen.phase", phaseName(s.phase));
        s.tags.put("gen.diversity", diversityName(s.diversity));
        s.tags.put("gen.noctx", noctx ? "true" : "false");

        int tok = approxQueryTokenCount(q0);
        if (tok <= 2) s.tags.put("response.sections", "summary");
        else s.tags.put("response.sections", "summary,evidence,actions");

        s.tags.put("gen.plan", Integer.toString(i % Math.max(1, Math.min(9, n))));
    }

    private static String phaseName(int phaseOrdinal) {
        try {
            CandidateControlSignals.Phase[] v = CandidateControlSignals.Phase.values();
            if (phaseOrdinal >= 0 && phaseOrdinal < v.length) return v[phaseOrdinal].name().toLowerCase(Locale.ROOT);
        } catch (Throwable ignored) {}
        return "unknown";
    }

    private static String diversityName(int divOrdinal) {
        try {
            CandidateControlSignals.Diversity[] v = CandidateControlSignals.Diversity.values();
            if (divOrdinal >= 0 && divOrdinal < v.length) return v[divOrdinal].name().toLowerCase(Locale.ROOT);
        } catch (Throwable ignored) {}
        return "unknown";
    }

    private List<Candidate> evaluateDrafts(String q0, List<Statement> context, ThoughtState baseState, List<String> drafts) {
        if (evaluator == null) return List.of();
        if (drafts == null || drafts.isEmpty()) return List.of();

        final String q = (q0 == null ? "" : q0);
        final long baseSeed = (baseState == null ? 0L : baseState.seed);
        final int iter = (baseState == null ? 0 : baseState.iteration);

        final int n = drafts.size();

        boolean canPar = (evalPool != null && evalParallelism > 1 && n > 2);
        if (!canPar) {
            ArrayList<Candidate> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) out.add(evalOne(q, context, baseState, drafts, baseSeed, iter, i, null));
            return out;
        }

        final Map<String, String> mdc = org.apache.logging.log4j.ThreadContext.getImmutableContext();
        @SuppressWarnings("unchecked")
        final CompletableFuture<Candidate>[] fut = new CompletableFuture[n];

        for (int i = 0; i < n; i++) {
            final int idx = i;
            fut[i] = CompletableFuture.supplyAsync(() -> evalOne(q, context, baseState, drafts, baseSeed, iter, idx, mdc), evalPool);
        }

        ArrayList<Candidate> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            try {
                out.add(fut[i].join());
            } catch (Throwable t) {
                log.warn("eval.join.fail i={} q='{}'", i, q, t);
                out.add(Candidate.empty(q));
            }
        }
        return out;
    }

    private Candidate evalOne(
            String q,
            List<Statement> context,
            ThoughtState baseState,
            List<String> drafts,
            long baseSeed,
            int iter,
            int i,
            Map<String, String> mdc
    ) {
        final String d = (drafts.get(i) == null ? "" : drafts.get(i));

        if (mdc != null && !mdc.isEmpty()) {
            org.apache.logging.log4j.ThreadContext.putAll(mdc);
        }

        try {
            ThoughtState s;
            if (baseState == null) {
                s = new ThoughtState();
                s.seed = baseSeed;
                s.iteration = iter;
                s.draftIndex = i;
            } else {
                s = baseState.copyForDraft(i);
                s.draftIndex = i;
            }

            Candidate c = new Candidate(q, d);
            c.seed = mix(baseSeed, 0xA0761D6478BD642FL + i);
            c.iteration = iter;
            c.draftIndex = i;

            CandidateEvaluator.Evaluation ev = evaluator.evaluate(q, context, s, d);
            if (ev != null) ev.syncNotes();

            c.evaluation = ev;
            c.score = (ev == null ? Double.NEGATIVE_INFINITY : ev.effectiveScore());
            c.critique = (ev == null ? "" : (ev.critique == null ? "" : ev.critique));
            return c;
        } catch (Throwable e) {
            log.warn("eval.fail i={} q='{}'", i, q, e);
            Candidate c = new Candidate(q, d);
            CandidateEvaluator.Evaluation bad = CandidateEvaluator.Evaluation.invalid(Double.NEGATIVE_INFINITY, "err=eval_exception", "");
            c.evaluation = bad;
            c.score = bad.effectiveScore();
            c.critique = bad.critique;
            return c;
        } finally {
            if (mdc != null) org.apache.logging.log4j.ThreadContext.clearMap();
        }
    }

    private Candidate pickBestCandidate(List<Candidate> evaluated) {
        if (evaluated == null || evaluated.isEmpty()) return Candidate.empty("");

        Candidate bestValid = null;
        double bestValidScore = Double.NEGATIVE_INFINITY;

        Candidate bestAny = null;
        double bestAnyScore = Double.NEGATIVE_INFINITY;

        for (Candidate c : evaluated) {
            if (c == null) continue;

            double sc = Double.isFinite(c.score) ? c.score : Double.NEGATIVE_INFINITY;

            if (bestAny == null || sc > bestAnyScore) {
                bestAnyScore = sc;
                bestAny = c;
            }

            if (c.evaluation != null && c.evaluation.valid) {
                if (bestValid == null || sc > bestValidScore) {
                    bestValidScore = sc;
                    bestValid = c;
                }
            }
        }

        return (bestValid != null) ? bestValid : (bestAny != null ? bestAny : Candidate.empty(""));
    }

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

    private List<Statement> mergeById(List<Statement> a, List<Statement> b) {
        if ((a == null || a.isEmpty()) && (b == null || b.isEmpty())) return List.of();
        LinkedHashMap<String, Statement> map = new LinkedHashMap<>();

        if (a != null) {
            for (Statement st : a) {
                if (st == null || st.text == null || st.text.isBlank()) continue;
                map.putIfAbsent(stableStmtKey(st), st);
            }
        }
        if (b != null) {
            for (Statement st : b) {
                if (st == null || st.text == null || st.text.isBlank()) continue;
                map.putIfAbsent(stableStmtKey(st), st);
            }
        }
        return map.isEmpty() ? List.of() : new ArrayList<>(map.values());
    }

    private String stableStmtKey(Statement st) {
        if (st == null) return "";
        if (st.id != null && !st.id.isBlank()) return st.id;
        String t = (st.text == null) ? "" : st.text.trim();
        if (t.length() > 96) t = t.substring(0, 96);
        return "t:" + t;
    }

    // --------- Strict verify / repair ---------

    Candidate strictVerifyCandidate(String q0, List<Statement> context, Candidate best) {
        if (best == null) return Candidate.empty(q0);

        ThoughtState s2 = new ThoughtState();
        s2.seed = mix(best.seed, 0x9E3779B97F4A7C15L);
        s2.iteration = best.iteration;
        s2.draftIndex = 999;

        s2.tags = new HashMap<>(16);
        s2.tags.put(Tags.VERIFY_STRICT, "true");
        s2.tags.put(Tags.REPAIR_STRICT, "true");
        s2.tags.put(Tags.REPAIR_ADD_EVIDENCE, "1");
        s2.tags.put(Tags.REPAIR_REDUCE_NOVELTY, "1");
        s2.tags.put(Tags.REPAIR_REQUIRE_EVIDENCE, "true");
        s2.tags.put(Tags.REPAIR_FORBID_GENERIC, "true");
        s2.tags.put(Tags.REPAIR_FIX_STRUCTURE, "true");
        s2.tags.put(Tags.REPAIR_AVOID_ECHO, "true");
        s2.tags.put(Tags.REPAIR_CITE_IDS, "true");

        s2.generationHint = best.text;

        List<String> drafts = generateDraftsDeterministic(q0, context, s2, 2);
        drafts = Drafts.sanitize(drafts, 2);

        List<Candidate> evaluated = evaluateDrafts(q0, context, s2, drafts);
        Candidate strictBest = pickBestCandidate(evaluated);

        if (strictBest != null && strictBest.text != null && knobs.TELEMETRY.matcher(strictBest.text).find()) {
            strictBest = Candidate.empty(q0);
        }

        return strictBest == null ? Candidate.empty(q0) : strictBest;
    }

    void applyStrictRepairTags(ThoughtState state, boolean veryShortQ) {
        if (state == null) return;
        if (state.tags == null) state.tags = new HashMap<>(16);

        state.tags.put(Tags.VERIFY_STRICT, "true");
        state.tags.put(Tags.REPAIR_STRICT, "true");

        state.tags.put(Tags.REPAIR_VERIFY_FAIL, "1");
        state.tags.put(Tags.REPAIR_ADD_EVIDENCE, "1");
        state.tags.put(Tags.REPAIR_REDUCE_NOVELTY, "1");
        state.tags.put(Tags.REPAIR_REQUIRE_EVIDENCE, "true");
        state.tags.put(Tags.REPAIR_FORBID_GENERIC, "true");
        state.tags.put(Tags.REPAIR_FIX_STRUCTURE, "true");
        state.tags.put(Tags.REPAIR_AVOID_ECHO, "true");
        state.tags.put(Tags.REPAIR_CITE_IDS, "true");

        if (veryShortQ) {
            state.tags.put(Tags.REPAIR_REQUIRE_GREETING, "true");
        }
    }

    private boolean shouldStop(int iter, Candidate best, int stagnation, boolean strictOk) {
        if (best != null && best.score >= targetScore) return true;
        if (patience <= 0) return (iter >= iterations);
        if (stagnation >= patience) return true;
        if (strictVerifyEnabled && strictOk && iter >= 2) return true;
        return iter >= iterations;
    }

    // --------- Compression ---------

    private List<Statement> compress(List<Statement> top, int sentences, int maxChars) {
        if (top == null || top.isEmpty()) return List.of();
        if (sentences <= 0) return top;
        if (maxChars <= 0) return top;

        ArrayList<Statement> out = new ArrayList<>(top.size());
        int total = 0;

        for (Statement st : top) {
            if (st == null || st.text == null || st.text.isBlank()) continue;

            String t = st.text.trim();
            t = truncateSentences(t, sentences);

            if (t.length() > maxChars) t = t.substring(0, maxChars);

            total += t.length();
            if (total > maxChars) break;

            Statement s2 = st.copyShallow();
            s2.text = t;
            out.add(s2);
        }

        return out.isEmpty() ? List.of() : out;
    }

    private String truncateSentences(String t, int n) {
        if (t == null) return "";
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

    private Set<String> buildQueryTokenSet(String q0) {
        if (q0 == null || q0.isBlank()) return Set.of();
        HashSet<String> qs = new HashSet<>(32);
        java.util.regex.Matcher m = knobs.WORD.matcher(q0.toLowerCase(Locale.ROOT));
        while (m.find()) qs.add(m.group());
        return qs.isEmpty() ? Set.of() : qs;
    }

    private double overlapScore(Set<String> qTokensLower, String text) {
        if (qTokensLower == null || qTokensLower.isEmpty() || text == null || text.isBlank()) return 0.0;

        int hit = 0;
        java.util.regex.Matcher m = knobs.WORD.matcher(text);
        while (m.find()) {
            String tok = m.group();
            if (tok == null) continue;
            if (qTokensLower.contains(tok.toLowerCase(Locale.ROOT))) hit++;
        }
        return hit / (double) Math.max(1, qTokensLower.size());
    }

    // --------- Retrieval post-processing & LTM (no magic) ---------

    private List<Statement> rerankAndCompress(Set<String> qTokensLower, List<Statement> retrieved, int k) {
        if (retrieved == null || retrieved.isEmpty()) return List.of();
        if (qTokensLower == null) qTokensLower = Set.of();

        ArrayList<ScoredStmt> scored = new ArrayList<>(retrieved.size());
        for (Statement st : retrieved) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            double sc = overlapScore(qTokensLower, st.text);
            scored.add(new ScoredStmt(st, sc));
        }
        if (scored.isEmpty()) return List.of();

        scored.sort((a, b) -> {
            int c = Double.compare(b.score, a.score);
            if (c != 0) return c;
            return stableStmtKey(a.stmt).compareTo(stableStmtKey(b.stmt));
        });

        int preTop = Math.min(scored.size(), Math.max(k, Math.min(knobs.rerankN, k * Math.max(1, knobs.rerankM))));
        ArrayList<Statement> top = new ArrayList<>(preTop);
        for (int i = 0; i < preTop; i++) top.add(scored.get(i).stmt);

        return compress(top, knobs.compressSentences, knobs.compressMaxChars);
    }

    private List<Statement> recallFromLtm(List<String> queries, Set<String> qTokensLower, int k, long tick) {
        if (!ltmEnabled || k <= 0) return List.of();
        if (ltm.isEmpty()) return List.of();

        HashSet<String> qTok = new HashSet<>(64);
        if (qTokensLower != null) qTok.addAll(qTokensLower);
        if (queries != null) {
            for (String q : queries) {
                if (q == null || q.isBlank()) continue;
                java.util.regex.Matcher m = knobs.WORD.matcher(q.toLowerCase(Locale.ROOT));
                while (m.find()) qTok.add(m.group());
            }
        }
        if (qTok.isEmpty()) return List.of();

        ArrayList<Map.Entry<String, LtmCell>> cells = new ArrayList<>(ltm.entrySet());
        cells.sort((e1, e2) -> {
            double s1 = recallScore(qTok, e1.getValue());
            double s2 = recallScore(qTok, e2.getValue());
            int c = Double.compare(s2, s1);
            if (c != 0) return c;
            return e1.getKey().compareTo(e2.getKey());
        });

        ArrayList<Statement> out = new ArrayList<>(Math.min(k, cells.size()));
        for (int i = 0; i < cells.size() && out.size() < k; i++) {
            LtmCell cell = cells.get(i).getValue();
            if (cell == null || cell.stmt == null || cell.stmt.text == null || cell.stmt.text.isBlank()) continue;

            Statement s = cell.stmt.copyShallow();
            out.add(s);

            // touch is deterministic; uses iteration tick (not an LTM counter)
            cell.lastSeenTick = tick;
        }
        return out;
    }

    private double recallScore(Set<String> qTok, LtmCell cell) {
        if (cell == null) return 0.0;
        if (cell.tokens == null || cell.tokens.isEmpty()) return 0.0;
        int hit = 0;
        for (String t : cell.tokens) {
            if (qTok.contains(t)) hit++;
        }
        double ov = hit / (double) Math.max(1, qTok.size());
        return 0.70 * ov + 0.30 * clamp01(cell.score);
    }

    /**
     * Apply LTM decay exactly once per tick.
     * If ticks were skipped, applies decay for the whole delta deterministically.
     */
    private void ltmDecay(long tick) {
        if (!ltmEnabled || ltm.isEmpty()) return;
        if (ltmDecayPerTick <= 0.0) return;

        if (tick <= ltmLastDecayTick) return;

        long delta = tick - ltmLastDecayTick;
        ltmLastDecayTick = tick;

        double per = Math.max(0.0, 1.0 - ltmDecayPerTick);
        double mult = Math.pow(per, (double) delta);

        for (LtmCell c : ltm.values()) {
            if (c == null) continue;
            c.score = clamp01(c.score * mult);
        }
    }

    private void maybeWriteLtm(long tick, String q0, List<Statement> ctx, Candidate best) {
        if (!ltmEnabled) return;
        if (ltmCapacity <= 0) return;
        if (best == null || best.evaluation == null || !best.evaluation.valid) return;
        if (best.evaluation.groundedness < ltmWriteMinGroundedness) return;
        if (ctx == null || ctx.isEmpty()) return;

        Set<String> aTok = tokenSet(best.text);
        if (aTok.isEmpty()) return;

        ArrayList<ScoredStmt> scored = new ArrayList<>(ctx.size());
        for (Statement st : ctx) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            double ov = overlapTokenSets(aTok, tokenSet(st.text));
            if (ov <= 0.0) continue;
            scored.add(new ScoredStmt(st, ov));
        }
        if (scored.isEmpty()) return;

        scored.sort((a, b) -> {
            int c = Double.compare(b.score, a.score);
            if (c != 0) return c;
            return stableStmtKey(a.stmt).compareTo(stableStmtKey(b.stmt));
        });

        int writeN = Math.min(12, scored.size());

        for (int i = 0; i < writeN; i++) {
            ScoredStmt ss = scored.get(i);
            if (ss.score < 0.12) break;

            Statement st = ss.stmt;
            String key = stableStmtKey(st);
            Set<String> stTok = tokenSet(st.text);

            LtmCell prev = ltm.get(key);
            if (prev == null) {
                Statement copy = st.copyShallow();
                LtmCell cell = new LtmCell(copy, stTok, tick, clamp01(0.25 + ltmPromotionBoost + ss.score));
                ltm.put(key, cell);
            } else {
                prev.lastSeenTick = tick;
                prev.score = clamp01(prev.score + ltmPromotionBoost + 0.25 * ss.score);
            }
        }
    }

    private Set<String> tokenSet(String text) {
        if (text == null || text.isBlank()) return Set.of();
        HashSet<String> out = new HashSet<>(64);
        java.util.regex.Matcher m = knobs.WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (m.find()) {
            out.add(m.group());
            if (out.size() > 4096) break;
        }
        return out.isEmpty() ? Set.of() : out;
    }

    private static double overlapTokenSets(Set<String> a, Set<String> b) {
        if (a == null || b == null || a.isEmpty() || b.isEmpty()) return 0.0;
        int hit = 0;
        for (String t : a) if (b.contains(t)) hit++;
        return hit / (double) Math.max(1, a.size());
    }

    // --------- Utilities ---------

    private static long mix(long x, long y) {
        long z = x + 0x9E3779B97F4A7C15L + y;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private static double clamp01(double x) {
        if (x < 0) return 0;
        if (x > 1) return 1;
        return x;
    }

    private static String normalizeUserText(String s) {
        if (s == null) return "";
        String x = Normalizer.normalize(s, Normalizer.Form.NFKC).trim();
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

    // --------- Small structs ---------

    private static final class ScoredStmt {
        final Statement stmt;
        final double score;

        ScoredStmt(Statement stmt, double score) {
            this.stmt = stmt;
            this.score = score;
        }
    }

    static final class LtmCell {
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
}