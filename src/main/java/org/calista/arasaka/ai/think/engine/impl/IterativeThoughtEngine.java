package org.calista.arasaka.ai.think.engine.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.think.*;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.engine.ThoughtCycleEngine;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;

import java.text.Normalizer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Enterprise deterministic think-loop:
 * base retrieval -> (optional) context refinement -> draft -> evaluate -> self-correct -> (optional) LTM.
 *
 * Enterprise rules:
 *  - NEVER feed telemetry/critique/internal markers into retriever queries (prevents cache poisoning)
 *  - NO domain semantic hardcode (identity/smalltalk lives in corpora)
 *  - deterministic ordering and stable keys
 */
public final class IterativeThoughtEngine implements ThoughtCycleEngine {

    private static final Logger log = LogManager.getLogger(IterativeThoughtEngine.class);

    // tokenization for recall/overlap (>=3 keeps noise low; still configurable via code edits if needed)
    private static final java.util.regex.Pattern WORD = java.util.regex.Pattern.compile("[\\p{L}\\p{Nd}_]{3,}");

    // telemetry signatures + internal markers (generic, non-domain)
    private static final java.util.regex.Pattern TELEMETRY = java.util.regex.Pattern.compile(
            "(?i)" +
                    "(^\\s*(noctx;|err=|sec=)\\S*\\s*$)|" +
                    "(\\w+=\\S+;)|" +
                    "(\\bqc=)|(\\bg=)|(\\bnov=)|(\\brep=)|(\\bmd=)|" +
                    "(\\bno_context\\b)|(\\bnoctx\\b)|(\\bmore_evidence\\b)|(\\bfix\\b)|(\\bnotes\\b)"
    );

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

    // token cache for LTM statements (big speed-up for recallScore)
    private final ConcurrentMap<String, Set<String>> ltmTokensByKey = new ConcurrentHashMap<>();

    // refinement inside each iteration
    private final int refineRounds;
    private final int refineQueryBudget;

    // pipelines
    private final List<QueryProvider> queryProviders;
    private final List<Predicate<String>> queryFilters;
    private final List<StopRule> stopRules;

    public IterativeThoughtEngine(Retriever retriever,
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
                                  double ltmWriteMinGroundedness) {
        this(retriever, intentDetector, strategy, evaluator, generator,
                iterations, retrieveK, draftsPerIteration, patience, targetScore,
                ltmEnabled, ltmCapacity, ltmRecallK, ltmWriteMinGroundedness,
                1, 16);
    }

    public IterativeThoughtEngine(Retriever retriever,
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
                                  int refineQueryBudget) {

        this.retriever = Objects.requireNonNull(retriever, "retriever");
        this.intentDetector = Objects.requireNonNull(intentDetector, "intentDetector");
        this.strategy = Objects.requireNonNull(strategy, "strategy");
        this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
        this.generator = generator;

        this.iterations = Math.max(1, iterations);
        this.retrieveK = Math.max(1, retrieveK);
        this.draftsPerIteration = Math.max(1, draftsPerIteration);
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

        final String q0 = normalizeUserText(userText);
        final Intent intent = intentDetector.detect(q0);

        ThoughtState state = new ThoughtState();
        state.intent = intent;
        state.generator = generator;
        state.query = q0;

        // IMPORTANT: protect against null tags in ThoughtState implementation
        if (state.tags == null) state.tags = new HashMap<>(16);

        initResponseSchema(state, q0, intent);

        Candidate globalBest = new Candidate("", Double.NEGATIVE_INFINITY, "", null);
        state.bestSoFar = globalBest;
        state.bestEvaluation = null;

        List<String> trace = new ArrayList<>(Math.max(4, iterations * 2));
        int stagnation = 0;

        for (int iter = 1; iter <= iterations; iter++) {
            state.iteration = iter;
            state.seed = mix(seed, iter);

            // generator hint only (never used for retrieval)
            state.generationHint = buildGenerationHint(state);

            // ---- queries (no critique, no telemetry) ----
            List<String> baseQueries = buildQueries(q0, state);

            // ---- retrieve base ----
            Map<String, Statement> mergedByKey = new ConcurrentHashMap<>();
            retrieveAndMerge(baseQueries, mergedByKey, state.seed);

            // ---- refine (data-driven) ----
            for (int r = 0; r < refineRounds; r++) {
                List<Statement> ctxNow = toDeterministicContext(mergedByKey);
                List<String> derived = deriveQueriesFromContext(ctxNow, refineQueryBudget);
                if (derived.isEmpty()) break;
                retrieveAndMerge(derived, mergedByKey, mix(state.seed, 0x9E3779B97F4A7C15L + r));
            }

            // ---- LTM recall ----
            List<Statement> recalled = ltmEnabled ? recallFromLtm(baseQueries) : List.of();
            state.recalledMemory = recalled;
            mergeStatements(recalled, mergedByKey);

            // ---- deterministic context ----
            List<Statement> context = toDeterministicContext(mergedByKey);
            state.lastContext = context;

            // ---- drafts ----
            List<String> drafts = Drafts.sanitize(generateDrafts(q0, context, state), draftsPerIteration);

            // ---- evaluate ----
            List<Candidate> evaluated = drafts.parallelStream()
                    .map(text -> Candidate.fromEvaluation(text, evaluator.evaluate(q0, text, context)))
                    .toList();

            Candidate bestIter = pickBest(evaluated).orElse(globalBest);

            state.lastCandidate = bestIter;
            state.lastCritique = bestIter.critique; // internal only
            state.lastEvaluation = bestIter.evaluation;

            boolean improved = bestIter.score > globalBest.score + 1e-9;
            if (improved) {
                globalBest = bestIter;
                stagnation = 0;
            } else {
                stagnation++;
            }

            state.bestSoFar = globalBest;
            state.bestEvaluation = globalBest.evaluation;

            // ---- LTM writeback (evidence-only) ----
            if (ltmEnabled && bestIter.evaluation != null && bestIter.evaluation.valid) {
                maybeWritebackToLtm(bestIter, context);
            }

            trace.add("iter=" + iter
                    + " ctx=" + context.size()
                    + " ltm_recall=" + recalled.size()
                    + " drafts=" + drafts.size()
                    + " bestIter=" + fmt(bestIter.score)
                    + " global=" + fmt(globalBest.score));

            log.debug("iter {} | ctx={} ltm={} drafts={} bestIter={} global={}",
                    iter, context.size(), recalled.size(), drafts.size(),
                    fmt(bestIter.score), fmt(globalBest.score));

            StopContext sc = new StopContext(iter, stagnation, globalBest);
            if (stopRules.stream().anyMatch(r -> r.shouldStop(sc))) break;
        }

        return new ThoughtResult(globalBest.text, trace, intent, globalBest, globalBest.evaluation);
    }

    // -------------------- queries --------------------

    private List<String> buildQueries(String userText, ThoughtState state) {
        return queryProviders.stream()
                .flatMap(p -> p.provide(userText, state))
                .map(IterativeThoughtEngine::normalizeQuery)
                .filter(andAll(queryFilters))
                .distinct()
                .toList();
    }

    private List<String> deriveQueriesFromContext(List<Statement> ctx, int budget) {
        if (ctx == null || ctx.isEmpty()) return List.of();

        Stream<String> tags = ctx.stream()
                .filter(Objects::nonNull)
                .flatMap(s -> Stream.ofNullable(s.tags).flatMap(Collection::stream))
                .filter(Objects::nonNull);

        Stream<String> ids = ctx.stream()
                .filter(Objects::nonNull)
                .map(s -> s.id)
                .filter(Objects::nonNull);

        Stream<String> terms = ctx.stream()
                .filter(Objects::nonNull)
                .map(s -> s.text)
                .filter(Objects::nonNull)
                .map(t -> TermExtractor.uniqueTerms(t, 6, 5))
                .filter(t -> t != null && !t.isBlank());

        return Stream.concat(Stream.concat(tags, ids), terms)
                .map(IterativeThoughtEngine::normalizeQuery)
                .filter(andAll(queryFilters))
                .distinct()
                .limit(Math.max(1, budget))
                .toList();
    }

    private static Predicate<String> andAll(List<Predicate<String>> preds) {
        return preds.stream().reduce(x -> true, Predicate::and);
    }

    // -------------------- retrieval/merge --------------------

    private void retrieveAndMerge(List<String> queries, Map<String, Statement> mergedByKey, long seed) {
        if (queries == null || queries.isEmpty()) return;

        // parallel retrieval is fine; merge is concurrent; final context is deterministic via sorting by key
        queries.parallelStream().forEach(q -> mergeStatements(
                retriever.retrieve(q, retrieveK, mix(seed, q.hashCode())),
                mergedByKey
        ));
    }

    private static void mergeStatements(List<Statement> stmts, Map<String, Statement> mergedByKey) {
        if (stmts == null || stmts.isEmpty()) return;
        for (Statement s : stmts) {
            if (s == null) continue;
            if (!isValidStatement(s)) continue;
            String key = stableStmtKey(s);
            if (key != null) mergedByKey.putIfAbsent(key, s);
        }
    }

    private static List<Statement> toDeterministicContext(Map<String, Statement> mergedByKey) {
        if (mergedByKey == null || mergedByKey.isEmpty()) return List.of();
        return mergedByKey.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(Map.Entry::getValue)
                .toList();
    }

    private static boolean isValidStatement(Statement s) {
        try { s.validate(); return true; }
        catch (Exception ignored) { return false; }
    }

    // -------------------- drafts --------------------

    private List<String> generateDrafts(String userText, List<Statement> context, ThoughtState state) {
        if (generator != null) return generator.generateN(userText, context, state, draftsPerIteration);

        ArrayList<String> out = new ArrayList<>(draftsPerIteration);
        for (int d = 0; d < draftsPerIteration; d++) {
            state.draftIndex = d;
            out.add(strategy.generate(userText, context, state));
        }
        return out;
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

    // -------------------- candidate selection --------------------

    private static Optional<Candidate> pickBest(List<Candidate> evaluated) {
        if (evaluated == null || evaluated.isEmpty()) return Optional.empty();

        Optional<Candidate> bestValid = evaluated.stream()
                .filter(c -> c != null && c.evaluation != null && c.evaluation.valid)
                .max(Comparator.comparingDouble(c -> c.score));

        return bestValid.isPresent()
                ? bestValid
                : evaluated.stream().filter(Objects::nonNull).max(Comparator.comparingDouble(c -> c.score));
    }

    // -------------------- generator hint --------------------

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

        Optional.ofNullable(state.lastCritique)
                .filter(v -> !v.isBlank())
                .ifPresent(v -> sb.append(";last_notes=").append(sanitizeHint(v)));

        return sb.toString();
    }

    private static String sanitizeHint(String v) {
        // do NOT leak newlines / control chars to generator side; keep it compact
        String s = v.replace('\n', ' ').replace('\r', ' ').trim();
        // hard cap to avoid prompt bloat
        if (s.length() > 240) s = s.substring(0, 239) + "…";
        return s;
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

        // IMPORTANT: no intent-specific schema hardcode.
        // If you want to change formatting based on intent, do it data-driven:
        // put desired keys into state.tags BEFORE generation (e.g. via corpora/LTM/config).
        // This engine stays neutral: one pipeline, one schema contract.
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

    private double recallScore(String key, Statement st, Set<String> qTokens) {
        Set<String> sTok = ltmTokensByKey.computeIfAbsent(key, kk -> tokens(st == null ? "" : st.text));
        int ov = overlap(sTok, qTokens);
        double pri = ltmPriorityByKey.getOrDefault(key, 0.0);
        double w = st == null ? 1.0 : Math.max(0.1, st.weight);
        return (ov * w) + (pri * 0.10);
    }

    private void maybeWritebackToLtm(Candidate best, List<Statement> context) {
        CandidateEvaluator.Evaluation ev = best.evaluation;
        if (ev == null || !ev.valid) return;
        if (ev.groundedness < ltmWriteMinGroundedness) return;

        List<Statement> topEvidence = topSupportingEvidence(best.text, context, 8);
        for (Statement s : topEvidence) {
            if (s == null || !isValidStatement(s)) continue;
            String key = stableStmtKey(s);
            if (key == null) continue;

            ltmByKey.putIfAbsent(key, s);
            ltmPriorityByKey.merge(key, 1.0, Double::sum);
            ltmTokensByKey.putIfAbsent(key, tokens(s.text));
        }

        int overflow = (ltmCapacity > 0) ? (ltmByKey.size() - ltmCapacity) : 0;
        if (overflow > 0) evictLtm(overflow);
    }

    private void evictLtm(int need) {
        if (need <= 0) return;

        ArrayList<String> keys = new ArrayList<>(ltmByKey.keySet());
        keys.sort((a, b) -> {
            double pa = ltmPriorityByKey.getOrDefault(a, 0.0);
            double pb = ltmPriorityByKey.getOrDefault(b, 0.0);
            int cmp = Double.compare(pa, pb);
            return (cmp != 0) ? cmp : a.compareTo(b);
        });

        for (int i = 0; i < Math.min(need, keys.size()); i++) {
            String key = keys.get(i);
            ltmByKey.remove(key);
            ltmPriorityByKey.remove(key);
            ltmTokensByKey.remove(key);
        }
    }

    private List<Statement> topSupportingEvidence(String answer, List<Statement> ctx, int k) {
        if (ctx == null || ctx.isEmpty()) return List.of();
        Set<String> aTok = tokens(answer);

        return ctx.stream()
                .filter(Objects::nonNull)
                .sorted(Comparator
                        .comparingInt((Statement s) -> overlap(tokens(s.text), aTok)).reversed()
                        .thenComparing(s -> normalizeQuery(s.text)))
                .limit(Math.max(0, k))
                .toList();
    }

    // -------------------- utilities --------------------

    private static String normalizeUserText(String s) {
        if (s == null) return "";
        String x = s.trim();
        if (x.isEmpty()) return "";
        // stable unicode normalization + collapse whitespace
        x = Normalizer.normalize(x, Normalizer.Form.NFKC);
        x = x.replace('\u00A0', ' ');
        x = x.replaceAll("\\s+", " ").trim();
        return x;
    }

    private static String normalizeQuery(String s) {
        if (s == null) return "";
        String x = s.trim();
        if (x.isEmpty()) return "";
        x = Normalizer.normalize(x, Normalizer.Form.NFKC);
        x = x.replace('\u00A0', ' ');
        x = x.replaceAll("\\s+", " ").trim();
        return x;
    }

    // SplitMix64 — быстрый, стабильный, хороший для сидов/подсидов
    private static long mix(long seed, long salt) {
        long z = seed + 0x9E3779B97F4A7C15L + salt;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private static double clamp01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        return Math.min(1.0, Math.max(0.0, v));
    }

    private static boolean isMostlyCyrillic(String s) {
        if (s == null || s.isBlank()) return false;
        int cyr = 0, other = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (Character.isWhitespace(ch)) continue;
            Character.UnicodeBlock b = Character.UnicodeBlock.of(ch);
            if (b == Character.UnicodeBlock.CYRILLIC
                    || b == Character.UnicodeBlock.CYRILLIC_SUPPLEMENTARY
                    || b == Character.UnicodeBlock.CYRILLIC_EXTENDED_A
                    || b == Character.UnicodeBlock.CYRILLIC_EXTENDED_B) cyr++;
            else other++;
        }
        return cyr > other;
    }

    private static Set<String> tokens(String s) {
        if (s == null || s.isBlank()) return Set.of();
        return WORD.matcher(s.toLowerCase(Locale.ROOT))
                .results()
                .map(r -> r.group())
                .limit(256)
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }

    private static int overlap(Set<String> a, Set<String> b) {
        if (a.isEmpty() || b.isEmpty()) return 0;
        int c = 0;
        for (String t : a) if (b.contains(t)) c++;
        return c;
    }

    private static String stableStmtKey(Statement s) {
        if (s == null) return null;
        if (s.id != null && !s.id.isBlank()) return "id:" + s.id.trim();
        if (s.text == null || s.text.isBlank()) return null;

        // normalized text key reduces accidental duplicates from whitespace variants
        String tx = normalizeQuery(s.text);
        return tx.isBlank() ? null : "tx:" + tx;
    }

    // -------------------- enterprise primitives --------------------

    private interface QueryProvider {
        Stream<String> provide(String userText, ThoughtState state);

        static QueryProvider userText() {
            return (u, s) -> Stream.ofNullable(u);
        }

        static QueryProvider intentName() {
            return (u, s) -> Stream.ofNullable(s)
                    .map(st -> st.intent)
                    .map(i -> i == null ? Intent.UNKNOWN : i)
                    .map(Intent::name);
        }

        /**
         * IMPORTANT: bestTerms must be hygienic to avoid pulling "no_context"/"fix=..." from bestSoFar text.
         */
        static QueryProvider bestTerms(int maxTerms, int minLen) {
            int mt = Math.max(1, maxTerms);
            int ml = Math.max(2, minLen);
            return (u, s) -> Stream.ofNullable(s)
                    .map(st -> st.bestSoFar)
                    .map(c -> c == null ? "" : c.text)
                    .map(IterativeThoughtEngine::normalizeQuery)
                    .filter(v -> !v.isEmpty())
                    .map(v -> TermExtractor.uniqueTerms(v, mt, ml))
                    .filter(v -> v != null && !v.isBlank())
                    .flatMap(v -> Stream.of(v));
        }
    }

    private interface StopRule {
        boolean shouldStop(StopContext ctx);

        static StopRule targetScore(double target) {
            return ctx -> ctx != null && ctx.best != null && ctx.best.score >= target;
        }

        static StopRule patience(int patience) {
            int p = Math.max(0, patience);
            return ctx -> ctx != null && p > 0 && ctx.stagnation >= p;
        }
    }

    private record StopContext(int iteration, int stagnation, Candidate best) {}

    private static final class TermExtractor {
        static String uniqueTerms(String text, int maxTerms, int minLen) {
            if (text == null || text.isBlank()) return "";
            LinkedHashSet<String> uniq = new LinkedHashSet<>();

            // split by non-letter/digit/_ (same spirit as WORD, но быстрее чем regex matcher на каждый токен)
            for (String raw : text.split("[^\\p{L}\\p{Nd}_]+")) {
                if (raw == null) continue;
                String t = raw.trim().toLowerCase(Locale.ROOT);
                if (t.length() < minLen) continue;
                if (TELEMETRY.matcher(t).find()) continue; // blocks no_context/noctx/more_evidence/fix/notes etc.
                uniq.add(t);
                if (uniq.size() >= maxTerms) break;
            }

            return String.join(" ", uniq);
        }
    }
}