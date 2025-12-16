package org.calista.arasaka.ai.think.engine.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.think.ResponseStrategy;
import org.calista.arasaka.ai.think.TextGenerator;
import org.calista.arasaka.ai.think.ThoughtResult;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.engine.ThoughtCycleEngine;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Enterprise-grade deterministic "think" loop:
 * query expansion -> retrieval (incl. LTM recall) -> draft -> evaluate -> self-correct -> (optional) writeback to LTM.
 *
 * No randomness: only injected components (Retriever/Generator) may vary, but must be deterministic w.r.t. (query, seed).
 */
public final class IterativeThoughtEngine implements ThoughtCycleEngine {

    private static final Logger log = LogManager.getLogger(IterativeThoughtEngine.class);
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{Nd}_]{3,}");

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

    // ---- long-term memory (LTM) ----
    private final boolean ltmEnabled;
    private final int ltmCapacity;
    private final int ltmRecallK;
    private final double ltmWriteMinGroundedness;

    /**
     * LTM is a plain evidence store.
     * Keys MUST be non-null and stable -> use stableStmtKey(stmt).
     */
    private final ConcurrentMap<String, Statement> ltmByKey = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, Double> ltmPriorityByKey = new ConcurrentHashMap<>();

    public IterativeThoughtEngine(Retriever retriever,
                                  IntentDetector intentDetector,
                                  ResponseStrategy strategy,
                                  CandidateEvaluator evaluator,
                                  TextGenerator generator,
                                  int iterations,
                                  int retrieveK,
                                  int draftsPerIteration,
                                  int patience,
                                  double targetScore) {
        this(retriever, intentDetector, strategy, evaluator, generator,
                iterations, retrieveK, draftsPerIteration, patience, targetScore,
                true, 50_000, 64, 0.55);
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
                                  double ltmWriteMinGroundedness) {
        this.retriever = Objects.requireNonNull(retriever);
        this.intentDetector = Objects.requireNonNull(intentDetector);
        this.strategy = Objects.requireNonNull(strategy);
        this.evaluator = Objects.requireNonNull(evaluator);
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
    }

    @Override
    public ThoughtResult think(String userText, long seed) {

        final String q0 = userText == null ? "" : userText.trim();
        Intent intent = intentDetector.detect(q0);

        ThoughtState state = new ThoughtState();
        state.intent = intent;
        state.generator = generator;
        state.query = q0;

        // Default response contract: configurable via state.tags; no scripted messages.
        initResponseSchema(state, q0, intent);

        Candidate globalBest = new Candidate("", Double.NEGATIVE_INFINITY, "", null);
        state.bestSoFar = globalBest;
        state.bestEvaluation = null;

        List<String> trace = new ArrayList<>(iterations * 2);
        int stagnation = 0;

        for (int iter = 1; iter <= iterations; iter++) {
            state.iteration = iter;
            state.seed = mix(seed, iter);

            // Deterministic self-correction hint (telemetry-like, non-domain).
            state.generationHint = buildGenerationHint(state);

            // 1) Build multiple semantic queries (deterministic)
            List<String> queries = buildQueries(q0, state);

            // 2) Retrieval pipeline (parallel)
            Map<String, Statement> mergedByKey = new ConcurrentHashMap<>();

            queries.parallelStream().forEach(q -> {
                List<Statement> stmts = retriever.retrieve(q, retrieveK, mix(state.seed, q.hashCode()));
                for (Statement s : stmts) {
                    if (s == null) continue;
                    try {
                        s.validate();
                    } catch (Exception ignored) {
                        continue;
                    }
                    String key = stableStmtKey(s);
                    if (key != null) mergedByKey.putIfAbsent(key, s);
                }
            });

            // LTM recall
            List<Statement> recalled = ltmEnabled ? recallFromLtm(queries) : List.of();
            state.recalledMemory = recalled;

            for (Statement s : recalled) {
                if (s == null) continue;
                try {
                    s.validate();
                } catch (Exception ignored) {
                    continue;
                }
                String key = stableStmtKey(s);
                if (key != null) mergedByKey.putIfAbsent(key, s);
            }

            // Deterministic order: never rely on ConcurrentHashMap.values() encounter order.
            List<Statement> context = mergedByKey.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .map(Map.Entry::getValue)
                    .toList();
            state.lastContext = context;

            // 3) Draft generation (N-best)
            List<String> drafts = generateDrafts(q0, context, state);

            // 4) Parallel evaluation
            List<Candidate> evaluated = drafts.parallelStream()
                    .map(text -> Candidate.fromEvaluation(text, evaluator.evaluate(q0, text, context)))
                    .collect(Collectors.toList());

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

            // 5) LTM writeback (evidence-only, stable keys)
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

            // 6) Early stop
            if (globalBest.score >= targetScore) break;
            if (stagnation >= patience) break;
        }

        CandidateEvaluator.Evaluation ev = globalBest == null ? null : globalBest.evaluation;
        return new ThoughtResult(globalBest == null ? "" : globalBest.text, trace, intent, globalBest, ev);
    }

    // -------------------- generation --------------------

    private List<String> generateDrafts(String userText, List<Statement> context, ThoughtState state) {
        if (generator != null) {
            return generator.generateN(userText, context, state, draftsPerIteration);
        }
        ArrayList<String> drafts = new ArrayList<>(draftsPerIteration);
        for (int d = 0; d < draftsPerIteration; d++) {
            state.draftIndex = d;
            drafts.add(strategy.generate(userText, context, state));
        }
        return drafts.isEmpty() ? List.of("") : drafts;
    }

    // -------------------- LTM recall/writeback --------------------

    private List<Statement> recallFromLtm(List<String> queries) {
        if (ltmByKey.isEmpty() || ltmRecallK <= 0) return List.of();

        Set<String> qTokens = new LinkedHashSet<>();
        for (String q : queries) qTokens.addAll(tokens(q));

        ArrayList<Map.Entry<String, Statement>> all = new ArrayList<>(ltmByKey.entrySet());
        all.sort((a, b) -> {
            double sa = recallScore(a.getKey(), a.getValue(), qTokens);
            double sb = recallScore(b.getKey(), b.getValue(), qTokens);
            int cmp = Double.compare(sb, sa);
            if (cmp != 0) return cmp;
            return a.getKey().compareTo(b.getKey());
        });

        int k = Math.min(ltmRecallK, all.size());
        ArrayList<Statement> out = new ArrayList<>(k);
        for (int i = 0; i < k; i++) out.add(all.get(i).getValue());
        return out;
    }

    private double recallScore(String key, Statement st, Set<String> qTokens) {
        Set<String> sTok = tokens(st == null ? "" : st.text);
        int ov = overlap(sTok, qTokens);
        double pri = ltmPriorityByKey.getOrDefault(key, 0.0);
        double w = st == null ? 1.0 : Math.max(0.1, st.weight);
        return (ov * w) + (pri * 0.10);
    }

    private void maybeWritebackToLtm(Candidate best, List<Statement> context) {
        CandidateEvaluator.Evaluation ev = best.evaluation;
        if (ev == null) return;
        if (!ev.valid) return;
        if (ev.groundedness < ltmWriteMinGroundedness) return;

        List<Statement> topEvidence = topSupportingEvidence(best.text, context, 8);
        for (Statement s : topEvidence) {
            if (s == null) continue;
            try {
                s.validate();
            } catch (Exception ignored) {
                continue;
            }

            String key = stableStmtKey(s);
            if (key == null) continue;

            ltmByKey.putIfAbsent(key, s);
            ltmPriorityByKey.merge(key, 1.0, Double::sum);
        }

        if (ltmCapacity > 0 && ltmByKey.size() > ltmCapacity) {
            evictLtm(ltmByKey.size() - ltmCapacity);
        }
    }

    private void evictLtm(int need) {
        if (need <= 0) return;

        ArrayList<String> keys = new ArrayList<>(ltmByKey.keySet());
        keys.sort((a, b) -> {
            double pa = ltmPriorityByKey.getOrDefault(a, 0.0);
            double pb = ltmPriorityByKey.getOrDefault(b, 0.0);
            int cmp = Double.compare(pa, pb);
            if (cmp != 0) return cmp;
            return a.compareTo(b);
        });

        int removed = 0;
        for (String key : keys) {
            if (removed >= need) break;
            ltmByKey.remove(key);
            ltmPriorityByKey.remove(key);
            removed++;
        }
    }

    private List<Statement> topSupportingEvidence(String answer, List<Statement> ctx, int k) {
        if (ctx == null || ctx.isEmpty()) return List.of();
        final Set<String> aTok = tokens(answer);

        return ctx.stream()
                .filter(Objects::nonNull)
                .sorted(Comparator.comparingInt((Statement s) -> overlap(tokens(s.text), aTok)).reversed()
                        .thenComparing(s -> String.valueOf(s.text)))
                .limit(Math.max(0, k))
                .toList();
    }

    // -------------------- query building --------------------

    private static List<String> buildQueries(String userText, ThoughtState state) {
        Set<String> queries = new LinkedHashSet<>();
        queries.add(userText == null ? "" : userText);

        if (state.bestSoFar != null && state.bestSoFar.text != null && !state.bestSoFar.text.isBlank()) {
            queries.add(extractTerms(state.bestSoFar.text));
        }
        if (state.lastCritique != null && !state.lastCritique.isBlank()) {
            queries.add(state.lastCritique);
        }
        queries.add(state.intent == null ? Intent.UNKNOWN.name() : state.intent.name());

        return new ArrayList<>(queries);
    }

    // -------------------- selection helpers --------------------

    private static Optional<Candidate> pickBest(List<Candidate> evaluated) {
        if (evaluated == null || evaluated.isEmpty()) return Optional.empty();

        Optional<Candidate> bestValid = evaluated.stream()
                .filter(c -> c != null && c.evaluation != null && c.evaluation.valid)
                .max(Comparator.comparingDouble(c -> c.score));

        if (bestValid.isPresent()) return bestValid;
        return evaluated.stream().filter(Objects::nonNull).max(Comparator.comparingDouble(c -> c.score));
    }

    private static String buildGenerationHint(ThoughtState state) {
        StringBuilder sb = new StringBuilder(256);
        String sections = (state == null || state.tags == null) ? null : state.tags.get("response.sections");
        String style = (state == null || state.tags == null) ? null : state.tags.get("response.style");
        if (sections == null || sections.isBlank()) sections = "summary,evidence,actions";
        if (style == null || style.isBlank()) style = "md";

        // 'format' is parsed by TextGenerator.Hint (keep key name stable).
        // 'sections' is forward-compatible for ML generators.
        sb.append("format=").append(style);
        sb.append(";sections=").append(sections);
        sb.append(";intent=").append(state.intent == null ? Intent.UNKNOWN.name() : state.intent.name());
        sb.append(";iter=").append(state.iteration);

        CandidateEvaluator.Evaluation last = state.lastEvaluation;
        if (last != null) {
            sb.append(";last_g=").append(fmt(last.groundedness));
            sb.append(";last_r=").append(fmt(last.contradictionRisk));
            sb.append(";last_v=").append(last.valid ? 1 : 0);
        }
        CandidateEvaluator.Evaluation best = state.bestEvaluation;
        if (best != null) {
            sb.append(";best_g=").append(fmt(best.groundedness));
            sb.append(";best_r=").append(fmt(best.contradictionRisk));
            sb.append(";best_v=").append(best.valid ? 1 : 0);
        }
        if (state.lastCritique != null && !state.lastCritique.isBlank()) {
            sb.append(";last_notes=").append(state.lastCritique.replace('\n', ' '));
        }
        return sb.toString();
    }

    private static void initResponseSchema(ThoughtState state, String userText, Intent intent) {
        if (state == null) return;
        if (state.tags == null) return;

        boolean ru = isMostlyCyrillic(userText);
        // Defaults (caller can override).
        state.tags.putIfAbsent("response.style", "md");
        state.tags.putIfAbsent("response.sections", "summary,evidence,actions");
        state.tags.putIfAbsent("response.label.summary", ru ? "Вывод" : "Conclusion");
        state.tags.putIfAbsent("response.label.evidence", ru ? "Опора на контекст" : "Evidence from context");
        state.tags.putIfAbsent("response.label.actions", ru ? "Следующие шаги" : "Next steps");

        // Intent-conditioned contraction, still deterministic and non-domain.
        if (intent == Intent.GREETING) {
            state.tags.put("response.style", "plain");
            state.tags.put("response.sections", "summary,actions");
        }
    }

    private static boolean isMostlyCyrillic(String s) {
        if (s == null || s.isBlank()) return false;
        int cyr = 0, other = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (Character.isWhitespace(ch)) continue;
            if (ch >= 'А' && ch <= 'я') cyr++;
            else other++;
        }
        return cyr > other;
    }

    private static String extractTerms(String text) {
        return Arrays.stream(text.split("\\W+"))
                .filter(t -> t.length() > 4)
                .limit(8)
                .collect(Collectors.joining(" "));
    }

    // -------------------- deterministic utils --------------------

    private static long mix(long a, long b) {
        long x = a ^ b;
        x ^= (x >>> 33);
        x *= 0xff51afd7ed558ccdL;
        x ^= (x >>> 33);
        return x;
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private static double clamp01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
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

    /**
     * Stable non-null key:
     * - id:<id> if present
     * - tx:<trim(text)> otherwise
     */
    private static String stableStmtKey(Statement s) {
        if (s == null) return null;
        String id = s.id;
        if (id != null && !id.isBlank()) return "id:" + id;
        String t = s.text;
        if (t == null || t.isBlank()) return null;
        return "tx:" + t.trim();
    }
}