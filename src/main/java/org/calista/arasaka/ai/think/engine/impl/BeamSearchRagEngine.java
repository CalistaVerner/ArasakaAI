package org.calista.arasaka.ai.think.engine.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.think.ResponseStrategy;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;
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
 * NeuralQuantumEngine
 *
 * Deterministic "quantum" search loop:
 * - keeps a beam ("frontier") of best candidates
 * - per iteration: build semantic queries -> retrieve (RAG + optional LTM) -> generate drafts per beam -> evaluate
 * - applies diversity penalty to avoid mode collapse
 * - optional verify pass (extra self-correction step) without hard-coded phrases
 *
 * NOTE:
 * - No randomness here. The only varying parts must be deterministic w.r.t. (userText, seed, state).
 * - Retrieval queries are sanitized to prevent telemetry/policy tokens from polluting the retriever.
 */
public final class BeamSearchRagEngine implements ThoughtCycleEngine {

    private static final Logger log = LogManager.getLogger(BeamSearchRagEngine.class);

    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{Nd}_]{3,}");

    // telemetry signatures + internal markers (generic, non-domain)
    private static final Pattern TELEMETRY = Pattern.compile(
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

    // beam search knobs
    private final int beamWidth;
    private final int draftsPerBeam;
    private final int maxDraftsPerIter;

    // early stop
    private final int patience;
    private final double targetScore;

    // diversity
    private final double diversityPenalty;     // penalty weight for high similarity
    private final double minDiversityJaccard;  // if similarity >= this, penalize

    // verify pass
    private final boolean verifyPassEnabled;

    // ---- long-term memory (LTM) ----
    private final boolean ltmEnabled;
    private final int ltmCapacity;
    private final int ltmRecallK;
    private final double ltmWriteMinGroundedness;

    private final ConcurrentMap<String, Statement> ltmByKey = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, Double> ltmPriorityByKey = new ConcurrentHashMap<>();

    public BeamSearchRagEngine(Retriever retriever,
                               IntentDetector intentDetector,
                               ResponseStrategy strategy,
                               CandidateEvaluator evaluator,
                               TextGenerator generator,
                               int iterations,
                               int retrieveK,
                               int beamWidth,
                               int draftsPerBeam,
                               int maxDraftsPerIter,
                               int patience,
                               double targetScore,
                               double diversityPenalty,
                               double minDiversityJaccard,
                               boolean verifyPassEnabled,
                               boolean ltmEnabled,
                               int ltmCapacity,
                               int ltmRecallK,
                               double ltmWriteMinGroundedness) {

        this.retriever = Objects.requireNonNull(retriever, "retriever");
        this.intentDetector = Objects.requireNonNull(intentDetector, "intentDetector");
        this.strategy = Objects.requireNonNull(strategy, "strategy");
        this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
        this.generator = generator;

        this.iterations = clampInt(iterations, 1, 16);
        this.retrieveK = clampInt(retrieveK, 1, 512);

        this.beamWidth = clampInt(beamWidth, 1, 64);
        this.draftsPerBeam = clampInt(draftsPerBeam, 1, 64);
        this.maxDraftsPerIter = clampInt(maxDraftsPerIter, 1, 1024);

        this.patience = clampInt(patience, 0, 32);
        this.targetScore = targetScore;

        this.diversityPenalty = clamp01(diversityPenalty);
        this.minDiversityJaccard = clamp01(minDiversityJaccard);

        this.verifyPassEnabled = verifyPassEnabled;

        this.ltmEnabled = ltmEnabled;
        this.ltmCapacity = Math.max(0, ltmCapacity);
        this.ltmRecallK = Math.max(0, ltmRecallK);
        this.ltmWriteMinGroundedness = clamp01(ltmWriteMinGroundedness);
    }

    @Override
    public ThoughtResult think(String userText, long seed) {
        final String q0 = userText == null ? "" : userText.trim();
        final Intent intent = intentDetector.detect(q0);

        ThoughtState state = new ThoughtState();
        state.intent = intent;
        state.generator = generator;
        state.query = q0;

        initResponseSchema(state, q0, intent);

        Candidate globalBest = new Candidate("", Double.NEGATIVE_INFINITY, "", null);
        state.bestSoFar = globalBest;
        state.bestEvaluation = null;

        List<Candidate> beam = new ArrayList<>(beamWidth);
        List<String> trace = new ArrayList<>(iterations * 2);

        int stagnation = 0;

        for (int iter = 1; iter <= iterations; iter++) {
            state.iteration = iter;
            state.seed = mix(seed, iter);

            // Deterministic generation hint (parseable; non-domain)
            state.generationHint = buildGenerationHint(state);

            // 1) Queries (sanitized)
            List<String> queries = buildQueries(q0, state, beam);

            // 2) Retrieval (RAG + optional LTM)
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

            // Deterministic context ordering (ConcurrentHashMap values order is undefined)
            List<Statement> context = mergedByKey.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .map(Map.Entry::getValue)
                    .toList();
            state.lastContext = context;

            // 3) Drafts per beam
            List<String> drafts = generateDraftsForBeam(q0, context, state, beam);

            // hard cap to avoid runaway compute
            if (drafts.size() > maxDraftsPerIter) {
                drafts = drafts.subList(0, maxDraftsPerIter);
            }

            // 4) Evaluate all drafts (parallel), apply diversity penalty against current beam
            List<Candidate> finalBeam = beam;
            final List<Candidate> evaluated = drafts.parallelStream()
                    .map(txt -> Candidate.fromEvaluation(txt, evaluator.evaluate(q0, txt, context)))
                    .map(c -> applyDiversityPenalty(c, finalBeam)).collect(Collectors.toList());

            // 5) Build new beam (best diverse candidates)
            List<Candidate> newBeam = selectBeam(evaluated, beamWidth, minDiversityJaccard);

            Candidate bestIter = newBeam.isEmpty()
                    ? (pickBest(evaluated).orElse(globalBest))
                    : newBeam.get(0);

            // optional verify pass: one extra self-correction step if the best is shaky
            if (verifyPassEnabled && shouldVerify(bestIter)) {
                Candidate verified = verifyCandidate(q0, context, state, bestIter, newBeam);
                if (verified != null && verified.score > bestIter.score + 1e-9) {
                    bestIter = verified;
                    // also keep it in beam deterministically
                    newBeam = mergeIntoBeam(newBeam, verified, beamWidth, minDiversityJaccard);
                }
            }

            // update state telemetry
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

            // writeback to LTM (evidence-only)
            if (ltmEnabled && bestIter.evaluation != null && bestIter.evaluation.valid) {
                maybeWritebackToLtm(bestIter, context);
            }

            // replace beam
            beam = newBeam.isEmpty() ? beam : newBeam;

            trace.add("iter=" + iter
                    + " ctx=" + context.size()
                    + " ltm_recall=" + recalled.size()
                    + " drafts=" + drafts.size()
                    + " beam=" + beam.size()
                    + " bestIter=" + fmt(bestIter.score)
                    + " global=" + fmt(globalBest.score));

            log.debug("NeuralQuantum iter {} | ctx={} ltm={} drafts={} beam={} bestIter={} global={}",
                    iter, context.size(), recalled.size(), drafts.size(), beam.size(),
                    fmt(bestIter.score), fmt(globalBest.score));

            // early stop
            if (globalBest.score >= targetScore) break;
            if (stagnation >= patience) break;
        }

        CandidateEvaluator.Evaluation ev = (globalBest == null) ? null : globalBest.evaluation;
        return new ThoughtResult(globalBest == null ? "" : globalBest.text, trace, intent, globalBest, ev);
    }

    // -------------------- generation --------------------

    private List<String> generateDraftsForBeam(String userText, List<Statement> context, ThoughtState state, List<Candidate> beam) {
        // If beam is empty, behave like an iterative engine "seed" iteration.
        if (beam == null || beam.isEmpty()) {
            return generateN(userText, context, state, Math.min(maxDraftsPerIter, Math.max(1, draftsPerBeam)));
        }

        ArrayList<String> drafts = new ArrayList<>(Math.min(maxDraftsPerIter, beam.size() * draftsPerBeam));

        // Deterministic order: sort beam by score DESC then text
        List<Candidate> ordered = new ArrayList<>(beam);
        ordered.sort(Comparator.comparingDouble((Candidate c) -> c == null ? Double.NEGATIVE_INFINITY : c.score)
                .reversed()
                .thenComparing(c -> c == null ? "" : c.text));

        int produced = 0;

        for (int b = 0; b < ordered.size(); b++) {
            Candidate base = ordered.get(b);
            if (base == null) continue;

            // anchor bestSoFar for generator/strategy
            state.bestSoFar = base;
            state.bestEvaluation = base.evaluation;

            for (int d = 0; d < draftsPerBeam; d++) {
                if (produced >= maxDraftsPerIter) break;
                state.draftIndex = (b * 997) + d; // deterministic mixing
                String s;
                if (generator != null) {
                    String prompt = generator.prepareUserText(userText, context, state);
                    s = generator.generate(prompt, context, state);
                } else {
                    s = strategy.generate(userText, context, state);
                }
                if (s != null && !s.isBlank()) {
                    drafts.add(s);
                    produced++;
                }
            }
            if (produced >= maxDraftsPerIter) break;
        }

        if (drafts.isEmpty()) drafts.add("");
        return drafts;
    }

    private List<String> generateN(String userText, List<Statement> context, ThoughtState state, int n) {
        if (generator != null) {
            return generator.generateN(userText, context, state, n);
        }
        ArrayList<String> out = new ArrayList<>(Math.max(1, n));
        int need = Math.max(1, n);
        for (int i = 0; i < need; i++) {
            state.draftIndex = i;
            out.add(strategy.generate(userText, context, state));
        }
        if (out.isEmpty()) out.add("");
        return out;
    }

    // -------------------- verify pass --------------------

    private boolean shouldVerify(Candidate c) {
        if (c == null || c.evaluation == null) return false;
        CandidateEvaluator.Evaluation e = c.evaluation;

        // No hard-coded phrases: only generic heuristics based on telemetry.
        if (!e.valid) return true;
        if (e.contradictionRisk > 0.65) return true;
        if (e.groundedness > 0 && e.groundedness < 0.30) return true;
        if (e.structureScore > 0 && e.structureScore < 0.25) return true;
        return false;
    }

    private Candidate verifyCandidate(String userText,
                                      List<Statement> context,
                                      ThoughtState state,
                                      Candidate currentBest,
                                      List<Candidate> beam) {

        // push a deterministic critique token (machine-readable), not a human phrase
        String prevCrit = state.lastCritique == null ? "" : state.lastCritique;
        state.lastCritique = (prevCrit.isBlank() ? "verify_pass" : (prevCrit + " verify_pass"));

        // anchor to current best
        state.bestSoFar = currentBest;
        state.bestEvaluation = currentBest == null ? null : currentBest.evaluation;

        // generate a small set of verify drafts
        int n = Math.min(3, Math.max(1, draftsPerBeam));
        List<String> drafts = generateN(userText, context, state, n);

        Candidate best = null;
        for (String s : drafts) {
            Candidate cand = Candidate.fromEvaluation(s, evaluator.evaluate(userText, s, context));
            cand = applyDiversityPenalty(cand, beam);
            if (best == null || cand.score > best.score + 1e-9) best = cand;
        }

        // restore critique
        state.lastCritique = prevCrit;
        return best;
    }

    // -------------------- diversity --------------------

    private Candidate applyDiversityPenalty(Candidate c, List<Candidate> beam) {
        if (c == null) return null;
        if (beam == null || beam.isEmpty()) return c;
        if (diversityPenalty <= 1e-9) return c;

        Set<String> ct = tokens(c.text);
        if (ct.isEmpty()) return c;

        double maxSim = 0.0;
        for (Candidate b : beam) {
            if (b == null) continue;
            Set<String> bt = tokens(b.text);
            if (bt.isEmpty()) continue;
            double j = jaccard(ct, bt);
            if (j > maxSim) maxSim = j;
        }

        double penalty = 0.0;
        if (maxSim >= minDiversityJaccard) {
            // penalize only if very similar (prevents over-penalizing reasonable alignment)
            double x = (maxSim - minDiversityJaccard) / Math.max(1e-9, (1.0 - minDiversityJaccard));
            penalty = clamp01(x) * diversityPenalty;
        }

        if (penalty <= 1e-12) return c;

        // keep critique/evaluation intact; only adjust ordering score deterministically
        return new Candidate(c.text, c.score - penalty, c.critique, c.evaluation);
    }

    private static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() || b.isEmpty()) return 0.0;
        int inter = 0;
        for (String t : a) if (b.contains(t)) inter++;
        int union = a.size() + b.size() - inter;
        if (union <= 0) return 0.0;
        return (double) inter / (double) union;
    }

    private static List<Candidate> selectBeam(List<Candidate> evaluated, int beamWidth, double minDivJ) {
        if (evaluated == null || evaluated.isEmpty()) return List.of();

        // deterministic sort by score DESC then text
        List<Candidate> sorted = evaluated.stream()
                .filter(Objects::nonNull)
                .sorted(Comparator
                        .comparingDouble((Candidate c) -> c.score).reversed()
                        .thenComparing(c -> c.text == null ? "" : c.text))
                .toList();

        ArrayList<Candidate> out = new ArrayList<>(Math.min(beamWidth, sorted.size()));

        for (Candidate c : sorted) {
            if (out.size() >= beamWidth) break;

            // diversity filter among selected
            boolean ok = true;
            Set<String> ct = tokens(c.text);
            for (Candidate p : out) {
                double j = jaccard(ct, tokens(p.text));
                if (j >= minDivJ) { ok = false; break; }
            }
            if (ok) out.add(c);
        }

        // If diversity filter was too strict and we got nothing, fall back to top-1.
        if (out.isEmpty() && !sorted.isEmpty()) out.add(sorted.get(0));

        return out;
    }

    private static List<Candidate> mergeIntoBeam(List<Candidate> beam, Candidate cand, int beamWidth, double minDivJ) {
        if (cand == null) return beam == null ? List.of() : beam;
        ArrayList<Candidate> all = new ArrayList<>();
        if (beam != null) all.addAll(beam);
        all.add(cand);
        return selectBeam(all, beamWidth, minDivJ);
    }

    // -------------------- query building (SANITIZED) --------------------

    private static List<String> buildQueries(String userText, ThoughtState state, List<Candidate> frontier) {
        final String q0 = userText == null ? "" : userText.trim();

        LinkedHashSet<String> out = new LinkedHashSet<>();
        if (!q0.isBlank()) out.add(q0);

        if (state != null && state.intent != null) out.add(state.intent.name());

        // From bestSoFar: terms only
        if (state != null && state.bestSoFar != null && state.bestSoFar.text != null && !state.bestSoFar.text.isBlank()) {
            String t = sanitizeRetrieveQuery(extractTerms(state.bestSoFar.text));
            if (!t.isBlank()) out.add(t);
        }

        // IMPORTANT: do NOT feed self-critique/telemetry back into retrieval.
        // Retrieval hygiene: only user text + intent + (sanitized) best text.

        // From frontier: terms only (best candidates text), NEVER critique.
        if (frontier != null && !frontier.isEmpty()) {
            frontier.stream()
                    .filter(Objects::nonNull)
                    .sorted(Comparator.comparingDouble((Candidate c) -> c.score).reversed()
                            .thenComparing(c -> c.text == null ? "" : c.text))
                    .limit(3)
                    .forEach(c -> {
                        if (c.text != null && !c.text.isBlank()) {
                            String t = sanitizeRetrieveQuery(extractTerms(c.text));
                            if (!t.isBlank()) out.add(t);
                        }
                    });
        }

        // Hard cap: never spam retriever with too many queries
        if (out.size() > 6) {
            ArrayList<String> cut = new ArrayList<>(6);
            int i = 0;
            for (String s : out) {
                cut.add(s);
                if (++i >= 6) break;
            }
            return cut;
        }

        return new ArrayList<>(out);
    }

    private static String sanitizeRetrieveQuery(String q) {
        if (q == null) return "";
        String s = q.trim().toLowerCase(Locale.ROOT);

        // strip telemetry-ish separators
        s = s.replace(';', ' ')
                .replace('=', ' ')
                .replace(':', ' ')
                .replace('|', ' ')
                .replace('[', ' ')
                .replace(']', ' ')
                .replace('{', ' ')
                .replace('}', ' ');

        String[] parts = s.split("\\s+");
        StringBuilder b = new StringBuilder(128);
        int kept = 0;

        for (String p : parts) {
            if (p.isBlank()) continue;
            if (p.length() < 3) continue;
            if (p.length() > 28) continue;
            if (looksLikeNumberOrMetric(p)) continue;
            if (TELEMETRY.matcher(p).find()) continue;

            boolean ok = true;
            for (int i = 0; i < p.length(); i++) {
                char ch = p.charAt(i);
                if (!(Character.isLetterOrDigit(ch) || ch == '_')) { ok = false; break; }
            }
            if (!ok) continue;

            if (b.length() > 0) b.append(' ');
            b.append(p);
            kept++;
            if (kept >= 10) break;
        }

        String r = b.toString().trim();
        if (r.length() > 160) r = r.substring(0, 160).trim();
        return r;
    }

    private static boolean looksLikeNumberOrMetric(String p) {
        boolean hasDigit = false;
        for (int i = 0; i < p.length(); i++) {
            char ch = p.charAt(i);
            if (Character.isDigit(ch)) hasDigit = true;
            else if (ch == '.' || ch == ',' || ch == '_' || ch == '-') {
                // ok
            } else if (!Character.isLetter(ch)) {
                return true;
            }
        }
        return hasDigit && p.length() <= 12;
    }

    private static String extractTerms(String text) {
        if (text == null || text.isBlank()) return "";
        // keep it short, deterministic, and "word-like"
        ArrayList<String> ts = new ArrayList<>(16);
        WORD.matcher(text.toLowerCase(Locale.ROOT))
                .results()
                .limit(64)
                .forEach(r -> {
                    String t = r.group();
                    if (t.length() >= 3 && t.length() <= 24 && !TELEMETRY.matcher(t).find()) ts.add(t);
                });

        // stable uniq
        LinkedHashSet<String> uniq = new LinkedHashSet<>(ts);
        StringBuilder b = new StringBuilder(120);
        int n = 0;
        for (String t : uniq) {
            if (n >= 10) break;
            if (b.length() > 0) b.append(' ');
            b.append(t);
            n++;
        }
        return b.toString();
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
        CandidateEvaluator.Evaluation ev = best == null ? null : best.evaluation;
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

    // -------------------- selection helpers --------------------

    private static Optional<Candidate> pickBest(List<Candidate> evaluated) {
        if (evaluated == null || evaluated.isEmpty()) return Optional.empty();

        Optional<Candidate> bestValid = evaluated.stream()
                .filter(c -> c != null && c.evaluation != null && c.evaluation.valid)
                .max(Comparator.comparingDouble(c -> c.score));

        if (bestValid.isPresent()) return bestValid;
        return evaluated.stream().filter(Objects::nonNull).max(Comparator.comparingDouble(c -> c.score));
    }

    // -------------------- schema/hints --------------------

    private static String buildGenerationHint(ThoughtState state) {
        StringBuilder sb = new StringBuilder(256);
        String sections = (state == null || state.tags == null) ? null : state.tags.get("response.sections");
        String style = (state == null || state.tags == null) ? null : state.tags.get("response.style");
        if (sections == null || sections.isBlank()) sections = "summary,evidence,actions";
        if (style == null || style.isBlank()) style = "md";

        sb.append("schema=").append(sections);
        sb.append(";style=").append(style);
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
            // IMPORTANT: this goes to generator only (conditioning), not to retriever
            sb.append(";last_notes=").append(state.lastCritique.replace('\n', ' '));
        }
        return sb.toString();
    }

    private static void initResponseSchema(ThoughtState state, String userText, Intent intent) {
        if (state == null || state.tags == null) return;

        boolean ru = isMostlyCyrillic(userText);

        state.tags.putIfAbsent("response.style", "md");
        state.tags.putIfAbsent("response.sections", "summary,evidence,actions");
        state.tags.putIfAbsent("response.label.summary", ru ? "Вывод" : "Conclusion");
        state.tags.putIfAbsent("response.label.evidence", ru ? "Опора на контекст" : "Evidence from context");
        state.tags.putIfAbsent("response.label.actions", ru ? "Следующие шаги" : "Next steps");

        // intent-conditioned contraction (still deterministic, non-domain)
        if (intent == Intent.GREETING) {
            state.tags.put("response.style", "plain");
            state.tags.put("response.sections", "summary");
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

    private static int clampInt(int v, int lo, int hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
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