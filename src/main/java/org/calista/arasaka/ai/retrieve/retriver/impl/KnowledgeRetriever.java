package org.calista.arasaka.ai.retrieve.retriver.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.Scored;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationConfig;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationStrategy;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;
import org.calista.arasaka.ai.retrieve.scorer.Scorer;

import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

public final class KnowledgeRetriever implements Retriever {
    private static final Logger log = LogManager.getLogger(KnowledgeRetriever.class);

    private final KnowledgeBase kb;
    private final Scorer scorer;
    private final ExplorationStrategy exploration;
    private final ExplorationConfig explorationCfg;

    private final Map<Long, List<Statement>> cache;
    private final int cacheCapacity;

    private volatile boolean prepared = false;

    public KnowledgeRetriever(
            KnowledgeBase kb,
            Scorer scorer,
            ExplorationStrategy exploration,
            ExplorationConfig explorationCfg,
            int cacheCapacity
    ) {
        this.kb = Objects.requireNonNull(kb, "kb");
        this.scorer = Objects.requireNonNull(scorer, "scorer");
        this.exploration = Objects.requireNonNull(exploration, "exploration");
        this.explorationCfg = Objects.requireNonNull(explorationCfg, "explorationCfg");

        this.cacheCapacity = Math.max(0, cacheCapacity);
        this.cache = (this.cacheCapacity <= 0)
                ? Map.of()
                : Collections.synchronizedMap(new LinkedHashMap<>(256, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Long, List<Statement>> eldest) {
                return size() > KnowledgeRetriever.this.cacheCapacity;
            }
        });
    }

    @Override
    public List<Statement> retrieve(String query, int k, long seed) {
        return retrieveTrace(query, k, seed).selected;
    }

    @Override
    public List<Scored<Statement>> retrieveScored(String query, int k, long seed) {
        return retrieveTrace(query, k, seed).finalRanked;
    }

    public Trace retrieveTrace(String query, int k, long seed) {
        final String q = query == null ? "" : query.trim();
        final int kk = Math.max(0, k);

        final long key = mix(seed, q.hashCode());
        if (cacheCapacity > 0) {
            List<Statement> cached = cache.get(key);
            if (cached != null) {
                if (log.isTraceEnabled()) log.trace("retriever cache hit key={} q='{}'", key, shortQ(q));
                return Trace.cached(q, cached.size() <= kk ? cached : cached.subList(0, kk));
            }
        }

        // Snapshot + dedup
        List<Statement> allRaw = kb.snapshotSorted();
        ArrayList<Statement> all = new ArrayList<>(allRaw.size());
        Set<String> dedup = new HashSet<>(Math.min(4096, allRaw.size()));
        for (Statement st : allRaw) {
            if (st == null) continue;
            String dk = dedupKey(st);
            if (dk != null && !dk.isBlank() && !dedup.add(dk)) continue;
            all.add(st);
        }

        // Prepare scorer once
        if (!prepared) {
            synchronized (this) {
                if (!prepared) {
                    try {
                        scorer.prepare(all);
                    } catch (Throwable t) {
                        log.warn("scorer.prepare(...) failed: {}", t.toString());
                    }
                    prepared = true;
                }
            }
        }

        HashMap<Statement, Double> agg = new HashMap<>(Math.max(256, all.size() / 8));
        ArrayList<Scored<Statement>> lastTop = new ArrayList<>(Math.min(256, Math.max(16, kk * 4)));

        String iterQuery = q;
        double iterWeight = 1.0;

        Trace trace = new Trace(q, kk);
        trace.totalCorpus = all.size();

        // Base query token set for coverage/validation/compress.
        final List<String> baseQTok = queryTokens(q, explorationCfg.candidateGateMinTokenLen);
        trace.queryTokenCount = baseQTok.size();
        final HashSet<String> baseQSet = baseQTok.isEmpty() ? null : new HashSet<>(baseQTok);

        // Stage 1: retrieve (wide, iterative)
        for (int iter = 0; iter < explorationCfg.iterations; iter++) {
            long t0 = System.nanoTime();

            final List<String> qTok = queryTokens(iterQuery, explorationCfg.candidateGateMinTokenLen);
            final HashSet<String> qSet = qTok.isEmpty() ? null : new HashSet<>(qTok);

            // Candidate gating should be deterministic and NOT biased by corpus iteration order.
            // If too many candidates match, we downsample deterministically using a hash-min sketch.
            ArrayList<Statement> candidates = gatedCandidates(all, qSet, explorationCfg.maxCandidatesPerIter, seed ^ (long) iter);

            double[] scores = scoreDeterministic(iterQuery, candidates);

            ArrayList<Scored<Statement>> scored = new ArrayList<>(candidates.size());
            for (int i = 0; i < candidates.size(); i++) {
                double s = scores[i];
                if (!Double.isFinite(s) || s < explorationCfg.minScore) continue;

                double weighted = s * iterWeight;
                Statement st = candidates.get(i);

                scored.add(new Scored<>(st, weighted));
                agg.merge(st, weighted, Double::sum);
            }

            scored.sort(Comparator.naturalOrder());

            lastTop.clear();
            int band = Math.min(Math.max(16, kk * 4), scored.size());
            for (int i = 0; i < band; i++) lastTop.add(scored.get(i));

            long t1 = System.nanoTime();
            trace.addIteration(iter, iterQuery, candidates.size(), scored.size(), band, (t1 - t0));

            double confNow = estimateConfidenceTop(scored);
            if (explorationCfg.earlyStopConfidence > 0.0 && confNow >= explorationCfg.earlyStopConfidence) {
                trace.earlyStopped = true;
                trace.earlyStopConfidence = confNow;
                break;
            }

            // refine: derive better query from evidence; anti-water via local IDF inside refineQuery(...)
            if (iter + 1 < explorationCfg.iterations && explorationCfg.refineTerms > 0 && !lastTop.isEmpty()) {
                iterQuery = refineQuery(q, lastTop, explorationCfg.refineTerms, scorer, baseQSet, explorationCfg.refineDfCut);
                iterWeight *= explorationCfg.iterationDecay;
            }
        }

        // Merge aggregated scores -> finalRanked (wide pool)
        ArrayList<Scored<Statement>> finalRanked = new ArrayList<>(agg.size());
        for (Map.Entry<Statement, Double> e : agg.entrySet()) {
            double s = e.getValue();
            if (Double.isFinite(s) && s >= explorationCfg.minScore) {
                finalRanked.add(new Scored<>(e.getKey(), s));
            }
        }
        finalRanked.sort(Comparator.naturalOrder());
        trace.finalRanked = finalRanked;

        trace.confidence = estimateConfidence(finalRanked);

        // Coverage/validation (cheap deterministic)
        if (explorationCfg.coverageK > 0 && baseQSet != null && !baseQSet.isEmpty()) {
            trace.coverage = estimateCoverage(finalRanked, baseQSet, explorationCfg.coverageK, scorer);
        } else {
            trace.coverage = Double.NaN;
        }

        int outK = kk;
        trace.quality = combineQuality(trace.confidence, trace.coverage);
        if (explorationCfg.qualityFloor > 0.0 && trace.quality < explorationCfg.qualityFloor) {
            outK = Math.min(outK, Math.max(1, kk / 2));
        }

        // Stage 2: rerank (precision) on top-N -> keep top-M
        List<Scored<Statement>> rerankedTop = finalRanked;
        if (explorationCfg.rerankN > 0 && !finalRanked.isEmpty()) {
            rerankedTop = rerankTop(q, finalRanked, explorationCfg.rerankN, scorer);
            if (explorationCfg.rerankM > 0 && rerankedTop.size() > explorationCfg.rerankM) {
                rerankedTop = rerankedTop.subList(0, Math.max(1, explorationCfg.rerankM));
            }
        }
        trace.rerankedTop = rerankedTop;

        // Selection should be done on rerankedTop (post-rerank)
        List<Statement> selected = exploration.select(rerankedTop, outK, explorationCfg, seed);
        trace.selected = selected;

        // Contradiction penalty on selected evidence
        trace.contradiction = (explorationCfg.contradictionPenalty > 0.0)
                ? estimateContradiction(selected)
                : 0.0;

        trace.quality = combineQuality(trace.confidence, trace.coverage)
                - (explorationCfg.contradictionPenalty * trace.contradiction);

        // Stage 3: compress (reduce noise) -> ready for prompt context
        if (explorationCfg.compressSentencesPerStatement > 0 && !selected.isEmpty()) {
            trace.compressedContext = compressContext(
                    q,
                    baseQSet == null ? Set.of() : baseQSet,
                    selected,
                    explorationCfg.compressSentencesPerStatement,
                    explorationCfg.compressMaxCharsPerStatement,
                    scorer
            );
        } else {
            trace.compressedContext = List.of();
        }

        if (cacheCapacity > 0) {
            cache.put(key, selected);
            if (log.isTraceEnabled()) log.trace("retriever cache miss key={} q='{}' -> {}", key, shortQ(q), selected.size());
        }

        return trace;
    }

    // ----------------- Stage 2: rerank -----------------

    /**
     * Rerank only topN results against the original query.
     * Deterministic: uses scorer.score(query, statement) (or scoreBatch when not parallel).
     */
    private List<Scored<Statement>> rerankTop(String query, List<Scored<Statement>> ranked, int topN, Scorer scorer) {
        int n = Math.min(Math.max(1, topN), ranked.size());
        ArrayList<Statement> cand = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Statement st = ranked.get(i).item;
            if (st != null && st.text != null && !st.text.isBlank()) cand.add(st);
        }
        if (cand.isEmpty()) return ranked;

        double[] s = scoreDeterministic(query, cand);

        ArrayList<Scored<Statement>> out = new ArrayList<>(cand.size());
        for (int i = 0; i < cand.size(); i++) {
            double v = s[i];
            if (!Double.isFinite(v)) continue;
            out.add(new Scored<>(cand.get(i), v));
        }
        out.sort(Comparator.naturalOrder());
        return out;
    }

    // ----------------- Stage 3: compress -----------------

    /**
     * Compress each Statement to the most relevant sentences (token overlap with query).
     * Returns list of compressed strings ready to be fed to generator.
     */
    private static List<String> compressContext(
            String query,
            Set<String> queryTokens,
            List<Statement> selected,
            int sentencesPerStatement,
            int maxCharsPerStatement,
            Scorer scorer
    ) {
        if (selected == null || selected.isEmpty() || sentencesPerStatement <= 0) return List.of();

        ArrayList<String> out = new ArrayList<>(selected.size());

        for (Statement st : selected) {
            if (st == null || st.text == null || st.text.isBlank()) continue;

            String text = st.text.trim();
            List<String> sentences = splitSentences(text);
            if (sentences.isEmpty()) continue;

            // Score sentences by overlap (fast & stable)
            ArrayList<Scored<String>> scored = new ArrayList<>(sentences.size());
            for (String s : sentences) {
                if (s == null) continue;
                String ss = s.trim();
                if (ss.isBlank()) continue;

                double ov = sentenceOverlap(queryTokens, ss, scorer);
                if (ov <= 0.0) continue;
                scored.add(new Scored<>(ss, ov));
            }

            scored.sort(Comparator.naturalOrder());

            StringBuilder sb = new StringBuilder(Math.min(1024, text.length()));
            int take = Math.min(sentencesPerStatement, scored.size());
            for (int i = 0; i < take; i++) {
                if (i > 0) sb.append(' ');
                sb.append(scored.get(i).item);
            }

            String compressed = sb.length() > 0 ? sb.toString() : bestFallback(text, maxCharsPerStatement);
            if (maxCharsPerStatement > 0 && compressed.length() > maxCharsPerStatement) {
                compressed = compressed.substring(0, Math.max(1, maxCharsPerStatement));
            }
            out.add(compressed);
        }

        return out;
    }

    private static double sentenceOverlap(Set<String> queryTokens, String sentence, Scorer scorer) {
        if (queryTokens == null || queryTokens.isEmpty() || sentence == null || sentence.isBlank()) return 0.0;

        // Cheap regex tokens (schema-agnostic)
        String[] parts = sentence.toLowerCase(Locale.ROOT).split("[^\\p{L}\\p{Nd}_]+");
        int hit = 0;
        int uniq = 0;
        HashSet<String> seen = new HashSet<>(Math.min(32, parts.length));
        for (String p : parts) {
            if (p == null) continue;
            String t = p.trim();
            if (t.length() < 3) continue;
            if (!seen.add(t)) continue;
            uniq++;
            if (queryTokens.contains(t)) hit++;
        }
        return uniq > 0 ? (double) hit / (double) uniq : 0.0;
    }

    private static List<String> splitSentences(String text) {
        if (text == null || text.isBlank()) return List.of();
        // Simple robust split: dot/exclamation/question/newline/semicolon.
        String[] parts = text.split("(?<=[.!?;])\\s+|\\n+");
        ArrayList<String> out = new ArrayList<>(Math.min(16, parts.length));
        for (String p : parts) {
            if (p == null) continue;
            String s = p.trim();
            if (s.length() < 3) continue;
            out.add(s);
        }
        return out;
    }

    private static String bestFallback(String text, int maxChars) {
        if (text == null) return "";
        String t = text.trim();
        if (maxChars > 0 && t.length() > maxChars) return t.substring(0, Math.max(1, maxChars));
        return t;
    }

    // ----------------- Deterministic scoring -----------------

    private double[] scoreDeterministic(String query, List<Statement> candidates) {
        if (candidates == null || candidates.isEmpty()) return new double[0];

        if (!explorationCfg.parallel) {
            return scorer.scoreBatch(query, candidates);
        }

        final int n = candidates.size();
        final double[] out = new double[n];

        if (explorationCfg.parallelism > 0) {
            // IMPORTANT: parallel streams ignore custom pools (they use the common pool).
            // For true deterministic bounded parallelism, do explicit chunked tasks.
            final int p = Math.max(1, explorationCfg.parallelism);
            ForkJoinPool pool = new ForkJoinPool(p);
            try {
                final int chunk = Math.max(256, (n + p - 1) / p);
                ArrayList<Future<?>> fs = new ArrayList<>(p * 2);
                for (int start = 0; start < n; start += chunk) {
                    final int s = start;
                    final int e = Math.min(n, start + chunk);
                    fs.add(pool.submit(() -> {
                        for (int i = s; i < e; i++) {
                            out[i] = scorer.score(query, candidates.get(i));
                        }
                    }));
                }
                for (Future<?> f : fs) {
                    try {
                        f.get();
                    } catch (Throwable ignored) {
                        // best-effort: leave defaults for failed chunk
                    }
                }
            } finally {
                pool.shutdown();
            }
        } else {
            IntStream.range(0, n).parallel().forEach(i -> out[i] = scorer.score(query, candidates.get(i)));
        }
        return out;
    }

    // ----------------- Candidate gating -----------------

    private boolean gateMatches(Set<String> qSet, Statement st) {
        try {
            String[] toks = scorer.tokens(st);
            if (toks != null && toks.length > 0) {
                for (String t : toks) {
                    if (t == null) continue;
                    if (qSet.contains(t)) return true;
                }
                return false;
            }
        } catch (Throwable ignored) {
        }
        return cheapMatches(qSet, st.text);
    }

    private static boolean cheapMatches(Set<String> qSet, String text) {
        if (text == null || text.isBlank()) return false;
        String low = text.toLowerCase(Locale.ROOT);
        for (String t : qSet) {
            if (t.length() >= 3 && low.contains(t)) return true;
        }
        return false;
    }

    /**
     * Deterministic candidate gating with order-unbiased downsampling.
     *
     * Why: collecting the "first N" matching statements biases results by corpus order.
     * Here we keep the N smallest mixed hashes (min-hash style), which is deterministic
     * and independent of snapshot ordering.
     */
    private ArrayList<Statement> gatedCandidates(
            List<Statement> all,
            Set<String> qSet,
            int maxCandidates,
            long iterSeed
    ) {
        if (all == null || all.isEmpty()) return new ArrayList<>(0);
        final int cap = Math.max(1, maxCandidates);

        // If cap is huge, just collect.
        if (cap >= all.size()) {
            ArrayList<Statement> out = new ArrayList<>(all.size());
            for (Statement st : all) {
                if (st == null || st.text == null || st.text.isBlank()) continue;
                if (qSet != null && !gateMatches(qSet, st)) continue;
                out.add(st);
            }
            return out;
        }

        // Keep the cap smallest hash values (min-hash sampling).
        PriorityQueue<Long> pq = new PriorityQueue<>(cap + 1, Comparator.reverseOrder());
        for (Statement st : all) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            if (qSet != null && !gateMatches(qSet, st)) continue;

            long h = mix(iterSeed, stableStmtHash(st));
            if (pq.size() < cap) {
                pq.add(h);
            } else if (h < pq.peek()) {
                pq.poll();
                pq.add(h);
            }
        }

        if (pq.isEmpty()) return new ArrayList<>(0);

        HashSet<Long> keep = new HashSet<>(pq.size() * 2);
        while (!pq.isEmpty()) keep.add(pq.poll());

        ArrayList<Statement> out = new ArrayList<>(Math.min(cap, keep.size()));
        for (Statement st : all) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            if (qSet != null && !gateMatches(qSet, st)) continue;

            long h = mix(iterSeed, stableStmtHash(st));
            if (keep.contains(h)) out.add(st);
            if (out.size() >= cap) break;
        }
        return out;
    }

    private static long stableStmtHash(Statement st) {
        if (st == null) return 0L;
        String id = st.id;
        if (id != null && !id.isBlank()) return (long) id.hashCode();
        String t = st.text == null ? "" : st.text;
        return (long) t.hashCode();
    }

    // ----------------- Query refine (anti-water: local IDF + DF cut) -----------------

    private static String refineQuery(
            String originalQuery,
            List<Scored<Statement>> top,
            int maxTerms,
            Scorer scorer,
            Set<String> baseQueryTokens,
            double dfCut
    ) {
        if (maxTerms <= 0 || top == null || top.isEmpty()) return originalQuery == null ? "" : originalQuery;

        final String oq = (originalQuery == null) ? "" : originalQuery.trim();
        final String oqLow = oq.toLowerCase(Locale.ROOT);

        // Local DF over top-docs
        HashMap<String, Integer> df = new HashMap<>(512);
        int docs = 0;

        // Also keep a relevance-weighted accumulator
        HashMap<String, Double> w = new HashMap<>(512);
        HashMap<String, Integer> firstSeen = new HashMap<>(512);

        int seen = 0;
        for (Scored<Statement> s : top) {
            Statement st = s.item;
            if (st == null || st.text == null || st.text.isBlank()) {
                seen++;
                continue;
            }

            docs++;

            String[] toks = null;
            try {
                toks = (scorer == null) ? null : scorer.tokens(st);
            } catch (Throwable ignored) {
            }
            if (toks == null || toks.length == 0) {
                String low = st.text.toLowerCase(Locale.ROOT);
                toks = low.split("[^\\p{L}\\p{Nd}_]+");
            }

            HashSet<String> uniq = new HashSet<>(Math.min(64, toks.length));
            for (String p : toks) {
                if (p == null) continue;
                String t = p.trim().toLowerCase(Locale.ROOT);
                if (t.length() < 3) continue;
                if (baseQueryTokens != null && baseQueryTokens.contains(t)) continue;
                if (!oqLow.isEmpty() && oqLow.contains(t)) continue;
                uniq.add(t);
            }

            for (String t : uniq) df.merge(t, 1, Integer::sum);

            double base = Math.max(0.0, s.score);
            if (!(base > 0.0)) base = 1e-6;

            for (String t : uniq) {
                double tw = 1.0;
                try {
                    tw = (scorer == null) ? 1.0 : scorer.termWeight(t);
                } catch (Throwable ignored) {
                }
                w.merge(t, base * Math.max(0.0, tw), Double::sum);
                firstSeen.putIfAbsent(t, seen);
            }

            seen++;
        }

        if (docs <= 0) return oq;

        ArrayList<String> terms = new ArrayList<>(w.keySet());
        int finalDocs = docs;
        terms.sort((a, b) -> {
            int dfa = df.getOrDefault(a, 0);
            int dfb = df.getOrDefault(b, 0);

            double ra = (double) dfa / (double) finalDocs;
            double rb = (double) dfb / (double) finalDocs;

            // If above dfCut, hard demote
            boolean wa = ra >= dfCut;
            boolean wb = rb >= dfCut;
            if (wa != wb) return wa ? 1 : -1;

            double idfa = Math.log((finalDocs + 1.0) / (dfa + 1.0)) + 1.0;
            double idfb = Math.log((finalDocs + 1.0) / (dfb + 1.0)) + 1.0;

            double sa = w.getOrDefault(a, 0.0) * idfa;
            double sb = w.getOrDefault(b, 0.0) * idfb;

            int cmp = Double.compare(sb, sa);
            if (cmp != 0) return cmp;
            return Integer.compare(firstSeen.getOrDefault(a, Integer.MAX_VALUE), firstSeen.getOrDefault(b, Integer.MAX_VALUE));
        });

        StringBuilder sb = new StringBuilder(256);
        if (!oq.isBlank()) {
            sb.append(oq);
            sb.append('\n');
        }
        sb.append("context: ");

        int added = 0;
        for (String t : terms) {
            if (added >= maxTerms) break;
            int d = df.getOrDefault(t, 0);
            double r = (double) d / (double) docs;
            if (r >= dfCut) continue; // hard cut water
            if (added > 0) sb.append(' ');
            sb.append(t);
            added++;
        }

        return sb.toString();
    }

    // ----------------- Confidence / Coverage / Quality -----------------

    private static double estimateConfidenceTop(List<Scored<Statement>> ranked) {
        if (ranked == null || ranked.size() < 2) return 0.0;
        double a = ranked.get(0).score;
        double b = ranked.get(1).score;
        if (!Double.isFinite(a) || !Double.isFinite(b)) return 0.0;
        if (a <= 0.0) return 0.0;
        double gap = Math.max(0.0, a - b);
        return gap / (gap + 1.0);
    }

    private static double estimateConfidence(List<Scored<Statement>> ranked) {
        if (ranked == null || ranked.isEmpty()) return 0.0;

        double top = ranked.get(0).score;
        if (!Double.isFinite(top) || top <= 0.0) return 0.0;

        double sum = 0.0;
        int n = Math.min(16, ranked.size());
        for (int i = 0; i < n; i++) {
            double s = ranked.get(i).score;
            if (Double.isFinite(s) && s > 0.0) sum += s;
        }
        if (!(sum > 0.0)) return 0.0;

        double dominance = top / sum;
        return 1.0 - Math.exp(-3.0 * dominance);
    }

    /**
     * Token coverage score: fraction of unique query tokens that appear in the topK evidence.
     * Uses scorer.tokens(...) when available (fast/cache), otherwise uses cheap substring checks.
     */
    private static double estimateCoverage(
            List<Scored<Statement>> ranked,
            Set<String> queryTokens,
            int topK,
            Scorer scorer
    ) {
        if (ranked == null || ranked.isEmpty() || queryTokens == null || queryTokens.isEmpty() || topK <= 0) {
            return 0.0;
        }

        HashSet<String> covered = new HashSet<>(Math.min(64, queryTokens.size() * 2));
        int n = Math.min(topK, ranked.size());
        for (int i = 0; i < n && covered.size() < queryTokens.size(); i++) {
            Statement st = ranked.get(i).item;
            if (st == null || st.text == null || st.text.isBlank()) continue;

            String[] toks = null;
            try {
                toks = (scorer == null) ? null : scorer.tokens(st);
            } catch (Throwable ignored) {
            }
            if (toks != null && toks.length > 0) {
                for (String t : toks) {
                    if (t == null) continue;
                    String x = t.trim().toLowerCase(Locale.ROOT);
                    if (x.length() < 3) continue;
                    if (queryTokens.contains(x)) covered.add(x);
                }
            } else {
                String low = st.text.toLowerCase(Locale.ROOT);
                for (String qt : queryTokens) {
                    if (covered.contains(qt)) continue;
                    if (qt.length() >= 3 && low.contains(qt)) covered.add(qt);
                }
            }
        }
        return (double) covered.size() / (double) queryTokens.size();
    }

    /**
     * Combine confidence + coverage into a single quality signal.
     * If coverage is NaN (disabled), quality==confidence.
     */
    private static double combineQuality(double confidence, double coverage) {
        if (!Double.isFinite(confidence) || confidence < 0.0) confidence = 0.0;
        if (!Double.isFinite(coverage)) return confidence;
        if (coverage < 0.0) coverage = 0.0;
        if (coverage > 1.0) coverage = 1.0;

        // Harmonic-like blend: punishes low coverage even when confidence is high.
        double a = confidence;
        double b = coverage;
        if (!(a > 0.0) || !(b > 0.0)) return 0.0;
        return (2.0 * a * b) / (a + b);
    }

    // ----------------- Contradiction -----------------

    /**
     * Contradiction estimator (cheap & schema-agnostic):
     *  - groups evidence by an inferred key (id/type/subject-like)
     *  - checks opposite polarity (negation markers) within a group
     * Returns [0..1] "how contradictory" the selection is.
     */
    private static double estimateContradiction(List<Statement> selected) {
        if (selected == null || selected.size() < 2) return 0.0;

        HashMap<String, int[]> group = new HashMap<>(64);
        int usable = 0;

        for (Statement st : selected) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            usable++;
            String key = contradictionKey(st);
            if (key == null || key.isBlank()) key = "__all__";
            boolean neg = isNegated(st.text);
            int[] c = group.computeIfAbsent(key, k -> new int[2]);
            c[neg ? 1 : 0]++;
        }

        if (usable < 2) return 0.0;

        int contradictoryPairs = 0;
        int possiblePairs = 0;
        for (int[] c : group.values()) {
            int pos = c[0];
            int neg = c[1];
            if (pos + neg < 2) continue;
            possiblePairs += (pos + neg) * (pos + neg - 1) / 2;
            contradictoryPairs += pos * neg;
        }
        if (possiblePairs <= 0) return 0.0;
        double v = (double) contradictoryPairs / (double) possiblePairs;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        return v;
    }

    private static boolean isNegated(String text) {
        if (text == null || text.isBlank()) return false;
        String low = text.toLowerCase(Locale.ROOT);
        return low.contains(" not ") || low.startsWith("not ") || low.contains("n't") ||
                low.contains(" не ") || low.startsWith("не ") || low.contains(" нет ") || low.contains("без ");
    }

    private static String contradictionKey(Statement st) {
        if (st == null) return null;
        if (st.id != null && !st.id.isBlank()) return "id:" + st.id;

        for (String f : new String[]{"type", "topic", "subject", "entity", "key"}) {
            Object v = tryGetField(st, f);
            if (v instanceof String s && !s.isBlank()) return f + ":" + s.trim();
        }

        String t = st.text == null ? "" : st.text.trim();
        int cut = t.indexOf('.');
        if (cut < 0) cut = t.indexOf('\n');
        if (cut < 0) cut = Math.min(80, t.length());
        return "tx:" + t.substring(0, Math.min(cut, t.length())).toLowerCase(Locale.ROOT);
    }

    private static Object tryGetField(Object o, String name) {
        try {
            var f = o.getClass().getField(name);
            return f.get(o);
        } catch (Throwable ignored) {
            return null;
        }
    }

    // ----------------- Tokenization / Dedup / Utils -----------------

    private static List<String> queryTokens(String query, int minLen) {
        if (query == null || query.isBlank()) return List.of();
        String low = query.toLowerCase(Locale.ROOT);
        String[] parts = low.split("[^\\p{L}\\p{Nd}_]+");
        if (parts.length == 0) return List.of();

        ArrayList<String> out = new ArrayList<>(Math.min(32, parts.length));
        for (String p : parts) {
            if (p == null) continue;
            String t = p.trim();
            if (t.length() < minLen) continue;
            out.add(t);
        }
        return out;
    }

    private static String dedupKey(Statement st) {
        if (st == null) return null;
        if (st.id != null && !st.id.isBlank()) return "id:" + st.id;
        if (st.text != null && !st.text.isBlank()) return "tx:" + st.text.trim();
        return null;
    }

    private static String shortQ(String q) {
        if (q == null) return "";
        String s = q.replaceAll("\\s+", " ").trim();
        return s.length() <= 120 ? s : s.substring(0, 117) + "...";
    }

    private static long mix(long a, long b) {
        long x = a ^ b;
        x ^= (x >>> 33);
        x *= 0xff51afd7ed558ccdL;
        x ^= (x >>> 33);
        x *= 0xc4ceb9fe1a85ec53L;
        x ^= (x >>> 33);
        return x;
    }

    // ----------------- Trace -----------------

    public static final class Trace {
        public final String query;
        public final int requestedK;

        public int totalCorpus;
        public double confidence;

        /** Number of unique query tokens used for coverage/validation. */
        public int queryTokenCount;

        /** Token coverage of base query by top evidence [0..1], or NaN if disabled. */
        public double coverage;

        /** Estimated contradiction ratio [0..1] within selected evidence. */
        public double contradiction;

        /** Aggregate quality signal (confidence + coverage - penalties). */
        public double quality;

        public boolean earlyStopped;
        public double earlyStopConfidence;

        /** Wide merged pool after retrieve+merge. */
        public List<Scored<Statement>> finalRanked = List.of();

        /** After rerank stage: top-N reranked, then trimmed to top-M. */
        public List<Scored<Statement>> rerankedTop = List.of();

        /** Selected evidence Statements (for explainability / debug). */
        public List<Statement> selected = List.of();

        /** Compressed context strings for prompt (best sentences only). */
        public List<String> compressedContext = List.of();

        public final List<Iter> iterations = new ArrayList<>(8);

        private Trace(String query, int requestedK) {
            this.query = query;
            this.requestedK = requestedK;
        }

        private static Trace cached(String query, List<Statement> selected) {
            Trace t = new Trace(query, selected == null ? 0 : selected.size());
            t.selected = selected == null ? List.of() : selected;
            t.confidence = Double.NaN;
            t.coverage = Double.NaN;
            t.quality = Double.NaN;
            return t;
        }

        private void addIteration(int idx, String iterQuery, int candidates, int scored, int band, long nanos) {
            iterations.add(new Iter(idx, iterQuery, candidates, scored, band, nanos));
        }

        public static final class Iter {
            public final int index;
            public final String query;
            public final int candidates;
            public final int scored;
            public final int topBand;
            public final long nanos;

            public Iter(int index, String query, int candidates, int scored, int topBand, long nanos) {
                this.index = index;
                this.query = query;
                this.candidates = candidates;
                this.scored = scored;
                this.topBand = topBand;
                this.nanos = nanos;
            }
        }
    }
}