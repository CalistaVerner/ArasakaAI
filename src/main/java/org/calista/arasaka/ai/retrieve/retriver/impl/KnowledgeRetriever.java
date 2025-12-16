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

        List<Statement> allRaw = kb.snapshotSorted();
        ArrayList<Statement> all = new ArrayList<>(allRaw.size());
        Set<String> dedup = new HashSet<>(Math.min(4096, allRaw.size()));
        for (Statement st : allRaw) {
            if (st == null) continue;
            String dk = dedupKey(st);
            if (dk != null && !dk.isBlank() && !dedup.add(dk)) continue;
            all.add(st);
        }

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

        for (int iter = 0; iter < explorationCfg.iterations; iter++) {
            long t0 = System.nanoTime();

            final List<String> qTok = queryTokens(iterQuery, explorationCfg.candidateGateMinTokenLen);
            final HashSet<String> qSet = qTok.isEmpty() ? null : new HashSet<>(qTok);

            ArrayList<Statement> candidates = new ArrayList<>(Math.min(all.size(), explorationCfg.maxCandidatesPerIter));
            for (Statement st : all) {
                if (st == null) continue;
                if (st.text == null || st.text.isBlank()) continue;
                if (qSet != null && !gateMatches(qSet, st)) continue;

                candidates.add(st);
                if (candidates.size() >= explorationCfg.maxCandidatesPerIter) break;
            }

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

            if (iter + 1 < explorationCfg.iterations && explorationCfg.refineTerms > 0 && !lastTop.isEmpty()) {
                iterQuery = refineQuery(q, lastTop, explorationCfg.refineTerms);
                iterWeight *= explorationCfg.iterationDecay;
            }
        }

        ArrayList<Scored<Statement>> finalRanked = new ArrayList<>(agg.size());
        for (Map.Entry<Statement, Double> e : agg.entrySet()) {
            double s = e.getValue();
            if (Double.isFinite(s) && s >= explorationCfg.minScore) {
                finalRanked.add(new Scored<>(e.getKey(), s));
            }
        }
        finalRanked.sort(Comparator.naturalOrder());

        double confidence = estimateConfidence(finalRanked);
        trace.confidence = confidence;

        if (log.isTraceEnabled()) {
            log.trace("retrieve q='{}' ranked={} confidence={}",
                    shortQ(q), finalRanked.size(), String.format(Locale.ROOT, "%.4f", confidence));
        }

        int outK = kk;
        if (explorationCfg.qualityFloor > 0.0 && confidence < explorationCfg.qualityFloor) {
            outK = Math.min(outK, Math.max(1, kk / 2));
        }

        List<Statement> selected = exploration.select(finalRanked, outK, explorationCfg, seed);
        trace.selected = selected;
        trace.finalRanked = finalRanked;

        if (cacheCapacity > 0) {
            cache.put(key, selected);
            if (log.isTraceEnabled()) log.trace("retriever cache miss key={} q='{}' -> {}", key, shortQ(q), selected.size());
        }
        return trace;
    }

    private double[] scoreDeterministic(String query, List<Statement> candidates) {
        if (candidates == null || candidates.isEmpty()) return new double[0];

        if (!explorationCfg.parallel) {
            return scorer.scoreBatch(query, candidates);
        }

        final int n = candidates.size();
        final double[] out = new double[n];

        if (explorationCfg.parallelism > 0) {
            ForkJoinPool pool = new ForkJoinPool(explorationCfg.parallelism);
            try {
                pool.submit(() ->
                        IntStream.range(0, n).parallel().forEach(i -> out[i] = scorer.score(query, candidates.get(i)))
                ).join();
            } finally {
                pool.shutdown();
            }
        } else {
            IntStream.range(0, n).parallel().forEach(i -> out[i] = scorer.score(query, candidates.get(i)));
        }
        return out;
    }

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

    private static String refineQuery(String originalQuery, List<Scored<Statement>> top, int maxTerms) {
        if (maxTerms <= 0 || top == null || top.isEmpty()) return originalQuery == null ? "" : originalQuery;

        HashMap<String, Double> w = new HashMap<>(256);
        HashMap<String, Integer> firstSeen = new HashMap<>(256);

        int seen = 0;
        for (Scored<Statement> s : top) {
            Statement st = s.item;
            if (st == null || st.text == null || st.text.isBlank()) continue;

            String low = st.text.toLowerCase(Locale.ROOT);
            String[] parts = low.split("[^\\p{L}\\p{Nd}_]+");
            if (parts.length == 0) continue;

            HashSet<String> uniq = new HashSet<>(Math.min(64, parts.length));
            for (String p : parts) {
                if (p == null) continue;
                String t = p.trim();
                if (t.length() < 3) continue;
                uniq.add(t);
            }

            double base = Math.max(0.0, s.score);
            if (!(base > 0.0)) base = 1e-6;

            for (String t : uniq) {
                w.merge(t, base, Double::sum);
                firstSeen.putIfAbsent(t, seen);
            }
            seen++;
        }

        ArrayList<String> terms = new ArrayList<>(w.keySet());
        terms.sort((a, b) -> {
            double wa = w.getOrDefault(a, 0.0);
            double wb = w.getOrDefault(b, 0.0);
            int cmp = Double.compare(wb, wa);
            if (cmp != 0) return cmp;
            return Integer.compare(firstSeen.getOrDefault(a, Integer.MAX_VALUE), firstSeen.getOrDefault(b, Integer.MAX_VALUE));
        });

        StringBuilder sb = new StringBuilder(256);
        if (originalQuery != null && !originalQuery.isBlank()) {
            sb.append(originalQuery.trim());
            sb.append('\n');
        }
        sb.append("context: ");

        int added = 0;
        for (String t : terms) {
            if (added >= maxTerms) break;
            if (originalQuery != null) {
                String oq = originalQuery.toLowerCase(Locale.ROOT);
                if (oq.contains(t)) continue;
            }
            if (added > 0) sb.append(' ');
            sb.append(t);
            added++;
        }

        return sb.toString();
    }

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

    public static final class Trace {
        public final String query;
        public final int requestedK;

        public int totalCorpus;
        public double confidence;

        public boolean earlyStopped;
        public double earlyStopConfidence;

        public List<Statement> selected = List.of();
        public List<Scored<Statement>> finalRanked = List.of();
        public final List<Iter> iterations = new ArrayList<>(8);

        private Trace(String query, int requestedK) {
            this.query = query;
            this.requestedK = requestedK;
        }

        private static Trace cached(String query, List<Statement> selected) {
            Trace t = new Trace(query, selected == null ? 0 : selected.size());
            t.selected = selected == null ? List.of() : selected;
            t.confidence = Double.NaN;
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