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

public final class KnowledgeRetriever implements Retriever {
    private static final Logger log = LogManager.getLogger(KnowledgeRetriever.class);

    private final KnowledgeBase kb;
    private final Scorer scorer;
    private final ExplorationStrategy exploration;
    private final ExplorationConfig explorationCfg;

    private final Map<Long, List<Statement>> cache;
    private final int cacheCapacity;

    // One-time warmup guard (per retriever instance)
    private volatile boolean prepared = false;

    public KnowledgeRetriever(
            KnowledgeBase kb,
            Scorer scorer,
            ExplorationStrategy exploration,
            ExplorationConfig explorationCfg
    ) {
        this(kb, scorer, exploration, explorationCfg, 10_000);
    }

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
                : java.util.Collections.synchronizedMap(new LinkedHashMap<>(256, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Long, List<Statement>> eldest) {
                return size() > KnowledgeRetriever.this.cacheCapacity;
            }
        });
    }

    @Override
    public List<Statement> retrieve(String query, int k, long seed) {
        final String q = query == null ? "" : query;
        final int kk = Math.max(0, k);

        final long key = mix(seed, q.hashCode());
        if (cacheCapacity > 0) {
            List<Statement> cached = cache.get(key);
            if (cached != null) {
                if (log.isTraceEnabled()) log.trace("retriever cache hit key={} q='{}'", key, shortQ(q));
                return cached.size() <= kk ? cached : cached.subList(0, kk);
            }
        }

        // Stable base order + deterministic de-dup.
        List<Statement> allRaw = kb.snapshotSorted();
        ArrayList<Statement> all = new ArrayList<>(allRaw.size());
        Set<String> dedup = new HashSet<>(Math.min(4096, allRaw.size()));
        for (Statement st : allRaw) {
            if (st == null) continue;
            String dk = dedupKey(st);
            if (dk != null && !dk.isBlank() && !dedup.add(dk)) continue;
            all.add(st);
        }

        // Warm up scorer once per retriever instance.
        if (!prepared) {
            synchronized (this) {
                if (!prepared) {
                    try {
                        scorer.prepare(all);
                    } catch (Throwable t) {
                        // Must not kill retrieval in prod; just log and continue.
                        log.warn("scorer.prepare(...) failed: {}", t.toString());
                    }
                    prepared = true;
                }
            }
        }

        // ------------------------------
        // Thinking pipeline: multi-iteration, deterministic, cacheable.
        //   Iteration: gate -> score -> aggregate(decay) -> refine query -> repeat
        // ------------------------------
        HashMap<Statement, Double> agg = new HashMap<>(Math.max(256, all.size() / 8));
        ArrayList<Scored<Statement>> lastTop = new ArrayList<>(Math.min(256, Math.max(16, kk * 4)));

        String iterQuery = q;
        double iterWeight = 1.0;

        for (int iter = 0; iter < explorationCfg.iterations; iter++) {
            final List<String> qTok = queryTokens(iterQuery, explorationCfg.candidateGateMinTokenLen);
            final HashSet<String> qSet = qTok.isEmpty() ? null : new HashSet<>(qTok);

            ArrayList<Scored<Statement>> scored = new ArrayList<>(Math.max(16, Math.min(all.size(), explorationCfg.maxCandidatesPerIter)));

            int processed = 0;
            for (Statement st : all) {
                if (st == null) continue;
                if (st.text == null || st.text.isBlank()) continue;

                // Candidate gating:
                if (qSet != null && !gateMatches(qSet, st)) continue;

                double s = scorer.score(iterQuery, st);
                if (!Double.isFinite(s) || s < explorationCfg.minScore) continue;

                double weighted = s * iterWeight;
                scored.add(new Scored<>(st, weighted));
                agg.merge(st, weighted, Double::sum);

                // Hard compute cap per iteration.
                processed++;
                if (processed >= explorationCfg.maxCandidatesPerIter) break;
            }

            scored.sort(Comparator
                    .comparingDouble((Scored<Statement> x) -> x.score).reversed()
                    .thenComparing(x -> x.item.id == null ? "" : x.item.id));

            // Keep a small band of best candidates for query refinement.
            lastTop.clear();
            int band = Math.min(Math.max(16, kk * 4), scored.size());
            for (int i = 0; i < band; i++) lastTop.add(scored.get(i));

            // Refine query for the next iteration using weighted terms from best candidates.
            if (iter + 1 < explorationCfg.iterations) {
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
        finalRanked.sort(Comparator
                .comparingDouble((Scored<Statement> x) -> x.score).reversed()
                .thenComparing(x -> x.item.id == null ? "" : x.item.id));

        // Quality estimation (confidence) – deterministic, used for logging and optional floor.
        double confidence = estimateConfidence(finalRanked);
        if (log.isTraceEnabled()) {
            log.trace("retrieve q='{}' ranked={} confidence={}", shortQ(q), finalRanked.size(), String.format(Locale.ROOT, "%.4f", confidence));
        }

        int outK = kk;
        if (explorationCfg.qualityFloor > 0.0 && confidence < explorationCfg.qualityFloor) {
            // If uncertain: return fewer items (high precision) instead of spraying noise.
            outK = Math.min(outK, Math.max(1, kk / 2));
        }

        List<Statement> selected = exploration.select(finalRanked, outK, explorationCfg, seed);

        if (cacheCapacity > 0) {
            cache.put(key, selected);
            if (log.isTraceEnabled()) log.trace("retriever cache miss key={} q='{}' -> {}", key, shortQ(q), selected.size());
        }
        return selected;
    }

    private boolean gateMatches(Set<String> qSet, Statement st) {
        // Fast path: use scorer.tokens cache if available.
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
            // fallback to cheap text contains
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
     * Deterministic query refinement:
     * - Start with the original query
     * - Add top-N terms from the best candidates (frequency-weighted, order-stable)
     */
    private static String refineQuery(String originalQuery, List<Scored<Statement>> top, int refineTerms) {
        if (refineTerms <= 0) return originalQuery == null ? "" : originalQuery;
        if (top == null || top.isEmpty()) return originalQuery == null ? "" : originalQuery;

        // term -> weight (sum of candidate scores where the term appears)
        HashMap<String, Double> w = new HashMap<>(256);

        // order memory to make tie-break deterministic and stable
        LinkedHashMap<String, Integer> firstSeen = new LinkedHashMap<>(256);

        int docs = 0;
        for (Scored<Statement> sc : top) {
            if (sc == null || sc.item == null || sc.item.text == null) continue;
            docs++;

            List<String> toks = queryTokens(sc.item.text, 3);
            HashSet<String> uniq = new HashSet<>(toks);
            for (String t : uniq) {
                if (t == null || t.isBlank()) continue;
                w.merge(t, Math.max(0.0, sc.score), Double::sum);
                firstSeen.putIfAbsent(t, firstSeen.size());
            }

            // Don't overfit: use only first several top docs.
            if (docs >= 12) break;
        }

        if (w.isEmpty()) return originalQuery == null ? "" : originalQuery;

        ArrayList<String> terms = new ArrayList<>(w.keySet());
        terms.sort((a, b) -> {
            double wa = w.getOrDefault(a, 0.0);
            double wb = w.getOrDefault(b, 0.0);
            int cmp = Double.compare(wb, wa);
            if (cmp != 0) return cmp;
            // stable tie-break: earlier firstSeen wins
            return Integer.compare(firstSeen.getOrDefault(a, Integer.MAX_VALUE), firstSeen.getOrDefault(b, Integer.MAX_VALUE));
        });

        StringBuilder sb = new StringBuilder(256);
        if (originalQuery != null && !originalQuery.isBlank()) {
            sb.append(originalQuery.trim());
            sb.append("\n");
        }
        sb.append("context: ");

        int added = 0;
        for (String t : terms) {
            if (t == null || t.isBlank()) continue;
            if (added++ > 0) sb.append(' ');
            sb.append(t);
            if (added >= refineTerms) break;
        }

        return sb.toString();
    }

    private static double estimateConfidence(List<Scored<Statement>> ranked) {
        if (ranked == null || ranked.isEmpty()) return 0.0;
        double top = ranked.get(0).score;
        if (!Double.isFinite(top) || top <= 0.0) return 0.0;

        double mass = 0.0;
        int m = Math.min(16, ranked.size());
        for (int i = 0; i < m; i++) {
            double s = ranked.get(i).score;
            if (Double.isFinite(s) && s > 0.0) mass += s;
        }
        return mass <= 0.0 ? 0.0 : (top / (mass + 1e-12));
    }

    private static List<String> queryTokens(String q, int minLen) {
        if (q == null || q.isBlank()) return List.of();
        String low = q.toLowerCase(Locale.ROOT);

        String[] parts = low.split("[^\\p{L}\\p{Nd}_]+");
        ArrayList<String> out = new ArrayList<>(Math.min(16, parts.length));
        for (String p : parts) {
            if (p == null) continue;
            String t = p.trim();
            if (t.length() < minLen) continue;
            out.add(t);
            if (out.size() >= 24) break;
        }

        // Dedup while keeping order (stable)
        LinkedHashSet<String> set = new LinkedHashSet<>(out);
        return new ArrayList<>(set);
    }

    private static String dedupKey(Statement st) {
        String id = st.id;
        if (id != null && !id.isBlank()) return "id:" + id;
        String t = st.text;
        if (t == null || t.isBlank()) return null;
        return "tx:" + t.trim();
    }

    private static String shortQ(String q) {
        if (q == null) return "";
        String s = q.replace('\n', ' ').replace('\r', ' ').trim();
        return s.length() <= 120 ? s : s.substring(0, 120);
    }

    private static long mix(long a, long b) {
        long x = a ^ b;
        x ^= (x >>> 33);
        x *= 0xff51afd7ed558ccdL;
        x ^= (x >>> 33);
        return x;
    }
}