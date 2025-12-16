package org.calista.arasaka.ai.retrieve.retriver.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationConfig;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationStrategy;
import org.calista.arasaka.ai.retrieve.Scored;
import org.calista.arasaka.ai.retrieve.scorer.Scorer;
import org.calista.arasaka.ai.retrieve.retriver.Retriever;

import java.util.*;
import java.util.stream.Collectors;

public final class KnowledgeRetriever implements Retriever {
    private static final Logger log = LogManager.getLogger(KnowledgeRetriever.class);

    private final KnowledgeBase kb;
    private final Scorer scorer;
    private final ExplorationStrategy exploration;
    private final ExplorationConfig explorationCfg;

    private final Map<Long, List<Statement>> cache;
    private final int cacheCapacity;

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

        // ------------------------------
        // "Thinking" retrieval: multi-iteration, deterministic, cacheable.
        // ------------------------------
        // Idea:
        //  1) run N iterations (ExplorationConfig#iterations)
        //  2) each iteration refines the query by extracting strong terms from the best candidates so far
        //  3) aggregate scores with decay (ExplorationConfig#iterationDecay)
        //  4) exploration stage does deterministic diversity-aware selection (SoftmaxSampler)

        // Stable base order + deterministic de-dup (so we can score the same statement in multiple iterations).
        List<Statement> allRaw = kb.snapshotSorted();
        ArrayList<Statement> all = new ArrayList<>(allRaw.size());
        Set<String> dedup = new HashSet<>(Math.min(4096, allRaw.size()));
        for (Statement st : allRaw) {
            if (st == null) continue;
            String dk = dedupKey(st);
            if (dk != null && !dk.isBlank() && !dedup.add(dk)) continue;
            all.add(st);
        }

        // Accumulate across iterations: Statement -> aggregated score
        HashMap<Statement, Double> agg = new HashMap<>(Math.max(256, all.size() / 8));
        ArrayList<Scored<Statement>> lastTop = new ArrayList<>(Math.min(256, Math.max(16, kk * 4)));

        String iterQuery = q;
        double iterWeight = 1.0;

        for (int iter = 0; iter < explorationCfg.iterations; iter++) {
            final List<String> qTok = queryTokens(iterQuery);
            ArrayList<Scored<Statement>> scored = new ArrayList<>(Math.max(16, all.size() / 6));

            for (Statement st : all) {
                if (st == null) continue;
                if (st.text == null || st.text.isBlank()) continue;

                if (!qTok.isEmpty() && !cheapMatches(qTok, st.text)) continue;

                double s = scorer.score(iterQuery, st);
                if (!Double.isFinite(s) || s < explorationCfg.minScore) continue;

                double weighted = s * iterWeight;
                scored.add(new Scored<>(st, weighted));
                agg.merge(st, weighted, Double::sum);
            }

            scored.sort(Comparator
                    .comparingDouble((Scored<Statement> x) -> x.score).reversed()
                    .thenComparing(x -> x.item.id == null ? "" : x.item.id));

            // Keep a small band of best candidates for query refinement.
            lastTop.clear();
            int band = Math.min(Math.max(16, kk * 4), scored.size());
            for (int i = 0; i < band; i++) lastTop.add(scored.get(i));

            // Refine query for the next iteration using the strongest terms found so far.
            if (iter + 1 < explorationCfg.iterations) {
                iterQuery = refineQuery(q, lastTop);
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

        List<Statement> selected = exploration.select(finalRanked, kk, explorationCfg, seed);

        if (cacheCapacity > 0) {
            cache.put(key, selected);
            if (log.isTraceEnabled()) log.trace("retriever cache miss key={} q='{}' -> {}", key, shortQ(q), selected.size());
        }
        return selected;
    }

    /**
     * Deterministic query refinement: start with the original query and append a small set of
     * high-signal terms from the current best candidates.
     */
    private static String refineQuery(String originalQuery, List<Scored<Statement>> top) {
        if (top == null || top.isEmpty()) return originalQuery == null ? "" : originalQuery;

        // Collect up to 12 unique terms from the top statements.
        LinkedHashSet<String> terms = new LinkedHashSet<>();
        for (Scored<Statement> sc : top) {
            if (sc == null || sc.item == null || sc.item.text == null) continue;
            for (String t : queryTokens(sc.item.text)) {
                terms.add(t);
                if (terms.size() >= 12) break;
            }
            if (terms.size() >= 12) break;
        }

        if (terms.isEmpty()) return originalQuery == null ? "" : originalQuery;

        StringBuilder sb = new StringBuilder(256);
        if (originalQuery != null && !originalQuery.isBlank()) {
            sb.append(originalQuery.trim());
            sb.append("\n");
        }
        sb.append("context: ");
        int i = 0;
        for (String t : terms) {
            if (i++ > 0) sb.append(' ');
            sb.append(t);
        }
        return sb.toString();
    }

    private static boolean cheapMatches(List<String> qTok, String text) {
        if (text == null || text.isBlank()) return false;
        String low = text.toLowerCase(Locale.ROOT);
        for (String t : qTok) {
            if (t.length() >= 3 && low.contains(t)) return true;
        }
        return false;
    }

    private static List<String> queryTokens(String q) {
        if (q == null || q.isBlank()) return List.of();
        String low = q.toLowerCase(Locale.ROOT);

        // Simple tokenization; deterministic and cheap.
        String[] parts = low.split("[^\\p{L}\\p{Nd}_]+");
        ArrayList<String> out = new ArrayList<>(Math.min(8, parts.length));
        for (String p : parts) {
            if (p == null) continue;
            String t = p.trim();
            if (t.length() < 3) continue;
            out.add(t);
            if (out.size() >= 8) break;
        }
        // Dedup while keeping order:
        return out.stream().distinct().collect(Collectors.toList());
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