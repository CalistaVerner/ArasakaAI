package org.calista.arasaka.ai.retrieve;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.Statement;

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

        // --- Prefilter: cheap, deterministic ---
        // Take up to 8 meaningful query tokens and require at least one to appear in statement text (lowercase contains).
        // This is a performance optimization before calling the heavier scorer.
        final List<String> qTok = queryTokens(q);

        List<Statement> all = kb.snapshotSorted(); // stable base order
        ArrayList<Scored<Statement>> scored = new ArrayList<>(Math.max(16, all.size() / 6));

        Set<String> seen = new HashSet<>(Math.min(4096, all.size()));

        for (Statement st : all) {
            if (st == null) continue;

            String dedupKey = dedupKey(st);
            if (dedupKey != null && !dedupKey.isBlank() && !seen.add(dedupKey)) continue;

            if (!qTok.isEmpty() && !cheapMatches(qTok, st.text)) continue;

            double s = scorer.score(q, st);
            if (s > 0.0) scored.add(new Scored<>(st, s));
        }

        scored.sort(Comparator
                .comparingDouble((Scored<Statement> x) -> x.score).reversed()
                .thenComparing(x -> x.item.id == null ? "" : x.item.id));

        List<Statement> selected = exploration.select(scored, kk, explorationCfg, seed);

        if (cacheCapacity > 0) {
            cache.put(key, selected);
            if (log.isTraceEnabled()) log.trace("retriever cache miss key={} q='{}' -> {}", key, shortQ(q), selected.size());
        }
        return selected;
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