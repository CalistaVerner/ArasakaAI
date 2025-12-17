// InMemoryKnowledgeBase.java
package org.calista.arasaka.ai.knowledge;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

/**
 * InMemoryKnowledgeBase — enterprise-grade, детерминированный retrieval engine.
 *
 * <p>
 * Цели:
 * - реальный, объяснимый retrieval (BM25 + сигналы)
 * - итеративный retrieval без раздувания запроса (IDF-based expansion)
 * - детерминизм (score desc, tie-break by id)
 * - TTL (expiresAtEpochMs)
 * - минимум классов: всё важное — внутри этого модуля
 * </p>
 */
public final class InMemoryKnowledgeBase implements KnowledgeBase {

    // =========================
    // Config (без хардкода коэффициентов)
    // =========================

    public static final class Config {
        // Query / evidence limits
        public int maxQueryTokens = 48;
        public int expandTokensPerStep = 12;
        public int maxEvidenceTokensPerStatement = 64;
        public int candidateCap = 50_000; // safety for large KB

        // BM25 params
        public double bm25K1 = 1.2;
        public double bm25B = 0.75;

        // Signal weights
        public double wBm25 = 1.0;
        public double wTag = 0.25;
        public double wRecency = 0.20;
        public double wStrength = 0.15;

        // Diversification (MMR)
        public boolean mmrEnabled = true;
        public double mmrLambda = 0.80;

        // Recency half-life
        public Duration recencyHalfLife = Duration.ofDays(14);
    }

    private final Config cfg;

    // =========================
    // Storage + index
    // =========================

    private final Map<String, Statement> byId = new ConcurrentHashMap<>();
    private final Map<String, Set<String>> invertedIndex = new ConcurrentHashMap<>();

    /**
     * Term frequencies per statement id.
     * Нужны и для BM25, и для "разумной" диверсификации.
     */
    private final Map<String, Map<String, Integer>> tfById = new ConcurrentHashMap<>();

    /** Corpus stats for BM25/IDF (maintained on reindex). */
    private final Map<String, Integer> docFreq = new ConcurrentHashMap<>();
    private volatile long totalDocs = 0L;
    private volatile long totalTokenCount = 0L;

    private final ReentrantReadWriteLock rw = new ReentrantReadWriteLock();

    public InMemoryKnowledgeBase() {
        this(new Config());
    }

    public InMemoryKnowledgeBase(Config cfg) {
        this.cfg = (cfg == null ? new Config() : cfg);
    }

    // =========================
    // Basic operations
    // =========================

    @Override
    public boolean upsert(Statement st) {
        Objects.requireNonNull(st, "st");
        st.validate();
        st.touchUpdatedNow(); // enterprise-friendly default: keep recency honest

        rw.writeLock().lock();
        try {
            Statement prev = byId.put(st.id, st);
            reindex(st);

            if (prev == null) return true;
            if (!Objects.equals(prev.text, st.text)) return true;
            if (!Objects.equals(prev.type, st.type)) return true;
            if (prev.weight != st.weight) return true;
            if (prev.confidence != st.confidence) return true;
            if (!Objects.equals(prev.tags, st.tags)) return true;
            if (prev.expiresAtEpochMs != st.expiresAtEpochMs) return true;
            // timestamps/meta changes are "changes" too
            if (prev.updatedAtEpochMs != st.updatedAtEpochMs) return true;
            if (!Objects.equals(prev.meta, st.meta)) return true;
            return false;
        } finally {
            rw.writeLock().unlock();
        }
    }

    @Override
    public Optional<Statement> get(String id) {
        if (id == null) return Optional.empty();
        rw.readLock().lock();
        try {
            return Optional.ofNullable(byId.get(id));
        } finally {
            rw.readLock().unlock();
        }
    }

    @Override
    public List<Statement> snapshotSorted() {
        rw.readLock().lock();
        try {
            ArrayList<Statement> out = new ArrayList<>(byId.values());
            out.sort(Comparator.comparing(a -> a.id));
            return Collections.unmodifiableList(out);
        } finally {
            rw.readLock().unlock();
        }
    }

    @Override
    public int size() {
        return byId.size();
    }

    // =========================
    // Retrieval
    // =========================

    @Override
    public Query buildQueryFromPrompt(String prompt) {
        LinkedHashSet<String> tokens = new LinkedHashSet<>(tokenizeList(prompt, cfg.maxQueryTokens));
        return new Query(Set.copyOf(tokens), Set.of());
    }

    @Override
    public List<ScoredStatement> search(Query query) {
        Objects.requireNonNull(query, "query");

        rw.readLock().lock();
        try {
            // TTL filtering handled per-statement
            Set<String> candidates = new LinkedHashSet<>();

            // Candidate generation: token hits
            for (String t : query.tokens) {
                Set<String> ids = invertedIndex.get(t);
                if (ids != null) candidates.addAll(ids);
                if (candidates.size() > cfg.candidateCap) break;
            }

            // safety cap (deterministic cut)
            if (candidates.size() > cfg.candidateCap) {
                candidates = candidates.stream()
                        .sorted()
                        .limit(cfg.candidateCap)
                        .collect(Collectors.toCollection(LinkedHashSet::new));
            }

            ArrayList<ScoredStatement> scored = new ArrayList<>(Math.min(candidates.size(), 4096));
            for (String id : candidates) {
                Statement st = byId.get(id);
                if (st == null) continue;
                if (st.isExpiredNow()) continue;

                scored.add(score(st, query));
            }

            scored.sort((a, b) -> {
                int c = Double.compare(b.score, a.score);
                if (c != 0) return c;
                return a.statement.id.compareTo(b.statement.id);
            });

            return Collections.unmodifiableList(scored);
        } finally {
            rw.readLock().unlock();
        }
    }

    @Override
    public RetrievalReport retrieveIterative(String prompt, int iterations, int topK) {
        if (prompt == null) prompt = "";
        if (iterations < 1) iterations = 1;
        if (topK < 1) topK = 1;

        LinkedHashSet<String> tokens = new LinkedHashSet<>(tokenizeList(prompt, cfg.maxQueryTokens));
        LinkedHashSet<String> tags = new LinkedHashSet<>();

        List<RetrievalStep> steps = new ArrayList<>(iterations);

        for (int i = 0; i < iterations; i++) {
            tokens = cap(tokens, cfg.maxQueryTokens);

            Query q = new Query(Set.copyOf(tokens), Set.copyOf(tags));
            List<ScoredStatement> ranked = search(q);

            List<ScoredStatement> evidence = cfg.mmrEnabled
                    ? mmrSelect(ranked, topK, cfg.mmrLambda)
                    : ranked.stream().limit(topK).collect(Collectors.toList());

            steps.add(new RetrievalStep(q, evidence));

            // refine: tags + top informative tokens from evidence
            tags.addAll(extractTags(evidence));
            tokens.addAll(expandByIdf(evidence, cfg.expandTokensPerStep));
        }

        return new RetrievalReport(steps);
    }

    // =========================
    // Scoring
    // =========================

    private ScoredStatement score(Statement s, Query q) {
        Map<String, Integer> docTf = tfById.get(s.id);
        if (docTf == null || docTf.isEmpty()) {
            docTf = termFreq(s.text, cfg.maxEvidenceTokensPerStatement);
        }

        Map<String, Integer> qTf = toQueryTf(q.tokens, cfg.maxQueryTokens);

        double avgLen = Math.max(1.0, (totalDocs <= 0 ? 1.0 : (double) totalTokenCount / (double) totalDocs));
        double bm25 = bm25(docTf, qTf, avgLen);

        double tag = normalizedIntersection(toTagSet(s.tags), q.tags);
        double strength = safe(s.weight) * safe01(s.confidence);
        double recency = recencyScore(s.updatedAtEpochMs, cfg.recencyHalfLife);

        double finalScore =
                cfg.wBm25 * bm25 +
                        cfg.wTag * tag +
                        cfg.wStrength * strength +
                        cfg.wRecency * recency;

        Map<String, Double> features = new LinkedHashMap<>();
        features.put("bm25", bm25);
        features.put("tag", tag);
        features.put("strength", strength);
        features.put("recency", recency);
        features.put("dl", (double) docLen(docTf));
        features.put("avgLen", avgLen);

        return new ScoredStatement(s, finalScore, Collections.unmodifiableMap(features));
    }

    private double bm25(Map<String, Integer> docTf, Map<String, Integer> qTf, double avgLen) {
        int dl = docLen(docTf);
        if (dl <= 0) return 0.0;

        double k1 = cfg.bm25K1;
        double b = cfg.bm25B;

        double sum = 0.0;
        for (String term : qTf.keySet()) {
            Integer f = docTf.get(term);
            if (f == null || f <= 0) continue;

            double idf = idf(term);
            double tf = f.doubleValue();
            double denom = tf + k1 * (1.0 - b + b * (dl / avgLen));
            double frac = (tf * (k1 + 1.0)) / Math.max(1e-9, denom);
            sum += idf * frac;
        }
        return sum;
    }

    private double idf(String term) {
        int df = Math.max(0, docFreq.getOrDefault(term, 0));
        long N = Math.max(1L, totalDocs);
        return Math.log(1.0 + ((N - df + 0.5) / (df + 0.5)));
    }

    // =========================
    // Indexing
    // =========================

    private void reindex(Statement st) {
        Map<String, Integer> newTf = termFreq(st.text, cfg.maxEvidenceTokensPerStatement);
        Map<String, Integer> oldTf = tfById.put(st.id, newTf);

        // remove old
        if (oldTf != null) {
            for (String t : oldTf.keySet()) {
                Set<String> ids = invertedIndex.get(t);
                if (ids != null) {
                    ids.remove(st.id);
                    if (ids.isEmpty()) invertedIndex.remove(t);
                }
                docFreq.computeIfPresent(t, (k, v) -> (v <= 1) ? null : (v - 1));
            }
            totalTokenCount -= docLen(oldTf);
        } else {
            totalDocs += 1;
        }

        // add new
        for (String t : newTf.keySet()) {
            invertedIndex
                    .computeIfAbsent(t, k -> ConcurrentHashMap.newKeySet())
                    .add(st.id);
            docFreq.merge(t, 1, Integer::sum);
        }
        totalTokenCount += docLen(newTf);
    }

    private static int docLen(Map<String, Integer> tf) {
        int sum = 0;
        for (int v : tf.values()) sum += Math.max(0, v);
        return sum;
    }

    // =========================
    // Query refinement helpers
    // =========================

    private static LinkedHashSet<String> cap(LinkedHashSet<String> in, int max) {
        if (in.size() <= max) return in;
        LinkedHashSet<String> out = new LinkedHashSet<>();
        int i = 0;
        for (String s : in) {
            out.add(s);
            if (++i >= max) break;
        }
        return out;
    }

    private static Set<String> extractTags(List<ScoredStatement> evidence) {
        LinkedHashSet<String> out = new LinkedHashSet<>();
        for (ScoredStatement ss : evidence) {
            Statement st = ss.statement;
            if (st.tags != null) out.addAll(st.tags);
        }
        return out;
    }

    private Set<String> expandByIdf(List<ScoredStatement> evidence, int limit) {
        if (limit <= 0 || evidence.isEmpty()) return Set.of();

        Map<String, Double> candidates = new HashMap<>();
        for (ScoredStatement ss : evidence) {
            Map<String, Integer> tf = tfById.get(ss.statement.id);
            if (tf == null) tf = termFreq(ss.statement.text, cfg.maxEvidenceTokensPerStatement);
            for (String t : tf.keySet()) {
                candidates.putIfAbsent(t, idf(t));
            }
        }

        return candidates.entrySet().stream()
                .sorted((a, b) -> {
                    int c = Double.compare(b.getValue(), a.getValue());
                    if (c != 0) return c;
                    return a.getKey().compareTo(b.getKey());
                })
                .limit(limit)
                .map(Map.Entry::getKey)
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }

    // =========================
    // Diversification (MMR) — deterministic
    // =========================

    private List<ScoredStatement> mmrSelect(List<ScoredStatement> ranked, int k, double lambda) {
        if (ranked == null || ranked.isEmpty()) return List.of();
        if (k <= 0) return List.of();

        ArrayList<ScoredStatement> selected = new ArrayList<>(k);
        LinkedHashSet<String> selectedIds = new LinkedHashSet<>();

        // first: best by rank
        ScoredStatement first = ranked.get(0);
        selected.add(first);
        selectedIds.add(first.statement.id);

        while (selected.size() < k) {
            ScoredStatement best = null;
            double bestScore = Double.NEGATIVE_INFINITY;

            for (ScoredStatement cand : ranked) {
                String cid = cand.statement.id;
                if (selectedIds.contains(cid)) continue;

                double rel = cand.score;
                double sim = maxSimilarityToSelected(cid, selectedIds);
                double mmr = lambda * rel - (1.0 - lambda) * sim;

                if (best == null
                        || mmr > bestScore
                        || (mmr == bestScore && cid.compareTo(best.statement.id) < 0)) {
                    bestScore = mmr;
                    best = cand;
                }
            }

            if (best == null) break;
            selected.add(best);
            selectedIds.add(best.statement.id);
        }

        return selected;
    }

    private double maxSimilarityToSelected(String candId, Set<String> selectedIds) {
        Map<String, Integer> a = tfById.get(candId);
        if (a == null || a.isEmpty()) return 0.0;

        Set<String> aKeys = a.keySet();
        double best = 0.0;

        for (String sid : selectedIds) {
            Map<String, Integer> b = tfById.get(sid);
            if (b == null || b.isEmpty()) continue;
            double sim = normalizedIntersection(aKeys, b.keySet());
            if (sim > best) best = sim;
        }
        return best;
    }

    // =========================
    // Tokenization / math (deterministic)
    // =========================

    private static Map<String, Integer> termFreq(String text, int capTokens) {
        List<String> toks = tokenizeList(text, capTokens);
        LinkedHashMap<String, Integer> tf = new LinkedHashMap<>();
        for (String t : toks) tf.merge(t, 1, Integer::sum);
        return tf;
    }

    private static Map<String, Integer> toQueryTf(Set<String> tokens, int max) {
        LinkedHashMap<String, Integer> out = new LinkedHashMap<>();
        int i = 0;
        for (String t : tokens) {
            out.put(t, 1);
            if (++i >= max) break;
        }
        return out;
    }

    private static List<String> tokenizeList(String text, int capTokens) {
        if (text == null || text.isBlank()) return List.of();
        String normalized = text
                .toLowerCase(Locale.ROOT)
                .replace('\u0451', '\u0435') // ё -> е
                .replaceAll("[^a-zа-я0-9 ]", " ");

        String[] parts = normalized.split("\\s+");
        ArrayList<String> out = new ArrayList<>(Math.min(parts.length, capTokens));
        for (String p : parts) {
            if (p == null) continue;
            if (p.length() < 3) continue;
            out.add(p);
            if (out.size() >= capTokens) break;
        }
        return out;
    }

    private static Set<String> toTagSet(List<String> tags) {
        if (tags == null || tags.isEmpty()) return Set.of();
        LinkedHashSet<String> out = new LinkedHashSet<>();
        for (String t : tags) {
            if (t == null) continue;
            String x = t.trim().toLowerCase(Locale.ROOT);
            if (x.isBlank()) continue;
            out.add(x);
        }
        return out;
    }

    private static double normalizedIntersection(Set<String> a, Set<String> b) {
        if (a == null || a.isEmpty() || b == null || b.isEmpty()) return 0.0;

        int inter = 0;
        // iterate smaller for speed, deterministic anyway
        Set<String> small = a.size() <= b.size() ? a : b;
        Set<String> big = a.size() <= b.size() ? b : a;

        for (String x : small) if (big.contains(x)) inter++;

        double denom = Math.sqrt((double) a.size() * (double) b.size());
        if (denom <= 0) return 0.0;
        return inter / denom;
    }

    private static double recencyScore(long updatedAtEpochMs, Duration halfLife) {
        if (updatedAtEpochMs <= 0) return 0.0;
        long ageMs = Math.max(0L, System.currentTimeMillis() - updatedAtEpochMs);
        double hl = Math.max(1.0, (double) (halfLife == null ? Duration.ofDays(14).toMillis() : halfLife.toMillis()));
        return Math.exp(-ageMs / hl);
    }

    private static double safe(double v) {
        if (!Double.isFinite(v)) return 0.0;
        return Math.max(0.0, v);
    }

    private static double safe01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        return Math.max(0.0, Math.min(1.0, v));
    }
}