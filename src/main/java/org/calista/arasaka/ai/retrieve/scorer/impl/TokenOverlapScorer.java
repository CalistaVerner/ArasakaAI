package org.calista.arasaka.ai.retrieve.scorer.impl;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.scorer.Scorer;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Enterprise-grade token overlap scorer:
 * - prepares DF/IDF stats once per corpus (prepare)
 * - caches per-statement token arrays (tokens)
 * - scores with IDF-weighted cosine-like overlap (stable + deterministic)
 */
public final class TokenOverlapScorer implements Scorer {
    private final Tokenizer tokenizer;

    // statementKey -> tokens
    private final Map<String, String[]> tokCache = new ConcurrentHashMap<>(64_000);

    // token -> idf
    private volatile Map<String, Double> idf = Map.of();
    private volatile boolean prepared = false;

    public TokenOverlapScorer(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    }

    @Override
    public void prepare(Collection<Statement> corpus) {
        if (prepared) return;
        if (corpus == null || corpus.isEmpty()) {
            prepared = true;
            return;
        }

        HashMap<String, Integer> df = new HashMap<>(64_000);
        int docs = 0;

        for (Statement st : corpus) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            docs++;

            String[] toks = tokenizeToArray(st.text);
            tokCache.put(keyOf(st), toks);

            HashSet<String> uniq = new HashSet<>(toks.length * 2);
            for (String t : toks) if (t != null && t.length() >= 3) uniq.add(t);
            for (String t : uniq) df.merge(t, 1, Integer::sum);
        }

        final double N = Math.max(1.0, (double) docs);
        HashMap<String, Double> idfLocal = new HashMap<>(df.size() * 2);
        for (var e : df.entrySet()) {
            double v = Math.log((N + 1.0) / (e.getValue() + 1.0)) + 1.0;
            idfLocal.put(e.getKey(), v);
        }

        this.idf = Collections.unmodifiableMap(idfLocal);
        this.prepared = true;
    }

    @Override
    public String[] tokens(Statement st) {
        if (st == null) return null;
        return tokCache.computeIfAbsent(keyOf(st), k -> tokenizeToArray(st.text));
    }

    @Override
    public double score(String query, Statement st) {
        if (st == null || st.text == null || st.text.isBlank()) return 0.0;
        if (query == null || query.isBlank()) return 0.0;

        String[] qt = tokenizeToArray(query);
        String[] dt = tokens(st);
        if (qt.length == 0 || dt == null || dt.length == 0) return 0.0;

        HashMap<String, Double> qw = new HashMap<>(qt.length * 2);
        double qnorm2 = 0.0;
        for (String t : qt) {
            if (t == null || t.length() < 3) continue;
            double w = idf.getOrDefault(t, 1.0);
            qw.merge(t, w, Double::sum);
        }
        for (double w : qw.values()) qnorm2 += w * w;

        double dot = 0.0;
        double dnorm2 = 0.0;

        HashMap<String, Double> dw = new HashMap<>(Math.min(256, dt.length * 2));
        for (String t : dt) {
            if (t == null || t.length() < 3) continue;
            double w = idf.getOrDefault(t, 1.0);
            dw.merge(t, w, Double::sum);
        }

        for (var e : dw.entrySet()) {
            double w = e.getValue();
            dnorm2 += w * w;
            Double qv = qw.get(e.getKey());
            if (qv != null) dot += qv * w;
        }

        if (dot <= 0.0) return 0.0;
        double denom = Math.sqrt(qnorm2) * Math.sqrt(dnorm2);
        if (denom <= 0.0) return 0.0;

        double cosineLike = dot / denom;

        double wst = Double.isFinite(st.weight) ? st.weight : 1.0;
        if (wst < 0.0) wst = 0.0;

        return cosineLike * wst;
    }

    private String[] tokenizeToArray(String text) {
        if (text == null || text.isBlank()) return new String[0];
        List<String> t = tokenizer.tokenize(text);
        if (t == null || t.isEmpty()) return new String[0];

        ArrayList<String> out = new ArrayList<>(t.size());
        for (String s : t) {
            if (s == null) continue;
            String x = s.trim().toLowerCase(Locale.ROOT);
            if (x.length() < 3) continue;
            out.add(x);
        }
        return out.toArray(new String[0]);
    }

    private static String keyOf(Statement st) {
        if (st.id != null && !st.id.isBlank()) return "id:" + st.id;
        return "ihc:" + System.identityHashCode(st);
    }
}