package org.calista.arasaka.ai.think;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;

import java.text.Normalizer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collectors;

/**
 * NeuronGraph — разреженный детерминированный граф признаков (feature graph).
 *
 * Это НЕ "нейросеть ради нейросети".
 * Это enterprise-узлы и веса, которые:
 *  - обучаются online (reward/penalty) без стохастики и без магии
 *  - дают динамическую добавку к AdvancedIntentDetector (feature -> intent)
 *  - умеют расширять запросы retrieval (token -> token) детерминированно
 *  - связывают token -> evidence-statement keys (token -> stmt:key) для будущего усиления recall/retrieval
 *
 * Устойчивость:
 *  - bounded weights
 *  - легкий decay per update
 *  - стабильные ключи statement: stmt:id:<id> или stmt:tx:<normalized_text>
 *
 * Производительность:
 *  - ConcurrentHashMap (lock-free in practice)
 *  - только sparse edges, никаких матриц
 */
public final class NeuronGraph {

    private static final Logger log = LogManager.getLogger(NeuronGraph.class);

    // ----------------------------- node prefixes -----------------------------

    /** Feature nodes: f:tok:hello, f:bg:добрый_день, f:sig:qm */
    public static final String P_FEATURE = "f:";

    /** Token nodes: t:hello (for query expansion) */
    public static final String P_TOKEN = "t:";

    /** Statement nodes: stmt:id:123 or stmt:tx:<normalized> */
    public static final String P_STMT = "stmt:";

    /** Intent nodes: intent:QUESTION */
    public static final String P_INTENT = "intent:";

    // ----------------------------- storage -----------------------------

    /** edges[from][to] = weight */
    private final ConcurrentMap<String, ConcurrentMap<String, Double>> edges = new ConcurrentHashMap<>();

    /** lightweight node stats (strength gate + lastSeen) */
    private final ConcurrentMap<String, NodeStats> stats = new ConcurrentHashMap<>();

    // ----------------------------- params -----------------------------

    private final double lrPos;
    private final double lrNeg;
    private final double maxAbsW;
    private final double decayPerUpdate;

    // optional caps to avoid map explosion
    private final int maxOutDegree;
    private final int maxNodes;

    public NeuronGraph() {
        this(0.08, 0.06, 2.50, 0.0015, 256, 200_000);
    }

    public NeuronGraph(double lrPos,
                       double lrNeg,
                       double maxAbsW,
                       double decayPerUpdate,
                       int maxOutDegree,
                       int maxNodes) {
        this.lrPos = clamp(lrPos, 0.0, 1.0);
        this.lrNeg = clamp(lrNeg, 0.0, 1.0);
        this.maxAbsW = clamp(maxAbsW, 0.1, 25.0);
        this.decayPerUpdate = clamp(decayPerUpdate, 0.0, 0.1);

        this.maxOutDegree = Math.max(16, maxOutDegree);
        this.maxNodes = Math.max(10_000, maxNodes);

        if (log.isInfoEnabled()) {
            log.info("NeuronGraph init: lrPos={} lrNeg={} maxAbsW={} decay={} maxOutDegree={} maxNodes={}",
                    fmt(lrPos), fmt(lrNeg), fmt(maxAbsW), fmt(decayPerUpdate), this.maxOutDegree, this.maxNodes);
        }
    }

    // ----------------------------- public API -----------------------------

    /**
     * Dynamic intent score from learned graph: sum(feature -> intent).
     * Returns small bounded delta so static weights remain dominant.
     *
     * @param intentName  intent enum name
     * @param features    features produced by detector (tok/bg/sig...)
     * @param tokenCounts optional token counts (used to scale tok:* features)
     */
    public double scoreIntent(String intentName,
                              List<String> features,
                              Map<String, Integer> tokenCounts) {
        if (intentName == null || intentName.isBlank()) return 0.0;
        if (features == null || features.isEmpty()) return 0.0;

        String intentNode = P_INTENT + intentName;
        double sum = 0.0;

        for (String f : features) {
            if (f == null || f.isBlank()) continue;
            String from = P_FEATURE + f;

            double w = edge(from, intentNode);
            if (Math.abs(w) < 1e-12) continue;

            // scale token features by count (if provided)
            double mul = 1.0;
            if (tokenCounts != null && f.startsWith("tok:")) {
                String tok = f.substring("tok:".length());
                mul = Math.max(1, tokenCounts.getOrDefault(tok, 1));
            }

            double gate = stats.getOrDefault(from, NodeStats.DEFAULT).strength;
            sum += (w * mul * gate);
        }

        // bounded: do not overpower static detector weights
        return clamp(sum, -1.5, 1.5);
    }

    /**
     * Suggest extra query terms from token nodes.
     * Deterministic: highest |weight| then lexicographic tie-break.
     */
    public List<String> suggestQueryTerms(Set<String> tokens, int budget) {
        int b = Math.max(0, budget);
        if (b == 0 || tokens == null || tokens.isEmpty()) return List.of();

        HashMap<String, Double> cand = new HashMap<>();

        for (String t : tokens) {
            if (t == null || t.isBlank()) continue;

            String from = P_TOKEN + t;
            Map<String, Double> out = edges.get(from);
            if (out == null || out.isEmpty()) continue;

            // gather only token->token edges
            for (Map.Entry<String, Double> e : out.entrySet()) {
                String to = e.getKey();
                if (to == null || !to.startsWith(P_TOKEN)) continue;

                double w = e.getValue() == null ? 0.0 : e.getValue();
                if (Math.abs(w) < 1e-9) continue;

                String term = to.substring(P_TOKEN.length());
                if (term.isBlank()) continue;
                if (tokens.contains(term)) continue;

                cand.merge(term, w, Double::sum);
            }
        }

        return cand.entrySet().stream()
                .sorted((a, b2) -> {
                    int cmp = Double.compare(Math.abs(b2.getValue()), Math.abs(a.getValue()));
                    if (cmp != 0) return cmp;
                    // stable order
                    return a.getKey().compareTo(b2.getKey());
                })
                .limit(b)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    /**
     * Update graph after an iteration: deterministic online learning.
     *
     * @param queryTokens  normalized query tokens (unique set)
     * @param features     detector features (tok/bg/sig...)
     * @param intentName   chosen intent enum name
     * @param evidenceKeys stable evidence keys (id:... or tx:...)
     * @param reward       positive (good/valid) or negative (bad/invalid), range suggested [-1..+1]
     */
    public void update(Set<String> queryTokens,
                       List<String> features,
                       String intentName,
                       List<String> evidenceKeys,
                       double reward) {

        if ((queryTokens == null || queryTokens.isEmpty())
                && (features == null || features.isEmpty())
                && (evidenceKeys == null || evidenceKeys.isEmpty())) {
            return;
        }

        // prevent uncontrolled growth
        if (stats.size() > maxNodes) {
            // light self-protection: decay all node strengths a bit and refuse adding too much
            if (log.isWarnEnabled()) log.warn("NeuronGraph: node cap reached ({}), applying soft pressure", maxNodes);
            softPressure();
        }

        final boolean pos = reward >= 0.0;
        final double lr = pos ? lrPos : lrNeg;
        final double mag = clamp(Math.abs(reward), 0.0, 1.0);
        final double delta = lr * mag;

        // 1) Feature -> intent links (adapts intent scoring)
        if (intentName != null && !intentName.isBlank() && features != null && !features.isEmpty()) {
            String intentNode = P_INTENT + intentName;
            touch(intentNode);

            for (String f : features) {
                if (f == null || f.isBlank()) continue;

                String from = P_FEATURE + f;
                touch(from);

                bump(from, intentNode, pos ? +delta : -delta);
            }
        }

        // 2) Token <-> token co-activation (helps query expansion)
        if (queryTokens != null && queryTokens.size() >= 2) {
            ArrayList<String> toks = new ArrayList<>(queryTokens);
            toks.removeIf(x -> x == null || x.isBlank());
            toks.sort(String::compareTo);

            // small complete graph on tokens (O(n^2), but n is small due to token limits upstream)
            for (int i = 0; i < toks.size(); i++) {
                String ti = toks.get(i);
                String ni = P_TOKEN + ti;
                touch(ni);

                for (int j = i + 1; j < toks.size(); j++) {
                    String tj = toks.get(j);
                    String nj = P_TOKEN + tj;

                    // weaker than intent updates
                    double dd = delta * 0.5;
                    bump(ni, nj, pos ? +dd : -dd);
                    bump(nj, ni, pos ? +dd : -dd);
                }
            }
        }

        // 3) Token -> evidence statement links (future: stronger recall by stmt keys)
        if (queryTokens != null && !queryTokens.isEmpty() && evidenceKeys != null && !evidenceKeys.isEmpty()) {
            ArrayList<String> ek = new ArrayList<>(evidenceKeys);
            ek.removeIf(x -> x == null || x.isBlank());
            Collections.sort(ek);

            for (String t : queryTokens) {
                if (t == null || t.isBlank()) continue;

                String from = P_TOKEN + t;
                touch(from);

                for (String k : ek) {
                    String to = P_STMT + k;
                    touch(to);
                    bump(from, to, pos ? +delta : -delta);
                }
            }
        }

        // cleanup / enforce out-degree caps (keep sparse)
        if (pos || mag > 0.40) {
            // only sometimes, avoids overhead
            enforceCaps(queryTokens, features, intentName);
        }
    }

    /**
     * Convenience: derive stable evidence keys from statements.
     * Uses id if present; else normalized text.
     */
    public static List<String> stableEvidenceKeys(List<Statement> stmts) {
        if (stmts == null || stmts.isEmpty()) return List.of();
        ArrayList<String> out = new ArrayList<>(stmts.size());
        for (Statement s : stmts) {
            if (s == null) continue;

            String key = stableKeyOfStatement(s);
            if (key != null) out.add(key);
        }
        out.sort(String::compareTo);
        return out;
    }

    /**
     * Stable key for a statement:
     * - id:<id>  if id exists
     * - tx:<normalized text> otherwise
     */
    public static String stableKeyOfStatement(Statement s) {
        if (s == null) return null;
        if (s.id != null && !s.id.isBlank()) return "id:" + s.id.trim();
        if (s.text == null || s.text.isBlank()) return null;

        String tx = normalizeTextKey(s.text);
        if (tx.isBlank()) return null;
        return "tx:" + tx;
    }

    // ----------------------------- internals -----------------------------

    private void bump(String from, String to, double dw) {
        if (from == null || to == null || from.isBlank() || to.isBlank()) return;

        // ensure nodes exist
        touch(from);
        touch(to);

        // apply per-update decay to involved nodes
        decayNode(from);
        decayNode(to);

        ConcurrentMap<String, Double> out = edges.computeIfAbsent(from, k -> new ConcurrentHashMap<>());
        out.compute(to, (k, old) -> {
            double v = (old == null ? 0.0 : old) + dw;
            return clamp(v, -maxAbsW, maxAbsW);
        });

        // keep out-degree sparse
        if (out.size() > maxOutDegree) trimOut(from, out, maxOutDegree);
    }

    private double edge(String from, String to) {
        Map<String, Double> out = edges.get(from);
        if (out == null) return 0.0;
        Double v = out.get(to);
        return v == null ? 0.0 : v;
    }

    private void touch(String id) {
        if (id == null || id.isBlank()) return;
        stats.computeIfAbsent(id, k -> new NodeStats()).touch();
    }

    private void decayNode(String id) {
        if (id == null || id.isBlank()) return;
        NodeStats st = stats.computeIfAbsent(id, k -> new NodeStats());
        st.decay(decayPerUpdate);
    }

    private void trimOut(String from, ConcurrentMap<String, Double> out, int keep) {
        if (out == null || out.size() <= keep) return;

        // deterministic trim: keep strongest by |w|, tie -> lexicographic
        List<Map.Entry<String, Double>> list = new ArrayList<>(out.entrySet());
        list.sort((a, b) -> {
            double wa = a.getValue() == null ? 0.0 : a.getValue();
            double wb = b.getValue() == null ? 0.0 : b.getValue();
            int cmp = Double.compare(Math.abs(wb), Math.abs(wa));
            if (cmp != 0) return cmp;
            return a.getKey().compareTo(b.getKey());
        });

        for (int i = keep; i < list.size(); i++) {
            out.remove(list.get(i).getKey());
        }
    }

    private void enforceCaps(Set<String> queryTokens, List<String> features, String intentName) {
        // ensure caps for a few hot nodes only (deterministic)
        ArrayList<String> hot = new ArrayList<>(32);

        if (intentName != null && !intentName.isBlank()) hot.add(P_INTENT + intentName);

        if (queryTokens != null) {
            ArrayList<String> t = new ArrayList<>(queryTokens);
            t.removeIf(x -> x == null || x.isBlank());
            t.sort(String::compareTo);
            for (int i = 0; i < Math.min(8, t.size()); i++) hot.add(P_TOKEN + t.get(i));
        }

        if (features != null && !features.isEmpty()) {
            ArrayList<String> f = new ArrayList<>(features);
            f.removeIf(x -> x == null || x.isBlank());
            Collections.sort(f);
            for (int i = 0; i < Math.min(12, f.size()); i++) hot.add(P_FEATURE + f.get(i));
        }

        for (String node : hot) {
            ConcurrentMap<String, Double> out = edges.get(node);
            if (out != null && out.size() > maxOutDegree) trimOut(node, out, maxOutDegree);
        }
    }

    /**
     * If graph explodes, apply mild strength decay on nodes, and trim some edges.
     * Deterministic: trims lexicographically among weakest.
     */
    private void softPressure() {
        // reduce strength everywhere a bit (cheap)
        for (NodeStats st : stats.values()) {
            if (st != null) st.decay(0.01);
        }

        // trim edges maps that are too large (cheap pass)
        for (Map.Entry<String, ConcurrentMap<String, Double>> e : edges.entrySet()) {
            ConcurrentMap<String, Double> out = e.getValue();
            if (out != null && out.size() > maxOutDegree) trimOut(e.getKey(), out, maxOutDegree);
        }
    }

    // ----------------------------- utils -----------------------------

    private static String normalizeTextKey(String s) {
        if (s == null) return "";
        String x = s.trim();
        if (x.isEmpty()) return "";
        x = Normalizer.normalize(x, Normalizer.Form.NFKC);
        x = x.replace('\u00A0', ' ');
        x = x.replaceAll("\\s+", " ").trim();
        return x;
    }

    private static double clamp(double v, double lo, double hi) {
        if (!Double.isFinite(v)) return 0.0;
        return Math.max(lo, Math.min(hi, v));
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }

    // ----------------------------- node stats -----------------------------

    private static final class NodeStats {
        static final NodeStats DEFAULT = new NodeStats(true);

        volatile double strength = 1.0;
        volatile long lastSeenNanos = System.nanoTime();

        NodeStats() {}

        NodeStats(boolean immutableDefault) {
            if (immutableDefault) {
                strength = 1.0;
                lastSeenNanos = 0L;
            }
        }

        void touch() { lastSeenNanos = System.nanoTime(); }

        void decay(double d) {
            double s = strength;
            s *= (1.0 - d);
            if (s < 0.25) s = 0.25;
            strength = s;
        }
    }

    // ----------------------------- diagnostics -----------------------------

    public int nodeCount() { return stats.size(); }

    public int edgeFromCount() { return edges.size(); }

    public int totalEdgeCountApprox() {
        // approximate: iterate quickly
        int sum = 0;
        for (Map<String, Double> m : edges.values()) {
            if (m != null) sum += m.size();
        }
        return sum;
    }

    @Override
    public String toString() {
        return "NeuronGraph{nodes=" + nodeCount()
                + ", fromNodes=" + edgeFromCount()
                + ", edges~=" + totalEdgeCountApprox()
                + ", maxOut=" + maxOutDegree
                + '}';
    }
}