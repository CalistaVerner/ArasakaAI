package org.calista.arasaka.ai.think.textGenerator.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * BigramBeamTextGenerator (thinking-cycle + retrieval + quality)
 *
 * Key upgrades (no new files/classes outside this file):
 * - Multi-iteration thinking cycle: retrieval -> draft -> validate -> improve -> pick best
 * - Deterministic draft diversity via draftIndex+iter rotation (no randomness)
 * - Context retrieval: build graph from most relevant Statements first (less noise, more focus)
 * - Quality scoring: balances context-coverage, uniqueness, query-coverage, anti-echo, connector ratio
 * - All tunable via state.tags without hardcoding scenario logic
 */
public final class BigramBeamTextGenerator implements TextGenerator {

    private static final Logger log = LogManager.getLogger(BigramBeamTextGenerator.class);

    private static final Pattern HAS_CYR = Pattern.compile(".*\\p{IsCyrillic}.*");
    private static final Pattern HAS_LAT = Pattern.compile(".*\\p{IsLatin}.*");

    private static final String STOP = "<eos>";

    private static final List<String> CONNECT_RU = List.of(
            "и", "но", "или", "поэтому", "так", "тоже", "это", "как", "что", "чтобы", "если"
    );

    private static final List<String> CONNECT_EN = List.of(
            "and", "or", "but", "so", "also", "this", "that", "as", "to", "if", "then"
    );

    private static final Set<String> BAN = Set.of(
            "unknown", "noctx", "no_context", "notes", "fix", "err", "invalid",

            "q_ent", "q_coh", "q_mincoh", "q_v",
            "sec", "md", "md_on", "st", "qc", "rep", "echo", "nov",
            "g", "r", "nd", "pd", "sp",
            "g_best_i", "g_best", "g_topk", "g_topk_mean", "nov_n", "nov_t", "rep_u", "rep_t"
    );

    private final Tokenizer tokenizer;

    private final int beamWidth;
    private final int minTokens;
    private final int maxTokens;

    private final int minUniqueTokens;
    private final int maxSameTokenRun;

    private final double repeatPenalty;
    private final double queryRepeatPenalty;
    private final double connectorPenalty;

    // --- anti-echo knobs ---
    private final double maxEchoRatio;        // if answer shares too much with query -> penalize/avoid EOS
    private final double queryTokenPenalty;   // always penalize query tokens a bit (prevents echo loops)

    public BigramBeamTextGenerator(Tokenizer tokenizer) {
        this(tokenizer, 12, 10, 32, 8, 2, 0.65, 1.15, 0.10,
                0.55, 0.12);
    }

    public BigramBeamTextGenerator(
            Tokenizer tokenizer,
            int beamWidth,
            int minTokens,
            int maxTokens,
            int minUniqueTokens,
            int maxSameTokenRun,
            double repeatPenalty,
            double queryRepeatPenalty,
            double connectorPenalty,
            double maxEchoRatio,
            double queryTokenPenalty
    ) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.beamWidth = clampInt(beamWidth, 1, 64);
        this.minTokens = clampInt(minTokens, 1, 64);
        this.maxTokens = clampInt(maxTokens, this.minTokens, 96);

        this.minUniqueTokens = clampInt(minUniqueTokens, 1, 64);
        this.maxSameTokenRun = clampInt(maxSameTokenRun, 1, 8);

        this.repeatPenalty = clamp01(repeatPenalty);
        this.queryRepeatPenalty = clamp01(queryRepeatPenalty);
        this.connectorPenalty = clamp01(connectorPenalty);

        this.maxEchoRatio = clamp01(maxEchoRatio);
        this.queryTokenPenalty = clamp01(queryTokenPenalty);
    }

    @Override
    public String generate(String userTextOrPrompt, List<Statement> context, ThoughtState state) {
        final long t0 = System.nanoTime();

        final String prompt = userTextOrPrompt == null ? "" : userTextOrPrompt.trim();
        final String userText = extractUser(prompt);
        final List<Statement> ctx = (context == null) ? List.of() : context;

        final Lang lang = detectLangPreferUser(userText, ctx);
        final List<String> connectors = (lang == Lang.RU) ? CONNECT_RU : CONNECT_EN;

        final Set<String> qTok = asTokenSet(tokenizer.tokenize(userText));

        // Allow deterministic runtime overrides via tags (engine/config can tune without changing code)
        int minTok = this.minTokens;
        int maxTok = this.maxTokens;
        if (state != null && state.tags != null) {
            minTok = clampInt(safeInt(state.tags.get("gen.minTok"), minTok), 4, 96);
            maxTok = clampInt(safeInt(state.tags.get("gen.maxTok"), maxTok), minTok, 160);
        }

        // If query is very short/identity-like, require more substance to avoid 3-4 token echo.
        if (qTok.size() <= 5) {
            minTok = Math.max(minTok, 18);
            maxTok = Math.max(maxTok, 36);
        }

        // --- NO CONTEXT: configurable fallback (no chains) ---
        if (ctx.isEmpty()) {
            String out = noContextReply(userText, lang, state);
            if (log.isDebugEnabled()) {
                long nanos = System.nanoTime() - t0;
                log.debug("BigramBeamTextGenerator | noctx=1 lang={} qTok={} dtMs={} out='{}'",
                        lang, qTok.size(), nanos / 1_000_000L, snippet(out, 180));
            }
            return out;
        }

        // --- THINKING CYCLE: retrieval -> draft -> validate -> improve (deterministic) ---
        int iters = 4;
        int ctxLimit = 24;
        if (state != null && state.tags != null) {
            iters = clampInt(safeInt(state.tags.get("gen.iters"), iters), 1, 8);
            ctxLimit = clampInt(safeInt(state.tags.get("gen.ctxLimit"), ctxLimit), 8, 160);
        }

        ArrayList<Draft> drafts = new ArrayList<>(iters);

        for (int iter = 0; iter < iters; iter++) {
            int lim = Math.min(ctx.size(), ctxLimit + iter * 12);

            // retrieval: pick most relevant statements first
            List<Statement> picked = selectRelevantContext(ctx, qTok, lim);
            if (picked.isEmpty()) picked = ctx;

            // generate draft
            Graph g = buildGraph(picked, connectors);

            int baseRot = (state == null) ? 0 : Math.max(0, state.draftIndex);
            int rot = baseRot + iter;

            Beam b = runBeam(g, qTok, connectors, minTok, maxTok, rot);
            if (b == null) continue;

            Quality q = evaluateQuality(userText, qTok, b.tokens, connectors, picked, minTok);
            drafts.add(new Draft(b, q, iter, picked.size()));

            // early exit if excellent
            if (q.total >= 0.90 && q.echoRatio <= (maxEchoRatio * 0.75)) break;
        }

        Draft bestDraft = drafts.stream()
                .sorted((a, b) -> {
                    int cmp = Double.compare(b.q.total, a.q.total);
                    if (cmp != 0) return cmp;
                    // tie-breakers: lower echo, higher context coverage, earlier iter
                    cmp = Double.compare(a.q.echoRatio, b.q.echoRatio);
                    if (cmp != 0) return cmp;
                    cmp = Double.compare(b.q.contextCoverage, a.q.contextCoverage);
                    if (cmp != 0) return cmp;
                    return Integer.compare(a.iter, b.iter);
                })
                .findFirst()
                .orElse(null);

        Beam best = (bestDraft == null) ? null : bestDraft.beam;
        String out = (best == null) ? "" : detokenize(best.tokens, lang);
        out = postFix(userText, out, lang);

        if (log.isDebugEnabled()) {
            long nanos = System.nanoTime() - t0;
            if (bestDraft != null) {
                log.debug("BigramBeamTextGenerator | think=1 iters={} bestIter={} bestQ={} echo={} qCov={} ctxCov={} connR={} uniqR={} picked={} qTok={} outTok={} dtMs={} out='{}'",
                        drafts.size(),
                        bestDraft.iter,
                        round3(bestDraft.q.total),
                        round3(bestDraft.q.echoRatio),
                        round3(bestDraft.q.queryCoverage),
                        round3(bestDraft.q.contextCoverage),
                        round3(bestDraft.q.connectorRatio),
                        round3(bestDraft.q.uniqueRatio),
                        bestDraft.pickedCount,
                        qTok.size(),
                        best == null ? 0 : best.tokens.size(),
                        nanos / 1_000_000L,
                        snippet(out, 240));
            } else {
                log.debug("BigramBeamTextGenerator | think=1 iters={} bestIter=-1 qTok={} dtMs={} out='{}'",
                        drafts.size(), qTok.size(), nanos / 1_000_000L, snippet(out, 240));
            }
        }

        if (out.isBlank()) return noContextReply(userText, lang, state);
        return out;
    }

    // -------------------- thinking: draft & quality --------------------

    private record Draft(Beam beam, Quality q, int iter, int pickedCount) {}

    private record Quality(
            double total,
            double echoRatio,
            double queryCoverage,
            double contextCoverage,
            double connectorRatio,
            double uniqueRatio,
            int len
    ) {}

    private List<Statement> selectRelevantContext(List<Statement> ctx, Set<String> qTok, int limit) {
        if (ctx == null || ctx.isEmpty()) return List.of();
        if (qTok == null) qTok = Set.of();
        final Set<String> q = qTok;

        return ctx.stream()
                .filter(Objects::nonNull)
                .filter(s -> s.text != null && !s.text.isBlank())
                .map(s -> new AbstractMap.SimpleEntry<>(s, relevanceScore(s, q)))
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(Math.max(4, limit))
                .map(Map.Entry::getKey)
                .toList();
    }

    private double relevanceScore(Statement st, Set<String> qTok) {
        List<String> toks = normalizeTokens(tokenizer.tokenize(st.text));
        if (toks.isEmpty()) return 0.0;

        int hit = 0;
        for (String t : toks) if (qTok.contains(t)) hit++;

        double cov = (qTok.isEmpty()) ? 0.0 : (double) hit / Math.max(1, qTok.size());
        double dens = (double) hit / Math.max(4, toks.size());

        double w = 0.0;
        try { w = st.weight; } catch (Exception ignored) {}

        double lenPenalty = Math.min(1.0, toks.size() / 40.0) * 0.15;

        return (cov * 1.20) + (dens * 0.80) + (Math.tanh(w) * 0.40) - lenPenalty;
    }

    private Quality evaluateQuality(
            String userText,
            Set<String> qTok,
            List<String> outTokens,
            List<String> connectors,
            List<Statement> usedCtx,
            int minTok
    ) {
        int len = 0;
        int echoHit = 0;
        int qHit = 0;
        int conn = 0;

        Set<String> uniq = new HashSet<>();
        for (String t : outTokens) {
            if (t == null || t.isBlank() || STOP.equals(t)) continue;
            len++;
            uniq.add(t);
            if (qTok != null && qTok.contains(t)) {
                echoHit++;
                qHit++;
            }
            if (connectors != null && connectors.contains(t)) conn++;
        }

        double echo = (len <= 0) ? 0.0 : (double) echoHit / (double) len;
        double qCov = (qTok == null || qTok.isEmpty()) ? 0.0 : (double) qHit / (double) Math.max(1, qTok.size());
        double connRatio = (len <= 0) ? 0.0 : (double) conn / (double) len;
        double uniqRatio = (len <= 0) ? 0.0 : (double) uniq.size() / (double) len;

        Set<String> ctxTok = new HashSet<>();
        if (usedCtx != null) {
            for (Statement s : usedCtx) {
                if (s == null || s.text == null) continue;
                ctxTok.addAll(normalizeTokens(tokenizer.tokenize(s.text)));
                if (ctxTok.size() > 4096) break;
            }
        }

        int ctxHit = 0;
        for (String t : uniq) if (ctxTok.contains(t)) ctxHit++;
        double ctxCov = (uniq.isEmpty()) ? 0.0 : (double) ctxHit / (double) uniq.size();

        double total = 0.0;
        total += (len >= minTok) ? 0.35 : -0.35;
        total += ctxCov * 0.65;
        total += uniqRatio * 0.25;
        total += Math.min(0.25, qCov * 0.20);
        total -= echo * 0.70;
        total -= Math.max(0.0, connRatio - 0.22) * 0.60;

        return new Quality(total, echo, qCov, ctxCov, connRatio, uniqRatio, len);
    }

    private static double round3(double v) {
        if (!Double.isFinite(v)) return 0.0;
        return Math.round(v * 1000.0) / 1000.0;
    }

    // -------------------- no-context reply --------------------

    private String noContextReply(String userText, Lang lang, ThoughtState state) {
        // No domain hardcode: use configurable templates in state.tags.
        // Recommended keys (set by engine/config/LTM):
        //  - fallback.noctx.ru / fallback.noctx.en
        //  - fallback.ask_more.ru / fallback.ask_more.en
        String tpl = tag(state,
                lang == Lang.RU ? "fallback.noctx.ru" : "fallback.noctx.en",
                "");

        if (!tpl.isBlank()) return tpl;

        // Minimal generic fallback (infrastructure-level, not scenario logic).
        return (lang == Lang.RU)
                ? "Нужны данные/контекст: сформулируй цель, ограничения и входные факты."
                : "I need more context: state the goal, constraints, and input facts.";
    }

    private static String tag(ThoughtState state, String key, String def) {
        if (state == null || state.tags == null) return def;
        String v = state.tags.get(key);
        return (v == null) ? def : v;
    }

    // -------------------- graph --------------------

    private record Graph(Map<String, Map<String, Double>> next, Map<String, Integer> freq) {}

    private Graph buildGraph(List<Statement> ctx, List<String> connectors) {
        Map<String, Map<String, Double>> next = new HashMap<>();
        Map<String, Integer> freq = new HashMap<>();

        for (Statement st : ctx) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            List<String> t = normalizeTokens(tokenizer.tokenize(st.text));
            if (t.size() < 2) continue;

            for (int i = 0; i < t.size(); i++) {
                String a = t.get(i);
                if (!isAllowed(a)) continue;
                freq.merge(a, 1, Integer::sum);
                if (i + 1 < t.size()) {
                    String b = t.get(i + 1);
                    if (!isAllowed(b)) continue;
                    next.computeIfAbsent(a, k -> new HashMap<>()).merge(b, 1.0, Double::sum);
                }
            }
        }

        if (!freq.isEmpty()) {
            List<String> top = freq.entrySet().stream()
                    .sorted((a, b) -> {
                        int cmp = Integer.compare(b.getValue(), a.getValue());
                        if (cmp != 0) return cmp;
                        return a.getKey().compareTo(b.getKey());
                    })
                    .limit(64)
                    .map(Map.Entry::getKey)
                    .toList();

            for (String from : top) {
                Map<String, Double> outs = next.computeIfAbsent(from, k -> new HashMap<>());
                for (String c : connectors) outs.merge(c, 0.10, Double::sum);
                outs.merge(STOP, 0.08, Double::sum);
            }
        }

        return new Graph(next, freq);
    }

    private List<String> startTokens(Graph g, Set<String> qTok, List<String> connectors, ThoughtState state) {
        int rot = (state == null) ? 0 : Math.max(0, state.draftIndex);
        return startTokens(g, qTok, connectors, rot);
    }

    private List<String> startTokens(Graph g, Set<String> qTok, List<String> connectors, int rot) {
        if (g.freq.isEmpty()) return connectors.stream().limit(3).toList();

        // If query is short, query-overlap start tokens cause mode collapse (echo).
        boolean allowOverlapStarts = qTok.size() >= 6;
        if (allowOverlapStarts) {
            List<String> overlap = g.freq.keySet().stream()
                    .filter(qTok::contains)
                    .filter(this::isAllowed)
                    .sorted()
                    .collect(Collectors.toList());
            if (!overlap.isEmpty()) return rotate(overlap, rot).stream().limit(8).toList();
        }

        // Prefer frequent non-connector tokens that are NOT in the query (anti-echo).
        List<String> top = g.freq.entrySet().stream()
                .map(Map.Entry::getKey)
                .filter(this::isAllowed)
                .filter(t -> !connectors.contains(t))
                .filter(t -> !qTok.contains(t))
                .sorted()
                .collect(Collectors.toList());
        if (!top.isEmpty()) return rotate(top, rot).stream().limit(8).toList();

        // fallback: frequent non-connector even if in query
        List<String> top2 = g.freq.entrySet().stream()
                .map(Map.Entry::getKey)
                .filter(this::isAllowed)
                .filter(t -> !connectors.contains(t))
                .sorted()
                .collect(Collectors.toList());
        if (!top2.isEmpty()) return rotate(top2, rot).stream().limit(8).toList();

        return connectors.stream().limit(3).toList();
    }

    private static <T> List<T> rotate(List<T> in, int rot) {
        if (in == null || in.isEmpty()) return List.of();
        int n = in.size();
        int r = (n == 0) ? 0 : (Math.floorMod(rot, n));
        if (r == 0) return in;
        ArrayList<T> out = new ArrayList<>(n);
        out.addAll(in.subList(r, n));
        out.addAll(in.subList(0, r));
        return out;
    }

    // -------------------- beam core --------------------

    private Beam runBeam(Graph g, Set<String> qTok, List<String> connectors, int minTok, int maxTok, int rot) {
        List<String> starts = startTokens(g, qTok, connectors, rot);

        List<Beam> beam = new ArrayList<>(beamWidth);
        for (String s : starts) beam.add(Beam.start(s));

        int steps = 1;
        while (steps < maxTok) {
            ArrayList<Beam> next = new ArrayList<>(beamWidth * 4);

            for (Beam b : beam) {
                if (b == null) continue;

                if (shouldStop(b, steps, minTok)) {
                    // discourage ending if we're still echoing the query too much
                    double er = echoRatio(b, qTok);
                    double eos = (er > maxEchoRatio) ? -0.25 : 0.05;
                    next.add(b.extend(STOP, eos));
                    continue;
                }

                String last = b.last();
                Map<String, Double> outs = g.next.get(last);

                if (outs == null || outs.isEmpty()) {
                    next.add(b.extend(STOP, 0.0));
                    for (String c : connectors) {
                        if (!isAllowed(c)) continue;
                        next.add(b.extend(c, scoreStep(b, c, 0.15, qTok, connectors, true, minTok)));
                    }
                    continue;
                }

                outs.entrySet().stream()
                        .sorted((a, x) -> {
                            int cmp = Double.compare(x.getValue(), a.getValue());
                            if (cmp != 0) return cmp;
                            return a.getKey().compareTo(x.getKey());
                        })
                        .limit(Math.max(10, beamWidth * 2L))
                        .forEach(e -> {
                            String tok = e.getKey();
                            if (!isAllowed(tok)) return;

                            boolean isConn = connectors.contains(tok);
                            double edgeW = e.getValue() == null ? 0.0 : e.getValue();

                            next.add(b.extend(tok, scoreStep(b, tok, edgeW, qTok, connectors, isConn, minTok)));
                        });

                if (b.tokens.size() >= minTok) {
                    double er = echoRatio(b, qTok);
                    double eos = (er > maxEchoRatio) ? -0.20 : 0.0;
                    next.add(b.extend(STOP, eos));
                }
            }

            next.sort(Comparator.comparingDouble((Beam b) -> b.score).reversed()
                    .thenComparing(Beam::textKey));

            beam = dedupeAndTop(next, beamWidth);

            if (!beam.isEmpty() && STOP.equals(beam.get(0).last())) break;

            steps++;
        }

        return pickBest(beam, minTok);
    }

    private static final class Beam {
        final ArrayList<String> tokens;
        final double score;
        final HashMap<String, Integer> freq;

        private Beam(ArrayList<String> tokens, double score, HashMap<String, Integer> freq) {
            this.tokens = tokens;
            this.score = score;
            this.freq = freq;
        }

        static Beam start(String tok) {
            ArrayList<String> t = new ArrayList<>(32);
            t.add(tok);
            HashMap<String, Integer> f = new HashMap<>();
            f.put(tok, 1);
            return new Beam(t, 0.0, f);
        }

        String last() { return tokens.get(tokens.size() - 1); }

        Beam extend(String tok, double delta) {
            ArrayList<String> t = new ArrayList<>(tokens.size() + 1);
            t.addAll(tokens);
            t.add(tok);

            HashMap<String, Integer> f = new HashMap<>(freq);
            f.merge(tok, 1, Integer::sum);

            return new Beam(t, score + delta, f);
        }

        int uniqueCount() {
            int u = 0;
            for (Map.Entry<String, Integer> e : freq.entrySet()) {
                if (e.getValue() != null && e.getValue() > 0 && !STOP.equals(e.getKey())) u++;
            }
            return u;
        }

        String textKey() {
            int n = Math.min(10, tokens.size());
            StringBuilder b = new StringBuilder(90);
            for (int i = 0; i < n; i++) {
                if (i > 0) b.append(' ');
                b.append(tokens.get(i));
            }
            return b.toString();
        }

        int runOfLast() {
            String last = last();
            int run = 0;
            for (int i = tokens.size() - 1; i >= 0; i--) {
                if (Objects.equals(tokens.get(i), last)) run++;
                else break;
            }
            return run;
        }
    }

    private boolean shouldStop(Beam b, int steps, int minTok) {
        if (b.tokens.size() < minTok) return false;
        if (b.uniqueCount() < Math.min(minUniqueTokens, Math.max(2, b.tokens.size() / 3))) return false;
        if (b.runOfLast() >= maxSameTokenRun) return true;
        return steps >= minTok + 2;
    }

    private double echoRatio(Beam b, Set<String> qTok) {
        if (b == null || qTok == null || qTok.isEmpty()) return 0.0;
        int tot = 0;
        int hit = 0;
        for (String t : b.tokens) {
            if (t == null || t.isBlank() || STOP.equals(t)) continue;
            tot++;
            if (qTok.contains(t)) hit++;
        }
        if (tot <= 0) return 0.0;
        return (double) hit / (double) tot;
    }

    private double scoreStep(Beam b, String tok, double edgeWeight, Set<String> qTok, List<String> connectors, boolean isConnector, int minTok) {
        if (STOP.equals(tok)) {
            double er = echoRatio(b, qTok);
            double ok = (b.tokens.size() >= minTok) ? 0.12 : -0.10;
            ok += (b.uniqueCount() >= 3) ? 0.05 : -0.05;
            if (er > maxEchoRatio) ok -= 0.22; // discourage ending while still echoing
            return ok;
        }

        double ev = Math.log1p(Math.max(0.0, edgeWeight)) * 0.35;

        // IMPORTANT: do not *reward* query tokens. Rewarding causes echo loops.
        double qHit = 0.0;

        int tokCount = b.freq.getOrDefault(tok, 0);
        double rep = (tokCount <= 0) ? 0.0 : (repeatPenalty * Math.min(1.0, tokCount / 2.0));

        // Always penalize query tokens a bit (prevents "Кто ты" -> "Кто ты").
        double qTokPen = (qTok != null && qTok.contains(tok)) ? queryTokenPenalty : 0.0;

        // Extra penalty if query token repeats inside the answer.
        double qRep = 0.0;
        if (qTok != null && qTok.contains(tok) && tokCount >= 1) {
            qRep = queryRepeatPenalty * Math.min(1.0, tokCount / 2.0);
        }

        double conn = 0.0;
        if (isConnector) {
            conn -= connectorPenalty;
            if (b.tokens.size() >= 2) {
                String prev = b.tokens.get(b.tokens.size() - 1);
                if (connectors.contains(prev)) conn -= connectorPenalty * 0.7;
            }
        }

        double stutter = (b.runOfLast() >= maxSameTokenRun - 1) ? 0.35 : 0.0;

        double uniq = 0.0;
        if (b.uniqueCount() < 3) uniq += 0.08;

        return ev + qHit + uniq - rep - qTokPen - qRep - conn - stutter;
    }

    private static List<Beam> dedupeAndTop(List<Beam> beams, int k) {
        LinkedHashMap<String, Beam> bestByKey = new LinkedHashMap<>();
        for (Beam b : beams) {
            String key = b.textKey();
            Beam prev = bestByKey.get(key);
            if (prev == null || b.score > prev.score) bestByKey.put(key, b);
        }
        ArrayList<Beam> out = new ArrayList<>(bestByKey.values());
        out.sort(Comparator.comparingDouble((Beam b) -> b.score).reversed().thenComparing(Beam::textKey));
        return (out.size() > k) ? out.subList(0, k) : out;
    }

    private Beam pickBest(List<Beam> beam, int minTok) {
        if (beam == null || beam.isEmpty()) return null;
        Beam best = null;
        for (Beam b : beam) {
            if (b == null) continue;
            boolean ended = STOP.equals(b.last());
            int len = ended ? b.tokens.size() - 1 : b.tokens.size();
            if (len < minTok) continue;
            if (b.uniqueCount() < 3) continue;

            if (best == null) best = b;
            else {
                boolean bestEnded = STOP.equals(best.last());
                if (ended && !bestEnded) best = b;
                else if (ended == bestEnded) {
                    if (b.score > best.score) best = b;
                    else if (Math.abs(b.score - best.score) < 1e-9 && len < (bestEnded ? best.tokens.size() - 1 : best.tokens.size())) {
                        best = b;
                    }
                }
            }
        }
        return best != null ? best : beam.get(0);
    }

    // -------------------- utils --------------------

    private enum Lang { RU, EN }

    private Lang detectLangPreferUser(String userText, List<Statement> ctx) {
        String u = userText == null ? "" : userText;
        boolean userCyr = HAS_CYR.matcher(u).matches();
        boolean userLat = HAS_LAT.matcher(u).matches();

        // CRITICAL: if userText has Cyrillic -> RU always (prevents "also" leaks)
        if (userCyr && !userLat) return Lang.RU;
        if (!userCyr && userLat) return Lang.EN;

        // fallback on context if user is ambiguous
        StringBuilder sb = new StringBuilder(256);
        if (ctx != null) {
            for (int i = 0; i < Math.min(3, ctx.size()); i++) {
                Statement s = ctx.get(i);
                if (s != null && s.text != null) sb.append(s.text).append(' ');
            }
        }
        String x = sb.toString();
        boolean c = HAS_CYR.matcher(x).matches();
        boolean l = HAS_LAT.matcher(x).matches();
        if (c && !l) return Lang.RU;
        if (c && l) return Lang.RU; // mixed: prefer RU to avoid EN leaks in RU dialog
        if (!c && l) return Lang.EN;

        // default
        return Lang.RU;
    }

    private boolean isAllowed(String tok) {
        if (tok == null) return false;
        String x = tok.trim().toLowerCase(Locale.ROOT);
        if (x.isBlank()) return false;
        if (STOP.equals(x)) return true;
        if (x.length() < 2) return false;
        if (x.length() > 32) return false;
        if (BAN.contains(x)) return false;
        return true;
    }

    private static List<String> normalizeTokens(List<String> toks) {
        if (toks == null || toks.isEmpty()) return List.of();
        ArrayList<String> out = new ArrayList<>(toks.size());
        for (String t : toks) {
            if (t == null) continue;
            String x = t.trim().toLowerCase(Locale.ROOT);
            if (x.isBlank()) continue;
            if (x.length() < 2) continue;
            if (x.length() > 32) continue;
            if (BAN.contains(x)) continue;
            out.add(x);
        }
        return out;
    }

    private static Set<String> asTokenSet(List<String> toks) {
        if (toks == null || toks.isEmpty()) return Set.of();
        LinkedHashSet<String> s = new LinkedHashSet<>();
        for (String t : toks) {
            if (t == null) continue;
            String x = t.trim().toLowerCase(Locale.ROOT);
            if (x.length() < 2) continue;
            if (x.length() > 32) continue;
            if (BAN.contains(x)) continue;
            s.add(x);
            if (s.size() >= 96) break;
        }
        return s;
    }

    private static String detokenize(List<String> toks, Lang lang) {
        if (toks == null || toks.isEmpty()) return "";
        StringBuilder b = new StringBuilder(toks.size() * 6);
        for (String t : toks) {
            if (t == null) continue;
            if (STOP.equals(t)) break;
            if (t.isBlank()) continue;
            if (b.length() > 0) b.append(' ');
            b.append(t);
        }
        String s = b.toString().trim();
        if (s.isBlank()) return s;
        char c0 = s.charAt(0);
        if (Character.isLetter(c0)) s = Character.toUpperCase(c0) + s.substring(1);
        if (!s.endsWith(".") && !s.endsWith("!") && !s.endsWith("?")) s += ".";
        return s;
    }

    private static String postFix(String userText, String out, Lang lang) {
        if (out == null) return "";
        String a = out.trim();
        if (a.isBlank()) return a;

        String q = (userText == null) ? "" : userText.trim().toLowerCase(Locale.ROOT);
        String al = a.toLowerCase(Locale.ROOT);

        if (!q.isBlank() && al.equals(q)) {
            return (lang == Lang.EN) ? (a + " How can I help?") : (a + " Чем помочь?");
        }
        return a;
    }

    private static String extractUser(String preparedPromptOrUserText) {
        if (preparedPromptOrUserText == null) return "";
        String s = preparedPromptOrUserText;
        int ix = s.lastIndexOf("User:");
        if (ix >= 0) {
            String tail = s.substring(ix + "User:".length()).trim();
            if (!tail.isBlank()) return tail;
        }
        return s.trim();
    }

    private static String snippet(String s, int max) {
        if (s == null) return "";
        String x = s.replace('\n', ' ').replace('\r', ' ').trim();
        if (x.length() <= max) return x;
        return x.substring(0, Math.max(0, max - 1)) + "…";
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

    private static int safeInt(String s, int def) {
        if (s == null) return def;
        try {
            return Integer.parseInt(s.trim());
        } catch (Exception ignored) {
            return def;
        }
    }
}