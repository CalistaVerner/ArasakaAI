package org.calista.arasaka.ai.think.candidate.impl;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.scorer.impl.TokenOverlapScorer;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Advanced deterministic evaluator (enterprise-grade):
 * - schema-aware structure contract (driven by state.tags, no domain hard-code)
 * - groundedness aggregation over context statements
 * - query coverage (addresses user input)
 * - novelty vs context (avoid hallucinations)
 * - repetition (avoid loops)
 *
 * IMPORTANT:
 * - deterministic: NO randomness
 * - must be fast in hot-path: avoid retokenizing same data repeatedly
 */
public final class BaseCandidateEvaluator implements CandidateEvaluator {

    // --- defaults (can be overridden by state.tags) ---
    private final double minGroundedness;
    private final double maxContradictionRisk;

    private final double minQueryCoverage;
    private final double maxNovelty;
    private final double maxRepetition;

    private final int minChars;
    private final int maxCharsSoft;
    private final int maxCharsHard;

    private final Tokenizer tokenizer;
    private final TokenOverlapScorer overlapScorer;

    // For deterministic relevance/structure checks.
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{Nd}_]{3,}");

    public BaseCandidateEvaluator(
            Tokenizer tokenizer,
            TokenOverlapScorer overlapScorer,
            double minGroundedness,
            double maxContradictionRisk,
            double minQueryCoverage,
            double maxNovelty,
            double maxRepetition,
            int minChars,
            int maxCharsSoft,
            int maxCharsHard
    ) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.overlapScorer = Objects.requireNonNull(overlapScorer, "overlapScorer");

        this.minGroundedness = clamp01(minGroundedness);
        this.maxContradictionRisk = clamp01(maxContradictionRisk);
        this.minQueryCoverage = clamp01(minQueryCoverage);
        this.maxNovelty = clamp01(maxNovelty);
        this.maxRepetition = clamp01(maxRepetition);

        this.minChars = Math.max(0, minChars);
        this.maxCharsSoft = Math.max(16, maxCharsSoft);
        this.maxCharsHard = Math.max(this.maxCharsSoft, maxCharsHard);
    }

    @Override
    public Evaluation evaluate(String userText, List<Statement> context, ThoughtState state, String draft) {
        final String q = (userText == null) ? "" : userText.trim();
        final String a = (draft == null) ? "" : draft.trim();

        final List<Statement> ctx = (context == null) ? List.of() : context;

        Evaluation ev = new Evaluation();
        ev.valid = false;
        ev.score = Double.NEGATIVE_INFINITY;

        // Basic length gate
        if (a.length() < minChars) {
            ev.validationNotes = "too_short";
            ev.notes = "Ответ слишком короткий.";
            ev.normalizeNotes();
            return ev;
        }
        if (a.length() > maxCharsHard) {
            ev.validationNotes = "too_long_hard";
            ev.notes = "Ответ слишком длинный.";
            ev.normalizeNotes();
            return ev;
        }

        // Token-cache for this (query, answer)
        // Hot-path optimization:
        // - query tokens are identical for all drafts within the same iteration
        // - so we cache qTok/qSet in ThoughtState.cache and reuse them
        TokCache tc = TokCache.of(tokenizer, state, q, a);

        // ---- 1) Structure ----
        final Structure s = structureOf(a);
        final double structureScore = structureScoreOf(s);

        // ---- No-context mode ----
        if (ctx.isEmpty()) {
            final double qc = queryCoverage(tc);
            final Repetition rep = repetition(tc);
            final double echo = echoPenalty(tc);

            // In no-context mode schema isn't mandatory; keep
            double score = 0.50 * qc - 0.30 * rep.rate - 0.15 * echo + 0.25 * structureScore;
            score = clamp01(score);

            ev.valid = (qc >= 0.20 && rep.rate <= 0.80);
            ev.score = score;

            ev.queryCoverage = qc;
            ev.repetition = rep.rate;
            ev.novelty = 1.0; // no context => novelty undefined; treat as high
            ev.groundedness = 0.20;
            ev.contradictionRisk = 0.20;
            ev.structureScore = structureScore;

            ev.notes = (ev.valid ? "" : "Нужно конкретнее и без повторов.");
            ev.validationNotes = "noctx;qc=" + fmt(qc) + ";rep=" + fmt(rep.rate) + ";st=" + fmt(structureScore);
            ev.normalizeNotes();
            return ev;
        }

        // ---- 2) Groundedness / risk from context ----
        final Grounding g = groundedness(a, ctx);  // ✅
        final double groundedness = g.grounded;
        final double risk = g.risk;

        // ---- 3) Query coverage ----
        final double qc = queryCoverage(tc);

        // ---- 4) Novelty vs context (CACHED union of ctx tokens via state.cache) ----
        final Novelty nov = noveltyVsContext(tc, ctx, state);

        // ---- 5) Repetition ----
        final Repetition rep = repetition(tc);

        // ---- 6) Misc penalties ----
        final double echo = echoPenalty(tc);
        final double softLenPenalty = (a.length() > maxCharsSoft) ? clamp01((a.length() - maxCharsSoft) / (double) (maxCharsHard - maxCharsSoft)) : 0.0;

        // ---- Policy thresholds (can be overridden by tags) ----
        Policy p = policyFromState(state);

        boolean valid =
                (groundedness >= p.gMin)
                        && (risk <= p.riskMax)
                        && (qc >= p.qcMin)
                        && (nov.rate <= p.novMax)
                        && (rep.rate <= p.repMax);

        // ---- Score composition (deterministic) ----
        // weights are stable; policy thresholds are configurable via tags.
        double score =
                0.34 * groundedness
                        + 0.22 * qc
                        + 0.18 * structureScore
                        + 0.12 * (1.0 - clamp01(risk))
                        + 0.10 * (1.0 - clamp01(nov.rate))
                        + 0.04 * (1.0 - clamp01(rep.rate));

        // penalties
        score -= 0.10 * echo;
        score -= 0.08 * softLenPenalty;

        score = clamp01(score);

        ev.valid = valid;
        ev.score = score;

        ev.groundedness = groundedness;
        ev.contradictionRisk = risk;
        ev.queryCoverage = qc;
        ev.novelty = nov.rate;
        ev.repetition = rep.rate;
        ev.structureScore = structureScore;

        StringBuilder notes = new StringBuilder(96);
        if (!valid) {
            if (groundedness < p.gMin) notes.append("Нужно больше опоры на контекст. ");
            if (risk > p.riskMax) notes.append("Есть риск противоречий. ");
            if (qc < p.qcMin) notes.append("Слабо отвечает на вопрос. ");
            if (nov.rate > p.novMax) notes.append("Слишком много нового без доказательств. ");
            if (rep.rate > p.repMax) notes.append("Слишком много повторов. ");
        }
        ev.notes = notes.toString().trim();
        ev.validationNotes =
                "g=" + fmt(groundedness)
                        + ";risk=" + fmt(risk)
                        + ";qc=" + fmt(qc)
                        + ";nov=" + fmt(nov.rate)
                        + ";rep=" + fmt(rep.rate)
                        + ";st=" + fmt(structureScore)
                        + (valid ? ";ok" : ";bad");

        ev.normalizeNotes();
        return ev;
    }

    // -------------------- Policy --------------------

    private static final class Policy {
        final double gMin;
        final double riskMax;
        final double qcMin;
        final double novMax;
        final double repMax;

        Policy(double gMin, double riskMax, double qcMin, double novMax, double repMax) {
            this.gMin = gMin;
            this.riskMax = riskMax;
            this.qcMin = qcMin;
            this.novMax = novMax;
            this.repMax = repMax;
        }
    }

    private Policy policyFromState(ThoughtState state) {
        double gMin = minGroundedness;
        double riskMax = maxContradictionRisk;
        double qcMin = minQueryCoverage;
        double novMax = maxNovelty;
        double repMax = maxRepetition;

        Map<String, String> tags = (state == null ? null : state.tags);
        if (tags != null && !tags.isEmpty()) {
            gMin = clamp01(readDouble(tags, "eval.grounded.min", gMin));
            riskMax = clamp01(readDouble(tags, "eval.risk.max", riskMax));
            qcMin = clamp01(readDouble(tags, "eval.qc.min", qcMin));
            novMax = clamp01(readDouble(tags, "eval.novelty.max", novMax));
            repMax = clamp01(readDouble(tags, "eval.repetition.max", repMax));
        }

        // Strict mode hooks (engine sets verify.strict / repair.strict)
        boolean strict = tags != null && ("true".equalsIgnoreCase(tags.get("verify.strict")) || "true".equalsIgnoreCase(tags.get("repair.strict")));
        if (strict) {
            // tighten slightly
            gMin = Math.min(0.95, gMin + 0.05);
            riskMax = Math.max(0.05, riskMax - 0.05);
            qcMin = Math.min(0.95, qcMin + 0.03);
            novMax = Math.max(0.05, novMax - 0.05);
            repMax = Math.max(0.05, repMax - 0.05);
        }

        return new Policy(gMin, riskMax, qcMin, novMax, repMax);
    }

    private static double readDouble(Map<String, String> tags, String key, double def) {
        try {
            String v = tags.get(key);
            if (v == null) return def;
            v = v.trim();
            if (v.isEmpty()) return def;
            return Double.parseDouble(v);
        } catch (Exception e) {
            return def;
        }
    }

    // -------------------- Token caches --------------------

    static final class TokCache {
        final List<String> qTok;
        final List<String> aTok;
        final Set<String> qSet;
        final Set<String> aSet;

        private TokCache(List<String> qTok, List<String> aTok, Set<String> qSet, Set<String> aSet) {
            this.qTok = qTok;
            this.aTok = aTok;
            this.qSet = qSet;
            this.aSet = aSet;
        }

        static TokCache of(Tokenizer tokenizer, ThoughtState state, String q, String a) {
            Objects.requireNonNull(tokenizer, "tokenizer");

            // Query cache is safe to share across drafts (deterministic).
            List<String> qt;
            Set<String> qs;

            QueryTokCache qc = getOrBuildQueryCache(tokenizer, state, q);
            qt = qc.qTok;
            qs = qc.qSet;

            List<String> at = tokenizeLimited(tokenizer, a, 768);
            Set<String> as = (at.isEmpty()) ? Set.of() : new HashSet<>(at);

            return new TokCache(qt, at, qs, as);
        }
    }

    private static final String CACHE_Q_TOK = "eval.qTok.v1";

    private static final class QueryTokCache {
        final String qRef;
        final List<String> qTok;
        final Set<String> qSet;

        QueryTokCache(String qRef, List<String> qTok, Set<String> qSet) {
            this.qRef = (qRef == null) ? "" : qRef;
            this.qTok = (qTok == null) ? List.of() : qTok;
            this.qSet = (qSet == null) ? Set.of() : qSet;
        }
    }

    private static QueryTokCache getOrBuildQueryCache(Tokenizer tokenizer, ThoughtState state, String q) {
        String qNorm = (q == null) ? "" : q;

        Map<String, Object> cache = (state == null) ? null : state.cache;
        if (cache != null) {
            Object v = cache.get(CACHE_Q_TOK);
            if (v instanceof QueryTokCache qc && qc.qRef.equals(qNorm)) {
                return qc;
            }
        }

        List<String> qt = tokenizeLimited(tokenizer, qNorm, 256);
        Set<String> qs = (qt.isEmpty()) ? Set.of() : new HashSet<>(qt);

        QueryTokCache qc = new QueryTokCache(qNorm, qt, qs);
        if (cache != null) cache.put(CACHE_Q_TOK, qc);
        return qc;
    }

    // -------------------- Context token cache (per-iteration) --------------------
    // Stored in ThoughtState.cache and shared across drafts within the same iteration.
    private static final String CACHE_CTX_TOK = "eval.ctxTok.v1";

    private static final class CtxTokCache {
        final List<Statement> ctxRef;
        final Set<String> tokens;

        CtxTokCache(List<Statement> ctxRef, Set<String> tokens) {
            this.ctxRef = ctxRef;
            this.tokens = (tokens == null) ? Set.of() : tokens;
        }
    }

    private Set<String> contextTokenUnionCached(ThoughtState state, List<Statement> ctx) {
        if (ctx == null || ctx.isEmpty()) return Set.of();

        final Map<String, Object> cache = (state == null) ? null : state.cache;
        if (cache != null) {
            Object v = cache.get(CACHE_CTX_TOK);
            if (v instanceof CtxTokCache c && c.ctxRef == ctx && c.tokens != null) {
                return c.tokens;
            }
        }

        // Build union of context tokens once per iteration, re-used by all drafts.
        HashSet<String> ctxTok = new HashSet<>(2048);
        for (Statement s : ctx) {
            if (s == null || s.text == null || s.text.isBlank()) continue;
            ctxTok.addAll(tokenSetLimited(s.text));
            if (ctxTok.size() > 20_000) break; // safety cap
        }

        Set<String> frozen = ctxTok.isEmpty() ? Set.of() : Collections.unmodifiableSet(ctxTok);

        if (state != null) {
            if (state.cache == null) state.cache = new HashMap<>(8);
            state.cache.put(CACHE_CTX_TOK, new CtxTokCache(ctx, frozen));
        }
        return frozen;
    }

    // -------------------- Metrics --------------------

    private static final class Grounding {
        final double grounded;
        final double risk;

        Grounding(double grounded, double risk) {
            this.grounded = grounded;
            this.risk = risk;
        }
    }

    private Grounding groundedness(String answerText, List<Statement> ctx) {
        if (answerText == null || answerText.isBlank()) return new Grounding(0.0, 1.0);

        double best = 0.0;
        for (Statement st : ctx) {
            if (st == null || st.text == null || st.text.isBlank()) continue;
            double ov = overlapScorer.score(answerText, st); // правильная сигнатура
            if (ov > best) best = ov;
        }

        // BM25-lite score > 1.0 вполне возможен, поэтому НУЖНА нормализация, а не clamp(best)
        // Стабильная "сжатая" нормализация: 1 - exp(-best)
        double grounded = clamp01(1.0 - Math.exp(-Math.max(0.0, best)));
        double risk = clamp01(1.0 - grounded);
        return new Grounding(grounded, risk);
    }



    private double queryCoverage(TokCache tc) {
        if (tc.qTok.isEmpty()) return 0.0;
        if (tc.aTok.isEmpty()) return 0.0;

        int hit = 0;
        for (String t : tc.qTok) {
            if (t == null) continue;
            if (tc.aSet.contains(t)) hit++;
        }
        return clamp01(hit / (double) Math.max(1, tc.qTok.size()));
    }

    private static final class Novelty {
        final double rate;
        final double supportedRate;
        final int unsupported;
        final int total;

        Novelty(double rate, double supportedRate, int unsupported, int total) {
            this.rate = rate;
            this.supportedRate = supportedRate;
            this.unsupported = unsupported;
            this.total = total;
        }
    }

    Novelty noveltyVsContext(TokCache tc, List<Statement> ctx, ThoughtState state) {
        if (tc.aTok.isEmpty()) return new Novelty(0.0, 1.0, 0, 0);

        Set<String> ctxTok = contextTokenUnionCached(state, ctx);

        int unsupported = 0;
        for (String t : tc.aTok) {
            if (t != null && !ctxTok.contains(t)) unsupported++;
        }

        int total = tc.aTok.size();
        double rate = unsupported / (double) Math.max(1, total);
        double supported = 1.0 - rate;

        return new Novelty(clamp01(rate), clamp01(supported), unsupported, total);
    }

    private static final class Repetition {
        final double rate;
        final int repeats;
        final int total;

        Repetition(double rate, int repeats, int total) {
            this.rate = rate;
            this.repeats = repeats;
            this.total = total;
        }
    }

    private Repetition repetition(TokCache tc) {
        if (tc.aTok.isEmpty()) return new Repetition(0.0, 0, 0);

        HashMap<String, Integer> freq = new HashMap<>(256);
        int rep = 0;

        for (String t : tc.aTok) {
            if (t == null) continue;
            int c = freq.getOrDefault(t, 0) + 1;
            freq.put(t, c);
            if (c >= 3) rep++;
        }

        double rate = rep / (double) Math.max(1, tc.aTok.size());
        return new Repetition(clamp01(rate), rep, tc.aTok.size());
    }

    private double echoPenalty(TokCache tc) {
        // Penalize if answer repeats query tokens too strongly (parroting).
        if (tc.qTok.isEmpty() || tc.aTok.isEmpty()) return 0.0;

        int hit = 0;
        for (String t : tc.qTok) {
            if (t == null) continue;
            if (tc.aSet.contains(t)) hit++;
        }
        double overlap = hit / (double) Math.max(1, tc.qTok.size());

        // Only penalize heavy echo.
        return (overlap <= 0.65) ? 0.0 : clamp01((overlap - 0.65) / 0.35);
    }

    // -------------------- Structure --------------------

    private static final class Structure {
        final boolean hasList;
        final boolean hasParagraphs;
        final boolean tooManyTelemetries;

        Structure(boolean hasList, boolean hasParagraphs, boolean tooManyTelemetries) {
            this.hasList = hasList;
            this.hasParagraphs = hasParagraphs;
            this.tooManyTelemetries = tooManyTelemetries;
        }
    }

    private Structure structureOf(String a) {
        if (a == null || a.isBlank()) return new Structure(false, false, false);

        boolean hasList = a.contains("\n- ") || a.contains("\n• ") || a.contains("\n* ");
        boolean hasParagraphs = a.contains("\n\n");

        // crude "telemetry-only" heuristic
        int eq = 0;
        for (int i = 0; i < a.length(); i++) if (a.charAt(i) == '=') eq++;
        boolean tooManyTelem = eq >= 6 && a.length() < 220;

        return new Structure(hasList, hasParagraphs, tooManyTelem);
    }

    private double structureScoreOf(Structure s) {
        if (s == null) return 0.0;
        double score = 0.0;
        if (s.hasList) score += 0.55;
        if (s.hasParagraphs) score += 0.35;
        if (s.tooManyTelemetries) score -= 0.60;
        return clamp01(score);
    }

    // -------------------- Tokenization helpers --------------------

    private static List<String> tokenizeLimited(Tokenizer tokenizer, String text, int limit) {
        if (text == null || text.isBlank()) return List.of();
        List<String> out = tokenizer.tokenize(text);
        if (out == null || out.isEmpty()) return List.of();
        if (out.size() <= limit) return out;
        return out.subList(0, limit);
    }

    private static Set<String> tokenSetLimited(String text) {
        if (text == null || text.isBlank()) return Set.of();

        HashSet<String> set = new HashSet<>(256);
        var m = WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (m.find()) {
            set.add(m.group());
            if (set.size() > 4096) break;
        }
        return set.isEmpty() ? Set.of() : set;
    }

    // -------------------- Utils --------------------

    private static double clamp01(double x) {
        if (x < 0) return 0;
        if (x > 1) return 1;
        return x;
    }

    private static String fmt(double x) {
        return String.format(Locale.ROOT, "%.4f", x);
    }
}