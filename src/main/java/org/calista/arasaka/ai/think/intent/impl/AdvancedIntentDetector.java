package org.calista.arasaka.ai.think.intent.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;

import java.text.Normalizer;
import java.util.*;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Advanced deterministic intent detector.
 *
 * Goals:
 * - zero "magic regex rules" for specific intents (policy lives in weights)
 * - stable + debuggable scoring (logs + explanations)
 * - better robustness than SimpleIntentDetector:
 *   - unicode normalization, latin/cyrillic safety
 *   - token + bigram phrase features
 *   - cheap punctuation/shape signals as configurable feature weights (NOT hard-coded to intents)
 *   - score normalization + margin-based confidence (prevents accidental triggers)
 *
 * Still deterministic and tunable via constructor config.
 */
public final class AdvancedIntentDetector implements IntentDetector {
    private static final Logger log = LogManager.getLogger(AdvancedIntentDetector.class);

    /** words >= 2 chars (letters/digits/_), unicode aware */
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{Nd}_]{2,}");
    private static final int MAX_TOKENS = 160;
    private static final int MAX_FEATURES = 256;

    /** Per-intent feature weights. Keys are features like: "tok:привет", "bg:добрый_день", "sig:qm", ... */
    private final EnumMap<Intent, Map<String, Double>> weights;

    /** Optional intent priors (bias). Example: QUESTION -> 0.05 */
    private final EnumMap<Intent, Double> priors;

    /** Confidence thresholds */
    private final double minConfidence;     // absolute confidence
    private final double minMargin;         // best - second best

    /** Normalization knobs */
    private final double lengthNormPow;     // 0..1 ; 0 disables
    private final boolean useBigrams;
    private final boolean uniqueTokens;     // set=true reproduces old behavior; false uses counts

    /** Debug */
    private final boolean debugLog;

    public AdvancedIntentDetector() {
        this(Config.builder().build());
    }

    public AdvancedIntentDetector(Config cfg) {
        Objects.requireNonNull(cfg, "cfg");
        this.weights = new EnumMap<>(Intent.class);
        for (Intent it : Intent.values()) {
            Map<String, Double> w = cfg.weights.get(it);
            this.weights.put(it, w == null ? Map.of() : new HashMap<>(w));
        }

        this.priors = new EnumMap<>(Intent.class);
        for (Intent it : Intent.values()) {
            double p = cfg.priors.getOrDefault(it, 0.0);
            this.priors.put(it, p);
        }

        this.minConfidence = clamp01(cfg.minConfidence);
        this.minMargin = Math.max(0.0, cfg.minMargin);

        this.lengthNormPow = clamp(cfg.lengthNormPow, 0.0, 1.0);
        this.useBigrams = cfg.useBigrams;
        this.uniqueTokens = cfg.uniqueTokens;

        this.debugLog = cfg.debugLog;
    }

    @Override
    public Intent detect(String userText) {
        if (userText == null) return Intent.UNKNOWN;

        String s0 = userText.trim();
        if (s0.isEmpty()) return Intent.UNKNOWN;

        // Normalize (deterministic; makes cyrillic/latin punctuation variants behave nicer)
        String s = normalizeText(s0);

        // Extract token stream (optionally counts)
        TokenStats stats = tokenize(s);

        // Feature set:
        // - tokens: tok:<t>
        // - bigrams: bg:<t1>_<t2>
        // - signals: sig:qm (question mark), sig:ex (exclamation), sig:code (code-ish), sig:len_short/len_long
        List<String> features = buildFeatures(s, stats);

        // Score
        EnumMap<Intent, Double> raw = new EnumMap<>(Intent.class);
        EnumMap<Intent, List<FeatureHit>> hits = debugLog ? new EnumMap<>(Intent.class) : null;

        for (Intent it : Intent.values()) raw.put(it, 0.0);

        for (Intent it : Intent.values()) {
            if (it == Intent.UNKNOWN) continue;

            double score = priors.getOrDefault(it, 0.0);
            Map<String, Double> w = weights.getOrDefault(it, Map.of());

            if (debugLog) hits.put(it, new ArrayList<>(16));

            for (String f : features) {
                Double wf = w.get(f);
                if (wf == null) continue;

                // if using token counts, scale only token features by count
                double mul = 1.0;
                if (!uniqueTokens && f.startsWith("tok:")) {
                    String tok = f.substring("tok:".length());
                    mul = Math.max(1, stats.counts.getOrDefault(tok, 1));
                }
                double add = wf * mul;
                score += add;

                if (debugLog) hits.get(it).add(new FeatureHit(f, wf, mul, add));
            }

            // Length normalization: prevents long prompts from dominating via many small hits
            if (lengthNormPow > 0.0) {
                double denom = Math.pow(Math.max(1, stats.tokenCount), lengthNormPow);
                score /= denom;
            }

            raw.put(it, score);
        }

        // Choose best + confidence
        Ranked r = rank(raw);

        // Convert to confidence in a deterministic way:
        // - "absolute" from softmax(best)
        // - plus margin gate (best-second) to reduce false positives
        double conf = softmaxConfidence(raw, r.best);
        double margin = r.bestScore - r.secondScore;

        boolean ok = conf >= minConfidence && margin >= minMargin;

        if (debugLog && log.isDebugEnabled()) {
            log.debug("IntentDetect: text='{}' norm='{}' tokens={} features={} best={} conf={} margin={} ok={}",
                    truncate(s0, 220),
                    truncate(s, 220),
                    stats.tokenCount,
                    features.size(),
                    r.best,
                    round(conf),
                    round(margin),
                    ok
            );
            dumpTopHits(hits, r.best);
        }

        return ok ? r.best : Intent.UNKNOWN;
    }

    // ------------------------- Config -------------------------

    public static final class Config {
        final EnumMap<Intent, Map<String, Double>> weights;
        final EnumMap<Intent, Double> priors;

        final double minConfidence;
        final double minMargin;

        final double lengthNormPow;
        final boolean useBigrams;
        final boolean uniqueTokens;

        final boolean debugLog;

        private Config(Builder b) {
            this.weights = new EnumMap<>(Intent.class);
            for (Intent it : Intent.values()) {
                Map<String, Double> w = b.weights.get(it);
                this.weights.put(it, w == null ? Map.of() : Map.copyOf(w));
            }

            this.priors = new EnumMap<>(Intent.class);
            this.priors.putAll(b.priors);

            this.minConfidence = b.minConfidence;
            this.minMargin = b.minMargin;

            this.lengthNormPow = b.lengthNormPow;
            this.useBigrams = b.useBigrams;
            this.uniqueTokens = b.uniqueTokens;

            this.debugLog = b.debugLog;
        }

        public static Builder builder() {
            Builder b = new Builder();
            // defaults: conservative but useful; ALL intent policy still lives in weights
            b.minConfidence = 0.62;
            b.minMargin = 0.10;
            b.lengthNormPow = 0.35;
            b.useBigrams = true;
            b.uniqueTokens = true;
            b.debugLog = false;

            // default priors = 0.0 for all
            for (Intent it : Intent.values()) b.priors.put(it, 0.0);

            // IMPORTANT: no hard-coded weights by default.
            // Load weights/priors from KB/LTM/config and inject via builder.weights(...).

            return b;
        }

        public static final class Builder {
            final EnumMap<Intent, Map<String, Double>> weights = new EnumMap<>(Intent.class);
            final EnumMap<Intent, Double> priors = new EnumMap<>(Intent.class);

            double minConfidence;
            double minMargin;

            double lengthNormPow;
            boolean useBigrams;
            boolean uniqueTokens;

            boolean debugLog;

            public Builder weights(Intent intent, Map<String, Double> w) {
                if (intent != null) weights.put(intent, w == null ? Map.of() : new HashMap<>(w));
                return this;
            }

            public Builder prior(Intent intent, double p) {
                if (intent != null) priors.put(intent, p);
                return this;
            }

            public Builder minConfidence(double v) { this.minConfidence = v; return this; }
            public Builder minMargin(double v) { this.minMargin = v; return this; }
            public Builder lengthNormPow(double v) { this.lengthNormPow = v; return this; }
            public Builder useBigrams(boolean v) { this.useBigrams = v; return this; }
            public Builder uniqueTokens(boolean v) { this.uniqueTokens = v; return this; }
            public Builder debugLog(boolean v) { this.debugLog = v; return this; }

            /**
             * Optional minimal bootstrap for dev/demo.
             * Do NOT use in production if you want "no hard code".
             */
            public Builder bootstrapRuEnMinimal() {
                this.weights.put(Intent.GREETING, Map.ofEntries(
                        Map.entry("tok:привет", 1.00),
                        Map.entry("tok:здравствуйте", 1.00),
                        Map.entry("tok:hello", 1.00),
                        Map.entry("tok:hi", 0.90),
                        Map.entry("bg:добрый_день", 1.05)
                ));
                this.weights.put(Intent.QUESTION, Map.ofEntries(
                        Map.entry("sig:qm", 0.35),
                        Map.entry("sig:whyshape", 0.10)
                ));
                this.weights.put(Intent.REQUEST, Map.ofEntries(
                        Map.entry("tok:сделай", 0.80),
                        Map.entry("tok:создай", 0.80),
                        Map.entry("tok:напиши", 0.75)
                ));
                return this;
            }

            public Config build() { return new Config(this); }
        }
    }

    // ------------------------- Internals -------------------------

    private static final class TokenStats {
        final List<String> tokens;
        final Map<String, Integer> counts;
        final int tokenCount;

        TokenStats(List<String> tokens, Map<String, Integer> counts) {
            this.tokens = tokens;
            this.counts = counts;
            this.tokenCount = tokens.size();
        }
    }

    private static final class Ranked {
        final Intent best;
        final Intent second;
        final double bestScore;
        final double secondScore;

        Ranked(Intent best, Intent second, double bestScore, double secondScore) {
            this.best = best;
            this.second = second;
            this.bestScore = bestScore;
            this.secondScore = secondScore;
        }
    }

    private static final class FeatureHit {
        final String feature;
        final double w;
        final double mul;
        final double add;

        FeatureHit(String feature, double w, double mul, double add) {
            this.feature = feature;
            this.w = w;
            this.mul = mul;
            this.add = add;
        }
    }

    private TokenStats tokenize(String s) {
        // We keep ordering (for bigrams) but still can apply unique set if configured
        List<String> toks = WORD.matcher(s)
                .results()
                .map(MatchResult::group)
                .limit(MAX_TOKENS)
                .collect(Collectors.toList());

        // Lowercase with ROOT locale (stable)
        for (int i = 0; i < toks.size(); i++) {
            toks.set(i, toks.get(i).toLowerCase(Locale.ROOT));
        }

        if (uniqueTokens) {
            // Preserve order while de-duplicating
            LinkedHashSet<String> set = new LinkedHashSet<>(toks.size());
            set.addAll(toks);
            toks = new ArrayList<>(set);
        }

        HashMap<String, Integer> counts = new HashMap<>();
        for (String t : toks) counts.merge(t, 1, Integer::sum);

        return new TokenStats(toks, counts);
    }

    private List<String> buildFeatures(String s, TokenStats stats) {
        ArrayList<String> feats = new ArrayList<>(Math.min(MAX_FEATURES, stats.tokens.size() * 2 + 8));

        // token features
        for (String t : stats.tokens) {
            feats.add("tok:" + t);
            if (feats.size() >= MAX_FEATURES) return feats;
        }

        // bigrams (phrase signals, still policy-driven by weights)
        if (useBigrams) {
            List<String> t = stats.tokens;
            for (int i = 0; i + 1 < t.size(); i++) {
                feats.add("bg:" + t.get(i) + "_" + t.get(i + 1));
                if (feats.size() >= MAX_FEATURES) return feats;
            }
        }

        // generic signals (configurable via weights; detector just emits features)
        if (s.indexOf('?') >= 0) feats.add("sig:qm");
        if (s.indexOf('!') >= 0) feats.add("sig:ex");

        // “whyshape”: starts with typical interrogatives (ru/en). This is NOT intent logic,
        // it’s just a feature your weights may use.
        String head = firstTokenOrEmpty(stats.tokens);
        if (!head.isEmpty()) {
            if (head.equals("как") || head.equals("почему") || head.equals("зачем") || head.equals("что")
                    || head.equals("where") || head.equals("why") || head.equals("how") || head.equals("what")) {
                feats.add("sig:whyshape");
            }
        }

        // code-ish text (braces, semicolons, import, package) as a feature for downstream policies
        if (looksLikeCode(s)) feats.add("sig:code");

        // length buckets
        int len = s.length();
        if (len <= 24) feats.add("sig:len_short");
        else if (len >= 220) feats.add("sig:len_long");

        // keep stable ordering; cap
        if (feats.size() > MAX_FEATURES) {
            return feats.subList(0, MAX_FEATURES);
        }
        return feats;
    }

    private static Ranked rank(EnumMap<Intent, Double> scores) {
        Intent best = Intent.UNKNOWN;
        Intent second = Intent.UNKNOWN;
        double bestScore = Double.NEGATIVE_INFINITY;
        double secondScore = Double.NEGATIVE_INFINITY;

        for (Map.Entry<Intent, Double> e : scores.entrySet()) {
            Intent it = e.getKey();
            if (it == Intent.UNKNOWN) continue;

            double v = e.getValue();
            if (v > bestScore) {
                second = best;
                secondScore = bestScore;
                best = it;
                bestScore = v;
            } else if (v > secondScore) {
                second = it;
                secondScore = v;
            }
        }

        if (best == Intent.UNKNOWN) {
            return new Ranked(Intent.UNKNOWN, Intent.UNKNOWN, 0.0, 0.0);
        }
        if (second == Intent.UNKNOWN) secondScore = bestScore;
        return new Ranked(best, second, bestScore, secondScore);
    }

    private static double softmaxConfidence(EnumMap<Intent, Double> raw, Intent best) {
        // Deterministic “probability-like” confidence (stable)
        double max = Double.NEGATIVE_INFINITY;
        for (Map.Entry<Intent, Double> e : raw.entrySet()) {
            if (e.getKey() == Intent.UNKNOWN) continue;
            max = Math.max(max, e.getValue());
        }

        double sum = 0.0;
        double bestExp = 0.0;
        for (Map.Entry<Intent, Double> e : raw.entrySet()) {
            Intent it = e.getKey();
            if (it == Intent.UNKNOWN) continue;

            // clamp to avoid overflow; still deterministic
            double z = clamp(e.getValue() - max, -30.0, 30.0);
            double ex = Math.exp(z);

            sum += ex;
            if (it == best) bestExp = ex;
        }

        if (sum <= 0.0) return 0.0;
        return bestExp / sum;
    }

    private void dumpTopHits(EnumMap<Intent, List<FeatureHit>> hits, Intent best) {
        if (hits == null) return;
        List<FeatureHit> list = hits.get(best);
        if (list == null || list.isEmpty()) return;

        list.sort((a, b) -> Double.compare(b.add, a.add));
        int n = Math.min(10, list.size());

        StringBuilder sb = new StringBuilder(256);
        sb.append("TopHits[").append(best).append("]: ");
        for (int i = 0; i < n; i++) {
            FeatureHit h = list.get(i);
            if (i > 0) sb.append(" | ");
            sb.append(h.feature)
                    .append(" w=").append(round(h.w))
                    .append(" x").append(round(h.mul))
                    .append(" => ").append(round(h.add));
        }
        log.debug(sb.toString());
    }

    private static boolean looksLikeCode(String s) {
        // deterministic shape heuristic; emitted as feature only
        int score = 0;
        if (s.indexOf('{') >= 0 || s.indexOf('}') >= 0) score += 2;
        if (s.indexOf(';') >= 0) score += 1;
        if (s.contains("package ")) score += 2;
        if (s.contains("import ")) score += 2;
        if (s.contains("class ")) score += 1;
        if (s.contains("public ")) score += 1;
        if (s.contains("private ")) score += 1;
        return score >= 3;
    }

    private static String normalizeText(String s) {
        // NFKC folds width variants, etc. Keep it deterministic.
        String x = Normalizer.normalize(s, Normalizer.Form.NFKC);

        // unify common quotes/dashes to spaces (not intent logic, just cleaning)
        x = x.replace('\u2014', '-') // em dash
                .replace('\u2013', '-') // en dash
                .replace('\u00A0', ' '); // NBSP

        return x;
    }

    private static String firstTokenOrEmpty(List<String> tokens) {
        return (tokens == null || tokens.isEmpty()) ? "" : tokens.get(0);
    }

    private static double clamp01(double v) { return clamp(v, 0.0, 1.0); }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, Math.max(0, max - 1)) + "…";
    }

    private static String round(double v) {
        // stable small formatting without BigDecimal overhead
        return String.format(Locale.ROOT, "%.4f", v);
    }
}