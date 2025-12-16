package org.calista.arasaka.ai.think.intent.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;

import java.text.Normalizer;
import java.util.*;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;

/**
 * HashedIntentDetector — enterprise-grade intent classifier:
 * - Deterministic feature hashing (fixed memory footprint, fast)
 * - Linear multiclass model (softmax over intents) with configurable thresholds
 * - Works great with offline-trained weights OR online-updatable weights
 *
 * Features (emitted, then hashed):
 * - tok:<token>
 * - bg:<t1>_<t2> (optional)
 * - sig:qm, sig:ex, sig:code, sig:len_short, sig:len_long, sig:whyshape (optional, still feature-only)
 *
 * Important:
 * - No "magic rules" for intents in code. The code emits generic features only.
 * - Intent policy is in weights (model).
 */
public final class HashedIntentDetector implements IntentDetector {
    private static final Logger log = LogManager.getLogger(HashedIntentDetector.class);

    /** words >= 2 chars (letters/digits/_), unicode aware */
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{Nd}_]{2,}");
    private static final int MAX_TOKENS = 160;

    private final int dim;                 // feature hashing space size
    private final int mask;                // dim must be power-of-two for fast masking

    private final boolean useBigrams;
    private final boolean uniqueTokens;

    private final double minConfidence;
    private final double minMargin;

    /** Softmax temperature for confidence calibration (1.0 = default). */
    private final double temperature;

    /** Optional debug. */
    private final boolean debugLog;

    /**
     * Weights layout:
     *  - intentsCount = number of intents excluding UNKNOWN
     *  - W[intentIndex][featureIndex] + bias[intentIndex]
     *
     * Intent index mapping stored in intentIndex / indexIntent.
     */
    private final double[][] W;
    private final double[] bias;

    private final EnumMap<Intent, Integer> intentIndex = new EnumMap<>(Intent.class);
    private final Intent[] indexIntent;

    public HashedIntentDetector() {
        this(Config.builder().build(), Model.emptyDefault());
    }

    public HashedIntentDetector(Config cfg, Model model) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(model, "model");

        this.dim = requirePow2(cfg.hashDim);
        this.mask = this.dim - 1;

        this.useBigrams = cfg.useBigrams;
        this.uniqueTokens = cfg.uniqueTokens;

        this.minConfidence = clamp01(cfg.minConfidence);
        this.minMargin = Math.max(0.0, cfg.minMargin);

        this.temperature = clamp(cfg.temperature, 0.25, 4.0);
        this.debugLog = cfg.debugLog;

        // Build intent index (exclude UNKNOWN)
        List<Intent> intents = new ArrayList<>();
        for (Intent it : Intent.values()) if (it != Intent.UNKNOWN) intents.add(it);

        this.indexIntent = intents.toArray(new Intent[0]);
        for (int i = 0; i < indexIntent.length; i++) intentIndex.put(indexIntent[i], i);

        int k = indexIntent.length;
        this.W = new double[k][dim];
        this.bias = new double[k];

        // Load model parameters (if provided)
        model.loadInto(this);
    }

    @Override
    public Intent detect(String userText) {
        if (userText == null) return Intent.UNKNOWN;
        String s0 = userText.trim();
        if (s0.isEmpty()) return Intent.UNKNOWN;

        String s = normalizeText(s0);

        TokenStats ts = tokenize(s);

        // Score logits
        double[] logits = new double[indexIntent.length];
        System.arraycopy(bias, 0, logits, 0, logits.length);

        // Emit hashed features and accumulate
        FeatureStream fs = new FeatureStream(ts.tokens, useBigrams, s);

        int featuresEmitted = 0;
        while (fs.hasNext()) {
            long featHash = fs.nextFeatureHash64();
            int idx = (int) featHash & mask;               // feature index
            int sign = ((featHash >>> 63) == 0) ? 1 : -1;  // signed hashing helps collisions a bit

            // Multiply by token count if uniqueTokens=false and this is token feature
            double mul = 1.0;
            if (!uniqueTokens && fs.isLastFeatureToken()) {
                String lastTok = fs.lastToken();
                mul = Math.max(1, ts.counts.getOrDefault(lastTok, 1));
            }

            for (int c = 0; c < logits.length; c++) {
                logits[c] += (W[c][idx] * sign) * mul;
            }

            if (++featuresEmitted >= cfgMaxFeatures(ts.tokens.size(), useBigrams)) break;
        }

        // Choose best + second
        int best = -1, second = -1;
        double bestLogit = Double.NEGATIVE_INFINITY, secondLogit = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < logits.length; i++) {
            double v = logits[i];
            if (v > bestLogit) {
                second = best; secondLogit = bestLogit;
                best = i; bestLogit = v;
            } else if (v > secondLogit) {
                second = i; secondLogit = v;
            }
        }

        if (best < 0) return Intent.UNKNOWN;
        if (second < 0) { second = best; secondLogit = bestLogit; }

        double conf = softmaxConfidence(logits, best, temperature);
        double margin = bestLogit - secondLogit;

        boolean ok = conf >= minConfidence && margin >= minMargin;

        if (debugLog && log.isDebugEnabled()) {
            log.debug("HashedIntent: text='{}' norm='{}' tok={} feat={} best={} conf={} margin={} ok={}",
                    truncate(s0, 220),
                    truncate(s, 220),
                    ts.tokenCount,
                    featuresEmitted,
                    indexIntent[best],
                    round(conf),
                    round(margin),
                    ok
            );
        }

        return ok ? indexIntent[best] : Intent.UNKNOWN;
    }

    // ---------------------------------------------------------------------
    // Model API (for loading weights from KB/LTM/config without hardcode)
    // ---------------------------------------------------------------------

    public interface Model {
        void loadInto(HashedIntentDetector detector);

        static Model emptyDefault() {
            return d -> {
                // Intentionally empty: truly data-driven by default.
                // You can load from JSON, DB, LTM, etc.
            };
        }
    }

    /**
     * Allows programmatic model injection if you already have weights in memory.
     * Shape must match [numIntentsExcludingUnknown][hashDim]
     */
    public void setWeights(double[][] w, double[] b) {
        if (w == null || b == null) throw new IllegalArgumentException("weights/bias are null");
        if (w.length != W.length) throw new IllegalArgumentException("weights intents mismatch");
        if (b.length != bias.length) throw new IllegalArgumentException("bias intents mismatch");
        for (int i = 0; i < w.length; i++) {
            if (w[i].length != dim) throw new IllegalArgumentException("weights dim mismatch at i=" + i);
            System.arraycopy(w[i], 0, W[i], 0, dim);
        }
        System.arraycopy(b, 0, bias, 0, bias.length);
    }

    // ---------------------------------------------------------------------
    // Config
    // ---------------------------------------------------------------------

    public static final class Config {
        final int hashDim;

        final boolean useBigrams;
        final boolean uniqueTokens;

        final double minConfidence;
        final double minMargin;

        final double temperature;
        final boolean debugLog;

        private Config(Builder b) {
            this.hashDim = b.hashDim;
            this.useBigrams = b.useBigrams;
            this.uniqueTokens = b.uniqueTokens;
            this.minConfidence = b.minConfidence;
            this.minMargin = b.minMargin;
            this.temperature = b.temperature;
            this.debugLog = b.debugLog;
        }

        public static Builder builder() {
            Builder b = new Builder();
            b.hashDim = 1 << 16;     // 65536 features (good default)
            b.useBigrams = true;
            b.uniqueTokens = true;
            b.minConfidence = 0.62;
            b.minMargin = 0.10;
            b.temperature = 1.0;
            b.debugLog = false;
            return b;
        }

        public static final class Builder {
            int hashDim;

            boolean useBigrams;
            boolean uniqueTokens;

            double minConfidence;
            double minMargin;

            double temperature;
            boolean debugLog;

            public Builder hashDim(int v) { this.hashDim = v; return this; }
            public Builder useBigrams(boolean v) { this.useBigrams = v; return this; }
            public Builder uniqueTokens(boolean v) { this.uniqueTokens = v; return this; }

            public Builder minConfidence(double v) { this.minConfidence = v; return this; }
            public Builder minMargin(double v) { this.minMargin = v; return this; }

            public Builder temperature(double v) { this.temperature = v; return this; }
            public Builder debugLog(boolean v) { this.debugLog = v; return this; }

            public Config build() { return new Config(this); }
        }
    }

    // ---------------------------------------------------------------------
    // Feature emission (no allocations for feature strings)
    // ---------------------------------------------------------------------

    private static final class FeatureStream {
        private final List<String> tokens;
        private final boolean useBigrams;
        private final String text;

        private int i = 0;
        private boolean emittingSignals = true;

        private int sigIdx = 0;
        private static final int SIG_QM = 0, SIG_EX = 1, SIG_CODE = 2, SIG_LEN_SHORT = 3, SIG_LEN_LONG = 4, SIG_WHYSHAPE = 5;
        private final boolean[] sig = new boolean[6];

        // last feature metadata
        private boolean lastIsToken = false;
        private String lastToken = "";

        FeatureStream(List<String> tokens, boolean useBigrams, String text) {
            this.tokens = tokens;
            this.useBigrams = useBigrams;
            this.text = text;

            // compute signals once
            sig[SIG_QM] = text.indexOf('?') >= 0;
            sig[SIG_EX] = text.indexOf('!') >= 0;
            sig[SIG_CODE] = looksLikeCode(text);

            int len = text.length();
            sig[SIG_LEN_SHORT] = len <= 24;
            sig[SIG_LEN_LONG] = len >= 220;

            String head = tokens.isEmpty() ? "" : tokens.get(0);
            sig[SIG_WHYSHAPE] = head.equals("как") || head.equals("почему") || head.equals("зачем") || head.equals("что")
                    || head.equals("where") || head.equals("why") || head.equals("how") || head.equals("what");
        }

        boolean hasNext() {
            if (emittingSignals) {
                for (int j = sigIdx; j < sig.length; j++) if (sig[j]) return true;
                emittingSignals = false;
            }
            if (i < tokens.size()) return true;
            if (useBigrams && tokens.size() >= 2 && i < tokens.size() - 1 + tokens.size()) return true;
            return false;
        }

        boolean isLastFeatureToken() { return lastIsToken; }
        String lastToken() { return lastToken; }

        long nextFeatureHash64() {
            // signals first
            if (emittingSignals) {
                while (sigIdx < sig.length && !sig[sigIdx]) sigIdx++;
                if (sigIdx < sig.length) {
                    lastIsToken = false;
                    lastToken = "";
                    return hash64Signal(sigIdx++);
                }
                emittingSignals = false;
            }

            // tokens then bigrams
            int tN = tokens.size();
            if (i < tN) {
                String t = tokens.get(i++);
                lastIsToken = true;
                lastToken = t;
                return hash64Token(t);
            }

            if (useBigrams && tN >= 2) {
                // map i in [tN, tN + (tN-1)) to bigram index
                int bi = i - tN; // 0..tN-2
                i++;
                if (bi >= 0 && bi < tN - 1) {
                    lastIsToken = false;
                    lastToken = "";
                    return hash64Bigram(tokens.get(bi), tokens.get(bi + 1));
                }
            }

            // fallback (should not happen if hasNext checked)
            lastIsToken = false;
            lastToken = "";
            return 0L;
        }

        private static long hash64Signal(int sigId) {
            // namespace "sig:"
            long h = 0xcbf29ce484222325L;
            h = fnv1a64(h, 's'); h = fnv1a64(h, 'i'); h = fnv1a64(h, 'g'); h = fnv1a64(h, ':');
            h = fnv1a64(h, (char) ('0' + sigId));
            return mix64(h);
        }

        private static long hash64Token(String t) {
            // namespace "tok:"
            long h = 0xcbf29ce484222325L;
            h = fnv1a64(h, 't'); h = fnv1a64(h, 'o'); h = fnv1a64(h, 'k'); h = fnv1a64(h, ':');
            for (int i = 0; i < t.length(); i++) h = fnv1a64(h, t.charAt(i));
            return mix64(h);
        }

        private static long hash64Bigram(String a, String b) {
            // namespace "bg:"
            long h = 0xcbf29ce484222325L;
            h = fnv1a64(h, 'b'); h = fnv1a64(h, 'g'); h = fnv1a64(h, ':');
            for (int i = 0; i < a.length(); i++) h = fnv1a64(h, a.charAt(i));
            h = fnv1a64(h, '_');
            for (int i = 0; i < b.length(); i++) h = fnv1a64(h, b.charAt(i));
            return mix64(h);
        }

        private static long fnv1a64(long h, char c) {
            h ^= (c & 0xFFFF);
            return h * 0x100000001b3L;
        }

        private static long mix64(long x) {
            x ^= (x >>> 33);
            x *= 0xff51afd7ed558ccdL;
            x ^= (x >>> 33);
            x *= 0xc4ceb9fe1a85ec53L;
            x ^= (x >>> 33);
            return x;
        }

        private static boolean looksLikeCode(String s) {
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
    }

    // ---------------------------------------------------------------------
    // Tokenization (same spirit as AdvancedIntentDetector but minimal allocs)
    // ---------------------------------------------------------------------

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

    private TokenStats tokenize(String s) {
        List<String> toks = WORD.matcher(s)
                .results()
                .map(MatchResult::group)
                .limit(MAX_TOKENS)
                .map(t -> t.toLowerCase(Locale.ROOT))
                .toList();

        List<String> finalToks;
        if (uniqueTokens) {
            LinkedHashSet<String> set = new LinkedHashSet<>(toks);
            finalToks = new ArrayList<>(set);
        } else {
            finalToks = new ArrayList<>(toks);
        }

        HashMap<String, Integer> counts = new HashMap<>();
        if (!uniqueTokens) {
            for (String t : toks) counts.merge(t, 1, Integer::sum);
        } else {
            for (String t : finalToks) counts.put(t, 1);
        }

        return new TokenStats(finalToks, counts);
    }

    // ---------------------------------------------------------------------
    // Math
    // ---------------------------------------------------------------------

    private static double softmaxConfidence(double[] logits, int best, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logits) max = Math.max(max, v);

        double sum = 0.0;
        double bestExp = 0.0;

        for (int i = 0; i < logits.length; i++) {
            double z = (logits[i] - max) / temperature;
            z = clamp(z, -30.0, 30.0);
            double ex = Math.exp(z);
            sum += ex;
            if (i == best) bestExp = ex;
        }

        if (sum <= 0.0) return 0.0;
        return bestExp / sum;
    }

    // ---------------------------------------------------------------------
    // Text normalization
    // ---------------------------------------------------------------------

    private static String normalizeText(String s) {
        String x = Normalizer.normalize(s, Normalizer.Form.NFKC);
        x = x.replace('\u2014', '-')
                .replace('\u2013', '-')
                .replace('\u00A0', ' ');
        return x;
    }

    // ---------------------------------------------------------------------
    // Utilities
    // ---------------------------------------------------------------------

    private static int cfgMaxFeatures(int tokenCount, boolean useBigrams) {
        // cheap deterministic cap
        int base = tokenCount;
        int bi = useBigrams ? Math.max(0, tokenCount - 1) : 0;
        int sig = 6;
        return Math.min(256, base + bi + sig);
    }

    private static int requirePow2(int v) {
        if (v < 256) throw new IllegalArgumentException("hashDim too small: " + v);
        if ((v & (v - 1)) != 0) throw new IllegalArgumentException("hashDim must be power-of-two: " + v);
        return v;
    }

    private static double clamp01(double v) { return clamp(v, 0.0, 1.0); }

    private static double clamp(double v, double lo, double hi) {
        if (!Double.isFinite(v)) return lo;
        return Math.max(lo, Math.min(hi, v));
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, Math.max(0, max - 1)) + "…";
    }

    private static String round(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }
}