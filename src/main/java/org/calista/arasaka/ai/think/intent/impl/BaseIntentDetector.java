// FILE: AdvancedIntentDetector.java
package org.calista.arasaka.ai.think.intent.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.think.intent.Intent;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * AdvancedIntentDetector — data-driven, deterministic intent classification.
 *
 * Design goals:
 *  - Deterministic for same input + same weight snapshot.
 *  - "No hardcode" in production: weights should be loaded from config/corpora/LTM as data.
 *  - Dev bootstrap is optional (autoBootstrap=true) for quick bring-up.
 *
 * Weight format (TSV; stable & simple):
 *   # optional comments
 *   meta\tversion\t<STRING>
 *   meta\tupdatedAt\t<ISO_INSTANT>
 *   INTENT\tTOKEN\tWEIGHT
 *
 * Example:
 *   meta\tversion\t2025-12-17.a
 *   meta\tupdatedAt\t2025-12-17T12:00:00Z
 *   GREETING\tпривет\t1.0
 *   QUESTION\tпочему\t0.65
 */
public final class BaseIntentDetector implements org.calista.arasaka.ai.think.intent.IntentDetector {

    private static final Logger log = LogManager.getLogger(BaseIntentDetector.class);

    private static final Charset FILE_CHARSET = StandardCharsets.UTF_8;

    // tokens >=2, letters/digits/_ in any Unicode script
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{Nd}_]{2,}");

    private final Config cfg;

    // immutable snapshot: weights + meta
    private final AtomicReference<Snapshot> snap = new AtomicReference<>();

    public BaseIntentDetector(Config config) {
        this.cfg = (config == null ? Config.builder().build() : config).freezeAndValidate();

        Snapshot loaded = null;

        // 1) Load from file if configured
        if (cfg.weightsPath != null) {
            loaded = tryLoad(cfg.weightsPath);
        }

        // 2) If not loaded and autoBootstrap enabled (dev), create bootstrap weights
        if (loaded == null && cfg.autoBootstrap) {
            loaded = Snapshot.bootstrapDev("bootstrap@" + Instant.now().toString(), Instant.now().toString());
        }

        // 3) If still null: empty snapshot (UNKNOWN only)
        if (loaded == null) {
            loaded = Snapshot.empty("empty", Instant.now().toString());
        }

        this.snap.set(loaded);

        if (log.isInfoEnabled()) {
            log.info("AdvancedIntentDetector init autoBootstrap={} weightsPath={} version={} updatedAt={} intentsLoaded={}",
                    cfg.autoBootstrap,
                    cfg.weightsPath == null ? "<none>" : cfg.weightsPath.toString(),
                    loaded.meta.version,
                    loaded.meta.updatedAtIso,
                    loaded.weights.size());
        }
    }

    @Override
    public Intent detect(String text) {
        final String s = normalize(text);
        if (s.isEmpty()) return Intent.UNKNOWN;

        final Snapshot ss = snap.get();
        final WeightsMeta meta = ss.meta;
        final Map<Intent, Map<String, Double>> weights = ss.weights;

        if (weights.isEmpty()) {
            return Intent.UNKNOWN;
        }

        // token -> tf
        final Map<String, Integer> tf = termFreq(s);
        if (tf.isEmpty()) return Intent.UNKNOWN;

        Intent best = Intent.UNKNOWN;
        double bestScore = cfg.unknownBias; // lets UNKNOWN win when nothing matches

        // Deterministic iteration order (EnumMap keys already stable; still guard)
        for (Intent intent : Intent.values()) {
            if (intent == Intent.UNKNOWN) continue;

            Map<String, Double> w = weights.get(intent);
            if (w == null || w.isEmpty()) continue;

            double score = 0.0;

            // sum(w[token] * (1 + log(1+tf)))
            for (Map.Entry<String, Integer> e : tf.entrySet()) {
                Double ww = w.get(e.getKey());
                if (ww == null) continue;

                int f = e.getValue();
                double tfw = 1.0 + Math.log1p(Math.max(0, f));
                score += ww * tfw;
            }

            // length normalization (optional)
            if (cfg.lengthNorm && tf.size() > 0) {
                score /= Math.sqrt(tf.size());
            }

            // small prior to prevent ties from flipping unpredictably
            score += cfg.intentPrior.getOrDefault(intent, 0.0);

            if (score > bestScore) {
                bestScore = score;
                best = intent;
            }
        }

        // threshold gate: if best is too weak -> UNKNOWN
        if (best != Intent.UNKNOWN && bestScore < cfg.minScore) {
            return Intent.UNKNOWN;
        }

        if (log.isTraceEnabled()) {
            log.trace("detect v={} qTok={} best={} score={}",
                    meta.version, tf.size(), best.name(), fmt(bestScore));
        }

        return best;
    }

    // -------------------- Public admin ops (optional) --------------------

    /**
     * Reload weights from cfg.weightsPath (if set). Deterministic, atomic swap.
     * Returns true if reload succeeded.
     */
    public boolean reload() {
        if (cfg.weightsPath == null) return false;
        Snapshot loaded = tryLoad(cfg.weightsPath);
        if (loaded == null) return false;
        snap.set(loaded);
        if (log.isInfoEnabled()) {
            log.info("AdvancedIntentDetector reloaded weightsPath={} version={} updatedAt={} intentsLoaded={}",
                    cfg.weightsPath, loaded.meta.version, loaded.meta.updatedAtIso, loaded.weights.size());
        }
        return true;
    }

    /**
     * Persist current snapshot to a file (TSV format described above).
     */
    public void saveSnapshot(Path out) throws IOException {
        Objects.requireNonNull(out, "out");
        Snapshot ss = snap.get();

        // ensure parent
        Path parent = out.getParent();
        if (parent != null) Files.createDirectories(parent);

        try (BufferedWriter w = Files.newBufferedWriter(out, FILE_CHARSET)) {
            w.write("meta\tversion\t" + ss.meta.version + "\n");
            w.write("meta\tupdatedAt\t" + ss.meta.updatedAtIso + "\n");

            // stable output ordering: Intent enum order, token lexical
            for (Intent intent : Intent.values()) {
                if (intent == Intent.UNKNOWN) continue;
                Map<String, Double> m = ss.weights.get(intent);
                if (m == null || m.isEmpty()) continue;

                ArrayList<String> keys = new ArrayList<>(m.keySet());
                keys.sort(String::compareTo);

                for (String tok : keys) {
                    double val = m.get(tok);
                    w.write(intent.name());
                    w.write('\t');
                    w.write(tok);
                    w.write('\t');
                    w.write(Double.toString(val));
                    w.write('\n');
                }
            }
        }
    }

    /**
     * Replace snapshot (atomic). Useful if you load weights from corpora/LTM externally.
     * This keeps detector "no hardcode" while letting enterprise pipeline manage data.
     */
    public void setSnapshot(Map<Intent, Map<String, Double>> weights, String version, String updatedAtIso) {
        Snapshot ss = Snapshot.of(weights, version, updatedAtIso);
        snap.set(ss);
    }

    public String getVersion() {
        return snap.get().meta.version;
    }

    // -------------------- Loading / parsing --------------------

    private Snapshot tryLoad(Path p) {
        try {
            if (!Files.exists(p) || !Files.isRegularFile(p)) {
                log.warn("Intent weights file not found: {}", p);
                return null;
            }
            Snapshot ss = Snapshot.loadTsv(p);
            if (ss == null) return null;
            if (ss.weights.isEmpty()) {
                log.warn("Intent weights loaded but empty: {}", p);
            }
            return ss;
        } catch (Exception e) {
            log.warn("Failed to load intent weights from {}: {}", p, e.toString());
            return null;
        }
    }

    // -------------------- Tokenization --------------------

    private static String normalize(String s) {
        if (s == null) return "";
        String x = s.trim();
        if (x.isEmpty()) return "";
        x = Normalizer.normalize(x, Normalizer.Form.NFKC);
        x = x.replace('\u00A0', ' ');
        x = x.replaceAll("\\s+", " ").trim();
        return x.toLowerCase(Locale.ROOT);
    }

    private static Map<String, Integer> termFreq(String s) {
        HashMap<String, Integer> tf = new HashMap<>(32);
        Matcher m = WORD.matcher(s);
        while (m.find()) {
            String t = m.group();
            if (t == null || t.isBlank()) continue;
            tf.merge(t, 1, Integer::sum);
            if (tf.size() >= 256) break;
        }
        return tf;
    }

    // -------------------- Config --------------------

    public static final class Config {
        private boolean frozen;

        /** Dev-only: if true and no file loaded => bootstrap minimal weights */
        public boolean autoBootstrap = true;

        /** Optional weights path (TSV). In prod set this and autoBootstrap=false. */
        public Path weightsPath;

        /** Minimum score to accept non-UNKNOWN */
        public double minScore = 0.22;

        /** Bias for UNKNOWN as baseline score */
        public double unknownBias = 0.0;

        /** Normalize score by query length */
        public boolean lengthNorm = true;

        /** Optional priors */
        public EnumMap<Intent, Double> intentPrior = new EnumMap<>(Intent.class);

        public static Builder builder() {
            return new Builder();
        }

        public Config freezeAndValidate() {
            if (frozen) return this;

            if (!Double.isFinite(minScore)) minScore = 0.22;
            if (!Double.isFinite(unknownBias)) unknownBias = 0.0;

            minScore = clamp(minScore, -10, 10);
            unknownBias = clamp(unknownBias, -10, 10);

            if (intentPrior == null) intentPrior = new EnumMap<>(Intent.class);

            // avoid UNKNOWN prior abuse
            intentPrior.remove(Intent.UNKNOWN);

            frozen = true;
            return this;
        }

        private static double clamp(double v, double lo, double hi) {
            if (v < lo) return lo;
            if (v > hi) return hi;
            return v;
        }

        public static final class Builder {
            private final Config c = new Config();

            public Builder autoBootstrap(boolean v) { c.autoBootstrap = v; return this; }
            public Builder weightsPath(Path p) { c.weightsPath = p; return this; }
            public Builder minScore(double v) { c.minScore = v; return this; }
            public Builder unknownBias(double v) { c.unknownBias = v; return this; }
            public Builder lengthNorm(boolean v) { c.lengthNorm = v; return this; }

            public Builder prior(Intent intent, double v) {
                if (intent != null && intent != Intent.UNKNOWN) c.intentPrior.put(intent, v);
                return this;
            }

            public Config build() { return c.freezeAndValidate(); }
        }
    }

    // -------------------- Snapshot --------------------

    private static final class Snapshot {
        final WeightsMeta meta;
        final EnumMap<Intent, Map<String, Double>> weights;

        private Snapshot(WeightsMeta meta, EnumMap<Intent, Map<String, Double>> weights) {
            this.meta = meta;
            this.weights = weights;
        }

        static Snapshot empty(String version, String updatedAtIso) {
            return new Snapshot(new WeightsMeta(version, updatedAtIso), new EnumMap<>(Intent.class));
        }

        static Snapshot of(Map<Intent, Map<String, Double>> in, String version, String updatedAtIso) {
            EnumMap<Intent, Map<String, Double>> out = new EnumMap<>(Intent.class);
            if (in != null) {
                for (Map.Entry<Intent, Map<String, Double>> e : in.entrySet()) {
                    Intent it = e.getKey();
                    if (it == null || it == Intent.UNKNOWN) continue;
                    Map<String, Double> m = e.getValue();
                    if (m == null || m.isEmpty()) continue;

                    // make immutable & sanitized snapshot
                    HashMap<String, Double> mm = new HashMap<>(m.size());
                    for (Map.Entry<String, Double> w : m.entrySet()) {
                        String tok = w.getKey();
                        Double val = w.getValue();
                        if (tok == null || tok.isBlank() || val == null || !Double.isFinite(val)) continue;
                        String t = tok.trim().toLowerCase(Locale.ROOT);
                        if (t.length() < 2) continue;
                        mm.put(t, val);
                    }
                    if (!mm.isEmpty()) out.put(it, Collections.unmodifiableMap(mm));
                }
            }
            return new Snapshot(new WeightsMeta(
                    (version == null || version.isBlank()) ? "manual" : version.trim(),
                    (updatedAtIso == null || updatedAtIso.isBlank()) ? Instant.now().toString() : updatedAtIso.trim()
            ), out);
        }

        static Snapshot loadTsv(Path p) throws IOException {
            String version = "file:" + p.getFileName();
            String updatedAt = Instant.now().toString();

            EnumMap<Intent, Map<String, Double>> tmp = new EnumMap<>(Intent.class);

            try (BufferedReader r = Files.newBufferedReader(p, FILE_CHARSET)) {
                String line;
                int ln = 0;
                while ((line = r.readLine()) != null) {
                    ln++;
                    String s = line.trim();
                    if (s.isEmpty() || s.startsWith("#")) continue;

                    String[] parts = s.split("\t");
                    if (parts.length < 3) continue;

                    if ("meta".equalsIgnoreCase(parts[0])) {
                        String k = parts[1].trim();
                        String v = parts[2].trim();
                        if ("version".equalsIgnoreCase(k) && !v.isBlank()) version = v;
                        if ("updatedAt".equalsIgnoreCase(k) && !v.isBlank()) updatedAt = v;
                        continue;
                    }

                    Intent intent;
                    try {
                        intent = Intent.valueOf(parts[0].trim().toUpperCase(Locale.ROOT));
                    } catch (Exception ex) {
                        // ignore unknown intent rows
                        continue;
                    }
                    if (intent == Intent.UNKNOWN) continue;

                    String tok = parts[1].trim().toLowerCase(Locale.ROOT);
                    if (tok.isBlank() || tok.length() < 2) continue;

                    double w;
                    try {
                        w = Double.parseDouble(parts[2].trim());
                    } catch (Exception ex) {
                        continue;
                    }
                    if (!Double.isFinite(w)) continue;

                    tmp.computeIfAbsent(intent, k -> new HashMap<>()).put(tok, w);
                }
            }

            // freeze
            EnumMap<Intent, Map<String, Double>> out = new EnumMap<>(Intent.class);
            for (Map.Entry<Intent, Map<String, Double>> e : tmp.entrySet()) {
                Map<String, Double> m = e.getValue();
                if (m == null || m.isEmpty()) continue;
                out.put(e.getKey(), Collections.unmodifiableMap(new HashMap<>(m)));
            }

            return new Snapshot(new WeightsMeta(version, updatedAt), out);
        }

        /**
         * Dev bootstrap: minimal generic weights. This is ONLY used when autoBootstrap=true.
         * In prod you should provide weightsPath or call setSnapshot(...).
         */
        static Snapshot bootstrapDev(String version, String updatedAtIso) {
            EnumMap<Intent, Map<String, Double>> w = new EnumMap<>(Intent.class);

            // Minimal, general-purpose seeds (small and generic; NOT business-specific)
            w.put(Intent.GREETING, Map.of(
                    "привет", 1.00, "здравствуйте", 1.00, "здравствуй", 0.95,
                    "hello", 1.00, "hi", 0.90, "hey", 0.85
            ));
            w.put(Intent.QUESTION, Map.of(
                    "как", 0.55, "почему", 0.70, "зачем", 0.55, "что", 0.55,
                    "когда", 0.55, "где", 0.55, "кто", 0.55, "какой", 0.55,
                    "какие", 0.55, "сколько", 0.60
            ));
            w.put(Intent.REQUEST, Map.of(
                    "сделай", 0.80, "создай", 0.80, "напиши", 0.75, "покажи", 0.70,
                    "build", 0.70, "make", 0.65, "create", 0.70, "generate", 0.65
            ));

            // freeze maps
            EnumMap<Intent, Map<String, Double>> out = new EnumMap<>(Intent.class);
            for (Map.Entry<Intent, Map<String, Double>> e : w.entrySet()) {
                out.put(e.getKey(), Collections.unmodifiableMap(new HashMap<>(e.getValue())));
            }
            return new Snapshot(new WeightsMeta(version, updatedAtIso), out);
        }
    }

    private record WeightsMeta(String version, String updatedAtIso) {}

    // -------------------- Formatting --------------------

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.3f", v);
    }
}