package org.calista.arasaka.ai.knowledge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.time.Instant;
import java.util.*;

/**
 * Statement — единица long-term memory / retrieval evidence.
 *
 * Enterprise goals:
 * - public fields for Jackson (minimal boilerplate)
 * - deterministic validation/normalization
 * - explicit ranking signals: weight, confidence, priority
 * - copy-on-write helpers for safe compression/rerank
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class Statement {

    // --------- Identity ---------

    /** Stable unique ID. */
    public String id;

    // --------- Content ---------

    /** Human-readable content (fact/rule/episode/etc). */
    public String text;

    /**
     * Type hint for retrieval/ranking.
     * examples: fact|rule|episode|instruction|preference|profile|...
     */
    public String type = "fact";

    // --------- Ranking signals ---------

    /** Weight/importance (>= 0). */
    public double weight = 1.0;

    /** Confidence 0..1 (model/user/system). */
    public double confidence = 1.0;

    /**
     * Priority 0..1 (enterprise routing/ranking signal).
     *
     * Meaning:
     * - 0.0  = lowest priority evidence
     * - 1.0  = highest priority evidence
     *
     * Use-cases:
     * - prefer "profile/preference" over generic facts
     * - prefer recent verified evidence over weak sources
     * - policy overrides without touching weight/confidence
     */
    public double priority = 0.5;

    /** Optional tags for grouping and query expansion. */
    public List<String> tags = List.of();

    /** Where it came from: dialogue/file/system/etc. */
    public String source;

    // --------- Time & lifecycle ---------

    /** Timestamps for recency scoring. */
    public long createdAtEpochMs = 0L;
    public long updatedAtEpochMs = 0L;

    /** TTL: if now > expiresAtEpochMs => ignore in retrieval. 0 => never. */
    public long expiresAtEpochMs = 0L;

    /** Free-form metadata (enterprise extensibility). */
    public Map<String, Object> meta = new HashMap<>();

    // ---------------------------------------------------------------------
    // Validation / normalization
    // ---------------------------------------------------------------------

    public void validate() {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("Statement.id is required");
        if (text == null || text.isBlank()) throw new IllegalArgumentException("Statement.text is required");

        if (type == null || type.isBlank()) type = "fact";
        type = type.trim().toLowerCase(Locale.ROOT);

        if (!Double.isFinite(weight) || weight < 0.0) weight = 1.0;

        if (!Double.isFinite(confidence)) confidence = 1.0;
        confidence = clamp01(confidence);

        if (!Double.isFinite(priority)) priority = 0.5;
        priority = clamp01(priority);

        if (tags == null || tags.isEmpty()) {
            tags = List.of();
        } else {
            // normalize tags (lowercase, non-empty, unique but stable order)
            LinkedHashSet<String> norm = new LinkedHashSet<>();
            for (String t : tags) {
                if (t == null) continue;
                String x = t.trim().toLowerCase(Locale.ROOT);
                if (x.isBlank()) continue;
                norm.add(x);
            }
            tags = norm.isEmpty() ? List.of() : new ArrayList<>(norm);
        }

        long now = System.currentTimeMillis();
        if (createdAtEpochMs <= 0L) createdAtEpochMs = now;
        if (updatedAtEpochMs <= 0L) updatedAtEpochMs = createdAtEpochMs;
        if (updatedAtEpochMs < createdAtEpochMs) updatedAtEpochMs = createdAtEpochMs;

        if (meta == null) meta = new HashMap<>();
    }

    // ---------------------------------------------------------------------
    // Copy helpers (for safe compression/rerank)
    // ---------------------------------------------------------------------

    /**
     * Shallow structural copy.
     *
     * <p>Notes:
     * - tags/meta references are shared (shallow) for performance
     * - use {@link #withMeta(String, Object)} / {@link #withTags(List)} if you need copy-on-write
     */
    public Statement copyShallow() {
        Statement s = new Statement();
        s.id = this.id;
        s.text = this.text;
        s.type = this.type;
        s.weight = this.weight;
        s.confidence = this.confidence;
        s.priority = this.priority;
        s.tags = this.tags;
        s.source = this.source;
        s.createdAtEpochMs = this.createdAtEpochMs;
        s.updatedAtEpochMs = this.updatedAtEpochMs;
        s.expiresAtEpochMs = this.expiresAtEpochMs;
        s.meta = this.meta;
        return s;
    }

    /** Copy-on-write: returns a shallow copy with replaced text. */
    public Statement withText(String newText) {
        Statement s = copyShallow();
        s.text = newText;
        return s;
    }

    /** Copy-on-write: returns a shallow copy with replaced type. */
    public Statement withType(String newType) {
        Statement s = copyShallow();
        s.type = (newType == null || newType.isBlank()) ? "fact" : newType.trim().toLowerCase(Locale.ROOT);
        return s;
    }

    /** Copy-on-write: returns a shallow copy with replaced tags (no normalization here; caller can call validate()). */
    public Statement withTags(List<String> newTags) {
        Statement s = copyShallow();
        s.tags = (newTags == null || newTags.isEmpty()) ? List.of() : new ArrayList<>(newTags);
        return s;
    }

    /**
     * Copy-on-write: returns a shallow copy with meta key set.
     * Meta map is copied (one level) to avoid cross-thread mutation issues.
     */
    public Statement withMeta(String key, Object value) {
        Statement s = copyShallow();
        HashMap<String, Object> m = (this.meta == null) ? new HashMap<>() : new HashMap<>(this.meta);
        if (key != null && !key.isBlank()) {
            if (value == null) m.remove(key);
            else m.put(key, value);
        }
        s.meta = m;
        return s;
    }

    /**
     * Copy-on-write: add a single tag (lowercased/trimmed), stable order, no duplicates.
     * This is handy for annotating evidence during rerank/compress.
     */
    public Statement withTag(String tag) {
        if (tag == null) return this;
        String x = tag.trim().toLowerCase(Locale.ROOT);
        if (x.isBlank()) return this;

        List<String> cur = (this.tags == null) ? List.of() : this.tags;
        if (cur.contains(x)) return this;

        Statement s = copyShallow();
        ArrayList<String> t = new ArrayList<>(cur.size() + 1);
        t.addAll(cur);
        t.add(x);
        s.tags = t;
        return s;
    }

    // ---------------------------------------------------------------------
    // Lifecycle helpers
    // ---------------------------------------------------------------------

    /** Enterprise-friendly: "touch" timestamps on upsert (unless caller manages them). */
    public void touchUpdatedNow() {
        long now = System.currentTimeMillis();
        if (createdAtEpochMs <= 0L) createdAtEpochMs = now;
        updatedAtEpochMs = now;
    }

    public boolean isExpiredNow() {
        return expiresAtEpochMs > 0L && System.currentTimeMillis() > expiresAtEpochMs;
    }

    public Instant createdAt() {
        return Instant.ofEpochMilli(createdAtEpochMs <= 0L ? System.currentTimeMillis() : createdAtEpochMs);
    }

    public Instant updatedAt() {
        return Instant.ofEpochMilli(updatedAtEpochMs <= 0L ? System.currentTimeMillis() : updatedAtEpochMs);
    }

    // ---------------------------------------------------------------------
    // Derived helpers (transparent, no magic)
    // ---------------------------------------------------------------------

    /** Base evidence strength signal. */
    public double effectiveWeight() {
        return weight * confidence;
    }

    private static double clamp01(double x) {
        if (x < 0.0) return 0.0;
        if (x > 1.0) return 1.0;
        return x;
    }

    @Override
    public String toString() {
        return "Statement{"
                + "id='" + id + '\''
                + ", type='" + type + '\''
                + ", w=" + weight
                + ", conf=" + confidence
                + ", pr=" + priority
                + ", expired=" + isExpiredNow()
                + '}';
    }
}