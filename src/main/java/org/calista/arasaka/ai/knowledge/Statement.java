// Statement.java
package org.calista.arasaka.ai.knowledge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.time.Instant;
import java.util.*;

/**
 * Statement — единица long-term memory.
 *
 * <p>
 * Поля public — удобно для Jackson и минимального бойлерплейта.
 * Без "магии": только метаданные, нужные для ранжирования/валидирования/TTL.
 * </p>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class Statement {

    /** Stable unique ID. */
    public String id;

    /** Human-readable content (fact/rule/episode/etc). */
    public String text;

    /**
     * Type hint for retrieval/ranking.
     * examples: fact|rule|episode|instruction|preference|profile|...
     */
    public String type = "fact";

    /** Weight/importance (>= 0). */
    public double weight = 1.0;

    /** Confidence 0..1 (model/user/system). */
    public double confidence = 1.0;

    /** Optional tags for grouping and query expansion. */
    public List<String> tags = List.of();

    /** Where it came from: dialogue/file/system/etc. */
    public String source;

    /** Timestamps for recency scoring. */
    public long createdAtEpochMs = 0L;
    public long updatedAtEpochMs = 0L;

    /** TTL: if now > expiresAtEpochMs => ignore in retrieval. 0 => never. */
    public long expiresAtEpochMs = 0L;

    /** Free-form metadata (enterprise extensibility). */
    public Map<String, Object> meta = new HashMap<>();

    public void validate() {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("Statement.id is required");
        if (text == null || text.isBlank()) throw new IllegalArgumentException("Statement.text is required");

        if (type == null || type.isBlank()) type = "fact";
        type = type.trim().toLowerCase(Locale.ROOT);

        if (!Double.isFinite(weight) || weight < 0) weight = 1.0;
        if (!Double.isFinite(confidence)) confidence = 1.0;
        confidence = Math.max(0.0, Math.min(1.0, confidence));

        if (tags == null) tags = List.of();
        // normalize tags (lowercase, non-empty, unique but stable order)
        if (!tags.isEmpty()) {
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
        if (createdAtEpochMs <= 0) createdAtEpochMs = now;
        if (updatedAtEpochMs <= 0) updatedAtEpochMs = createdAtEpochMs;
        if (updatedAtEpochMs < createdAtEpochMs) updatedAtEpochMs = createdAtEpochMs;
    }

    /** Enterprise-friendly: "touch" timestamps on upsert (unless caller manages them). */
    public void touchUpdatedNow() {
        long now = System.currentTimeMillis();
        if (createdAtEpochMs <= 0) createdAtEpochMs = now;
        updatedAtEpochMs = now;
    }

    public boolean isExpiredNow() {
        return expiresAtEpochMs > 0 && System.currentTimeMillis() > expiresAtEpochMs;
    }

    public Instant createdAt() {
        return Instant.ofEpochMilli(createdAtEpochMs <= 0 ? System.currentTimeMillis() : createdAtEpochMs);
    }

    public Instant updatedAt() {
        return Instant.ofEpochMilli(updatedAtEpochMs <= 0 ? System.currentTimeMillis() : updatedAtEpochMs);
    }
}