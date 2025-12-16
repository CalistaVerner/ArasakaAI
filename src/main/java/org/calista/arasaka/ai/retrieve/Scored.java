package org.calista.arasaka.ai.retrieve;

import java.util.Objects;

/**
 * Small immutable scored wrapper.
 * Comparable uses descending score order, with a stable tie-break.
 */
public final class Scored<T> implements Comparable<Scored<T>> {
    public final T item;
    public final double score;

    public Scored(T item, double score) {
        this.item = item;
        this.score = score;
    }

    public static <T> Scored<T> of(T item, double score) {
        return new Scored<>(item, score);
    }

    /**
     * Stable key used for deterministic tie-breaks. Avoid identity-based hash codes.
     */
    public String stableKey() {
        if (item == null) return "null";

        // Fast path for common pattern: public fields id/text.
        // We intentionally avoid any dependency on a specific model class here.
        try {
            Class<?> c = item.getClass();

            String id = null;
            try {
                var fId = c.getField("id");
                Object v = fId.get(item);
                if (v instanceof String s && !s.isBlank()) id = s;
            } catch (NoSuchFieldException ignored) {
            }

            String text = null;
            try {
                var fText = c.getField("text");
                Object v = fText.get(item);
                if (v instanceof String s && !s.isBlank()) text = s;
            } catch (NoSuchFieldException ignored) {
            }

            if (id != null) return "id:" + id;
            if (text != null) return "tx:" + text;
        } catch (Throwable ignored) {
        }

        return String.valueOf(item);
    }

    @Override
    public int compareTo(Scored<T> o) {
        int c = Double.compare(o.score, this.score);
        if (c != 0) return c;
        return this.stableKey().compareTo(o.stableKey());
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) return true;
        if (!(other instanceof Scored<?> s)) return false;
        return Double.doubleToLongBits(score) == Double.doubleToLongBits(s.score)
                && Objects.equals(item, s.item);
    }

    @Override
    public int hashCode() {
        return Objects.hash(item, Double.doubleToLongBits(score));
    }

    @Override
    public String toString() {
        return "Scored{score=" + score + ", item=" + item + '}';
    }
}