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
