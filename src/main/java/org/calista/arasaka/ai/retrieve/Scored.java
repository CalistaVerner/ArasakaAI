package org.calista.arasaka.ai.retrieve;

public final class Scored<T> {
    public final T item;
    public final double score;

    public Scored(T item, double score) {
        this.item = item;
        this.score = score;
    }
}