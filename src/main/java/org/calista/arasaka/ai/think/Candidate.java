package org.calista.arasaka.ai.think;

public final class Candidate {
    public final String text;
    public final double score;
    public final String critique;

    public Candidate(String text, double score, String critique) {
        this.text = text;
        this.score = score;
        this.critique = critique;
    }
}