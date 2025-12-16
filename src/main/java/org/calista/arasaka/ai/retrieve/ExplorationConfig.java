package org.calista.arasaka.ai.retrieve;

public final class ExplorationConfig {
    public final double temperature;
    public final int topK;

    public ExplorationConfig(double temperature, int topK) {
        if (temperature <= 0.0) throw new IllegalArgumentException("temperature must be > 0");
        if (topK < 1) throw new IllegalArgumentException("topK must be >= 1");
        this.temperature = temperature;
        this.topK = topK;
    }
}