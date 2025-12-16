package org.calista.arasaka.ai.think;

public interface ThoughtCycleEngine {
    ThoughtResult think(String userText, long seed);
}