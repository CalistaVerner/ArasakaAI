package org.calista.arasaka.ai.think.engine;

import org.calista.arasaka.ai.think.ThoughtResult;

public interface ThoughtCycleEngine {
    ThoughtResult think(String userText, long seed);
}