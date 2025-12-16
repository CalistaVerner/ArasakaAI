package org.calista.arasaka.ai.think;

import java.util.List;

public final class ThoughtResult {
    public final String answer;
    public final List<String> trace; // полезно для отладки/обучения/логов

    public ThoughtResult(String answer, List<String> trace) {
        this.answer = answer;
        this.trace = trace;
    }
}