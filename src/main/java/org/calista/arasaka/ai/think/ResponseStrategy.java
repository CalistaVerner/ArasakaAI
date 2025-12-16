package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.List;

public interface ResponseStrategy {
    /**
     * Backward-compatible hook for router-based designs.
     *
     * <p>In the modern "single smart strategy" pipeline, the engine may ignore
     * intent routing entirely and always call {@link #generate}.</p>
     */
    default boolean supports(Intent intent) {
        return true;
    }

    String generate(String userText, List<Statement> context, ThoughtState state);
}