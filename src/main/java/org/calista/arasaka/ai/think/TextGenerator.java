package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import java.util.List;

/**
 * Pluggable generation backend.
 *
 * Implementations may call your BeamSearch/TensorFlow generator (token-by-token) or any other model.
 * The contract is intentionally minimal to avoid locking the system to a specific ML stack.
 */
@FunctionalInterface
public interface TextGenerator {
    /**
     * Generate a response given the user input, selected context, and current thought state.
     * Implementations should be deterministic for the same inputs.
     */
    String generate(String userText, List<Statement> context, ThoughtState state);
}