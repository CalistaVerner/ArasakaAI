package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.List;

public final class GreetingStrategy implements ResponseStrategy {
    private final ResponseStrategy delegate;

    public GreetingStrategy() {
        this(new ContextAnswerStrategy());
    }

    public GreetingStrategy(ResponseStrategy delegate) {
        this.delegate = java.util.Objects.requireNonNull(delegate);
    }

    @Override
    public boolean supports(Intent intent) {
        return intent == Intent.GREETING;
    }

    @Override
    public String generate(String userText, List<Statement> context, ThoughtState state) {
        // No scripted phrases. Just steer formatting towards a short answer.
        if (state != null && state.tags != null) {
            state.tags.putIfAbsent("response.sections", "summary,actions");
            state.tags.putIfAbsent("response.style", "plain");
        }
        return delegate.generate(userText, context, state);
    }
}