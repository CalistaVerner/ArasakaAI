package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.List;

public final class RequestStrategy implements ResponseStrategy {
    private final ResponseStrategy delegate;

    public RequestStrategy() {
        this(new ContextAnswerStrategy());
    }

    public RequestStrategy(ResponseStrategy delegate) {
        this.delegate = java.util.Objects.requireNonNull(delegate);
    }

    @Override
    public boolean supports(Intent intent) {
        return intent == Intent.REQUEST;
    }

    @Override
    public String generate(String userText, List<Statement> context, ThoughtState state) {
        // No scripted phrasing. Prefer actionable plan + evidence.
        if (state != null && state.tags != null) {
            state.tags.putIfAbsent("response.sections", "summary,evidence,actions");
            state.tags.putIfAbsent("response.style", "md");
            // Encourage more actions for request-like intent.
            state.tags.putIfAbsent("response.label.actions", defaultActionsLabel(userText));
        }
        return delegate.generate(userText, context, state);
    }

    private static String defaultActionsLabel(String userText) {
        boolean ru = userText != null && userText.chars().anyMatch(c -> c >= 'А' && c <= 'я');
        return ru ? "План выполнения" : "Implementation plan";
    }
}
