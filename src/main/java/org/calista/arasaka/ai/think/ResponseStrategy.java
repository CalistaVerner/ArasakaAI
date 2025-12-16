package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.List;

public interface ResponseStrategy {
    boolean supports(Intent intent);
    String generate(String userText, List<Statement> context, ThoughtState state);
}