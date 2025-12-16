package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.List;

public final class GreetingStrategy implements ResponseStrategy {
    @Override
    public boolean supports(Intent intent) {
        return intent == Intent.GREETING;
    }

    @Override
    public String generate(String userText, List<Statement> context, ThoughtState state) {
        // Нормальный человеческий ответ + мягкий переход к цели
        return "Привет! Я на месте 🙂\n" +
                "Скажи, что сейчас важнее: (1) улучшить мышление/генерацию, (2) обучение памяти, (3) архитектуру/ООП.\n" +
                "Если дашь 1–2 примера ожидаемых диалогов — я подстрою поведение.";
    }
}