package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.List;

public final class RequestStrategy implements ResponseStrategy {
    @Override
    public boolean supports(Intent intent) {
        return intent == Intent.REQUEST;
    }

    @Override
    public String generate(String userText, List<Statement> context, ThoughtState state) {
        StringBuilder sb = new StringBuilder(900);
        sb.append("Ок, делаем.\n\n");
        sb.append("Сначала уточню рамки (чтобы сделать правильно):\n");
        sb.append("1) Вход/выход модуля?\n");
        sb.append("2) Где хранить знания (jsonl snapshot/events уже есть) — достаточно?\n");
        sb.append("3) Нужна ли многопоточность/серверный режим?\n\n");

        sb.append("Опорные принципы из базы:\n");
        int limit = Math.min(4, context.size());
        for (int i = 0; i < limit; i++) sb.append("• ").append(context.get(i).text).append("\n");

        sb.append("\nСкажи: ты хочешь следующий шаг про (A) BeamSearch генерацию, (B) граф ассоциаций, (C) обучение весов?");
        return sb.toString();
    }
}