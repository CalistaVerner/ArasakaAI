package org.calista.arasaka.ai.learn;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.List;
import java.util.Map;

/**
 * Learner: извлекает устойчивые "утверждения" из текста и сохраняет/усиливает их в KB.
 * Должен быть детерминированным (без рандома) и устойчивым к шуму.
 */
public interface Learner {

    /**
     * Извлекает утверждения из текста, добавляет в KB (или усиливает существующие) и возвращает список обработанных.
     *
     * @param text исходный текст
     * @param tag  доменный тег/контекст (например "dialog", "user", "wiki", "memory")
     * @return список утверждений, которые были добавлены/усилены
     */
    List<Statement> learnFromText(String text, String tag);
    List<Statement> learnFromText(String text, String tag, Map<String, String> context);
}