package org.calista.arasaka.ai.think;

public final class SimpleIntentDetector implements IntentDetector {
    @Override
    public Intent detect(String userText) {
        if (userText == null) return Intent.UNKNOWN;
        String s = userText.trim().toLowerCase();
        if (s.isEmpty()) return Intent.UNKNOWN;

        // приветствия
        if (s.matches("^(привет|здравствуй|здравствуйте|hello|hi|hey)\\b.*")) return Intent.GREETING;

        // вопрос
        if (s.contains("?") || s.matches("^(как|почему|зачем|что|когда|где|кто)\\b.*")) return Intent.QUESTION;

        // просьба/задача
        if (s.matches("^(сделай|создай|напиши|реализуй|покажи|объясни|давай)\\b.*")) return Intent.REQUEST;

        return Intent.UNKNOWN;
    }
}