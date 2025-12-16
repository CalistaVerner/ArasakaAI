package org.calista.arasaka.ai.text;

import java.util.ArrayList;
import java.util.List;

public final class SimpleTokenizer implements Tokenizer {
    @Override
    public List<String> tokenize(String text) {
        if (text == null || text.isBlank()) return List.of();
        String s = text.toLowerCase();

        // нормализация “по-взрослому”: оставляем буквы/цифры, остальное -> пробел
        StringBuilder b = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isLetterOrDigit(c)) b.append(c);
            else b.append(' ');
        }

        String[] parts = b.toString().trim().split("\\s+");
        ArrayList<String> out = new ArrayList<>(parts.length);
        for (String p : parts) if (!p.isBlank()) out.add(p);
        return out;
    }
}