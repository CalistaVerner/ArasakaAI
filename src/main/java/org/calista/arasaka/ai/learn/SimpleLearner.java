package org.calista.arasaka.ai.learn;

import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;

public final class SimpleLearner {
    private final KnowledgeBase kb;
    private final Tokenizer tokenizer;
    private final double newWeight;
    private final double reinforceStep;

    public SimpleLearner(KnowledgeBase kb, Tokenizer tokenizer, double newWeight, double reinforceStep) {
        this.kb = kb;
        this.tokenizer = tokenizer;
        this.newWeight = newWeight;
        this.reinforceStep = reinforceStep;
    }

    /** Извлекаем “утверждения” из текста как стабильные предложения. */
    public List<Statement> learnFromText(String text, String tag) {
        if (text == null || text.isBlank()) return List.of();

        // Разбиение на предложения (простое, но стабильное)
        String[] parts = text.split("[\\.\\!\\?\\n]+");
        ArrayList<Statement> out = new ArrayList<>();

        for (String p : parts) {
            String s = p == null ? "" : p.trim();
            if (s.length() < 12) continue; // мусор
            if (tokenizer.tokenize(s).size() < 4) continue;

            String id = "learn:" + tag + ":" + crc32(s);
            Statement st = kb.get(id).orElseGet(() -> {
                Statement n = new Statement();
                n.id = id;
                n.text = s;
                n.weight = newWeight;
                n.tags = List.of("learned", tag);
                return n;
            });

            // усиление
            st.weight = Math.min(3.0, st.weight + reinforceStep);
            kb.upsert(st);
            out.add(st);
        }
        return out;
    }

    private static String crc32(String s) {
        CRC32 c = new CRC32();
        c.update(s.getBytes(StandardCharsets.UTF_8));
        return Long.toHexString(c.getValue());
    }
}