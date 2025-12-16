package org.calista.arasaka.ai.learn.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.learn.Learner;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.text.BreakIterator;
import java.util.*;
import java.util.zip.CRC32;

/**
 * Enterprise-версия "простого" обучателя:
 * - стабильное разбиение на предложения (BreakIterator)
 * - нормализация текста (вырезаем мусор/контрольные символы, склеиваем пробелы)
 * - анти-дубликаты в рамках одного learn() (по сигнатуре)
 * - детерминированный id (crc32 от нормализованного предложения + tag)
 * - аккуратное усиление веса и лимиты
 * - log4j2
 */
@Deprecated
public final class SimpleLearner implements Learner {
    private static final Logger log = LogManager.getLogger(SimpleLearner.class);

    private final KnowledgeBase kb;
    private final Tokenizer tokenizer;

    private final double newWeight;
    private final double reinforceStep;
    private final double maxWeight;

    private final int minChars;
    private final int minTokens;

    public SimpleLearner(
            KnowledgeBase kb,
            Tokenizer tokenizer,
            double newWeight,
            double reinforceStep
    ) {
        this(kb, tokenizer, newWeight, reinforceStep,
                3.0,       // maxWeight
                12,        // minChars
                4          // minTokens
        );
    }

    public SimpleLearner(
            KnowledgeBase kb,
            Tokenizer tokenizer,
            double newWeight,
            double reinforceStep,
            double maxWeight,
            int minChars,
            int minTokens
    ) {
        this.kb = Objects.requireNonNull(kb, "kb");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");

        if (newWeight <= 0) throw new IllegalArgumentException("newWeight must be > 0");
        if (reinforceStep <= 0) throw new IllegalArgumentException("reinforceStep must be > 0");
        if (maxWeight <= 0) throw new IllegalArgumentException("maxWeight must be > 0");
        if (minChars < 1) throw new IllegalArgumentException("minChars must be >= 1");
        if (minTokens < 1) throw new IllegalArgumentException("minTokens must be >= 1");

        this.newWeight = newWeight;
        this.reinforceStep = reinforceStep;
        this.maxWeight = maxWeight;
        this.minChars = minChars;
        this.minTokens = minTokens;
    }

    @Override
    public List<Statement> learnFromText(String text, String tag) {
        if (text == null || text.isBlank()) return List.of();
        final String safeTag = normalizeTag(tag);

        final String normalized = normalizeText(text);
        if (normalized.isBlank()) return List.of();

        // анти-дубликаты внутри одного вызова (даже если текст повторяется/зациклен)
        final HashSet<String> seenSigs = new HashSet<>(128);
        final ArrayList<Statement> out = new ArrayList<>(64);

        final BreakIterator it = BreakIterator.getSentenceInstance(Locale.ROOT);
        it.setText(normalized);

        int start = it.first();
        for (int end = it.next(); end != BreakIterator.DONE; start = end, end = it.next()) {
            String sentence = normalized.substring(start, end).trim();
            sentence = cleanupSentence(sentence);

            if (!passesQualityGate(sentence)) continue;

            // стабильная сигнатура для дедупа (без тега)
            final String sig = crc32(sentence);
            if (!seenSigs.add(sig)) continue;

            final String id = "learn:" + safeTag + ":" + sig;

            String finalSentence = sentence;
            Statement st = kb.get(id).orElseGet(() -> {
                Statement n = new Statement();
                n.id = id;
                n.text = finalSentence;
                n.weight = newWeight;
                n.tags = List.of("learned", safeTag);
                return n;
            });

            // усиление веса (детерминированно)
            final double before = st.weight;
            st.weight = clamp(before + reinforceStep, 0.0, maxWeight);

            // гарантируем, что тег присутствует, но не плодим мусор
            st.tags = mergeTags(st.tags, "learned", safeTag);

            kb.upsert(st);
            out.add(st);

            if (log.isTraceEnabled()) {
                log.trace("Learned: id={}, w:{}->{} text='{}'",
                        st.id, round3(before), round3(st.weight), abbreviate(st.text, 140));
            }
        }

        if (log.isDebugEnabled()) {
            log.debug("learnFromText(tag={}): produced {} statements (textLen={})",
                    safeTag, out.size(), text.length());
        }

        return out;
    }

    @Override
    public List<Statement> learnFromText(String text, String tag, Map<String, String> context) {
        return List.of();
    }

    // ---------- quality / normalization ----------

    private boolean passesQualityGate(String s) {
        if (s.isBlank()) return false;
        if (s.length() < minChars) return false;

        // токены — через ваш Tokenizer (без хардкода на regex-токенизацию)
        int tokens = tokenizer.tokenize(s).size();
        if (tokens < minTokens) return false;

        // легкий анти-шум: слишком много знаков подряд / слишком мало букв
        int letters = 0;
        int weird = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isLetter(c)) letters++;
            if ("@#$%^*_=<>[]{}".indexOf(c) >= 0) weird++;
        }
        if (letters < Math.max(3, s.length() / 8)) return false;
        if (weird > Math.max(2, s.length() / 20)) return false;

        return true;
    }

    private static String normalizeText(String text) {
        // убираем управляющие, нормализуем пробелы
        String s = text
                .replace('\u00A0', ' ') // NBSP
                .replaceAll("[\\p{Cntrl}&&[^\r\n\t]]", " ")
                .replaceAll("[\\t\\r\\n]+", " ")
                .replaceAll(" +", " ")
                .trim();
        return s;
    }

    private static String cleanupSentence(String s) {
        // выкидываем мусорные хвосты, приводим пробелы к норме
        String x = s
                .replaceAll(" +", " ")
                .replaceAll("^[\\p{Punct}\\s]+", "")
                .replaceAll("[\\p{Punct}\\s]+$", "")
                .trim();
        // если после чистки стало пусто — вернем пустую строку
        return x;
    }

    private static String normalizeTag(String tag) {
        String t = (tag == null || tag.isBlank()) ? "generic" : tag.trim();
        t = t.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9._-]+", "_");
        if (t.length() > 48) t = t.substring(0, 48);
        return t;
    }

    private static List<String> mergeTags(List<String> existing, String... toAdd) {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        if (existing != null) {
            for (String e : existing) {
                if (e != null && !e.isBlank()) set.add(e.trim());
            }
        }
        for (String a : toAdd) {
            if (a != null && !a.isBlank()) set.add(a.trim());
        }
        return List.copyOf(set);
    }

    // ---------- utils ----------

    private static String crc32(String s) {
        CRC32 c = new CRC32();
        c.update(s.getBytes(StandardCharsets.UTF_8));
        return Long.toHexString(c.getValue());
    }

    private static double clamp(double v, double min, double max) {
        return Math.max(min, Math.min(max, v));
    }

    private static double round3(double v) {
        return Math.rint(v * 1000.0) / 1000.0;
    }

    private static String abbreviate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, Math.max(0, max - 1)) + "…";
    }
}