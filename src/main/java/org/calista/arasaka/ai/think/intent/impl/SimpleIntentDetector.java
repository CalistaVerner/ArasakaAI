package org.calista.arasaka.ai.think.intent.impl;

import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.intent.IntentDetector;

import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;

/**
 * Deterministic intent detector without hard-coded regex "magic".
 *
 * <p>Policy is expressed as keyword weights (configurable via constructor),
 * so you can tune it from config/tests and keep behavior stable.</p>
 */
@Deprecated
public final class SimpleIntentDetector implements IntentDetector {

    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{Nd}_]{2,}");

    private final Map<Intent, Map<String, Double>> weights;
    private final double minConfidence;

    /**
     * Default detector with conservative weights (can be tuned later).
     */
    public SimpleIntentDetector() {
        this(defaultWeights(), 0.60);
    }

    public SimpleIntentDetector(Map<Intent, Map<String, Double>> weights, double minConfidence) {
        this.weights = new EnumMap<>(Intent.class);
        if (weights != null) this.weights.putAll(weights);
        this.minConfidence = Math.max(0.0, minConfidence);
    }

    @Override
    public Intent detect(String userText) {
        if (userText == null) return Intent.UNKNOWN;
        String s = userText.trim();
        if (s.isEmpty()) return Intent.UNKNOWN;

        // Cheap signal: question mark.
        double qMark = s.indexOf('?') >= 0 ? 0.35 : 0.0;

        Map<Intent, Double> scores = new EnumMap<>(Intent.class);
        for (Intent it : Intent.values()) scores.put(it, 0.0);

        for (String tok : tokens(s)) {
            for (Map.Entry<Intent, Map<String, Double>> e : weights.entrySet()) {
                Double w = e.getValue().get(tok);
                if (w != null) scores.put(e.getKey(), scores.get(e.getKey()) + w);
            }
        }

        // Merge punctuation signal into QUESTION.
        scores.put(Intent.QUESTION, scores.get(Intent.QUESTION) + qMark);

        Intent best = Intent.UNKNOWN;
        double bestScore = 0.0;

        for (Map.Entry<Intent, Double> e : scores.entrySet()) {
            if (e.getKey() == Intent.UNKNOWN) continue;
            if (e.getValue() > bestScore) {
                bestScore = e.getValue();
                best = e.getKey();
            }
        }

        return bestScore >= minConfidence ? best : Intent.UNKNOWN;
    }

    private static Set<String> tokens(String s) {
        return WORD.matcher(s.toLowerCase(Locale.ROOT))
                .results()
                .map(MatchResult::group)
                .limit(128)
                .collect(java.util.stream.Collectors.toSet());
    }

    private static Map<Intent, Map<String, Double>> defaultWeights() {
        EnumMap<Intent, Map<String, Double>> m = new EnumMap<>(Intent.class);

        m.put(Intent.GREETING, Map.ofEntries(
                Map.entry("привет", 1.00),
                Map.entry("здравствуй", 1.00),
                Map.entry("здравствуйте", 1.00),
                Map.entry("hello", 1.00),
                Map.entry("hi", 0.90),
                Map.entry("hey", 0.90)
        ));

        m.put(Intent.QUESTION, Map.ofEntries(
                Map.entry("как", 0.55),
                Map.entry("почему", 0.65),
                Map.entry("зачем", 0.55),
                Map.entry("что", 0.55),
                Map.entry("когда", 0.55),
                Map.entry("где", 0.55),
                Map.entry("кто", 0.55),
                Map.entry("какой", 0.55),
                Map.entry("какие", 0.55),
                Map.entry("сколько", 0.60),
                Map.entry("можно", 0.45)
        ));

        m.put(Intent.REQUEST, Map.ofEntries(
                Map.entry("сделай", 0.80),
                Map.entry("создай", 0.80),
                Map.entry("напиши", 0.75),
                Map.entry("реализуй", 0.85),
                Map.entry("покажи", 0.60),
                Map.entry("объясни", 0.60),
                Map.entry("давай", 0.55),
                Map.entry("нужно", 0.55),
                Map.entry("исправь", 0.70),
                Map.entry("обнови", 0.70)
        ));

        // UNKNOWN intentionally absent (computed as fallback)
        return m;
    }
}