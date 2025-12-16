package org.calista.arasaka.ai.think;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Единственная "умная" стратегия ответа:
 *  - собирает ответ по секциям (краткий вывод → аргументы из контекста → шаги)
 *  - самоисправляется (используя state.bestSoFar и state.lastCritique)
 *  - готова к подключению BeamSearch/TensorFlow генератора (через state.generator)
 */
public final class ContextAnswerStrategy implements ResponseStrategy {
    private static final Logger log = LogManager.getLogger(ContextAnswerStrategy.class);

    private final int maxContextBullets;

    public ContextAnswerStrategy() {
        this(6);
    }

    public ContextAnswerStrategy(int maxContextBullets) {
        this.maxContextBullets = Math.max(2, maxContextBullets);
    }

    @Override
    public boolean supports(Intent intent) {
        // "Единая стратегия" — работает для любого интента.
        return true;
    }

    @Override
    public String generate(String userText, List<Statement> context, ThoughtState state) {
        Objects.requireNonNull(state, "state");
        String user = userText == null ? "" : userText.trim();
        List<Statement> ctx = context == null ? List.of() : context;

        String bestHint = bestSoFarHint(state);
        String critiqueHint = critiqueHint(state);

        String summary = draftSummary(user, ctx, state, bestHint, critiqueHint);
        List<String> bullets = buildContextBullets(user, ctx, state);
        String steps = draftSteps(user, ctx, state, bestHint, critiqueHint);

        String answer = assemble(summary, bullets, steps);
        answer = sanitize(answer);

        log.debug("strategy.generate iter={} intent={} ctx={} summaryChars={} stepsChars={} hasGenerator={}",
                state.iteration, state.intent, ctx.size(), summary.length(), steps.length(), state.generator != null);
        return answer;
    }

    private String draftSummary(String user, List<Statement> ctx, ThoughtState state, String bestHint, String critiqueHint) {
        if (state.generator != null) {
            String out = state.generator.generate(user, ctx, state);
            if (out != null && out.trim().length() >= 40) {
                return out.trim();
            }
        }

        String topic = pickTopic(user);
        if (state.intent == Intent.GREETING) {
            return "Привет! Давай сделаем это по-взрослому. Скажи, какой результат ты хочешь на выходе (код, план, архитектура, тесты)?";
        }
        if (topic.isBlank()) {
            return "Ок. Сформулирую ответ так, чтобы он был проверяемым: вывод, опора на контекст, и конкретные шаги.";
        }
        String prefix = (state.iteration <= 1)
                ? "Суть: "
                : "Уточнил и улучшил: ";

        String improve = (critiqueHint.isBlank() ? "" : (" (учёл: " + critiqueHint + ")"));
        return prefix + "по теме «" + topic + "» можно построить решение как конвейер: выбор данных → сбор черновика → валидация → улучшение." + improve;
    }

    private String draftSteps(String user, List<Statement> ctx, ThoughtState state, String bestHint, String critiqueHint) {
        if (state.generator != null) {
            String prevHint = state.generationHint;
            state.generationHint = "steps";
            try {
                String out = state.generator.generate(user, ctx, state);
                if (out != null && out.trim().length() >= 40) return out.trim();
            } finally {
                state.generationHint = prevHint;
            }
        }

        ArrayList<String> steps = new ArrayList<>(6);
        steps.add("Сделать единый цикл: retrieve(query) → draft(answer) → evaluate(metrics) → revise(query/answer) с ранней остановкой.");
        steps.add("Перенести критику внутрь CandidateEvaluator (метрики + текст критики), а движок держать тупым и чистым.");
        steps.add("Стратегия ответа всегда выдаёт: (1) вывод, (2) аргументы из контекста, (3) шаги и проверки.");
        steps.add("Валидация: проверять покрытие запроса, опору на контекст, стиль (не раскрывать внутренности), и наличие следующего действия.");
        steps.add("Подготовить интерфейс TextGenerator и подключить BeamSearchResponder/TensorFlow как backend без изменения движка.");

        if (!critiqueHint.isBlank()) {
            steps.add("Адресно улучшить: " + critiqueHint + ".");
        }
        if (!bestHint.isBlank() && state.iteration >= 2) {
            steps.add("Слить лучшее из прошлой версии с текущими правками (bestSoFar) и пересчитать оценку.");
        }

        StringBuilder sb = new StringBuilder(600);
        for (int i = 0; i < steps.size(); i++) {
            sb.append(i + 1).append(") ").append(steps.get(i)).append('\n');
        }
        return sb.toString().trim();
    }

    private List<String> buildContextBullets(String user, List<Statement> ctx, ThoughtState state) {
        int limit = Math.min(maxContextBullets, ctx.size());
        if (limit == 0) {
            return List.of("Контекст пока пустой — либо расширь базу знаний, либо уточни формулировку (что именно строим: retrieval, генерация, оценка, память?).");
        }
        ArrayList<String> out = new ArrayList<>(limit);
        for (int i = 0; i < limit; i++) {
            String t = ctx.get(i).text == null ? "" : ctx.get(i).text.trim();
            if (!t.isBlank()) out.add(t);
        }
        if (out.size() < maxContextBullets && state.iteration >= 2 && !state.lastCritique.isBlank() && ctx.size() > out.size()) {
            String extra = ctx.get(out.size()).text == null ? "" : ctx.get(out.size()).text.trim();
            if (!extra.isBlank()) out.add(extra);
        }
        return List.copyOf(out);
    }

    private static String assemble(String summary, List<String> bullets, String steps) {
        StringBuilder sb = new StringBuilder(1400);

        sb.append(summary == null ? "" : summary.trim()).append("\n\n");

        sb.append("Опора на контекст:\n");
        for (String b : bullets) {
            sb.append("• ").append(b).append('\n');
        }

        sb.append("\nШаги:\n");
        sb.append(steps == null ? "" : steps.trim());

        return sb.toString().trim();
    }

    private static String bestSoFarHint(ThoughtState state) {
        if (state.bestSoFar == null || state.bestSoFar.text == null) return "";
        String s = state.bestSoFar.text.trim();
        if (s.isEmpty()) return "";
        int cap = Math.min(120, s.length());
        return s.substring(0, cap);
    }

    private static String critiqueHint(ThoughtState state) {
        String c = state.lastCritique == null ? "" : state.lastCritique.trim();
        if (c.isEmpty()) return "";
        c = c.replace("итера", "");
        if (c.length() > 180) c = c.substring(0, 180);
        return c;
    }

    private static String sanitize(String s) {
        if (s == null) return "";
        String x = s;
        x = x.replaceAll("(?i)контекст знаний\\s*:\\s*", "");
        x = x.replaceAll("(?i)план ответа\\s*:\\s*", "");
        x = x.replaceAll("(?i)trace\\s*:\\s*", "");
        return x.trim();
    }

    private static String pickTopic(String user) {
        if (user == null) return "";
        String u = user.trim();
        if (u.isEmpty()) return "";
        int cut = u.indexOf('\n');
        if (cut < 0) cut = u.indexOf('.');
        if (cut < 0) cut = u.indexOf('!');
        if (cut < 0) cut = u.indexOf('?');
        if (cut < 0) cut = Math.min(64, u.length());
        String t = u.substring(0, Math.min(cut, u.length())).trim();
        if (t.length() > 64) t = t.substring(0, 64);
        return t;
    }
}
