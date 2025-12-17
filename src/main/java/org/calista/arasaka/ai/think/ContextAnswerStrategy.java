package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.*;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public final class ContextAnswerStrategy implements ResponseStrategy {

    // Tokenization for deterministic lightweight relevance.
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{Nd}_]{3,}");

    private static String summarize(String userText, List<Statement> ctx, ThoughtState state) {
        // Use bestSoFar as anchor if it exists (stability across iterations).
        if (state != null && state.bestSoFar != null && state.bestSoFar.text != null && !state.bestSoFar.text.isBlank()) {
            String best = extractSection1(state.bestSoFar.text);
            if (!best.isBlank()) return truncate(best, 220);
        }
        if (ctx == null || ctx.isEmpty()) return truncate(userText, 200);
        return truncate(String.valueOf(ctx.get(0).text), 240);
    }

    private static String arguments(List<Statement> ctx, ThoughtState state) {
        // No magic markers in user-visible output. If no evidence, omit the section.
        if (ctx == null || ctx.isEmpty()) return "";

        Set<String> ltmIds = (state == null || state.recalledMemory == null)
                ? Set.of()
                : state.recalledMemory.stream()
                .filter(Objects::nonNull)
                .map(s -> s.id)
                .filter(id -> id != null && !id.isBlank())
                .collect(Collectors.toSet());

        return ctx.stream()
                .filter(Objects::nonNull)
                .limit(8)
                .map(s -> {
                    String id = s.id;
                    String t = (s.text == null) ? "" : s.text;
                    String p = (id != null && !id.isBlank() && ltmIds.contains(id)) ? "• [LTM] " : "• ";
                    return p + truncate(t, 240);
                })
                .collect(Collectors.joining("\n"));
    }

    private static String steps(List<Statement> ctx, ThoughtState state) {
        // This block is user-visible. Keep it clean: no internal markers (fix=..., notes=...).
        // If you want self-critique / repair guidance, feed it via ThoughtState.generationHint
        // so the generator can self-correct without leaking telemetry into the final answer.

        if (ctx == null || ctx.isEmpty()) return "";

        StringBuilder sb = new StringBuilder(320);
        for (Statement s : ctx.stream().filter(Objects::nonNull).limit(4).toList()) {
            String t = String.valueOf(s.text).trim();
            if (t.isEmpty()) continue;
            sb.append("- ").append(truncate(t, 220)).append('\n');
        }
        return sb.toString().trim();
    }

    private static List<Statement> rankContext(String userText, List<Statement> context) {
        if (context == null || context.isEmpty()) return List.of();
        Set<String> q = tokens(userText);

        return context.stream()
                .filter(Objects::nonNull)
                .sorted(Comparator
                        .comparingInt((Statement s) -> overlap(tokens(String.valueOf(s.text)), q))
                        .reversed()
                        .thenComparing(s -> String.valueOf(s.text)))
                .toList();
    }

    private static Set<String> tokens(String s) {
        if (s == null || s.isBlank()) return Set.of();
        return WORD.matcher(s.toLowerCase(Locale.ROOT))
                .results()
                .map(MatchResult::group)
                .limit(256)
                .collect(Collectors.toSet());
    }

    private static int overlap(Set<String> a, Set<String> b) {
        if (a.isEmpty() || b.isEmpty()) return 0;
        int c = 0;
        for (String t : a) if (b.contains(t)) c++;
        return c;
    }

    private static String extractSection1(String text) {
        // Back-compat: try old numeric contract; fall back to first markdown section.
        if (text == null) return "";

        int idx = text.indexOf("1)");
        if (idx >= 0) {
            int start = idx + 2;
            int end = text.indexOf("2)", start);
            if (end < 0) end = Math.min(text.length(), start + 260);
            return text.substring(start, end).trim();
        }

        // Markdown: take content after first "##" header until next header.
        int h = text.indexOf("##");
        if (h >= 0) {
            int startBody = text.indexOf('\n', h);
            if (startBody >= 0) {
                startBody++;
                int next = text.indexOf("\n##", startBody);
                if (next < 0) next = Math.min(text.length(), startBody + 260);
                return text.substring(startBody, next).trim();
            }
        }

        // Plain fallback: first non-empty line.
        for (String line : text.split("\\R")) {
            String t = line.trim();
            if (!t.isEmpty()) return t;
        }
        return "";
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max);
    }

    @Override
    public boolean supports(Intent intent) {
        return true; // single "smart" strategy
    }

    @Override
    public String generate(String userText, List<Statement> context, ThoughtState state) {
        Objects.requireNonNull(state);

        // Sort context deterministically by relevance to the query (no ML / no magic).
        List<Statement> ranked = rankContext(userText, context);

        String section1 = summarize(userText, ranked, state);
        String section2 = arguments(ranked, state);
        String section3 = steps(ranked, state);

        Locale locale = localeOf(userText);
        String style = tag(state, "response.style", "md");
        String sections = tag(state, "response.sections", "summary,evidence,actions");

        String labelSummary = tag(state, "response.label.summary", defaultLabel("summary", locale));
        String labelEvidence = tag(state, "response.label.evidence", defaultLabel("evidence", locale));
        String labelActions = tag(state, "response.label.actions", defaultLabel("actions", locale));

        Map<String, String> blocks = new HashMap<>();
        blocks.put("summary", section1);
        blocks.put("evidence", section2);
        blocks.put("actions", section3);

        Map<String, String> labels = Map.of(
                "summary", labelSummary,
                "evidence", labelEvidence,
                "actions", labelActions
        );

        return render(sections, style, blocks, labels);
    }

    private static String tag(ThoughtState state, String key, String def) {
        if (state == null || state.tags == null) return def;
        String v = state.tags.get(key);
        return (v == null || v.isBlank()) ? def : v;
    }

    private static String render(String sectionsCsv,
                                 String style,
                                 Map<String, String> blocks,
                                 Map<String, String> labels) {
        String[] parts = (sectionsCsv == null ? "" : sectionsCsv).split("\\s*,\\s*");
        boolean md = style != null && style.toLowerCase(Locale.ROOT).contains("md");

        StringBuilder out = new StringBuilder(1400);
        for (String p : parts) {
            if (p == null || p.isBlank()) continue;
            String key = p.trim().toLowerCase(Locale.ROOT);
            String label = labels.getOrDefault(key, key);
            String body = blocks.getOrDefault(key, "").trim();
            if (body.isBlank()) continue;

            if (md) {
                out.append("## ").append(label).append('\n');
                out.append(body).append("\n\n");
            } else {
                out.append(label).append(':').append('\n');
                out.append(body).append("\n\n");
            }
        }
        return out.toString().trim();
    }

    private static Locale localeOf(String s) {
        if (s == null) return Locale.ROOT;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 'А' && c <= 'я') return new Locale("ru");
        }
        return Locale.ROOT;
    }

    private static String defaultLabel(String key, Locale locale) {
        boolean ru = locale != null && "ru".equalsIgnoreCase(locale.getLanguage());
        return switch (key) {
            case "summary" -> ru ? "Вывод" : "Conclusion";
            case "evidence" -> ru ? "Опора на контекст" : "Evidence from context";
            case "actions" -> ru ? "Следующие шаги" : "Next steps";
            default -> key;
        };
    }
}