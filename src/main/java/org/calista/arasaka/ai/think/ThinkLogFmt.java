package org.calista.arasaka.ai.think;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * ThinkLogFmt — компактный helper для красивого форматирования логов.
 *
 * <p>Не тянет внешние зависимости, не влияет на публичный API Think.
 */
public final class ThinkLogFmt {

    private ThinkLogFmt() {}

    /**
     * Рендерит параметры в "коробку".
     */
    public static String box(String title, Consumer<BoxBuilder> fill) {
        Objects.requireNonNull(title, "title");
        Objects.requireNonNull(fill, "fill");

        BoxBuilder b = new BoxBuilder();
        fill.accept(b);
        return renderBox(title, b.lines);
    }

    // ---------------------------------------------------------------------
    // Builder
    // ---------------------------------------------------------------------

    public static final class BoxBuilder {
        private final List<String> lines = new ArrayList<>(32);

        public BoxBuilder kv(String key, Object value) {
            String k = (key == null) ? "" : key;
            String v = String.valueOf(value);
            lines.add(k + ": " + v);
            return this;
        }

        public BoxBuilder sep() {
            lines.add("--");
            return this;
        }
    }

    // ---------------------------------------------------------------------
    // Rendering
    // ---------------------------------------------------------------------

    private static String renderBox(String title, List<String> lines) {
        int contentWidth = title.length();
        for (String l : lines) {
            if (l == null) continue;
            if ("--".equals(l)) continue;
            contentWidth = Math.max(contentWidth, l.length());
        }

        // немного воздуха по краям
        int w = Math.max(24, contentWidth + 2);

        StringBuilder out = new StringBuilder((lines.size() + 5) * (w + 8));

        out.append("┌").append(repeat("─", w)).append("┐\n");
        out.append("│ ").append(padRight(title, w - 1)).append("│\n");
        out.append("├").append(repeat("─", w)).append("┤\n");

        for (String l : lines) {
            if (l == null) continue;
            if ("--".equals(l)) {
                out.append("│").append(repeat("─", w)).append("│\n");
                continue;
            }
            out.append("│ ").append(padRight(l, w - 1)).append("│\n");
        }

        out.append("└").append(repeat("─", w)).append("┘");
        return out.toString();
    }

    private static String padRight(String s, int width) {
        if (s == null) s = "";
        if (s.length() >= width) return s;
        StringBuilder b = new StringBuilder(width);
        b.append(s);
        while (b.length() < width) b.append(' ');
        return b.toString();
    }

    private static String repeat(String s, int n) {
        if (n <= 0) return "";
        StringBuilder b = new StringBuilder(n * s.length());
        for (int i = 0; i < n; i++) b.append(s);
        return b.toString();
    }
}