package org.calista.arasaka.ai.think.textGenerator.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;
import org.calista.arasaka.ai.think.ThoughtState;

import java.util.List;
import java.util.Locale;

/**
 * DefaultTextGenerator
 *
 * Minimal deterministic baseline implementation:
 * - delegates generation to a stable, domain-agnostic template
 * - intended as fallback until BeamSearch/TensorFlow backend is plugged in
 *
 * IMPORTANT:
 * This implementation is intentionally conservative and deterministic.
 */
@Deprecated
public final class DefaultTextGenerator implements TextGenerator {

    private static final Logger log = LogManager.getLogger(DefaultTextGenerator.class);

    private final int maxEvidenceLines;

    public DefaultTextGenerator() {
        this(6);
    }

    public DefaultTextGenerator(int maxEvidenceLines) {
        this.maxEvidenceLines = Math.max(0, maxEvidenceLines);
    }

    @Override
    public String generate(String userText, List<Statement> context, ThoughtState state) {
        final String prompt = userText == null ? "" : userText.trim();
        final List<Statement> ctx = context == null ? List.of() : context;

        // parse format from hint (md/plain)
        String format = "";
        String hint = (state == null || state.generationHint == null) ? "" : state.generationHint;
        TextGenerator.Hint h = TextGenerator.Hint.parse(hint);
        if (!h.format.isBlank()) format = h.format.trim().toLowerCase(Locale.ROOT);

        boolean md = !"plain".equals(format);

        StringBuilder out = new StringBuilder(512);

        // Very conservative: if no context, acknowledge insufficiency deterministically.
        if (ctx.isEmpty()) {
            if (md) {
                out.append("## Вывод\n");
                out.append(extractUser(prompt)).append("\n\n");
                out.append("## Опора на контекст\n");
                out.append("• (no_context)\n");
            } else {
                out.append(extractUser(prompt));
            }
            if (log.isDebugEnabled()) {
                log.debug("DefaultTextGenerator: ctx=0 format={} draftIndex={}", format, state == null ? -1 : state.draftIndex);
            }
            return out.toString();
        }

        if (md) {
            out.append("## Вывод\n");
            out.append(extractUser(prompt)).append("\n\n");
            out.append("## Опора на контекст\n");
            int added = 0;
            for (Statement s : ctx) {
                if (s == null || s.text == null || s.text.isBlank()) continue;
                out.append("• ").append(s.text.trim()).append("\n");
                if (++added >= maxEvidenceLines) break;
            }
        } else {
            // plain: just return a short supported summary-like output
            out.append(extractUser(prompt));
        }

        if (log.isDebugEnabled()) {
            log.debug("DefaultTextGenerator: ctx={} format={} draftIndex={}", ctx.size(), format, state == null ? -1 : state.draftIndex);
        }

        return out.toString();
    }

    private static String extractUser(String preparedPromptOrUserText) {
        if (preparedPromptOrUserText == null) return "";
        String s = preparedPromptOrUserText;
        int ix = s.lastIndexOf("User:");
        if (ix >= 0) {
            String tail = s.substring(ix + "User:".length()).trim();
            if (!tail.isBlank()) return tail;
        }
        return s.trim();
    }
}