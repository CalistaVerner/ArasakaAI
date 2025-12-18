package org.calista.arasaka.ai.think.utils;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Drafts — deterministic draft sanitization utilities.
 *
 * Goals:
 * - keep list size stable (exactly n)
 * - normalize whitespace + remove nulls
 * - de-duplicate stably
 * - cap per-draft size to avoid runaway memory/log spam
 *
 * IMPORTANT:
 * - must be deterministic
 * - must not depend on locale/time/random
 */
public final class Drafts {
    private Drafts() {}

    // Enterprise safe limits (can be tuned)
    public static final int MAX_DRAFT_CHARS = 4000;

    /**
     * Sanitizes drafts and returns an EXACT size of {@code n} by padding with empty strings.
     *
     * Preserved for backward compatibility with callers that expect stable size.
     * For modern high-performance iterative generation, prefer
     * {@link #sanitizeKeepNonEmpty(List, int)} and let the engine re-generate missing drafts.
     */

    public static List<String> sanitize(List<String> drafts, int n) {
        if (n <= 0) return List.of();
        if (drafts == null || drafts.isEmpty()) return padToN(List.of(), n);

        // 1) normalize + trim + cap
        ArrayList<String> norm = new ArrayList<>(drafts.size());
        for (String d : drafts) {
            String x = normalize(d);
            if (x.isBlank()) continue;
            norm.add(x);
        }
        if (norm.isEmpty()) return padToN(List.of(), n);

        // 2) stable de-dup
        LinkedHashSet<String> uniq = new LinkedHashSet<>(Math.min(64, norm.size() * 2));
        for (String x : norm) {
            if (uniq.size() >= n) break;
            uniq.add(x);
        }

        // 3) output exact size n
        return padToN(new ArrayList<>(uniq), n);
    }

    /**
     * Sanitizes drafts and returns up to {@code n} NON-EMPTY unique drafts (no padding).
     *
     * This is the preferred mode for iterative engines: if less than {@code n} drafts are
     * produced after sanitization, the engine should generate additional drafts deterministically
     * rather than wasting evaluation budget on empty strings.
     */
    public static List<String> sanitizeKeepNonEmpty(List<String> drafts, int n) {
        if (n <= 0) return List.of();
        if (drafts == null || drafts.isEmpty()) return List.of();

        ArrayList<String> norm = new ArrayList<>(drafts.size());
        for (String d : drafts) {
            String x = normalize(d);
            if (x.isBlank()) continue;
            norm.add(x);
        }
        if (norm.isEmpty()) return List.of();

        LinkedHashSet<String> uniq = new LinkedHashSet<>(Math.min(64, norm.size() * 2));
        for (String x : norm) {
            if (uniq.size() >= n) break;
            uniq.add(x);
        }
        return uniq.isEmpty() ? List.of() : new ArrayList<>(uniq);
    }

    private static List<String> padToN(List<String> in, int n) {
        ArrayList<String> out = new ArrayList<>(n);
        if (in != null && !in.isEmpty()) {
            for (int i = 0; i < in.size() && out.size() < n; i++) {
                out.add(in.get(i));
            }
        }
        while (out.size() < n) out.add("");
        return out;
    }

    public static String normalize(String s) {
        if (s == null) return "";
        String x = s;

        // Remove NULs and trim
        x = x.replace("\u0000", "");
        x = x.trim();

        // Collapse whitespace deterministically
        x = x.replaceAll("[\\s\\u00A0]+", " ");

        if (x.length() > MAX_DRAFT_CHARS) x = x.substring(0, MAX_DRAFT_CHARS);
        return x;
    }
}