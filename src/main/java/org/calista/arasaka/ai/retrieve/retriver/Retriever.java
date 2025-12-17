package org.calista.arasaka.ai.retrieve.retriver;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.Scored;

import java.util.*;

/**
 * Retriever returns the most relevant knowledge statements for a query.
 *
 * Backward compatibility:
 * - Main method remains: {@link #retrieve(String, int, long)}
 *
 * Enterprise additions:
 * - Multi-query retrieval with deterministic merge: {@link #retrieve(List, int, long)}
 * - Scored variants for debugging/quality: {@link #retrieveScored(String, int, long)} and {@link #retrieveScored(List, int, long)}
 */
public interface Retriever {

    /**
     * Retrieve top-k statements for a single query.
     * Must be deterministic for the same (query, k, seed).
     */
    List<Statement> retrieve(String query, int k, long seed);

    /**
     * Retrieve top-k for multiple queries and merge deterministically.
     *
     * Default implementation:
     * - calls single-query retrieve(...) per query
     * - merges with stable order (first win)
     * - dedups by Statement.id (fallback to normalized text key)
     *
     * NOTE: Implementers may override to do a single batched search for better performance.
     */
    default List<Statement> retrieve(List<String> queries, int k, long seed) {
        if (queries == null || queries.isEmpty() || k <= 0) return List.of();

        final int nQ = queries.size();
        final int perQ = Math.max(1, (int) Math.ceil(k / (double) Math.max(1, nQ)));

        LinkedHashMap<String, Statement> merged = new LinkedHashMap<>(Math.max(16, k * 2));

        for (int i = 0; i < nQ; i++) {
            String q = queries.get(i);
            if (q == null || q.isBlank()) continue;

            long s = mix(seed, 0xD1B54A32D192ED03L ^ (i * 0x9E3779B97F4A7C15L));

            List<Statement> part = retrieve(q, perQ, s);
            if (part == null || part.isEmpty()) continue;

            for (Statement st : part) {
                if (st == null) continue;

                String key = stableKey(st);
                if (key.isEmpty()) continue;

                merged.putIfAbsent(key, st);

                if (merged.size() >= k) break;
            }

            if (merged.size() >= k) break;
        }

        return merged.isEmpty() ? List.of() : new ArrayList<>(merged.values());
    }

    /**
     * Scored retrieval for single query (debug/quality).
     *
     * Default: wraps {@link #retrieve(String, int, long)} and sets score=NaN if not available.
     */
    default List<Scored<Statement>> retrieveScored(String query, int k, long seed) {
        List<Statement> items = retrieve(query, k, seed);
        if (items == null || items.isEmpty()) return List.of();

        ArrayList<Scored<Statement>> out = new ArrayList<>(items.size());
        for (Statement st : items) out.add(new Scored<>(st, Double.NaN));
        return out;
    }

    /**
     * Scored retrieval for multi-query.
     *
     * Default: calls {@link #retrieve(List, int, long)} and wraps score=NaN.
     * Implementers may override for real scores.
     */
    default List<Scored<Statement>> retrieveScored(List<String> queries, int k, long seed) {
        List<Statement> items = retrieve(queries, k, seed);
        if (items == null || items.isEmpty()) return List.of();

        ArrayList<Scored<Statement>> out = new ArrayList<>(items.size());
        for (Statement st : items) out.add(new Scored<>(st, Double.NaN));
        return out;
    }

    // ---------------------------------------------------------------------
    // Helpers (kept inside interface to avoid new utility classes)
    // ---------------------------------------------------------------------

    private static String stableKey(Statement st) {
        if (st == null) return "";
        if (st.id != null && !st.id.isBlank()) return st.id;

        // fallback: normalized text key
        return stableTextKey(st.text);
    }

    private static String stableTextKey(String text) {
        if (text == null) return "";
        String s = text.trim().toLowerCase(Locale.ROOT);
        if (s.isEmpty()) return "";

        StringBuilder b = new StringBuilder(Math.min(96, s.length()));
        boolean ws = false;

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isWhitespace(c) || c == '\u00A0') {
                if (!ws) {
                    b.append(' ');
                    ws = true;
                }
            } else if (!Character.isISOControl(c)) {
                b.append(c);
                ws = false;
            }
            if (b.length() >= 160) break; // clamp
        }
        return b.toString().trim();
    }

    /** Deterministic seed mixer (SplitMix64-like). */
    private static long mix(long a, long b) {
        long x = a ^ (b + 0x9E3779B97F4A7C15L);
        x ^= (x >>> 30);
        x *= 0xBF58476D1CE4E5B9L;
        x ^= (x >>> 27);
        x *= 0x94D049BB133111EBL;
        x ^= (x >>> 31);
        return x;
    }
}