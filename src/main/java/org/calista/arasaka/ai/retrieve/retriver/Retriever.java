package org.calista.arasaka.ai.retrieve.retriver;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.Scored;

import java.util.ArrayList;
import java.util.List;

/**
 * Retriever returns the most relevant knowledge statements for a query.
 *
 * Note: return type is {@code List<Statement>} for backward compatibility.
 * Use {@link #retrieveScored(String, int, long)} when you want scores for debugging/quality.
 */
public interface Retriever {
    List<Statement> retrieve(String query, int k, long seed);

    default List<Scored<Statement>> retrieveScored(String query, int k, long seed) {
        List<Statement> items = retrieve(query, k, seed);
        if (items == null || items.isEmpty()) return List.of();
        // Unknown scorer here; keep score=NaN as "not provided".
        ArrayList<Scored<Statement>> out = new ArrayList<>(items.size());
        for (Statement st : items) out.add(new Scored<>(st, Double.NaN));
        return out;
    }
}