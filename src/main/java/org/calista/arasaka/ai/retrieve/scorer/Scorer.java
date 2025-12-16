package org.calista.arasaka.ai.retrieve.scorer;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.Collection;
import java.util.List;

/**
 * A scorer estimates relevance of a {@link Statement} to the query.
 *
 * Design goals:
 *  - deterministic (no RNG required)
 *  - fast (optional prepare + token cache)
 *  - extensible (batch scoring hook without forcing implementations)
 */
public interface Scorer {
    double score(String query, Statement st);

    /**
     * Optional warmup step for enterprise-grade performance.
     * Implementations may precompute statistics/caches over the corpus.
     */
    default void prepare(Collection<Statement> corpus) {
        // no-op
    }

    /**
     * Optional fast tokens accessor for gating/diversity.
     * May return null if not supported.
     */
    default String[] tokens(Statement st) {
        return null;
    }

    /**
     * Optional batch scoring hook.
     * Default implementation just calls {@link #score(String, Statement)} in a loop.
     *
     * IMPORTANT: must preserve the order of {@code candidates}.
     */
    default double[] scoreBatch(String query, List<Statement> candidates) {
        if (candidates == null || candidates.isEmpty()) return new double[0];
        double[] out = new double[candidates.size()];
        for (int i = 0; i < candidates.size(); i++) {
            out[i] = score(query, candidates.get(i));
        }
        return out;
    }
}