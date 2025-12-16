package org.calista.arasaka.ai.retrieve.scorer;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.Collection;

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
     * Optional fast tokens accessor for diversity/filters.
     * May return null if not supported.
     */
    default String[] tokens(Statement st) {
        return null;
    }
}
