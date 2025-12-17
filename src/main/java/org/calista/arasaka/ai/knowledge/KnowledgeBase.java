// KnowledgeBase.java
package org.calista.arasaka.ai.knowledge;

import java.util.*;

/**
 * KnowledgeBase — интерфейс модуля памяти.
 *
 * <p>Практика проекта соблюдена: интерфейс + реализация.</p>
 * <p>Backwards-compatible: существующие методы сохранены.</p>
 */
public interface KnowledgeBase {

    /**
     * Upsert statement.
     *
     * @return true если данные реально изменились (полезно для логов/метрик)
     */
    boolean upsert(Statement st);

    Optional<Statement> get(String id);

    /** Всегда один и тот же порядок: по id (важно для снапшотов/диффов/отладки). */
    List<Statement> snapshotSorted();

    int size();

    // =========================
    // Retrieval / "thinking" support
    // =========================

    /**
     * Ranked search over KB.
     * Реализация должна быть детерминированной (score desc, tie-break by id).
     */
    default List<ScoredStatement> search(Query query) {
        return List.of();
    }

    /**
     * Единая точка построения Query из промпта/входного текста.
     * Реализация может использовать corpus stats / tokenization rules / stopwords и т.д.
     */
    default Query buildQueryFromPrompt(String prompt) {
        return new Query(Set.of(), Set.of());
    }

    /**
     * Iterative retrieval: retrieve → refine query → retrieve ... (без магии).
     */
    default RetrievalReport retrieveIterative(String prompt, int iterations, int topK) {
        return new RetrievalReport(Collections.emptyList());
    }

    // =========================
    // Nested DTOs (без "миллиона классов")
    // =========================

    final class Query {
        public final Set<String> tokens;
        public final Set<String> tags;

        public Query(Set<String> tokens, Set<String> tags) {
            this.tokens = tokens == null ? Set.of() : Collections.unmodifiableSet(tokens);
            this.tags = tags == null ? Set.of() : Collections.unmodifiableSet(tags);
        }
    }

    final class ScoredStatement {
        public final Statement statement;
        public final double score;
        /** Explainable features for logs/QA/debug. */
        public final Map<String, Double> features;

        public ScoredStatement(Statement statement, double score, Map<String, Double> features) {
            this.statement = Objects.requireNonNull(statement, "statement");
            this.score = score;
            this.features = features == null ? Map.of() : Collections.unmodifiableMap(features);
        }
    }

    final class RetrievalStep {
        public final Query query;
        public final List<ScoredStatement> evidence;

        public RetrievalStep(Query query, List<ScoredStatement> evidence) {
            this.query = Objects.requireNonNull(query, "query");
            this.evidence = evidence == null ? List.of() : Collections.unmodifiableList(evidence);
        }
    }

    final class RetrievalReport {
        public final List<RetrievalStep> steps;

        public RetrievalReport(List<RetrievalStep> steps) {
            this.steps = steps == null ? List.of() : Collections.unmodifiableList(steps);
        }
    }
}