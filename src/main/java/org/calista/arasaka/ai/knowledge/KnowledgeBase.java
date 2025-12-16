package org.calista.arasaka.ai.knowledge;

import java.util.List;
import java.util.Optional;

public interface KnowledgeBase {
    boolean upsert(Statement st);
    Optional<Statement> get(String id);

    /** Всегда один и тот же порядок: по id (важно для снапшотов/диффов/отладки). */
    List<Statement> snapshotSorted();
    int size();
}