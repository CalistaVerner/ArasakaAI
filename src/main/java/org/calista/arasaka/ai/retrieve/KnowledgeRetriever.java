package org.calista.arasaka.ai.retrieve;

import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.Statement;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class KnowledgeRetriever implements Retriever {
    private final KnowledgeBase kb;
    private final Scorer scorer;
    private final ExplorationStrategy exploration;
    private final ExplorationConfig explorationCfg;

    public KnowledgeRetriever(
            KnowledgeBase kb,
            Scorer scorer,
            ExplorationStrategy exploration,
            ExplorationConfig explorationCfg
    ) {
        this.kb = kb;
        this.scorer = scorer;
        this.exploration = exploration;
        this.explorationCfg = explorationCfg;
    }

    @Override
    public List<Statement> retrieve(String query, int k, long seed) {
        List<Statement> all = kb.snapshotSorted(); // стабильный базовый порядок
        ArrayList<Scored<Statement>> scored = new ArrayList<>(all.size());

        for (Statement st : all) {
            double s = scorer.score(query, st);
            if (s > 0.0) scored.add(new Scored<>(st, s));
        }

        scored.sort(Comparator
                .comparingDouble((Scored<Statement> x) -> x.score).reversed()
                .thenComparing(x -> x.item.id)); // стабильно

        return exploration.select(scored, k, explorationCfg, seed);
    }
}