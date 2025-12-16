package org.calista.arasaka.ai.retrieve.scorer.impl;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.scorer.Scorer;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class TokenOverlapScorer implements Scorer {
    private final Tokenizer tokenizer;

    public TokenOverlapScorer(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public double score(String query, Statement st) {
        List<String> q = tokenizer.tokenize(query);
        List<String> t = tokenizer.tokenize(st.text);

        if (q.isEmpty() || t.isEmpty()) return 0.0;

        Set<String> qset = new HashSet<>(q);
        int hit = 0;
        for (String tok : t) if (qset.contains(tok)) hit++;

        // “взрослая” нормализация + учитываем weight утверждения
        double overlap = (double) hit / Math.sqrt((q.size() * t.size()));
        return overlap * st.weight;
    }
}