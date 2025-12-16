package org.calista.arasaka.ai.retrieve.scorer;

import org.calista.arasaka.ai.knowledge.Statement;

public interface Scorer {
    double score(String query, Statement st);
}