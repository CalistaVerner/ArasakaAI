package org.calista.arasaka.ai.retrieve;

import org.calista.arasaka.ai.knowledge.Statement;

public interface Scorer {
    double score(String query, Statement st);
}