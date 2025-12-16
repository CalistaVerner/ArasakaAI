package org.calista.arasaka.ai.retrieve.retriver;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.List;

public interface Retriever {
    List<Statement> retrieve(String query, int k, long seed);
}