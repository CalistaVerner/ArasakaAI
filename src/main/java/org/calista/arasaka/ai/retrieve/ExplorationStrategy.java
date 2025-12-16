package org.calista.arasaka.ai.retrieve;

import java.util.List;

public interface ExplorationStrategy {
    <T> List<T> select(List<Scored<T>> ranked, int k, ExplorationConfig cfg, long seed);
}