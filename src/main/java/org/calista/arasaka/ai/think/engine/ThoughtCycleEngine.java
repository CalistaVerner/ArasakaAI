package org.calista.arasaka.ai.think.engine;

import org.calista.arasaka.ai.think.ThoughtResult;

/**
 * ThoughtCycleEngine — внутренний контракт исполнения цикла размышления.
 *
 * <p>ВАЖНО:
 * <ul>
 *   <li>НЕ orchestrator</li>
 *   <li>НЕ builder</li>
 *   <li>НЕ владеет зависимостями</li>
 * </ul>
 *
 * <p>Engine выполняет ОДИН детерминированный thinking-loop,
 * полностью управляемый внешним оркестратором (Think).</p>
 *
 * <p>Все политики, конфиги, lifecycle, пулы, генераторы и память
 * находятся выше — в Think.</p>
 */
public interface ThoughtCycleEngine {

    /**
     * Execute one thinking cycle.
     *
     * @param userText normalized user input
     * @param seed deterministic seed (provided by Think)
     * @return ThoughtResult with best answer + diagnostics
     */
    ThoughtResult think(String userText, long seed);
}