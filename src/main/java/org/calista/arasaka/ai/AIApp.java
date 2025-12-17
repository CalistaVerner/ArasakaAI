package org.calista.arasaka.ai;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.core.AIKernel;
import org.calista.arasaka.ai.core.AIComposer;
import org.calista.arasaka.ai.events.AIEvent;
import org.calista.arasaka.ai.learn.Learner;
import org.calista.arasaka.ai.learn.impl.AdvancedLearner;
import org.calista.arasaka.ai.think.engine.ThoughtCycleEngine;
import org.calista.arasaka.ai.tokenizer.Tokenizer;
import org.calista.arasaka.ai.tokenizer.impl.AdvancedTokenizer;
import org.calista.arasaka.ai.tokenizer.impl.HybridTokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Scanner;

public final class AIApp {
    private static final Logger log = LogManager.getLogger(AIApp.class);
    AIKernel kernel;
    Tokenizer tokenizer;
    AIComposer AIComposer;
    ThoughtCycleEngine thoughtCycleEngine;
    Learner learner;

    public static void main(String[] args) throws Exception {
        new AIApp();
    }

    AIApp() throws IOException {
        Path cfg = Path.of("config/config.json");
        kernel = AIKernel.boot(cfg);
        tokenizer = new AdvancedTokenizer();
        AIComposer = new AIComposer(this);
        thoughtCycleEngine = AIComposer.buildEngine(kernel, tokenizer);
        learner = new AdvancedLearner(kernel.knowledge(), tokenizer, kernel.config().learning.newStatementWeight, kernel.config().learning.reinforceStep);

        String sessionId = "sess-" + Long.toHexString(System.nanoTime());
        long turns = 0;

        log.info("AI started. knowledge.size=" + kernel.knowledge().size());
        log.info("Type 'exit' to quit.\n");

        try (Scanner sc = new Scanner(System.in)) {
            while (true) {
                System.out.print("> ");
                String user = sc.nextLine();
                if (user == null) break;
                if (user.trim().equalsIgnoreCase("exit")) break;

                long now = System.currentTimeMillis();
                kernel.eventStore().append(AIEvent.of("USER", sessionId, user, now));

                long seed = System.nanoTime(); // “живость”: разные траектории поиска/мысли
                var result = thoughtCycleEngine.think(user, seed);

                kernel.eventStore().append(AIEvent.of("ASSISTANT", sessionId, result.answer, System.currentTimeMillis()));
                System.out.println("\n" + result.answer + "\n");

                // learning
                if (kernel.config().learning.enabled) {
                    learner.learnFromText(user, "user");
                    learner.learnFromText(result.answer, "assistant");
                    kernel.eventStore().append(AIEvent.of("LEARNING", sessionId, "learned_from_turn", System.currentTimeMillis()));
                }

                turns++;
                if (turns % kernel.config().knowledge.autoSnapshotEveryTurns == 0) {
                    kernel.snapshotStore().save(kernel.knowledge());
                    kernel.eventStore().append(AIEvent.of("SNAPSHOT", sessionId, "saved", System.currentTimeMillis()));
                }
            }
            kernel.snapshotStore().save(kernel.knowledge());
            System.out.println("Bye. Snapshot saved.");
        }
    }

    public AIKernel getKernel() {
        return kernel;
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public AIComposer getGraalAIComposer() {
        return AIComposer;
    }

    public ThoughtCycleEngine getThoughtCycleEngine() {
        return thoughtCycleEngine;
    }

    public Learner getLearner() {
        return learner;
    }
}