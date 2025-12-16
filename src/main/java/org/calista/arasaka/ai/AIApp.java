package org.calista.arasaka.ai;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.core.AIComposer;
import org.calista.arasaka.ai.core.AIKernel;
import org.calista.arasaka.ai.events.AIEvent;
import org.calista.arasaka.ai.learn.SimpleLearner;
import org.calista.arasaka.ai.tokenizer.impl.AdvancedTokenizer;

import java.nio.file.Path;
import java.util.Scanner;

public final class AIApp {
    private static final Logger log = LogManager.getLogger(AIApp.class);

    public static void main(String[] args) throws Exception {
        Path cfg = args.length > 0 ? Path.of(args[0]) : Path.of("config/config.json");
        AIKernel kernel = AIKernel.boot(cfg);

        var tokenizer = new AdvancedTokenizer();
        var engine = AIComposer.buildThinking(kernel, tokenizer);

        SimpleLearner learner = new SimpleLearner(
                kernel.knowledge(),
                tokenizer,
                kernel.config().learning.newStatementWeight,
                kernel.config().learning.reinforceStep
        );

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
                var result = engine.think(user, seed);

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
        }

        // финальный снапшот
        kernel.snapshotStore().save(kernel.knowledge());
        System.out.println("Bye. Snapshot saved.");
    }
}