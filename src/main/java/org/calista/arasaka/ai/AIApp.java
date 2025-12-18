package org.calista.arasaka.ai;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.core.AIKernel;
import org.calista.arasaka.ai.core.AIComposer;
import org.calista.arasaka.ai.events.AIEvent;
import org.calista.arasaka.ai.learn.Learner;
import org.calista.arasaka.ai.learn.impl.BasicLearner;
import org.calista.arasaka.ai.think.Think;
import org.calista.arasaka.ai.tokenizer.Tokenizer;
import org.calista.arasaka.ai.tokenizer.impl.AdvancedTokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Scanner;

/**
 * AIApp — interactive console runner.
 *
 * Updated lifecycle:
 *  1) build kernel (no bootstrap inside build)
 *  2) kernel.bootstrap()
 *  3) compose Think + Learner
 *  4) run loop
 *  5) close Think (owns eval pool) + Kernel (owns JS context)
 */
public final class AIApp {

    private static final Logger log = LogManager.getLogger(AIApp.class);

    private Path cfgPath;
    private AIKernel kernel;
    private Tokenizer tokenizer;
    private AIComposer composer;
    private Think think;
    private Learner learner;

    public static void main(String[] args) throws Exception {
        new AIApp().run();
    }

    public AIApp() {
        this.cfgPath = Path.of("config/config.json");
    }

    public void run() throws IOException {
        try {
            // 1) Build kernel container (loads/creates config, init IO/stores, NO corpora load yet)
            kernel = AIKernel.builder()
                    .configRoot(Path.of("."))
                    .enableJs(true)
                    .allowHostAccess(false)
                    .allowIO(false)
                    .build(cfgPath);

            // 2) Explicit bootstrap step (loads corpora into KB)
            kernel.bootstrap();

            tokenizer = new AdvancedTokenizer();

            composer = new AIComposer(this);

            // 3) Compose Think orchestrator (Think owns its eval pool unless injected)
            think = composer.buildEngine(kernel, tokenizer);

            learner = new BasicLearner(
                    kernel.knowledge(),
                    tokenizer,
                    kernel.config().learning.newStatementWeight,
                    kernel.config().learning.reinforceStep
            );

            runConsoleLoop();
        } finally {
            shutdown();
        }
    }

    private void runConsoleLoop() throws IOException {
        String sessionId = "sess-" + Long.toHexString(System.nanoTime());
        long turns = 0;

        log.info("AI started. knowledge.size={}", kernel.knowledge().size());
        log.info("Type 'exit' to quit.\n");

        try (Scanner sc = new Scanner(System.in)) {
            while (true) {
                System.out.print("> ");
                String user = sc.nextLine();
                if (user == null) break;

                user = user.trim();
                if (user.equalsIgnoreCase("exit")) break;
                if (user.isEmpty()) continue;

                long now = System.currentTimeMillis();
                kernel.eventStore().append(AIEvent.of("USER", sessionId, user, now));

                // deterministic per-turn seed source
                long seed = System.nanoTime();
                var result = think.think(user, seed);

                long at = System.currentTimeMillis();
                kernel.eventStore().append(AIEvent.of("ASSISTANT", sessionId, result.answer, at));
                System.out.println("\n" + result.answer + "\n");

                // learning
                if (kernel.config().learning.enabled) {
                    learner.learnFromText(user, "user");
                    learner.learnFromText(result.answer, "assistant");
                    kernel.eventStore().append(AIEvent.of("LEARNING", sessionId, "learned_from_turn", System.currentTimeMillis()));
                }

                // snapshots
                turns++;
                int every = kernel.config().knowledge.autoSnapshotEveryTurns;
                if (every > 0 && (turns % every == 0)) {
                    kernel.snapshotStore().save(kernel.knowledge());
                    kernel.eventStore().append(AIEvent.of("SNAPSHOT", sessionId, "saved", System.currentTimeMillis()));
                }
            }

            kernel.snapshotStore().save(kernel.knowledge());
            System.out.println("Bye. Snapshot saved.");
        }
    }

    private void shutdown() {
        // Ensure owned resources are released
        try {
            if (think != null) think.close();      // closes owned eval pool (if owned)
        } catch (Throwable ignored) {}

        try {
            if (kernel != null) kernel.close();    // closes JS context
        } catch (Throwable ignored) {}
    }

    // ---------------------------------------------------------------------
    // Accessors (for composer / integrations)
    // ---------------------------------------------------------------------

    public AIKernel getKernel() { return kernel; }

    public Tokenizer getTokenizer() { return tokenizer; }

    public AIComposer getComposer() { return composer; }

    public Think getThink() { return think; }

    public Learner getLearner() { return learner; }

    public Path getCfgPath() { return cfgPath; }
}