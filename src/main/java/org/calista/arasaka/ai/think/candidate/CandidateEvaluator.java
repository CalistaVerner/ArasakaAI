package org.calista.arasaka.ai.think.candidate;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * CandidateEvaluator — контракт оценки кандидатов (скор + диагностическая телеметрия).
 *
 * ВАЖНО:
 *  - score — базовый скор (legacy)
 *  - effectiveScore() — то, что должны использовать движки (Iterative / Neural / Beam)
 */
public interface CandidateEvaluator {

    /**
     * Full evaluation entry point.
     * По умолчанию — legacy-режим.
     */
    default Evaluation evaluate(String userText, String candidateText, List<Statement> context) {
        double s = score(userText, candidateText, context);
        return Evaluation.basic(s);
    }

    /**
     * Minimal scoring function (legacy-compatible).
     */
    double score(String userText, String candidateText, List<Statement> context);

    /**
     * Immutable evaluation telemetry.
     */
    final class Evaluation {

        // --- legacy ---
        public final double score;
        public final String critique;
        public final double coverage;
        public final double contextSupport;
        public final double stylePenalty;

        // --- reasoning ---
        public final double groundedness;        // 0..1
        public final double contradictionRisk;   // 0..1
        public final double structureScore;      // 0..1  <<< ВАЖНО
        public final boolean valid;
        public final String validationNotes;

        // --- optional telemetry ---
        public final int tokens;
        public final long nanos;

        /**
         * Full constructor (enterprise).
         */
        public Evaluation(
                double score,
                String critique,
                double coverage,
                double contextSupport,
                double stylePenalty,
                double groundedness,
                double contradictionRisk,
                double structureScore,
                boolean valid,
                String validationNotes,
                int tokens,
                long nanos
        ) {
            this.score = finite(score, -1.0);
            this.critique = critique == null ? "" : critique;

            this.coverage = clamp01(coverage);
            this.contextSupport = clamp01(contextSupport);
            this.stylePenalty = clamp01(stylePenalty);

            this.groundedness = clamp01(groundedness);
            this.contradictionRisk = clamp01(contradictionRisk);
            this.structureScore = clamp01(structureScore);

            this.valid = valid;
            this.validationNotes = validationNotes == null ? "" : validationNotes;

            this.tokens = Math.max(0, tokens);
            this.nanos = Math.max(0L, nanos);
        }

        /**
         * Backward-compatible constructor.
         */
        public Evaluation(
                double score,
                String critique,
                double coverage,
                double contextSupport,
                double stylePenalty,
                double groundedness,
                double contradictionRisk,
                double structureScore,
                boolean valid,
                String validationNotes
        ) {
            this(score, critique, coverage, contextSupport, stylePenalty,
                    groundedness, contradictionRisk, structureScore,
                    valid, validationNotes, 0, 0L);
        }

        /**
         * Legacy minimal evaluation.
         */
        public static Evaluation basic(double score) {
            return new Evaluation(
                    score,
                    "",
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    true,
                    "",
                    0,
                    0L
            );
        }

        /**
         * Итоговый скор для движков мышления.
         */
        public double effectiveScore() {
            double s = score;

            // усиливаем хорошие сигналы
            s += 0.20 * groundedness;
            s += 0.15 * contextSupport;
            s += 0.20 * structureScore;

            // штрафы
            s -= 0.15 * stylePenalty;

            if (!valid) {
                s -= (1.0 + contradictionRisk);
            } else {
                s -= 0.30 * contradictionRisk;
            }
            return s;
        }

        public boolean isSane() {
            return Double.isFinite(score)
                    && in01(groundedness)
                    && in01(contradictionRisk)
                    && in01(structureScore);
        }

        public String brief() {
            return String.format(Locale.ROOT,
                    "s=%.3f eff=%.3f g=%.2f r=%.2f st=%.2f v=%s",
                    score, effectiveScore(),
                    groundedness, contradictionRisk, structureScore, valid);
        }

        @Override
        public String toString() {
            return "Evaluation{" +
                    "score=" + score +
                    ", effectiveScore=" + effectiveScore() +
                    ", groundedness=" + groundedness +
                    ", contradictionRisk=" + contradictionRisk +
                    ", structureScore=" + structureScore +
                    ", valid=" + valid +
                    '}';
        }

        // ---- utils ----

        private static double finite(double v, double fallback) {
            return Double.isFinite(v) ? v : fallback;
        }

        private static boolean in01(double v) {
            return v >= 0.0 && v <= 1.0 && Double.isFinite(v);
        }

        private static double clamp01(double v) {
            if (!Double.isFinite(v)) return 0.0;
            if (v < 0.0) return 0.0;
            if (v > 1.0) return 1.0;
            return v;
        }
    }

    // ---- helpers for implementations ----

    static List<Statement> safeContext(List<Statement> ctx) {
        return ctx == null ? Collections.emptyList() : ctx;
    }

    static String nn(String s) {
        return s == null ? "" : s;
    }

    static void requireInputs(String userText, String candidateText) {
        Objects.requireNonNull(userText, "userText");
        Objects.requireNonNull(candidateText, "candidateText");
    }
}