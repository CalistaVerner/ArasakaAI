package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.List;

public interface CandidateEvaluator {

    default Evaluation evaluate(String userText, String candidateText, List<Statement> context) {
        return new Evaluation(score(userText, candidateText, context), "", 0.0, 0.0, 0.0);
    }

    double score(String userText, String candidateText, List<Statement> context);

    final class Evaluation {
        public final double score;
        public final String critique;
        public final double coverage;
        public final double contextSupport;
        public final double stylePenalty;

        public Evaluation(double score, String critique, double coverage, double contextSupport, double stylePenalty) {
            this.score = score;
            this.critique = critique == null ? "" : critique;
            this.coverage = coverage;
            this.contextSupport = contextSupport;
            this.stylePenalty = stylePenalty;
        }
    }
}