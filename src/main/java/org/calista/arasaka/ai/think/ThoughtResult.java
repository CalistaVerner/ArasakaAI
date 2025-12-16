package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.List;

/**
 * Final output of a thinking cycle.
 */
public final class ThoughtResult {

    public final String answer;
    public final List<String> trace; // useful for debugging/training/logs

    // optional telemetry
    public final Intent intent;
    public final Candidate bestCandidate;
    public final CandidateEvaluator.Evaluation evaluation;

    public ThoughtResult(String answer, List<String> trace) {
        this(answer, trace, Intent.UNKNOWN, null, null);
    }

    public ThoughtResult(String answer,
                         List<String> trace,
                         Intent intent,
                         Candidate bestCandidate,
                         CandidateEvaluator.Evaluation evaluation) {
        this.answer = answer == null ? "" : answer;
        this.trace = trace == null ? List.of() : trace;
        this.intent = intent == null ? Intent.UNKNOWN : intent;
        this.bestCandidate = bestCandidate;
        this.evaluation = evaluation;
    }
}