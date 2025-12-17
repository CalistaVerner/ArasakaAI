package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.List;

/**
 * Final output of a thinking cycle.
 *
 * Compatibility note:
 * - Engine/logging may refer to result.text
 * - Older code may refer to result.answer
 *
 * This class exposes BOTH (answer and text) pointing to the same value.
 */
public final class ThoughtResult {

    /** Primary output text (canonical). */
    public final String text;

    /** Backward-compatible alias for text. */
    public final String answer;

    /** Debug/training trace (deterministic, safe to log). */
    public final List<String> trace;

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

        String a = (answer == null ? "" : answer);

        // canonical output
        this.text = a;

        // alias
        this.answer = a;

        this.trace = (trace == null ? List.of() : trace);
        this.intent = (intent == null ? Intent.UNKNOWN : intent);
        this.bestCandidate = bestCandidate;
        this.evaluation = evaluation;
    }

    @Override
    public String toString() {
        return "ThoughtResult{len=" + text.length()
                + ", intent=" + (intent == null ? "UNKNOWN" : intent.name())
                + ", hasCandidate=" + (bestCandidate != null)
                + '}';
    }
}