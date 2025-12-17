package org.calista.arasaka.ai.think.response;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.ThoughtResult;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.ArrayList;
import java.util.List;

/**
 * ResponseStrategy — final answer builder.
 *
 * Backward compatibility:
 * - Old implementations only implement generate(userText, context, state)
 * - New engine can call build(userText, state, bestCandidate)
 *
 * Rule of thumb:
 * - generate(...) produces plain text
 * - build(...) produces ThoughtResult (text + telemetry)
 */
public interface ResponseStrategy {

    /**
     * Backward-compatible hook for router-based designs.
     *
     * In the modern "single smart strategy" pipeline,
     * the engine may ignore intent routing entirely and always call build()/generate().
     */
    default boolean supports(Intent intent) {
        return true;
    }

    /**
     * Legacy API: generate a final textual answer.
     * Implementations should not do retrieval; they must use provided context/state.
     */
    String generate(String userText, List<Statement> context, ThoughtState state);

    /**
     * Modern API: build full ThoughtResult including telemetry.
     *
     * Default implementation bridges to legacy generate().
     * Engine may pass bestCandidate == null.
     */
    default ThoughtResult build(String userText, ThoughtState state, Candidate bestCandidate) {
        List<Statement> ctx = (bestCandidate != null && bestCandidate.evaluation != null && bestCandidate.evaluation.valid)
                ? stateContextOrEmpty(state)   // prefer state-owned context if your engine stores it
                : stateContextOrEmpty(state);

        // Generate final text (legacy)
        String answer = generate(userText, ctx, state);

        // Build trace (best-effort; stable and safe)
        List<String> trace = buildTrace(state, bestCandidate);

        Intent intent = state != null && state.intent != null ? state.intent : Intent.UNKNOWN;

        return new ThoughtResult(
                answer,
                trace,
                intent,
                bestCandidate,
                bestCandidate == null ? null : bestCandidate.evaluation
        );
    }

    // --------- helpers (default, no new public types) ---------

    /**
     * If you store context in state in your project, adapt this method.
     * By default we don't assume state has context.
     */
    private static List<Statement> stateContextOrEmpty(ThoughtState state) {
        // ThoughtState in your current project does not expose context.
        // Keep this as empty to avoid implicit retrieval here.
        return List.of();
    }

    private static List<String> buildTrace(ThoughtState state, Candidate bestCandidate) {
        ArrayList<String> t = new ArrayList<>(8);

        if (state != null) {
            t.add("iter=" + state.iteration);
            t.add("phase=" + state.phase);
            t.add("diversity=" + state.diversity);

            if (state.tags != null && !state.tags.isEmpty()) {
                // keep it short; do NOT dump everything (enterprise logs only)
                t.add("tags.n=" + state.tags.size());
            }
            if (state.lastCritique != null && !state.lastCritique.isBlank()) {
                t.add("lastCritique=" + safeShort(state.lastCritique, 160));
            }
        }

        if (bestCandidate != null) {
            t.add("best.score=" + bestCandidate.score);
            if (bestCandidate.evaluation != null) {
                t.add("best.valid=" + bestCandidate.evaluation.valid);
                t.add("best.grounded=" + bestCandidate.evaluation.groundedness);
                t.add("best.risk=" + bestCandidate.evaluation.contradictionRisk);
                t.add("best.struct=" + bestCandidate.evaluation.structureScore);
            }
        }

        return t;
    }

    private static String safeShort(String s, int max) {
        if (s == null) return "";
        String x = s.trim();
        if (x.length() <= max) return x;
        return x.substring(0, max);
    }
}
