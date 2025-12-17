package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ThoughtState — mutable per-request state used by the think-loop.
 *
 * Enterprise rules:
 * - No hardcoded user-visible text here.
 * - Tags + generationHint are the primary control plane for the generator/evaluator.
 * - copyForDraft() MUST clone tags to avoid cross-draft interference.
 * - Context is stored explicitly (no implicit retrieval inside strategies).
 */
public final class ThoughtState {

    // -------------------- Determinism / iteration --------------------

    public long seed;
    public int iteration;
    public int draftIndex;

    /** Engine phase ordinal (EXPLORE/EXPLOIT/REPAIR/VERIFY...). */
    public int phase;

    /** Diversity ordinal (LOW/MED/HIGH...). */
    public int diversity;

    // -------------------- Intent --------------------

    public Intent intent;

    // -------------------- Generation control --------------------

    /** Soft hint for generator (structured, parseable; may include previous metrics). */
    public String generationHint;

    /** Canonical tag-map used to control generation/strategy/evaluator (deterministic). */
    public Map<String, String> tags;

    // -------------------- Context snapshot (explicit) --------------------

    /**
     * Context snapshot used by the engine for the CURRENT best answer.
     * This is the only context that final ResponseStrategy should use.
     */
    public List<Statement> lastContext;

    /** Deterministic query list used for retrieval on the last iteration (debug/telemetry). */
    public List<String> lastQueries;

    // -------------------- LTM recall signals --------------------

    /** Evidence recalled from LTM (optional; subset). */
    public List<Statement> recalledMemory;

    /** Optional: raw recall query terms (debug/telemetry; deterministic). */
    public List<String> recallQueries;

    // -------------------- Best-so-far tracking --------------------

    public Candidate bestSoFar;
    public CandidateEvaluator.Evaluation bestEvaluation;

    // -------------------- Last snapshot --------------------

    public Candidate lastCandidate;
    public CandidateEvaluator.Evaluation lastEvaluation;
    public String lastCritique;

    // -------------------- Engine telemetry/bookkeeping --------------------

    public String engineNotes;
    public double scoreDelta;
    public int stagnation;

    /** Deterministic per-iteration trace lines (compact, safe-to-log). */
    public List<String> trace;

    public ThoughtState() {}

    /**
     * Copy state for a specific draft.
     * - tags are cloned (copy-on-draft)
     * - context/memory snapshots are shared references (treated as read-only lists)
     * - candidates/evaluations are shared references (treated as immutable snapshots by convention)
     */
    public ThoughtState copyForDraft(int newDraftIndex) {
        ThoughtState s = new ThoughtState();

        s.seed = this.seed;
        s.iteration = this.iteration;
        s.draftIndex = newDraftIndex;

        s.phase = this.phase;
        s.diversity = this.diversity;

        s.intent = this.intent;
        s.generationHint = this.generationHint;

        if (this.tags != null && !this.tags.isEmpty()) {
            s.tags = new HashMap<>(this.tags);
        }

        // explicit context snapshots
        s.lastContext = this.lastContext;
        s.lastQueries = this.lastQueries;

        // memory/context signals
        s.recalledMemory = this.recalledMemory;
        s.recallQueries = this.recallQueries;

        // best/last snapshots
        s.bestSoFar = this.bestSoFar;
        s.bestEvaluation = this.bestEvaluation;

        s.lastCandidate = this.lastCandidate;
        s.lastEvaluation = this.lastEvaluation;
        s.lastCritique = this.lastCritique;

        // telemetry
        s.engineNotes = this.engineNotes;
        s.scoreDelta = this.scoreDelta;
        s.stagnation = this.stagnation;

        s.trace = this.trace; // shared; append only from engine thread

        return s;
    }

    // -------------------- Tag + trace helpers --------------------

    public Map<String, String> ensureTags() {
        if (tags == null) tags = new HashMap<>(16);
        return tags;
    }

    public void putTag(String key, String value) {
        ensureTags().put(key, value);
    }

    public void putTagIfAbsent(String key, String value) {
        ensureTags().putIfAbsent(key, value);
    }

    public void removeTag(String key) {
        if (tags != null) tags.remove(key);
    }

    public List<String> ensureTrace() {
        if (trace == null) trace = new ArrayList<>(16);
        return trace;
    }

    public void addTrace(String line) {
        if (line == null || line.isBlank()) return;
        ensureTrace().add(line);
    }

    @Override
    public String toString() {
        return "ThoughtState{iter=" + iteration
                + ", draft=" + draftIndex
                + ", phase=" + phase
                + ", diversity=" + diversity
                + ", tags=" + (tags == null ? 0 : tags.size())
                + ", ctx=" + (lastContext == null ? 0 : lastContext.size())
                + ", ltm=" + (recalledMemory == null ? 0 : recalledMemory.size())
                + '}';
    }
}