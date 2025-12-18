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
 * - Must be deterministic for the same inputs (seed + userText + context).
 * - It is OK to store runtime caches, but they must be transient and never affect determinism.
 * - copyForDraft() MUST clone tags to avoid cross-draft interference.
 * - Context is stored explicitly (no implicit retrieval inside strategies).
 */
public final class ThoughtState {

    // -------------------- Determinism / iteration --------------------

    public long seed;
    public int iteration;
    public int draftIndex;

    /** Engine phase enum ordinal (see CandidateControlSignals.Phase). */
    public int phase;

    /** Diversity enum ordinal (see CandidateControlSignals.Diversity). */
    public int diversity;

    public Intent intent;

    // -------------------- Generation control --------------------

    /**
     * Optional hint from previous best answer or repair pass.
     * Generator should treat it as "anchor" but must remain deterministic.
     */
    public String generationHint;

    /** Canonical tag-map used to control generation/strategy/evaluator (deterministic). */
    public Map<String, String> tags;

    /**
     * Runtime per-request/per-iteration cache.
     *
     * Not part of the determinism contract; must be treated as ephemeral.
     * Intended for sharing precomputed artifacts (e.g., context token unions) across drafts.
     */
    public transient Map<String, Object> cache;

    // -------------------- Context snapshot (explicit) --------------------

    /**
     * Context snapshot used by the engine for the CURRENT best answer.
     * This is the only context that final response strategy must rely on.
     */
    public List<Statement> lastContext;

    /** Queries used for the most recent retrieval step. */
    public List<String> lastQueries;

    // -------------------- Memory / recall snapshot --------------------

    /** Evidence recalled from LTM (if enabled). */
    public List<Statement> recalledMemory;

    /** Queries used for LTM recall in the most recent iteration. */
    public List<String> recallQueries;

    // -------------------- Candidate snapshots (optional, for strategies) --------------------

    /** Best candidate so far (across iterations). */
    public Candidate bestSoFar;

    /** Evaluation of best candidate so far (cached convenience). */
    public CandidateEvaluator.Evaluation bestEvaluation;

    /** Last evaluated candidate in current iteration (for debugging/strategies). */
    public Candidate lastCandidate;

    /** Last evaluation (for debugging/strategies). */
    public CandidateEvaluator.Evaluation lastEvaluation;

    /** Last critique string (generator-safe). */
    public String lastCritique;

    // -------------------- Engine telemetry (never feed to generator) --------------------

    /** Engine notes / flags (telemetry only). */
    public String engineNotes;

    /** Best score delta on last iteration (telemetry). */
    public double scoreDelta;

    /** Stagnation counter (telemetry). */
    public int stagnation;

    /** Append-only trace lines (telemetry). */
    public List<String> trace;

    // -------------------- Lifecycle helpers --------------------

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

        // runtime cache (shared across drafts within the same iteration)
        s.cache = this.cache;

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
        Map<String, String> t = ensureTags();
        t.putIfAbsent(key, value);
    }

    public String tag(String key) {
        if (tags == null) return null;
        return tags.get(key);
    }

    public boolean tagTrue(String key) {
        String v = tag(key);
        if (v == null) return false;
        v = v.trim();
        return "1".equals(v) || "true".equalsIgnoreCase(v) || "yes".equalsIgnoreCase(v);
    }

    public List<String> ensureTrace() {
        if (trace == null) trace = new ArrayList<>(64);
        return trace;
    }

    public void trace(String line) {
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