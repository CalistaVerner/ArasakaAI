package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.intent.Intent;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ThoughtState — mutable per-request state used by the think-loop.
 *
 * Rules:
 * - No hardcoded user-visible text here.
 * - Tags + generationHint are the primary control plane for the generator.
 * - copyForDraft() MUST clone tags to avoid cross-draft interference.
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

    /** Soft hint for generator (e.g., previous best answer or critique summary). */
    public String generationHint;

    /** Canonical tag-map used to control generation/strategy. */
    public Map<String, String> tags;

    // -------------------- Context / memory signals --------------------

    /**
     * Evidence recalled from LTM (optional).
     * Used by strategies to mark [LTM] evidence and by engine to bias ranking.
     *
     * IMPORTANT:
     * - This is NOT the full retrieved context; it's the subset that came from memory.
     * - May be empty or null.
     */
    public List<Statement> recalledMemory;

    /**
     * Optional: raw recall query terms (debug/telemetry; deterministic).
     * Engine may set this to show what it asked LTM for.
     */
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

    public ThoughtState() {}

    /**
     * Copy state for a specific draft.
     * - tags are cloned (copy-on-draft)
     * - recalledMemory/queries are shared references (treated as read-only lists)
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
        } else {
            s.tags = null;
        }

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

        return s;
    }

    // -------------------- Tag helpers --------------------

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

    @Override
    public String toString() {
        return "ThoughtState{iter=" + iteration
                + ", draft=" + draftIndex
                + ", phase=" + phase
                + ", diversity=" + diversity
                + ", tags=" + (tags == null ? 0 : tags.size())
                + ", ltm=" + (recalledMemory == null ? 0 : recalledMemory.size())
                + '}';
    }
}