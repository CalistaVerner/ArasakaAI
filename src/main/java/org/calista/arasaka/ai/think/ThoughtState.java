package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.candidate.Candidate;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.think.intent.Intent;
import org.calista.arasaka.ai.think.textGenerator.TextGenerator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Mutable state for a single reasoning episode.
 *
 * <p>Important: long-term memory is NOT stored here. The engine may attach
 * recalled memory items into {@link #recalledMemory} for transparency and evaluation.</p>
 */
public final class ThoughtState {

    // --- episode info ---
    public int iteration = 0;
    public Intent intent = Intent.UNKNOWN;
    public String query = "";

    /**
     * Engine-provided deterministic seed per iteration.
     */
    public long seed = 0L;

    /**
     * Draft index inside an iteration (0..draftsPerIter-1).
     */
    public int draftIndex = 0;

    /**
     * Optional per-iteration plan (engine -> generator). Mirrors encoded fields in generationHint.
     * These are NOT required by all generators, but allow strong backends to avoid re-parsing strings.
     */
    public int draftsRequested = 1;
    public int phase = 0; // CandidateControlSignals.Phase ordinal
    public int diversity = 1; // CandidateControlSignals.Diversity ordinal
    public int beamWidth = 12;
    public int maxTokens = 96;
    public double evidenceStrictness = 0.65;
    public boolean requireSections = true;
    public boolean forbidGeneric = true;


    /**
     * Light hint for generation (contract/constraints), NOT hard-coded domain rules.
     * Think of it as telemetry tokens for BeamSearch/TensorFlow (parseable, stable).
     */
    public String generationHint = "";

    /**
     * Optional opaque tags produced by engine/validators (stable keys).
     * Useful for tracing, tests, A/B, and future ML conditioning.
     *
     * Expected keys (examples):
     *  - response.style = md|plain
     *  - response.sections = summary,evidence,actions
     *  - response.label.summary / evidence / actions
     */
    public Map<String, String> tags = new HashMap<>();

    // --- self-correction loop ---
    public String lastCritique = "";
    public Candidate bestSoFar = null;

    /**
     * Telemetry of the best candidate so far (mirrors {@link #bestSoFar}).
     */
    public CandidateEvaluator.Evaluation bestEvaluation = null;

    public Candidate lastCandidate = null;

    /**
     * Telemetry of the last evaluated candidate (mirrors {@link #lastCandidate}).
     */
    public CandidateEvaluator.Evaluation lastEvaluation = null;

    // --- evidence ---
    public List<Statement> lastContext = List.of();

    /**
     * Evidence recalled from long-term memory (if enabled by the engine).
     * These items are merged into {@link #lastContext}, but kept separately for debugging.
     */
    public List<Statement> recalledMemory = List.of();

    /**
     * Optional ML backend (BeamSearch/TensorFlow).
     */
    public TextGenerator generator = null;
}