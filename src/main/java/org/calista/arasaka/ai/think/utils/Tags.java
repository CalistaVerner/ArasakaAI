package org.calista.arasaka.ai.think.utils;

/**
 * Tags — canonical keys for ThoughtState.tags.
 *
 * Rules:
 * - Keys are stable (do not rename without migration).
 * - Values are typically "true"/"false" or "1"/"0".
 * - Tags are hints for generator/evaluator/engine (must not break determinism).
 */
public final class Tags {
    private Tags() {}

    // ---- strict policy ----
    public static final String VERIFY_STRICT = "verify.strict";
    public static final String REPAIR_STRICT = "repair.strict";

    // ---- repair hints (generator-safe) ----
    public static final String REPAIR_VERIFY_FAIL = "repair.verifyFail";
    public static final String REPAIR_ADD_EVIDENCE = "repair.addEvidence";
    public static final String REPAIR_REDUCE_NOVELTY = "repair.reduceNovelty";
    public static final String REPAIR_REQUIRE_EVIDENCE = "repair.requireEvidence";
    public static final String REPAIR_FORBID_GENERIC = "repair.forbidGeneric";
    public static final String REPAIR_FIX_STRUCTURE = "repair.fixStructure";
    public static final String REPAIR_AVOID_ECHO = "repair.avoidEcho";
    public static final String REPAIR_CITE_IDS = "repair.citeIds";
    public static final String REPAIR_REQUIRE_GREETING = "repair.requireGreeting";
}