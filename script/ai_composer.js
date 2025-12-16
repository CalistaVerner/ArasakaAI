"use strict";

/* ============================================================
 * buildPlan.js (GraalVM)
 * ============================================================
 * - Нормализация thinking-конфига
 * - Автоматический вывод данных в консоль
 * - Без module.exports / require.main (не Node)
 * ============================================================
 */

const BUILDPLAN_LOG = {
    enabled: true,
    colors: true,
    showInput: true,
    showPlan: true
};

function buildPlan(cfg) {
    const t = cfg?.thinking ?? {};
    const engine = String(t.engine ?? "iterative").trim().toLowerCase();

    const plan = {
        engine,
        iterations: clampInt(intVal(t.iterations, 2), 1, 8),
        retrieveK: clampInt(intVal(t.retrieveK, 16), 1, 128),
        patience: clampInt(intVal(t.patience, 2), 0, 6),
        targetScore: doubleVal(t.targetScore, 0.75),

        draftsPerIteration: clampInt(
            intVal(t.draftsPerIteration, intVal(t.drafts, 8)),
            1, 32
        ),

        beamWidth: clampInt(intVal(t.beamWidth, 12), 1, 32),
        draftsPerBeam: clampInt(intVal(t.draftsPerBeam, 2), 1, 16),
        maxDraftsPerIter: clampInt(
            intVal(
                t.maxDraftsPerIter,
                Math.max(
                    8,
                    intVal(t.beamWidth, 12) * intVal(t.draftsPerBeam, 2)
                )
            ),
            1, 256
        ),

        diversityPenalty: doubleVal(t.diversityPenalty, 0.18),
        minDiversityJaccard: doubleVal(t.minDiversityJaccard, 0.92),
        verifyPassEnabled: boolVal(t.verifyPassEnabled, true),

        ltmEnabled: boolVal(t.ltmEnabled, false),
        ltmCapacity: clampInt(intVal(t.ltmCapacity, 2048), 0, 200_000),
        ltmRecallK: clampInt(intVal(t.ltmRecallK, 8), 0, 128),
        ltmWriteMinGroundedness: doubleVal(t.ltmWriteMinGroundedness, 0.55),

        refineRounds: clampInt(intVal(t.refineRounds, 1), 0, 8),
        refineQueryBudget: clampInt(intVal(t.refineQueryBudget, 16), 1, 128)
    };

    if (BUILDPLAN_LOG.enabled) {
        logBuildPlan(cfg?.thinking, plan, BUILDPLAN_LOG);
    }

    return plan;
}

/* =========================
 * Logging / Output
 * ========================= */

function logBuildPlan(input, plan, opts) {
    const colors = !!opts?.colors;

    const C = colors ? {
        reset: "\x1b[0m",
        title: "\x1b[1m\x1b[36m",
        key: "\x1b[33m",
        num: "\x1b[32m",
        bool: "\x1b[35m",
        str: "\x1b[36m",
        dim: "\x1b[2m"
    } : {
        reset: "", title: "", key: "", num: "", bool: "", str: "", dim: ""
    };

    console.log(`\n${C.title}🧠 buildPlan()${C.reset}`);

    if (opts?.showInput) {
        console.log(`${C.dim}┌─ input.thinking${C.reset}`);
        console.log(safeStringify(input ?? {}, 2));
        console.log(`${C.dim}└─${C.reset}`);
    }

    if (opts?.showPlan) {
        printPlan(plan, { colors });
    }
}

function printPlan(plan, opts = {}) {
    const colors = !!opts.colors;

    const C = colors ? {
        reset: "\x1b[0m",
        title: "\x1b[1m\x1b[36m",
        key: "\x1b[33m",
        num: "\x1b[32m",
        bool: "\x1b[35m",
        str: "\x1b[36m"
    } : {
        reset: "", title: "", key: "", num: "", bool: "", str: ""
    };

    console.log(`${C.title}┌─ resolved thinking plan${C.reset}`);

    for (const [k, v] of Object.entries(plan)) {
        let c = C.str;
        if (typeof v === "number") c = C.num;
        else if (typeof v === "boolean") c = C.bool;

        console.log(`  ${C.key}${k.padEnd(30)}${C.reset} : ${c}${v}${C.reset}`);
    }

    console.log(`${C.title}└─ end${C.reset}`);
}

function safeStringify(obj, space) {
    const seen = new Set();
    return JSON.stringify(obj, function (k, v) {
        if (v && typeof v === "object") {
            if (seen.has(v)) return "[Circular]";
            seen.add(v);
        }
        return v;
    }, space);
}

/* =========================
 * Utils
 * ========================= */

function intVal(v, d) {
    if (typeof v === "number") return v | 0;
    if (typeof v === "string" && v.trim() !== "" && !isNaN(+v)) return (+v) | 0;
    return d;
}

function doubleVal(v, d) {
    if (typeof v === "number") return v;
    if (typeof v === "string" && v.trim() !== "" && !isNaN(+v)) return +v;
    return d;
}

function boolVal(v, d) {
    if (typeof v === "boolean") return v;
    if (typeof v === "number") return v !== 0;
    if (typeof v === "string") {
        const x = v.trim().toLowerCase();
        if (["true", "1", "yes", "y", "on", "enabled"].includes(x)) return true;
        if (["false", "0", "no", "n", "off", "disabled"].includes(x)) return false;
    }
    return d;
}

function clampInt(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

/* =========================
 * Export to global (GraalVM friendly)
 * ========================= */

globalThis.buildPlan = buildPlan;
globalThis.printPlan = printPlan;
globalThis.BUILDPLAN_LOG = BUILDPLAN_LOG;