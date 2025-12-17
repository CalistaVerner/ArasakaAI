package org.calista.arasaka.ai.think.textGenerator;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.think.ThoughtState;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Pluggable generation backend.
 *
 * <h3>Determinism contract</h3>
 * For the same (userText, context, state) the output must be deterministic.
 *
 * <p>For multi-draft generation the engine will vary (state.seed, state.draftIndex, tags/signals).
 * Generator must treat them as conditioning inputs.</p>
 *
 * <h3>Thinking vs Generation contract</h3>
 * Generator must NOT do retrieval/reranking. It must only:
 * - use the provided {@code context} as evidence,
 * - respect engine-provided constraints via {@link ThoughtState#generationHint} and {@code state.tags},
 * - produce diverse drafts deterministically when requested.
 */
public interface TextGenerator {

    /**
     * Generate a draft given userText (or prepared prompt), context, and state.
     * Contract: deterministic for the same inputs.
     *
     * <p>IMPORTANT: Do not perform retrieval here. Use only {@code context}.</p>
     */
    String generate(String userText, List<Statement> context, ThoughtState state);

    /**
     * Prepare a deterministic generation prompt derived from:
     * - {@link ThoughtState#generationHint} (engine-generated, parseable)
     * - {@link ThoughtState#bestSoFar} and {@link ThoughtState#lastCritique} (self-correction anchors)
     * - {@code state.tags} (phase/explore-exploit, section plan, evidence strictness, anti-generic)
     *
     * <p>BeamSearch/TensorFlow backends can treat it as a stable "system" prefix.</p>
     *
     * <p>Contract: for same (userText, context, state.generationHint, state.tags, state.bestSoFar, state.lastCritique)
     * the result must be identical.</p>
     */
    default String prepareUserText(String userText, List<Statement> context, ThoughtState state) {
        final String q = userText == null ? "" : userText.trim();
        final String hintRaw = (state == null || state.generationHint == null) ? "" : state.generationHint.trim();
        final Hint h = Hint.parse(hintRaw);

        final Map<String, String> tags = (state == null || state.tags == null) ? Collections.emptyMap() : state.tags;

        // Engine knobs (Explore/Exploit) live in tags to avoid new classes.
        final String phase = norm(tags.getOrDefault("think.phase", ""));
        final String diversity = norm(tags.getOrDefault("gen.diversity", ""));
        final String sections = norm(tags.getOrDefault("response.sections", ""));
        final String forbidGeneric = norm(tags.getOrDefault("repair.forbid_generic", ""));
        final String requireEvidence = norm(tags.getOrDefault("repair.require_evidence", ""));
        final String citeIds = norm(tags.getOrDefault("repair.cite_ids", ""));
        final String minGround = norm(tags.getOrDefault("eval.grounded.min", ""));
        final String minValid = norm(tags.getOrDefault("eval.minValid", ""));

        // Keep prefix short, stable, parseable. No “creative” phrasing here.
        StringBuilder sb = new StringBuilder(q.length() + 1024);

        sb.append("[contract] ");
        if (!h.format.isBlank()) sb.append("format=").append(h.format).append(' ');
        if (!h.intent.isBlank()) sb.append("intent=").append(h.intent).append(' ');
        if (h.iter >= 0) sb.append("iter=").append(h.iter).append(' ');
        if (!phase.isBlank()) sb.append("phase=").append(phase).append(' ');
        if (!diversity.isBlank()) sb.append("diversity=").append(diversity).append(' ');
        if (!sections.isBlank()) sb.append("sections=").append(sections).append(' ');
        if (!minValid.isBlank()) sb.append("min_valid=").append(minValid).append(' ');
        if (!minGround.isBlank()) sb.append("min_ground=").append(minGround).append(' ');
        if (!forbidGeneric.isBlank()) sb.append("forbid_generic=").append(forbidGeneric).append(' ');
        if (!requireEvidence.isBlank()) sb.append("require_evidence=").append(requireEvidence).append(' ');
        if (!citeIds.isBlank()) sb.append("cite_ids=").append(citeIds).append(' ');

        // Generic quality knobs derived from previous telemetry (hint).
        if (h.lastG >= 0 && h.lastG < 0.35) sb.append("need=more_evidence ");
        if (h.lastR >= 0 && h.lastR > 0.60) sb.append("need=reduce_unbacked_claims ");
        if (h.lastV == 0) sb.append("need=fix_validity ");

        // Self-correction anchor: improve previous best draft if available.
        if (state != null && state.bestSoFar != null && state.bestSoFar.text != null && !state.bestSoFar.text.isBlank()) {
            sb.append("; improve_previous_draft=true\n");
            sb.append("[prev_best]\n");
            sb.append(state.bestSoFar.text.trim()).append("\n");
        } else {
            sb.append("\n");
        }

        // Critique anchor (structured critique is preferred; we keep it as a single line).
        if (state != null && state.lastCritique != null && !state.lastCritique.isBlank()) {
            sb.append("[critique]\n");
            sb.append(state.lastCritique.trim().replace('\n', ' ')).append("\n");
        }

        // Context reminder: we do NOT inline full context here (it is passed separately).
        sb.append("[evidence]\n");
        if (!citeIds.isBlank()) {
            sb.append("Use the specified evidence ids when making factual claims: ").append(citeIds).append(".\n");
        } else {
            sb.append("Use statements from context as evidence for factual claims.\n");
        }

        // Task instruction: stable and strict (engine controls strictness via tags).
        sb.append("[task]\n");
        sb.append("Answer the user grounded in the provided context. ");
        if ("true".equalsIgnoreCase(requireEvidence)) {
            sb.append("Do not introduce facts not supported by context. ");
        } else {
            sb.append("Prefer context for factual claims; if insufficient, say so. ");
        }
        if ("true".equalsIgnoreCase(forbidGeneric)) {
            sb.append("Avoid generic filler. Be specific and actionable. ");
        }
        if (!sections.isBlank()) {
            sb.append("Follow the requested section plan. ");
        }
        sb.append("User: ").append(q);
        return sb.toString();
    }

    /**
     * Optional n-best generation. Default: call generate() N times.
     *
     * <p>BeamSearch/TensorFlow implementation should override this for real n-best.</p>
     *
     * <p>Determinism: must be deterministic for the same inputs; diversity is achieved by conditioning on
     * (state.seed, state.draftIndex) and tags like gen.diversity/think.phase.</p>
     */
    default List<String> generateN(String userText, List<Statement> context, ThoughtState state, int n) {
        int need = Math.max(1, n);
        ArrayList<String> out = new ArrayList<>(need);

        // Protect callers: restore draftIndex after generation to avoid surprising shared-state effects.
        int prevIdx = (state == null) ? 0 : state.draftIndex;
        try {
            for (int i = 0; i < need; i++) {
                if (state != null) state.draftIndex = i;

                // Feed conditioning prefix based on generationHint + tags (still deterministic).
                String prompt = prepareUserText(userText, context, state);
                String s = generate(prompt, context, state);

                if (s != null && !s.isBlank()) out.add(s);
            }
        } finally {
            if (state != null) state.draftIndex = prevIdx;
        }

        if (out.isEmpty()) out.add("");
        return out;
    }

    private static String norm(String s) {
        return s == null ? "" : s.trim();
    }

    /**
     * Minimal parser for {@link ThoughtState#generationHint}.
     * Expected keys: last_g, last_r, last_v, best_g, best_r, best_v, iter, intent, format.
     *
     * <p>Hint format: key=value;key2=value2 ... (engine-generated).</p>
     */
    final class Hint {
        public final String format;
        public final String intent;
        public final int iter;
        public final double lastG;
        public final double lastR;
        public final int lastV;

        private Hint(String format, String intent, int iter, double lastG, double lastR, int lastV) {
            this.format = format == null ? "" : format;
            this.intent = intent == null ? "" : intent;
            this.iter = iter;
            this.lastG = lastG;
            this.lastR = lastR;
            this.lastV = lastV;
        }

        public static Hint parse(String generationHint) {
            if (generationHint == null || generationHint.isBlank()) {
                return new Hint("", "", -1, -1, -1, -1);
            }
            String fmt = "";
            String intent = "";
            int iter = -1;
            double lastG = -1;
            double lastR = -1;
            int lastV = -1;

            String[] parts = generationHint.split(";");
            for (String p : parts) {
                int eq = p.indexOf('=');
                if (eq <= 0) continue;
                String k = p.substring(0, eq).trim();
                String v = p.substring(eq + 1).trim();

                switch (k) {
                    case "format" -> fmt = v;
                    case "intent" -> intent = v;
                    case "iter" -> iter = safeInt(v, -1);
                    case "last_g" -> lastG = safeDouble(v, -1);
                    case "last_r" -> lastR = safeDouble(v, -1);
                    case "last_v" -> lastV = safeInt(v, -1);
                    default -> {
                        // ignore unknown keys (forward-compatible)
                    }
                }
            }
            return new Hint(fmt, intent, iter, lastG, lastR, lastV);
        }

        private static int safeInt(String s, int def) {
            try {
                return Integer.parseInt(s);
            } catch (Exception ignored) {
                return def;
            }
        }

        private static double safeDouble(String s, double def) {
            try {
                return Double.parseDouble(s);
            } catch (Exception ignored) {
                return def;
            }
        }
    }
}