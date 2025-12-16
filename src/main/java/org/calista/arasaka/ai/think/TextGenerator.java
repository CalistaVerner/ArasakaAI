package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;

import java.util.ArrayList;
import java.util.List;

/**
 * Pluggable generation backend.
 * <p>
 * Determinism contract:
 * - deterministic for same (userText, context, state)
 * - for multi-draft use (state.seed, state.draftIndex) as conditioning
 */
@FunctionalInterface
public interface TextGenerator {

    String generate(String userText, List<Statement> context, ThoughtState state);

    /**
     * Prepare a deterministic generation prompt derived from {@link ThoughtState#generationHint}.
     * <p>
     * This is intentionally <b>domain-agnostic</b>: it only encodes formatting and quality constraints
     * (grounding / avoiding unsupported claims / using evidence). BeamSearch/TensorFlow backends can
     * treat it as a stable "system" prefix.
     *
     * <p>Contract: for same (userText, context, state.generationHint, state.bestSoFar, state.lastCritique)
     * the result must be identical.</p>
     */
    default String prepareUserText(String userText, List<Statement> context, ThoughtState state) {
        String q = userText == null ? "" : userText.trim();
        String hint = (state == null || state.generationHint == null) ? "" : state.generationHint.trim();

        // Parseable hint: key=value;key2=value2 ... (engine-generated).
        Hint h = Hint.parse(hint);

        // Keep the prefix short and deterministic (no "creative" phrasing).
        StringBuilder sb = new StringBuilder(q.length() + 512);
        sb.append("[contract] ");
        if (!h.format.isBlank()) sb.append("format=").append(h.format).append(' ');
        if (!h.intent.isBlank()) sb.append("intent=").append(h.intent).append(' ');
        if (h.iter >= 0) sb.append("iter=").append(h.iter).append(' ');

        // Generic quality knobs derived from previous telemetry.
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

        if (state != null && state.lastCritique != null && !state.lastCritique.isBlank()) {
            sb.append("[critique]\n");
            sb.append(state.lastCritique.trim().replace('\n', ' ')).append("\n");
        }

        // Evidence is passed separately as context, but we add a stable "use evidence" reminder.
        sb.append("[task]\n");
        sb.append("Answer the user using only supported statements from context when making factual claims. ");
        sb.append("If context is insufficient, say so. ");
        sb.append("User: ").append(q);
        return sb.toString();
    }

    /**
     * Optional n-best generation. Default: call generate() N times.
     * BeamSearch/TensorFlow implementation should override this for real n-best.
     */
    default List<String> generateN(String userText, List<Statement> context, ThoughtState state, int n) {
        int need = Math.max(1, n);
        ArrayList<String> out = new ArrayList<>(need);
        for (int i = 0; i < need; i++) {
            state.draftIndex = i;
            // Feed a conditioning prefix based on generationHint (still deterministic).
            String prompt = prepareUserText(userText, context, state);
            String s = generate(prompt, context, state);
            if (s != null && !s.isBlank()) out.add(s);
        }
        if (out.isEmpty()) out.add("");
        return out;
    }

    /**
     * Minimal parser for {@link ThoughtState#generationHint}.
     * Expected keys: last_g, last_r, last_v, best_g, best_r, best_v, iter, intent, format.
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