package org.calista.arasaka.ai.think.candidate.impl;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.retrieve.scorer.impl.TokenOverlapScorer;
import org.calista.arasaka.ai.think.ThoughtState;
import org.calista.arasaka.ai.think.candidate.CandidateEvaluator;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Advanced deterministic evaluator (enterprise-grade):
 * - schema-aware structure contract (driven by state.tags, no hard-coded domains)
 * - groundedness aggregation over context statements
 * - query coverage (does it address user input)
 * - novelty / unsupported-assertion risk (tokens not backed by context)
 * - repetition / low-information penalties
 * - length/verbosity guardrails
 *
 * No randomness. No domain phrases. Only general heuristics and tunable thresholds via tags.
 */
public final class BaseCandidateEvaluator implements CandidateEvaluator {

    // --- defaults (can be overridden by state.tags) ---
    private final double minGroundedness;
    private final double maxContradictionRisk;
    private final double minQueryCoverage;
    private final double maxNovelty;
    private final double maxRepetition;
    private final int minChars;
    private final int maxCharsSoft;
    private final int maxCharsHard;

    private final Tokenizer tokenizer;
    private final TokenOverlapScorer overlapScorer;

    // Numbered sections: "1) ..." at line start (1..99)
    private static final Pattern NUMBERED_SECTION = Pattern.compile("(?m)^\\s*\\d{1,2}\\)\\s+");
    // Markdown headers: ## / ###
    private static final Pattern MD_HEADER = Pattern.compile("(?m)^\\s*#{2,3}\\s+.+$");
    private static final Pattern BULLETISH = Pattern.compile("(?m)^\\s*([-*•]|\\d+\\.|\\d+\\))\\s+");

    public BaseCandidateEvaluator(Tokenizer tokenizer) {
        this(
                tokenizer,
                new TokenOverlapScorer(Objects.requireNonNull(tokenizer, "tokenizer")),
                0.30, 0.70,
                0.20,
                0.70,
                0.35,
                40,
                1600,
                3500
        );
    }

    public BaseCandidateEvaluator(
            Tokenizer tokenizer,
            TokenOverlapScorer overlapScorer,
            double minGroundedness,
            double maxContradictionRisk,
            double minQueryCoverage,
            double maxNovelty,
            double maxRepetition,
            int minChars,
            int maxCharsSoft,
            int maxCharsHard
    ) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.overlapScorer = Objects.requireNonNull(overlapScorer, "overlapScorer");

        this.minGroundedness = clamp01(minGroundedness);
        this.maxContradictionRisk = clamp01(maxContradictionRisk);
        this.minQueryCoverage = clamp01(minQueryCoverage);
        this.maxNovelty = clamp01(maxNovelty);
        this.maxRepetition = clamp01(maxRepetition);

        this.minChars = Math.max(0, minChars);
        this.maxCharsSoft = Math.max(this.minChars + 1, maxCharsSoft);
        this.maxCharsHard = Math.max(this.maxCharsSoft + 1, maxCharsHard);
    }

    @Override
    public Evaluation evaluate(String userText, List<Statement> context, ThoughtState state, String draft) {
        final long t0 = System.nanoTime();

        final String q = (userText == null) ? "" : userText.trim();
        final String a = (draft == null) ? "" : draft.trim();
        final List<Statement> ctx = (context == null) ? List.of() : context;

        final int tokenCount = safeTokenCount(a);

        if (a.isEmpty()) return invalid(-1.0, "err=empty", tokenCount, System.nanoTime() - t0);
        if (a.length() > maxCharsHard) return invalid(-0.6, "err=too_long_hard;len=" + a.length(), tokenCount, System.nanoTime() - t0);

        // ---- policy overrides from tags (enterprise control-plane) ----
        Policy p = policyFromState(state);

        // ---- 1) Structure ----
        Structure s = structureOf(a);
        double structureScore = structureScoreOf(s);

        // ---- no-context mode ----
        if (ctx.isEmpty()) {
            double qc = queryCoverage(q, a);
            Repetition rep = repetition(a);
            double echo = echoPenalty(q, a);

            // no-context: structure is soft, but we still keep it as a quality hint
            double structureSoft = clamp01(s.structurePenalty) * 0.25;

            double verbosityPenalty = 0.0;
            if (a.length() > maxCharsSoft) {
                double over = (double) (a.length() - maxCharsSoft) / (double) Math.max(1, (maxCharsHard - maxCharsSoft));
                verbosityPenalty = clamp01(over) * 0.35;
            }

            double stylePenalty = structureSoft + verbosityPenalty + rep.repetitionPenalty + echo;

            double contradictionRisk = clamp01(
                    0.10
                            + 0.35 * echo
                            + 0.25 * rep.repetition
                            + 0.10 * (s.hasAllSections ? 0.0 : 1.0)
            );

            boolean valid =
                    qc >= (p.minQueryCoverage * 0.50)
                            && echo <= 0.60
                            && rep.repetition <= p.maxRepetition
                            && a.length() <= maxCharsHard;

            double score =
                    (0.75 * qc)
                            - (0.95 * contradictionRisk)
                            - (0.90 * stylePenalty)
                            + (0.15 * structureScore)
                            + (0.10 * s.actionability);

            String telemetry = "noctx"
                    + ";qc=" + fmt2(qc)
                    + ";echo=" + fmt2(echo)
                    + ";rep=" + fmt2(rep.repetition)
                    + ";sec=" + s.sectionCount
                    + ";md=" + s.mdHeaders
                    + ";st=" + fmt2(structureScore)
                    + ";v=" + (valid ? 1 : 0);

            Evaluation ev = new Evaluation();
            ev.score = score;
            ev.valid = valid;
            ev.groundedness = 0.0;
            ev.contradictionRisk = contradictionRisk;
            ev.structureScore = structureScore;
            ev.repetition = rep.repetition;
            ev.novelty = 0.0;
            ev.coherence = qc;
            ev.critique = telemetry;
            ev.validationNotes = telemetry;
            ev.syncNotes();
            return ev;
        }

        // ---- context mode ----
        if (a.length() < minChars) return invalid(-0.8, "err=too_short;len=" + a.length(), tokenCount, System.nanoTime() - t0);

        GroundingAgg g = groundednessAgg(a, ctx);
        double groundedness = g.groundedness;

        double qc = queryCoverage(q, a);
        Novelty nov = noveltyVsContext(a, ctx);
        Repetition rep = repetition(a);

        double numericDensity = (double) countDigits(a) / (double) Math.max(1, a.length());
        double punctuationDensity = (double) countPunctuation(a) / (double) Math.max(1, a.length());

        double contradictionRisk = clamp01(
                0.15
                        + 0.40 * clamp01(numericDensity / 0.18)
                        + 0.15 * clamp01(punctuationDensity / 0.22)
                        + 0.45 * clamp01(nov.novelty / Math.max(1e-9, p.maxNovelty)) * 0.5
                        + (s.hasAllSections ? 0.0 : 0.10)
                        - 0.55 * groundedness
        );

        double verbosityPenalty = 0.0;
        if (a.length() > maxCharsSoft) {
            double over = (double) (a.length() - maxCharsSoft) / (double) Math.max(1, (maxCharsHard - maxCharsSoft));
            verbosityPenalty = clamp01(over) * 0.35;
        }

        double stylePenalty = s.structurePenalty + verbosityPenalty + rep.repetitionPenalty;

        // schema requirement driven by response.sections
        boolean mustHaveSections = p.requireSections;

        boolean valid =
                (!mustHaveSections || s.hasAllSections)
                        && (!mustHaveSections || structureScore >= 0.35)
                        && groundedness >= p.minGroundedness
                        && contradictionRisk <= p.maxContradictionRisk
                        && qc >= p.minQueryCoverage
                        && nov.novelty <= p.maxNovelty
                        && rep.repetition <= p.maxRepetition
                        && a.length() <= maxCharsHard;

        double score =
                (0.50 * groundedness)
                        + (0.25 * qc)
                        + (0.15 * structureScore)
                        + (0.10 * s.actionability)
                        - (0.55 * contradictionRisk)
                        - (0.55 * stylePenalty)
                        - (0.35 * nov.novelty)
                        - (0.35 * rep.repetition);

        if (mustHaveSections && !s.hasAllSections) score -= 0.35;

        String telemetry = "qc=" + fmt2(qc)
                + ";g=" + fmt2(groundedness)
                + ";nov=" + fmt2(nov.novelty)
                + ";rep=" + fmt2(rep.repetition)
                + ";sec=" + s.sectionCount
                + ";md=" + s.mdHeaders
                + ";st=" + fmt2(structureScore)
                + ";risk=" + fmt2(contradictionRisk)
                + ";v=" + (valid ? 1 : 0);

        Evaluation ev = new Evaluation();
        ev.score = score;
        ev.valid = valid;
        ev.groundedness = groundedness;
        ev.contradictionRisk = contradictionRisk;
        ev.structureScore = structureScore;
        ev.repetition = rep.repetition;
        ev.novelty = nov.novelty;
        ev.coherence = clamp01(0.60 * groundedness + 0.40 * qc);
        ev.critique = telemetry;
        ev.validationNotes = telemetry;
        ev.syncNotes();
        return ev;
    }

    // -------------------- policy --------------------

    private record Policy(
            double minGroundedness,
            double maxContradictionRisk,
            double minQueryCoverage,
            double maxNovelty,
            double maxRepetition,
            boolean requireSections
    ) {}

    private Policy policyFromState(ThoughtState state) {
        double gMin = minGroundedness;
        double riskMax = maxContradictionRisk;
        double qcMin = minQueryCoverage;
        double novMax = maxNovelty;
        double repMax = maxRepetition;

        boolean requireSections = true;

        Map<String, String> tags = (state == null ? null : state.tags);
        if (tags != null && !tags.isEmpty()) {
            gMin = clamp01(readDouble(tags, "eval.grounded.min", gMin));
            riskMax = clamp01(readDouble(tags, "eval.risk.max", riskMax));
            qcMin = clamp01(readDouble(tags, "eval.qc.min", qcMin));
            novMax = clamp01(readDouble(tags, "eval.novelty.max", novMax));
            repMax = clamp01(readDouble(tags, "eval.rep.max", repMax));

            String sections = readString(tags, "response.sections", "");
            if (!sections.isBlank()) {
                String s = sections.toLowerCase(Locale.ROOT);
                if (s.equals("summary") || (!s.contains("evidence") && !s.contains("actions"))) {
                    requireSections = false;
                }
            }
        }

        return new Policy(gMin, riskMax, qcMin, novMax, repMax, requireSections);
    }

    private static double readDouble(Map<String, String> tags, String key, double def) {
        String v = tags.get(key);
        if (v == null || v.isBlank()) return def;
        try { return Double.parseDouble(v.trim()); }
        catch (Exception ignore) { return def; }
    }

    private static String readString(Map<String, String> tags, String key, String def) {
        String v = tags.get(key);
        return (v == null) ? def : v;
    }

    // -------------------- structure --------------------

    private record Structure(
            boolean hasAllSections,
            int numberedSections,
            int mdHeaders,
            double structurePenalty,
            double actionability,
            int sectionCount
    ) {}

    private static Structure structureOf(String answer) {
        int numbered = countMatches(NUMBERED_SECTION, answer);
        int md = countMatches(MD_HEADER, answer);
        int bullets = countMatches(BULLETISH, answer);

        boolean hasAll = (numbered >= 3) || (md >= 3);

        double penalty = hasAll ? 0.0 : 0.70;
        // slight penalty if barely meets threshold (keeps it honest)
        if (hasAll) {
            if (numbered > 0 && numbered < 3) penalty += 0.20;
            if (md > 0 && md < 3) penalty += 0.20;
        }

        double actionability = clamp01(bullets / 8.0);
        int sectionCount = (numbered >= 3) ? numbered : md;

        return new Structure(hasAll, numbered, md, clamp01(penalty), actionability, sectionCount);
    }

    private static int countMatches(Pattern p, String s) {
        if (s == null || s.isBlank()) return 0;
        int c = 0;
        var m = p.matcher(s);
        while (m.find()) c++;
        return c;
    }

    private static double structureScoreOf(Structure s) {
        if (s == null) return 0.0;
        double base = 1.0 - clamp01(s.structurePenalty);
        double score = (0.80 * base) + (0.20 * clamp01(s.actionability));
        if (s.hasAllSections) score += 0.05;
        return clamp01(score);
    }

    // -------------------- groundedness --------------------

    private record GroundingAgg(double groundedness, int bestIdx, double bestScore, int topK, double topKMean) {}

    private GroundingAgg groundednessAgg(String answer, List<Statement> ctx) {
        if (ctx == null || ctx.isEmpty()) return new GroundingAgg(0.0, -1, 0.0, 0, 0.0);

        double best = 0.0;
        int bestIdx = -1;

        final int K = Math.min(4, ctx.size());
        double[] top = new double[K];

        for (int i = 0; i < ctx.size(); i++) {
            Statement st = ctx.get(i);
            if (st == null || st.text == null || st.text.isBlank()) continue;

            double s = overlapScorer.score(answer, st);
            if (!Double.isFinite(s)) s = 0.0;
            s = clamp01(s);

            if (s > best) { best = s; bestIdx = i; }

            for (int j = 0; j < K; j++) {
                if (s > top[j]) {
                    for (int k = K - 1; k > j; k--) top[k] = top[k - 1];
                    top[j] = s;
                    break;
                }
            }
        }

        double sum = 0.0;
        int cnt = 0;
        for (double v : top) {
            if (v > 0) { sum += v; cnt++; }
        }

        double mean = (cnt == 0) ? 0.0 : (sum / cnt);
        double agg = clamp01(0.70 * best + 0.30 * mean);

        return new GroundingAgg(agg, bestIdx, best, cnt, mean);
    }

    // -------------------- novelty --------------------

    private record Novelty(double novelty, double contextSupport, int unsupportedTokens, int totalTokens) {}

    private Novelty noveltyVsContext(String answer, List<Statement> ctx) {
        Set<String> ctxTok = new HashSet<>(2048);
        for (Statement s : ctx) {
            if (s == null || s.text == null || s.text.isBlank()) continue;
            ctxTok.addAll(tokenSet(s.text));
            if (ctxTok.size() > 20_000) break;
        }

        List<String> aTok = tokenize(answer);
        if (aTok.isEmpty()) return new Novelty(0.0, 1.0, 0, 0);

        int unsupported = 0;
        for (String t : aTok) {
            if (t == null) continue;
            if (!ctxTok.contains(t)) unsupported++;
        }

        double novelty = clamp01((double) unsupported / (double) Math.max(1, aTok.size()));
        return new Novelty(novelty, 1.0 - novelty, unsupported, aTok.size());
    }

    // -------------------- query coverage --------------------

    private double queryCoverage(String query, String answer) {
        if (query == null || query.isBlank()) return 0.0;
        if (answer == null || answer.isBlank()) return 0.0;

        List<String> qTok = tokenize(query);
        List<String> aTok = tokenize(answer);
        if (qTok.isEmpty() || aTok.isEmpty()) return 0.0;

        Set<String> aSet = new HashSet<>(aTok);
        int inter = 0;
        for (String t : qTok) if (aSet.contains(t)) inter++;

        return clamp01((double) inter / (double) Math.max(1, qTok.size()));
    }

    // -------------------- repetition --------------------

    private record Repetition(double repetition, double repetitionPenalty) {}

    private Repetition repetition(String answer) {
        List<String> tok = tokenize(answer);
        if (tok.isEmpty()) return new Repetition(0.0, 0.0);

        Map<String, Integer> c = new HashMap<>(tok.size() * 2);
        for (String t : tok) c.merge(t, 1, Integer::sum);

        int max = 1;
        for (int v : c.values()) if (v > max) max = v;

        double rep = clamp01((double) max / (double) Math.max(1, tok.size() / 6));
        double pen = clamp01(rep / Math.max(1e-9, maxRepetition)) * 0.35;

        return new Repetition(rep, pen);
    }

    private double echoPenalty(String q, String a) {
        if (q == null || q.isBlank() || a == null || a.isBlank()) return 0.0;

        List<String> qt = tokenize(q);
        List<String> at = tokenize(a);
        if (qt.isEmpty() || at.isEmpty()) return 0.0;

        Set<String> qs = new HashSet<>(qt);
        Set<String> as = new HashSet<>(at);

        int inter = 0;
        for (String t : qs) if (as.contains(t)) inter++;

        double jacc = (double) inter / (double) Math.max(1, (qs.size() + as.size() - inter));
        return clamp01(jacc);
    }

    // -------------------- token utils --------------------

    private List<String> tokenize(String s) {
        if (s == null || s.isBlank()) return List.of();
        List<String> out = tokenizer.tokenize(s);
        if (out == null || out.isEmpty()) return List.of();

        if (out.size() > 512) {
            // IMPORTANT: subList view is easy to misuse; return a stable copy
            return new ArrayList<>(out.subList(0, 512));
        }
        return out;
    }

    private Set<String> tokenSet(String s) {
        List<String> t = tokenize(s);
        if (t.isEmpty()) return Set.of();
        return new HashSet<>(t);
    }

    private int safeTokenCount(String s) {
        if (s == null || s.isBlank()) return 0;
        List<String> t = tokenizer.tokenize(s);
        return (t == null) ? 0 : t.size();
    }

    private static int countDigits(String s) {
        int c = 0;
        for (int i = 0; i < s.length(); i++) if (Character.isDigit(s.charAt(i))) c++;
        return c;
    }

    private static int countPunctuation(String s) {
        int c = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (!Character.isLetterOrDigit(ch) && !Character.isWhitespace(ch)) c++;
        }
        return c;
    }

    // -------------------- invalid helper --------------------

    private Evaluation invalid(double score, String note, int tokens, long nanos) {
        Evaluation ev = new Evaluation();
        ev.score = score;
        ev.valid = false;
        ev.groundedness = 0.0;
        ev.contradictionRisk = 1.0;
        ev.structureScore = 0.0;

        ev.novelty = 1.0;
        ev.repetition = 1.0;
        ev.coherence = 0.0;

        ev.critique = (note == null) ? "" : note;
        ev.validationNotes = ev.critique;

        ev.syncNotes();
        return ev;
    }

    // -------------------- misc math --------------------

    private static double clamp01(double v) {
        if (!Double.isFinite(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    private static String fmt2(double v) {
        if (!Double.isFinite(v)) return "0.00";
        double x = Math.round(v * 100.0) / 100.0;
        String s = Double.toString(x);
        int dot = s.indexOf('.');
        if (dot < 0) return s + ".00";
        int dec = s.length() - dot - 1;
        if (dec == 0) return s + "00";
        if (dec == 1) return s + "0";
        if (dec > 2) return s.substring(0, dot + 3);
        return s;
    }
}