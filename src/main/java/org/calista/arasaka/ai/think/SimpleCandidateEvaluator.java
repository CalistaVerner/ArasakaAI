package org.calista.arasaka.ai.think;

import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.text.Tokenizer;

import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Heuristic-but-useful evaluator that provides:
 *  - overall score
 *  - metrics (coverage, contextSupport, stylePenalty)
 *  - machine-generated critique used as a self-correction signal
 *
 * The engine must not hardcode critique rules; this class is the single place where
 * these heuristics live (and can later be replaced by a learned evaluator).
 */
public final class SimpleCandidateEvaluator implements CandidateEvaluator {
    private final Tokenizer tokenizer;

    public SimpleCandidateEvaluator(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public Evaluation evaluate(String userText, String candidateText, List<Statement> context) {
        String user = userText == null ? "" : userText;
        String cand = candidateText == null ? "" : candidateText;
        List<Statement> ctx = context == null ? List.of() : context;

        List<String> q = tokenizer.tokenize(user);
        List<String> a = tokenizer.tokenize(cand);

        double coverage = tokenCoverage(q, a);
        double contextSupport = contextSupport(a, ctx);
        double stylePenalty = stylePenalty(cand);

        double actionBonus = containsAction(cand) ? 0.10 : 0.0;
        double lengthPenalty = lengthPenalty(cand);

        double score = 0.45 * coverage + 0.45 * contextSupport + actionBonus - stylePenalty - lengthPenalty;
        if (!Double.isFinite(score)) score = 0.0;

        String critique = buildCritique(coverage, contextSupport, stylePenalty, lengthPenalty, actionBonus, ctx);
        return new Evaluation(score, critique, coverage, contextSupport, stylePenalty + lengthPenalty);
    }

    @Override
    public double score(String userText, String candidateText, List<Statement> context) {
        return evaluate(userText, candidateText, context).score;
    }

    private static double tokenCoverage(List<String> q, List<String> a) {
        if (q == null || q.isEmpty() || a == null || a.isEmpty()) return 0.0;
        Set<String> aset = new HashSet<>(a);
        int hit = 0;
        for (String t : q) if (aset.contains(t)) hit++;
        return (double) hit / (double) q.size();
    }

    private double contextSupport(List<String> aTokens, List<Statement> ctx) {
        if (aTokens == null || aTokens.isEmpty() || ctx == null || ctx.isEmpty()) return 0.0;
        Set<String> a = new HashSet<>(aTokens);

        int n = Math.min(6, ctx.size());
        double sum = 0.0;
        double wsum = 0.0;
        for (int i = 0; i < n; i++) {
            Statement st = ctx.get(i);
            if (st == null || st.text == null || st.text.isBlank()) continue;
            List<String> t = tokenizer.tokenize(st.text);
            if (t.isEmpty()) continue;
            int hit = 0;
            for (String tok : t) if (a.contains(tok)) hit++;
            double overlap = (double) hit / Math.sqrt((double) (t.size() * Math.max(1, aTokens.size())));
            double w = Math.max(0.05, st.weight);
            sum += overlap * w;
            wsum += w;
        }
        if (wsum <= 0.0) return 0.0;
        double v = sum / wsum;
        if (!Double.isFinite(v)) return 0.0;
        return Math.max(0.0, Math.min(1.0, v));
    }

    private static double stylePenalty(String cand) {
        if (cand == null || cand.isBlank()) return 0.4;
        String s = cand.toLowerCase(Locale.ROOT);
        double p = 0.0;

        if (s.contains("chain of thought") || s.contains("hidden") || s.contains("внутрен")) p += 0.20;
        if (s.contains("контекст знаний") || s.contains("план ответа") || s.contains("trace:")) p += 0.12;
        if (s.contains("seed") || s.contains("итерац")) p += 0.08;

        if (cand.length() > 220 && !(cand.contains("\n") || cand.contains("•") || cand.contains("1)"))) p += 0.08;

        return Math.max(0.0, Math.min(0.6, p));
    }

    private static double lengthPenalty(String cand) {
        if (cand == null) return 0.25;
        int len = cand.length();
        if (len < 80) return 0.15;
        if (len > 2500) return Math.min(0.30, (len - 2500) / 2500.0);
        return 0.0;
    }

    private static boolean containsAction(String cand) {
        if (cand == null) return false;
        String s = cand.toLowerCase(Locale.ROOT);
        return s.contains("шаг") || s.contains("сделать") || s.contains("реализ") || s.contains("провер") || s.contains("тест");
    }

    private static String buildCritique(
            double coverage,
            double contextSupport,
            double stylePenalty,
            double lengthPenalty,
            double actionBonus,
            List<Statement> ctx
    ) {
        StringBuilder c = new StringBuilder(220);

        if (coverage < 0.35) c.append("точнее ответить на формулировку пользователя; ");
        if (contextSupport < 0.30) {
            if (ctx == null || ctx.isEmpty()) c.append("нужно больше релевантного контекста из базы; ");
            else c.append("сильнее опереться на контекст: сослаться на 3–6 фактов из retrieved; ");
        }
        if (actionBonus <= 0.0) c.append("добавить конкретные шаги/проверки; ");
        if (lengthPenalty > 0.0) {
            if (lengthPenalty >= 0.20) c.append("сильно сократить и убрать воду; ");
            else c.append("немного сократить; ");
        }
        if (stylePenalty > 0.0) c.append("убрать внутренние термины/планирование, отвечать напрямую; ");

        if (c.length() == 0) {
            c.append("улучшить ясность формулировок и сделать шаги более проверяемыми");
        }
        String out = c.toString().trim();
        while (out.endsWith(";")) out = out.substring(0, out.length() - 1).trim();
        return out;
    }
}