package org.calista.arasaka.ai.learn.impl;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.ai.learn.Learner;
import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.text.BreakIterator;
import java.util.*;
import java.util.regex.Pattern;
import java.util.zip.CRC32;

public final class BasicLearner implements Learner {
    private static final Logger log = LogManager.getLogger(BasicLearner.class);

    private final KnowledgeBase kb;
    private final Tokenizer tokenizer;

    // веса
    private final double newWeight;
    private final double reinforceStep;
    private final double maxWeight;

    // качество
    private final int minChars;
    private final int minTokens;
    private final int maxTokens; // слишком длинные предложения дробим

    // стратегия
    private final int iterations;
    private final int topKPerIteration;
    private final double minScoreToLearn;

    // --- formatting / boilerplate guards (generic, not domain-specific)
    private static final Pattern MARKDOWN_HEADING = Pattern.compile("(?m)^\\s{0,3}#{1,6}\\s+.*$");
    private static final Pattern LIST_LINE = Pattern.compile("(?m)^\\s{0,6}([\\-*•]|\\d+[\\.)])\\s+.*$");
    private static final Pattern CODE_FENCE = Pattern.compile("(?s)```.*?```");
    private static final Pattern HR_LINE = Pattern.compile("(?m)^\\s{0,3}(-{3,}|_{3,}|\\*{3,})\\s*$");

    public BasicLearner(
            KnowledgeBase kb,
            Tokenizer tokenizer,
            double newWeight,
            double reinforceStep
    ) {
        this(kb, tokenizer, newWeight, reinforceStep,
                3.0,     // maxWeight
                12,      // minChars
                4,       // minTokens
                40,      // maxTokens
                3,       // iterations
                48,      // topKPerIteration
                0.35     // minScoreToLearn
        );
    }

    public BasicLearner(
            KnowledgeBase kb,
            Tokenizer tokenizer,
            double newWeight,
            double reinforceStep,
            double maxWeight,
            int minChars,
            int minTokens,
            int maxTokens,
            int iterations,
            int topKPerIteration,
            double minScoreToLearn
    ) {
        this.kb = Objects.requireNonNull(kb, "kb");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");

        if (newWeight <= 0) throw new IllegalArgumentException("newWeight must be > 0");
        if (reinforceStep <= 0) throw new IllegalArgumentException("reinforceStep must be > 0");
        if (maxWeight <= 0) throw new IllegalArgumentException("maxWeight must be > 0");
        if (minChars < 1) throw new IllegalArgumentException("minChars must be >= 1");
        if (minTokens < 1) throw new IllegalArgumentException("minTokens must be >= 1");
        if (maxTokens < minTokens) throw new IllegalArgumentException("maxTokens must be >= minTokens");
        if (iterations < 1) throw new IllegalArgumentException("iterations must be >= 1");
        if (topKPerIteration < 1) throw new IllegalArgumentException("topKPerIteration must be >= 1");
        if (minScoreToLearn < 0 || minScoreToLearn > 1) throw new IllegalArgumentException("minScoreToLearn must be in [0..1]");

        this.newWeight = newWeight;
        this.reinforceStep = reinforceStep;
        this.maxWeight = maxWeight;

        this.minChars = minChars;
        this.minTokens = minTokens;
        this.maxTokens = maxTokens;

        this.iterations = iterations;
        this.topKPerIteration = topKPerIteration;
        this.minScoreToLearn = minScoreToLearn;
    }

    @Override
    public List<Statement> learnFromText(String text, String tag) {
        return learnFromText(text, tag, Map.of());
    }

    @Override
    public List<Statement> learnFromText(String text, String tag, Map<String, String> context) {
        if (text == null || text.isBlank()) return List.of();

        final String safeTag = normalizeTag(tag);
        final Map<String, String> ctx = (context == null) ? Map.of() : context;

        final boolean fromAssistant = isAssistantTag(safeTag) || "assistant".equalsIgnoreCase(ctx.get("source"));

        // 0) Normalize + strip formatting/boilerplate BEFORE learning
        String normalized = normalizeText(text);
        normalized = stripFormatting(normalized);
        normalized = normalizeText(normalized);

        // If assistant output still looks like a structured template, skip entirely.
        if (fromAssistant && looksLikeTemplate(normalized)) {
            if (log.isDebugEnabled()) {
                log.debug("AdvancedLearner: skipped template-like assistant text (len={})", normalized.length());
            }
            return List.of();
        }

        if (normalized.isBlank()) return List.of();

        // token count cache for this learn() call (big perf win)
        final HashMap<String, Integer> tokCache = new HashMap<>(1024);

        // Многоитерационный цикл: каждый проход делает кандидатов чуть “чище” и точнее
        final ArrayList<Statement> learned = new ArrayList<>(64);

        // Дедуп на весь запуск, чтобы не усилять одно и то же 10 раз на разных итерациях
        final HashSet<String> globalSeenSig = new HashSet<>(256);

        // Пул входных сегментов для итераций (в начале весь текст)
        List<String> segments = List.of(normalized);

        for (int it = 1; it <= iterations; it++) {
            final long t0 = System.nanoTime();

            // 1) Extract candidates from segments
            final ArrayList<Candidate> candidates = new ArrayList<>(256);
            for (String seg : segments) {
                extractCandidates(seg, candidates, tokCache, fromAssistant);
            }

            // 2) Score candidates (детерминированно)
            for (Candidate c : candidates) {
                c.score = scoreCandidate(c.text, ctx, tokCache);
            }

            // 3) Filter + TopK
            candidates.sort((a, b) -> {
                int cmp = Double.compare(b.score, a.score);
                if (cmp != 0) return cmp;
                // стабильная сортировка: по sig
                return a.sig.compareTo(b.sig);
            });

            final ArrayList<Candidate> selected = new ArrayList<>(Math.min(topKPerIteration, candidates.size()));
            for (Candidate c : candidates) {
                if (c.score < minScoreToLearn) continue;
                if (selected.size() >= topKPerIteration) break;
                if (!globalSeenSig.add(c.sig)) continue;
                selected.add(c);
            }

            // 4) Upsert + reinforce
            int upserted = 0;
            for (Candidate c : selected) {
                Statement st = upsertLearned(c.text, safeTag, c.sig, c.score);
                if (st != null) {
                    learned.add(st);
                    upserted++;
                }
            }

            // 5) Refine: следующая итерация учится уже на “лучших” кусках (и дробит длинные)
            segments = refineSegmentsFromTop(selected, tokCache);

            final long dtMs = (System.nanoTime() - t0) / 1_000_000L;
            if (log.isDebugEnabled()) {
                log.debug("AdvancedLearner[it={}/{} tag={}]: candidates={}, selected={}, upserted={}, nextSegments={}, dtMs={}",
                        it, iterations, safeTag, candidates.size(), selected.size(), upserted, segments.size(), dtMs);
            }

            // если сегментов больше нет — смысла крутить дальше нет
            if (segments.isEmpty()) break;
        }

        return learned;
    }

    // ------------------------- core pipeline -------------------------

    private void extractCandidates(String text, List<Candidate> out, Map<String, Integer> tokCache, boolean fromAssistant) {
        // 1) sentence split (stable)
        BreakIterator it = BreakIterator.getSentenceInstance(Locale.ROOT);
        it.setText(text);

        int start = it.first();
        for (int end = it.next(); end != BreakIterator.DONE; start = end, end = it.next()) {
            String s = text.substring(start, end).trim();
            s = cleanupSentence(s);
            if (fromAssistant) s = stripInlineFormatting(s);

            if (!passesQualityGateBasic(s, tokCache, fromAssistant)) continue;

            // 2) если слишком длинно — дробим на клаузы (детерминированно)
            List<String> parts = (tokenCount(s, tokCache) > maxTokens) ? splitIntoClauses(s) : List.of(s);
            for (String p : parts) {
                String x = cleanupSentence(p);
                if (fromAssistant) x = stripInlineFormatting(x);

                if (!passesQualityGateBasic(x, tokCache, fromAssistant)) continue;
                String sig = crc32(x);
                out.add(new Candidate(sig, x));
            }
        }
    }

    private double scoreCandidate(String s, Map<String, String> ctx, Map<String, Integer> tokCache) {
        // Скорая “умность” без магии/рандома: чистые эвристики + токены.
        // Результат в [0..1], детерминированный.

        int len = s.length();
        int tokens = tokenCount(s, tokCache);

        // База: “содержательность” по токенам
        double tokenScore = clamp01((tokens - minTokens) / 16.0); // после ~20 токенов почти насыщение

        // Штрафы за мусор
        double noisePenalty = 0.0;
        int letters = 0;
        int digits = 0;
        int weird = 0;
        int punctRun = 0;
        int maxPunctRun = 0;

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isLetter(c)) letters++;
            if (Character.isDigit(c)) digits++;
            if ("@#$%^*_=<>[]{}".indexOf(c) >= 0) weird++;

            if (isPunctLike(c)) {
                punctRun++;
                maxPunctRun = Math.max(maxPunctRun, punctRun);
            } else {
                punctRun = 0;
            }
        }

        if (letters < Math.max(3, len / 8)) noisePenalty += 0.25;
        if (weird > Math.max(2, len / 25)) noisePenalty += 0.25;
        if (maxPunctRun >= 4) noisePenalty += 0.15;

        // Штраф за вопросительность/не-утверждение
        if (looksLikeQuestion(s)) noisePenalty += 0.25;

        // Бонус: контекст-ориентированность (если есть домен/роль — аккуратно)
        double ctxBonus = 0.0;
        String domain = safeLower(ctx.get("domain"));
        if (!domain.isBlank() && s.toLowerCase(Locale.ROOT).contains(domain)) ctxBonus += 0.05;

        // Бонус за “структурность”: наличие связок (очень мягко, без NLP зависимостей)
        double structureBonus = 0.0;
        String low = s.toLowerCase(Locale.ROOT);
        if (low.contains(" is ") || low.contains(" are ") || low.contains(" это ") || low.contains(" является ")) {
            structureBonus += 0.05;
        }

        double score = 0.55 * tokenScore + 0.10 * structureBonus + ctxBonus;

        // Нормируем по длине (слишком короткое/слишком длинное — хуже)
        double lenScore = 1.0;
        if (len < 20) lenScore *= 0.85;
        if (len > 240) lenScore *= 0.85;
        score *= lenScore;

        score -= noisePenalty;

        return clamp01(score);
    }

    private Statement upsertLearned(String sentence, String tag, String sig, double score) {
        // id стабилен
        final String id = "learn:" + tag + ":" + sig;

        Statement st = kb.get(id).orElseGet(() -> {
            Statement n = new Statement();
            n.id = id;
            n.text = sentence;
            n.weight = newWeight;
            n.tags = List.of("learned", tag);
            return n;
        });

        // усиление зависит от score (но детерминированно)
        double before = st.weight;
        double step = reinforceStep * (0.6 + 0.4 * score); // шаг в [0.6..1.0]*reinforceStep
        st.weight = clamp(before + step, 0.0, maxWeight);

        st.tags = mergeTags(st.tags, "learned", tag);

        kb.upsert(st);

        if (log.isTraceEnabled()) {
            log.trace("Learned: id={}, score={}, w:{}->{} text='{}'",
                    st.id, round3(score), round3(before), round3(st.weight), abbreviate(st.text, 160));
        }

        return st;
    }

    private List<String> refineSegmentsFromTop(List<Candidate> selected, Map<String, Integer> tokCache) {
        if (selected.isEmpty()) return List.of();

        // Берем лучшие и делаем “локальный контекст” для следующей итерации:
        // - короткие: как есть
        // - длинные: дробим на клаузы
        ArrayList<String> segs = new ArrayList<>(Math.min(32, selected.size()));
        for (int i = 0; i < selected.size() && segs.size() < 24; i++) {
            String s = selected.get(i).text;
            if (tokenCount(s, tokCache) > maxTokens) {
                for (String p : splitIntoClauses(s)) {
                    String x = cleanupSentence(p);
                    if (passesQualityGateBasic(x, tokCache, false)) segs.add(x);
                    if (segs.size() >= 24) break;
                }
            } else {
                segs.add(s);
            }
        }
        return segs;
    }

    // ------------------------- gating / text utils -------------------------

    private boolean passesQualityGateBasic(String s, Map<String, Integer> tokCache, boolean fromAssistant) {
        if (s == null || s.isBlank()) return false;
        if (s.length() < minChars) return false;

        // For assistant text we avoid learning “section labels” and list fragments (format-driven).
        if (fromAssistant) {
            String t = s.trim();
            if (t.startsWith("#")) return false;
            if (t.startsWith("-") || t.startsWith("•") || t.matches("^\\d+[\\.)].*")) return false;
        }

        int tokens = tokenCount(s, tokCache);
        if (tokens < minTokens) return false;

        // мягкий анти-шум
        int letters = 0;
        int weird = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isLetter(c)) letters++;
            if ("@#$%^*_=<>[]{}".indexOf(c) >= 0) weird++;
        }
        if (letters < Math.max(3, s.length() / 8)) return false;
        if (weird > Math.max(2, s.length() / 20)) return false;

        return true;
    }

    private int tokenCount(String s, Map<String, Integer> cache) {
        if (s == null || s.isBlank()) return 0;
        Integer v = cache.get(s);
        if (v != null) return v;
        int n = tokenizer.tokenize(s).size();
        cache.put(s, n);
        return n;
    }

    private static List<String> splitIntoClauses(String s) {
        // детерминированно, без NLP-зависимостей:
        // делим по ; : , — но не превращаем в “мельницу” (минимальная длина)
        ArrayList<String> out = new ArrayList<>();
        String[] parts = s.split("[;:，、]|\\s+-\\s+|\\s+—\\s+");
        for (String p : parts) {
            String x = cleanupSentence(p);
            if (x.length() >= 16) out.add(x);
        }
        // fallback
        if (out.isEmpty()) return List.of(s);
        return out;
    }

    private static boolean looksLikeQuestion(String s) {
        if (s.indexOf('?') >= 0) return true;
        String low = s.toLowerCase(Locale.ROOT).trim();
        return low.startsWith("why ")
                || low.startsWith("how ")
                || low.startsWith("what ")
                || low.startsWith("кто ")
                || low.startsWith("что ")
                || low.startsWith("почему ")
                || low.startsWith("как ");
    }

    private static boolean isPunctLike(char c) {
        return Character.getType(c) == Character.OTHER_PUNCTUATION
                || Character.getType(c) == Character.DASH_PUNCTUATION
                || ",.!?:;".indexOf(c) >= 0;
    }

    private static String normalizeText(String text) {
        return text
                .replace('\u00A0', ' ')
                .replaceAll("[\\p{Cntrl}&&[^\r\n\t]]", " ")
                .replaceAll("[\\t\\r\\n]+", " ")
                .replaceAll(" +", " ")
                .trim();
    }

    private static String cleanupSentence(String s) {
        if (s == null) return "";
        return s
                .replaceAll(" +", " ")
                .replaceAll("^[\\p{Punct}\\s]+", "")
                .replaceAll("[\\p{Punct}\\s]+$", "")
                .trim();
    }

    private static String normalizeTag(String tag) {
        String t = (tag == null || tag.isBlank()) ? "generic" : tag.trim();
        t = t.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9._-]+", "_");
        if (t.length() > 48) t = t.substring(0, 48);
        return t;
    }

    private static List<String> mergeTags(List<String> existing, String... toAdd) {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        if (existing != null) {
            for (String e : existing) {
                if (e != null && !e.isBlank()) set.add(e.trim());
            }
        }
        for (String a : toAdd) {
            if (a != null && !a.isBlank()) set.add(a.trim());
        }
        return List.copyOf(set);
    }

    private static String safeLower(String s) {
        return (s == null) ? "" : s.toLowerCase(Locale.ROOT).trim();
    }

    private static String crc32(String s) {
        CRC32 c = new CRC32();
        c.update(s.getBytes(StandardCharsets.UTF_8));
        return Long.toHexString(c.getValue());
    }

    private static double clamp(double v, double min, double max) {
        return Math.max(min, Math.min(max, v));
    }

    private static double clamp01(double v) {
        return clamp(v, 0.0, 1.0);
    }

    private static double round3(double v) {
        return Math.rint(v * 1000.0) / 1000.0;
    }

    private static String abbreviate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, Math.max(0, max - 1)) + "…";
    }

    // ------------------------- assistant/template hygiene -------------------------

    private static boolean isAssistantTag(String safeTag) {
        // deterministic + flexible
        return safeTag.contains("assistant") || "bot".equals(safeTag) || safeTag.endsWith(":assistant");
    }

    private static String stripFormatting(String s) {
        if (s == null || s.isBlank()) return "";

        // remove fenced code blocks
        s = CODE_FENCE.matcher(s).replaceAll(" ");

        // remove horizontal rules
        s = HR_LINE.matcher(s).replaceAll(" ");

        // remove markdown headings (whole lines)
        s = MARKDOWN_HEADING.matcher(s).replaceAll(" ");

        // remove list lines (whole lines) - they are usually scaffolding
        s = LIST_LINE.matcher(s).replaceAll(" ");

        // collapse excess whitespace
        s = s.replaceAll("[\\t\\r\\n]+", " ");
        s = s.replaceAll(" +", " ").trim();

        return s;
    }

    private static String stripInlineFormatting(String s) {
        if (s == null || s.isBlank()) return "";
        // inline markdown emphasis/links are noise for memory
        String x = s;
        x = x.replaceAll("\\*\\*(.*?)\\*\\*", "$1");
        x = x.replaceAll("\\*(.*?)\\*", "$1");
        x = x.replaceAll("`([^`]+)`", "$1");
        x = x.replaceAll("\\[([^\\]]+)]\\([^\\)]+\\)", "$1");
        return x.trim();
    }

    private static boolean looksLikeTemplate(String s) {
        if (s == null) return false;
        String t = s.trim();
        if (t.isEmpty()) return false;

        // Generic template signals: many section markers, many bullets, very low sentence diversity.
        int headings = countMatches(t, "##");
        int bullets = countMatches(t, "\n-") + countMatches(t, "\n•") + countMatches(t, "\n1.");
        if (headings >= 2) return true;
        if (bullets >= 4) return true;

        // If text is dominated by very short segments after stripping, it's likely scaffolding.
        String[] parts = t.split("[.!?]+");
        int shortOnes = 0;
        int total = 0;
        for (String p : parts) {
            String x = p.trim();
            if (x.isEmpty()) continue;
            total++;
            if (x.length() < 18) shortOnes++;
        }
        return total >= 3 && shortOnes >= (int) Math.ceil(total * 0.7);
    }

    private static int countMatches(String s, String sub) {
        int c = 0;
        int i = 0;
        while (true) {
            int p = s.indexOf(sub, i);
            if (p < 0) break;
            c++;
            i = p + sub.length();
        }
        return c;
    }

    // ------------------------- internal data -------------------------

    private static final class Candidate {
        final String sig;
        final String text;
        double score;

        Candidate(String sig, String text) {
            this.sig = sig;
            this.text = text;
        }
    }
}