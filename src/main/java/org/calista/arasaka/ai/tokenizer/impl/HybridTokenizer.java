package org.calista.arasaka.ai.tokenizer.impl;

import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.text.Normalizer;
import java.util.*;

/**
 * HybridTokenizer (enterprise-ready):
 * - Unicode normalization (NFKC)
 * - Optional diacritics stripping (NFD + remove combining marks)
 * - Locale.ROOT lowercase (deterministic)
 * - URL/email as atomic tokens (strict heuristics)
 * - Keeps #hashtags / @mentions
 * - Keeps inner connectors: - _ ' ’ . + # when surrounded by token chars (code/versions/domains)
 * - Optional n-grams (bigrams) without extra classes
 * - Fast: single pass, no regex split, minimal allocations
 *
 * Designed for:
 * - retrieval scoring
 * - intent detection
 * - candidate evaluation metrics
 * Not a subword tokenizer for LLM (BPE/WordPiece should be separate).
 */
public final class HybridTokenizer implements Tokenizer {

    public static final int DEFAULT_MIN_TOKEN_LEN = 1;
    public static final int DEFAULT_MAX_TOKEN_LEN = 64;

    private final int minLen;
    private final int maxLen;

    private final boolean keepUrls;
    private final boolean keepEmails;
    private final boolean keepHashtags;
    private final boolean keepMentions;

    private final boolean stripDiacritics;

    /** If true: emits bigrams as additional tokens "bg:<t1>_<t2>" (useful for intent/retrieval). */
    private final boolean emitBigrams;
    private final String bigramPrefix;

    /** If true: deduplicate tokens but preserve order (LinkedHashSet). */
    private final boolean unique;

    public HybridTokenizer() {
        this(new Config());
    }

    public HybridTokenizer(Config cfg) {
        Objects.requireNonNull(cfg, "cfg");

        this.minLen = Math.max(1, cfg.minTokenLength);
        this.maxLen = Math.max(this.minLen, cfg.maxTokenLength);

        this.keepUrls = cfg.keepUrls;
        this.keepEmails = cfg.keepEmails;
        this.keepHashtags = cfg.keepHashtags;
        this.keepMentions = cfg.keepMentions;

        this.stripDiacritics = cfg.stripDiacritics;

        this.emitBigrams = cfg.emitBigrams;
        this.bigramPrefix = (cfg.bigramPrefix == null ? "bg:" : cfg.bigramPrefix);

        this.unique = cfg.unique;
    }

    @Override
    public List<String> tokenize(String text) {
        if (text == null) return List.of();
        String raw = text.trim();
        if (raw.isEmpty()) return List.of();

        // 1) Normalize to NFKC + lowercase (stable).
        String s = Normalizer.normalize(raw, Normalizer.Form.NFKC).toLowerCase(Locale.ROOT);

        // 2) Optional: strip diacritics by NFD and removing combining marks.
        if (stripDiacritics) {
            s = Normalizer.normalize(s, Normalizer.Form.NFD);
        }

        ArrayList<String> out = new ArrayList<>(Math.min(64, s.length() / 4));
        StringBuilder tok = new StringBuilder(32);

        final int n = s.length();
        int i = 0;

        while (i < n) {
            char c = s.charAt(i);

            if (stripDiacritics && isCombiningMark(c)) {
                i++;
                continue;
            }

            // URL atomic token
            if (keepUrls && looksLikeUrlStart(s, i)) {
                int j = consumeUrl(s, i);
                addToken(out, s, i, j);
                i = j;
                continue;
            }

            // Email atomic token
            if (keepEmails && looksLikeEmailStart(s, i)) {
                int j = consumeEmail(s, i);
                if (j > i) {
                    addToken(out, s, i, j);
                    i = j;
                    continue;
                }
            }

            // #hashtag / @mention
            if ((c == '#' && keepHashtags) || (c == '@' && keepMentions)) {
                int j = i + 1;
                if (j < n && isBaseTokenChar(s.charAt(j))) {
                    j = consumeTokenBody(s, j);
                    addToken(out, s, i, j);
                    i = j;
                    continue;
                }
                // otherwise treat symbol as separator
            }

            // Regular token
            if (isBaseTokenChar(c)) {
                tok.setLength(0);
                tok.append(c);
                i++;

                while (i < n) {
                    char x = s.charAt(i);

                    if (stripDiacritics && isCombiningMark(x)) {
                        i++;
                        continue;
                    }

                    if (isBaseTokenChar(x)) {
                        tok.append(x);
                        i++;
                        continue;
                    }

                    // allow connectors if surrounded by base token chars
                    if (isInnerConnector(x) && i + 1 < n) {
                        char prev = tok.charAt(tok.length() - 1);
                        char next = s.charAt(i + 1);

                        if (isBaseTokenChar(prev) && isBaseTokenChar(next)) {
                            tok.append(x);
                            i++;
                            continue;
                        }
                    }

                    break;
                }

                addToken(out, tok);
                continue;
            }

            i++;
        }

        if (out.isEmpty()) return out;

        // Optional uniqueness (stable)
        if (unique) {
            LinkedHashSet<String> set = new LinkedHashSet<>(out);
            out = new ArrayList<>(set);
        }

        // Optional bigrams
        if (emitBigrams && out.size() >= 2) {
            int base = out.size();
            for (int k = 0; k + 1 < base; k++) {
                String a = out.get(k);
                String b = out.get(k + 1);
                if (a.isEmpty() || b.isEmpty()) continue;
                out.add(bigramPrefix + a + "_" + b);
            }
            if (unique) {
                LinkedHashSet<String> set = new LinkedHashSet<>(out);
                out = new ArrayList<>(set);
            }
        }

        return out;
    }

    // ---------------- Config ----------------

    public static final class Config {
        public int minTokenLength = DEFAULT_MIN_TOKEN_LEN;
        public int maxTokenLength = DEFAULT_MAX_TOKEN_LEN;

        public boolean keepUrls = true;
        public boolean keepEmails = true;
        public boolean keepHashtags = true;
        public boolean keepMentions = true;

        public boolean stripDiacritics = false;

        public boolean emitBigrams = false;
        public String bigramPrefix = "bg:";

        public boolean unique = false;
    }

    // ---------------- token rules ----------------

    private static boolean isBaseTokenChar(char c) {
        return Character.isLetterOrDigit(c);
    }

    /**
     * Inner connectors useful for code/tech text:
     * foo-bar, snake_case, o’neill, node.js, c++, c#, v2.1.0
     */
    private static boolean isInnerConnector(char c) {
        return c == '-' || c == '_' || c == '\'' || c == '’' || c == '.' || c == '+' || c == '#';
    }

    private static boolean isCombiningMark(char c) {
        int t = Character.getType(c);
        return t == Character.NON_SPACING_MARK
                || t == Character.COMBINING_SPACING_MARK
                || t == Character.ENCLOSING_MARK;
    }

    // ---------------- URL / Email parsing ----------------

    private static boolean looksLikeUrlStart(String s, int i) {
        int n = s.length();
        if (i >= n) return false;

        // http:// or https://
        if (i + 7 < n && s.regionMatches(true, i, "http://", 0, 7)) return true;
        if (i + 8 < n && s.regionMatches(true, i, "https://", 0, 8)) return true;

        // www.
        return i + 4 < n && s.regionMatches(true, i, "www.", 0, 4);
    }

    private static int consumeUrl(String s, int i) {
        // URL ends on whitespace/control; strip trailing punctuation
        int end = consumeUntilWhitespaceOrControl(s, i);
        while (end > i) {
            char last = s.charAt(end - 1);
            if (last == '.' || last == ',' || last == ';' || last == ':' || last == '!' || last == '?' || last == ')' || last == ']' || last == '}') {
                end--;
            } else {
                break;
            }
        }
        return end;
    }

    private static boolean looksLikeEmailStart(String s, int i) {
        // must start with letter/digit; then contain '@' before whitespace/control
        if (i >= s.length()) return false;
        char c = s.charAt(i);
        if (!isBaseTokenChar(c)) return false;

        int n = s.length();
        int j = i;
        while (j < n) {
            char x = s.charAt(j);
            if (Character.isWhitespace(x) || Character.isISOControl(x)) break;
            if (x == '@') return j > i && j + 1 < n;
            j++;
        }
        return false;
    }

    private static int consumeEmail(String s, int i) {
        int n = s.length();
        int j = i;

        // local part
        int localStart = j;
        while (j < n) {
            char c = s.charAt(j);
            if (Character.isWhitespace(c) || Character.isISOControl(c)) return -1;
            if (c == '@') break;

            // allow local part chars
            if (isBaseTokenChar(c) || c == '.' || c == '_' || c == '+' || c == '-' ) {
                j++;
                continue;
            }
            return -1;
        }
        if (j <= localStart || j >= n || s.charAt(j) != '@') return -1;
        j++; // skip '@'

        // domain part: must have at least one dot, and domain labels are alnum/hyphen
        int domainStart = j;
        boolean hasDot = false;

        while (j < n) {
            char c = s.charAt(j);
            if (Character.isWhitespace(c) || Character.isISOControl(c)) break;

            if (c == '.') {
                hasDot = true;
                j++;
                continue;
            }
            if (Character.isLetterOrDigit(c) || c == '-') {
                j++;
                continue;
            }
            break;
        }

        if (j <= domainStart || !hasDot) return -1;

        int end = j;

        // trim trailing punctuation
        while (end > i) {
            char last = s.charAt(end - 1);
            if (last == '.' || last == ',' || last == ';' || last == ':' || last == '!' || last == '?' || last == ')' || last == ']' || last == '}') {
                end--;
            } else {
                break;
            }
        }
        return end;
    }

    private static int consumeUntilWhitespaceOrControl(String s, int i) {
        int n = s.length();
        int j = i;
        while (j < n) {
            char c = s.charAt(j);
            if (Character.isWhitespace(c) || Character.isISOControl(c)) break;
            j++;
        }
        return j;
    }

    private static int consumeTokenBody(String s, int i) {
        int n = s.length();
        int j = i;
        while (j < n && isBaseTokenChar(s.charAt(j))) j++;
        return j;
    }

    // ---------------- add token ----------------

    private void addToken(List<String> out, CharSequence token) {
        if (token == null) return;
        int len = token.length();
        if (len < minLen) return;

        // filter ultra-noisy tokens (mostly punctuation/mostly digits)
        if (!passesQualityGate(token)) return;

        if (len > maxLen) out.add(token.subSequence(0, maxLen).toString());
        else out.add(token.toString());
    }

    private void addToken(List<String> out, String s, int from, int to) {
        if (from >= to) return;
        int len = to - from;
        if (len < minLen) return;

        if (len > maxLen) {
            CharSequence sub = s.subSequence(from, from + maxLen);
            if (!passesQualityGate(sub)) return;
            out.add(sub.toString());
        } else {
            CharSequence sub = s.subSequence(from, to);
            if (!passesQualityGate(sub)) return;
            out.add(sub.toString());
        }
    }

    private static boolean passesQualityGate(CharSequence tok) {
        int n = tok.length();
        if (n == 0) return false;

        int letters = 0, digits = 0, other = 0;
        for (int i = 0; i < n; i++) {
            char c = tok.charAt(i);
            if (Character.isLetter(c)) letters++;
            else if (Character.isDigit(c)) digits++;
            else other++;
        }

        // allow: emails/urls/hashtags/mentions may include more symbols -> keep if has letters
        if (letters == 0 && digits == 0) return false;

        // if it's almost all digits and long -> likely junk ids; keep short ones
        if (letters == 0 && digits >= 1) {
            return n <= 12;
        }

        // too much punctuation for a "word token"
        double otherRatio = other / (double) n;
        return otherRatio <= 0.45;
    }
}