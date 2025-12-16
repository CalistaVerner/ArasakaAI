package org.calista.arasaka.ai.tokenizer.impl;

import org.calista.arasaka.ai.tokenizer.Tokenizer;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Advanced tokenizer:
 * - Unicode normalization (NFKC)
 * - Lowercasing with Locale.ROOT
 * - Keeps URL/email as single tokens
 * - Keeps #hashtags and @mentions
 * - Allows inner '-' '_' '\'' '’' when surrounded by letters/digits
 * - Fast: single pass, no regex split
 */
public final class AdvancedTokenizer implements Tokenizer {

    public static final int DEFAULT_MIN_TOKEN_LEN = 1;
    public static final int DEFAULT_MAX_TOKEN_LEN = 64;

    private final int minLen;
    private final int maxLen;
    private final boolean keepUrls;
    private final boolean keepEmails;
    private final boolean keepHashtags;
    private final boolean keepMentions;
    private final boolean stripDiacritics;

    public AdvancedTokenizer() {
        this(new Config());
    }

    public AdvancedTokenizer(Config cfg) {
        this.minLen = Math.max(1, cfg.minTokenLength);
        this.maxLen = Math.max(this.minLen, cfg.maxTokenLength);
        this.keepUrls = cfg.keepUrls;
        this.keepEmails = cfg.keepEmails;
        this.keepHashtags = cfg.keepHashtags;
        this.keepMentions = cfg.keepMentions;
        this.stripDiacritics = cfg.stripDiacritics;
    }

    @Override
    public List<String> tokenize(String text) {
        if (text == null || text.isBlank()) return List.of();

        // Unicode normalize + lowercase
        String s = Normalizer.normalize(text, Normalizer.Form.NFKC)
                .toLowerCase(Locale.ROOT);

        if (stripDiacritics) {
            // NFD + remove combining marks
            s = Normalizer.normalize(s, Normalizer.Form.NFD);
        }

        ArrayList<String> out = new ArrayList<>(Math.min(64, s.length() / 4));
        StringBuilder tok = new StringBuilder(32);

        final int n = s.length();
        int i = 0;

        while (i < n) {
            char c = s.charAt(i);

            // Optionally remove combining marks after NFD
            if (stripDiacritics && isCombiningMark(c)) {
                i++;
                continue;
            }

            // URL/email fast-path: consume as a whole token
            if (keepUrls && looksLikeUrlStart(s, i)) {
                int j = consumeUntilWhitespaceOrControl(s, i);
                addToken(out, s.substring(i, j));
                i = j;
                continue;
            }
            if (keepEmails && looksLikeEmailStart(s, i)) {
                int j = consumeEmailLike(s, i);
                addToken(out, s.substring(i, j));
                i = j;
                continue;
            }

            // #hashtag / @mention
            if ((c == '#' && keepHashtags) || (c == '@' && keepMentions)) {
                int j = i + 1;
                if (j < n && isTokenChar(s.charAt(j))) {
                    j = consumeSimpleToken(s, j);
                    addToken(out, s.substring(i, j));
                    i = j;
                    continue;
                }
                // otherwise treat as separator
            }

            // Main tokenization: letters/digits are base token chars
            if (isTokenChar(c)) {
                tok.setLength(0);
                tok.append(c);
                i++;

                while (i < n) {
                    char x = s.charAt(i);

                    if (stripDiacritics && isCombiningMark(x)) {
                        i++;
                        continue;
                    }

                    if (isTokenChar(x)) {
                        tok.append(x);
                        i++;
                        continue;
                    }

                    // allow inner connectors if surrounded by token chars: foo-bar, it's, o’neill, snake_case
                    if (isInnerConnector(x) && i + 1 < n) {
                        char prev = tok.charAt(tok.length() - 1);
                        char next = s.charAt(i + 1);

                        if (isTokenChar(prev) && isTokenChar(next)) {
                            tok.append(x);
                            i++;
                            continue;
                        }
                    }

                    // stop token
                    break;
                }

                addToken(out, tok);
                continue;
            }

            // otherwise separator
            i++;
        }

        return out;
    }

    // ---- helpers ----

    public static final class Config {
        public int minTokenLength = DEFAULT_MIN_TOKEN_LEN;
        public int maxTokenLength = DEFAULT_MAX_TOKEN_LEN;
        public boolean keepUrls = true;
        public boolean keepEmails = true;
        public boolean keepHashtags = true;
        public boolean keepMentions = true;

        /** If true: removes accents/diacritics (useful for mixed Latin text) */
        public boolean stripDiacritics = false;
    }

    private void addToken(List<String> out, CharSequence token) {
        if (token == null) return;
        int len = token.length();
        if (len < minLen) return;

        // clamp very long tokens (protects memory + prevents junk like huge base64 blobs)
        if (len > maxLen) {
            out.add(token.subSequence(0, maxLen).toString());
        } else {
            out.add(token.toString());
        }
    }

    private boolean isTokenChar(char c) {
        return Character.isLetterOrDigit(c);
    }

    private boolean isInnerConnector(char c) {
        return c == '-' || c == '_' || c == '\'' || c == '’';
    }

    private boolean isCombiningMark(char c) {
        int t = Character.getType(c);
        return t == Character.NON_SPACING_MARK
                || t == Character.COMBINING_SPACING_MARK
                || t == Character.ENCLOSING_MARK;
    }

    private boolean looksLikeUrlStart(String s, int i) {
        // http://, https://, www.
        int n = s.length();
        if (i + 4 <= n && s.regionMatches(i, "http", 0, 4)) return true;
        if (i + 4 <= n && s.regionMatches(i, "www.", 0, 4)) return true;
        return false;
    }

    private boolean looksLikeEmailStart(String s, int i) {
        // Heuristic: if we see something like name@domain later in the same "chunk"
        // We only trigger if current char is letter/digit to avoid garbage.
        if (i >= s.length()) return false;
        if (!isTokenChar(s.charAt(i))) return false;

        int n = s.length();
        int j = i;
        int atPos = -1;
        while (j < n) {
            char c = s.charAt(j);
            if (Character.isWhitespace(c) || Character.isISOControl(c)) break;
            if (c == '@') { atPos = j; break; }
            j++;
        }
        return atPos > i && atPos + 1 < n;
    }

    private int consumeUntilWhitespaceOrControl(String s, int i) {
        int n = s.length();
        int j = i;
        while (j < n) {
            char c = s.charAt(j);
            if (Character.isWhitespace(c) || Character.isISOControl(c)) break;
            j++;
        }
        return j;
    }

    private int consumeEmailLike(String s, int i) {
        // read until whitespace/control, then trim trailing punctuation
        int end = consumeUntilWhitespaceOrControl(s, i);
        while (end > i) {
            char last = s.charAt(end - 1);
            if (last == '.' || last == ',' || last == ';' || last == ':' || last == '!' || last == '?' || last == ')' || last == ']') {
                end--;
            } else {
                break;
            }
        }
        return end;
    }

    private int consumeSimpleToken(String s, int i) {
        int n = s.length();
        int j = i;
        while (j < n && isTokenChar(s.charAt(j))) j++;
        return j;
    }
}