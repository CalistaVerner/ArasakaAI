package org.calista.arasaka.ai.text;

import java.util.List;

public interface Tokenizer {
    List<String> tokenize(String text);
}