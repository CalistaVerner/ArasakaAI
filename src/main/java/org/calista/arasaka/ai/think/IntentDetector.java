package org.calista.arasaka.ai.think;

public interface IntentDetector {
    Intent detect(String userText);
}