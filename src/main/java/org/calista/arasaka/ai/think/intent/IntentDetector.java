package org.calista.arasaka.ai.think.intent;

public interface IntentDetector {
    Intent detect(String userText);
}