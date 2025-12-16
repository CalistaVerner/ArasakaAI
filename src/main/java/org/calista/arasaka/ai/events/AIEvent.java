package org.calista.arasaka.ai.events;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public final class AIEvent {
    public String type;        // "USER", "ASSISTANT", "LEARN_UPSERT", "SNAPSHOT"
    public long tsEpochMs;
    public String sessionId;
    public String text;        // payload (user msg / assistant msg / etc.)

    public static AIEvent of(String type, String sessionId, String text, long tsEpochMs) {
        AIEvent e = new AIEvent();
        e.type = type;
        e.sessionId = sessionId;
        e.text = text;
        e.tsEpochMs = tsEpochMs;
        return e;
    }
}