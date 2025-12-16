package org.calista.arasaka.ai.knowledge;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public final class Statement {
    public String id;
    public String text;
    public double weight = 1.0;
    public List<String> tags = List.of();

    public void validate() {
        if (id == null || id.isBlank()) throw new IllegalArgumentException("Statement.id is required");
        if (text == null || text.isBlank()) throw new IllegalArgumentException("Statement.text is required");
    }
}