package org.calista.arasaka.ai.events;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.calista.arasaka.io.FileIO;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public final class EventStore {
    private final FileIO io;
    private final ObjectMapper mapper;
    private final Path file;

    public EventStore(FileIO io, ObjectMapper mapper, Path file) {
        this.io = io;
        this.mapper = mapper;
        this.file = file;
    }

    public void append(AIEvent e) throws IOException {
        String line = mapper.writeValueAsString(e);
        io.appendJsonl(file, line);
    }

    public List<String> readAllRawLines() throws IOException {
        return io.readJsonl(file);
    }
}