package org.calista.arasaka.ai.knowledge;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.calista.arasaka.io.FileIO;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public final class KnowledgeSnapshotStore {
    private final FileIO io;
    private final ObjectMapper mapper;
    private final Path snapshotFile;

    public KnowledgeSnapshotStore(FileIO io, ObjectMapper mapper, Path snapshotFile) {
        this.io = io;
        this.mapper = mapper;
        this.snapshotFile = snapshotFile;
    }

    public void save(KnowledgeBase kb) throws IOException {
        List<Statement> st = kb.snapshotSorted();

        // атомарно через WriterHandle из твоего FileIO
        FileIO.WriterHandle h = io.openWriter(snapshotFile);
        try {
            for (Statement s : st) {
                h.writer.write(mapper.writeValueAsString(s));
                h.writer.newLine();
            }
            io.commitWriter(h);
        } catch (Exception e) {
            io.rollbackWriter(h);
            throw (e instanceof IOException) ? (IOException) e : new IOException(e);
        }
    }
}