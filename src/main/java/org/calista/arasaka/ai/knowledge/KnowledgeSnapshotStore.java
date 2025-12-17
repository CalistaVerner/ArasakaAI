// KnowledgeSnapshotStore.java
package org.calista.arasaka.ai.knowledge;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.calista.arasaka.io.FileIO;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * KnowledgeSnapshotStore — persist/load KB snapshots.
 *
 * <p>
 * Формат: JSONL (один Statement на строку).
 * Пишем атомарно через FileIO.
 * Читаем через публичный API FileIO.jsonlStream(...).
 * </p>
 *
 * <p>
 * Схема: первой строкой пишем {"_schema":"kb-jsonl-v1"} для миграций.
 * При чтении: строка со схемой игнорируется, ошибки на строках не валят загрузку.
 * </p>
 */
public final class KnowledgeSnapshotStore {

    private static final String SCHEMA_LINE = "{\"_schema\":\"kb-jsonl-v1\"}";

    private final ObjectMapper mapper;
    private final FileIO io;
    private final Path snapshotFile;

    public KnowledgeSnapshotStore(FileIO io,  ObjectMapper mapper, Path snapshotFile) {
        this.io = io;
        this.mapper = mapper;
        this.snapshotFile = snapshotFile;
    }

    public void save(KnowledgeBase kb) throws IOException {
        List<Statement> st = kb.snapshotSorted();

        FileIO.WriterHandle h = io.openWriter(snapshotFile);
        try {
            // schema header (v1)
            h.writer.write(SCHEMA_LINE);
            h.writer.newLine();

            for (Statement s : st) {
                h.writer.write(mapper.writeValueAsString(s));
                h.writer.newLine();
            }
            io.commitWriter(h);
        } catch (Exception e) {
            io.rollbackWriter(h);
            if (e instanceof IOException) throw (IOException) e;
            throw new IOException("Failed to save snapshot: " + snapshotFile, e);
        }
    }

    /**
     * Load snapshot into KB (upsert).
     *
     * @return how many statements were loaded
     */
    public int load(KnowledgeBase kb) throws IOException {
        int loaded = 0;

        try (java.util.stream.Stream<String> lines = io.jsonlStream(snapshotFile)) {
            java.util.Iterator<String> it = lines.iterator();
            while (it.hasNext()) {
                String line = it.next();
                if (line == null || line.isBlank()) continue;
                if (line.contains("\"_schema\"")) continue;

                try {
                    Statement s = mapper.readValue(line, Statement.class);
                    if (s == null) continue;
                    s.validate();
                    kb.upsert(s);
                    loaded++;
                } catch (Exception rowErr) {
                    // enterprise behavior: do not kill entire load because of one broken row
                    // (caller can observe loaded count; detailed logging should be done higher level)
                    System.err.println("KnowledgeSnapshotStore: skip broken row: " + rowErr.getMessage());
                }
            }
        }
        return loaded;
    }
}