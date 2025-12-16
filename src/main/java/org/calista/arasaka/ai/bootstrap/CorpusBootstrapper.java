package org.calista.arasaka.ai.bootstrap;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.Statement;
import org.calista.arasaka.io.FileIO;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public final class CorpusBootstrapper {
    private static final Logger log = LogManager.getLogger(CorpusBootstrapper.class);

    private final FileIO io;
    private final ObjectMapper mapper;

    public CorpusBootstrapper(FileIO io, ObjectMapper mapper) {
        this.io = io;
        this.mapper = mapper;
    }

    public Report loadInto(KnowledgeBase kb, Path jsonlFile, boolean failFast) throws IOException {
        int ok = 0, bad = 0;

        if (!io.exists(jsonlFile)) {
            log.warn("Corpus not found: {}", jsonlFile);
            return new Report(jsonlFile, 0, 0);
        }

        List<String> lines = io.readJsonl(jsonlFile); // trim+skip empty (у тебя уже есть)
        for (String line : lines) {
            try {
                Statement st = mapper.readValue(line, Statement.class);
                st.validate();
                kb.upsert(st);
                ok++;
            } catch (Exception e) {
                bad++;
                log.warn("Bad corpus line in {}: {}", jsonlFile, e.toString());
                if (failFast) throw new IOException("Bad corpus line in " + jsonlFile + ": " + e, e);
            }
        }

        log.info("Corpus loaded: {} (ok={}, bad={})", jsonlFile, ok, bad);
        return new Report(jsonlFile, ok, bad);
    }

    public static final class Report {
        public final Path file;
        public final int ok;
        public final int bad;

        public Report(Path file, int ok, int bad) {
            this.file = file;
            this.ok = ok;
            this.bad = bad;
        }
    }
}