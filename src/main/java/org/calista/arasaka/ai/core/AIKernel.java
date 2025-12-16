package org.calista.arasaka.ai.core;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.calista.arasaka.ai.bootstrap.CorpusBootstrapper;
import org.calista.arasaka.ai.events.EventStore;
import org.calista.arasaka.ai.knowledge.InMemoryKnowledgeBase;
import org.calista.arasaka.ai.knowledge.KnowledgeBase;
import org.calista.arasaka.ai.knowledge.KnowledgeSnapshotStore;
import org.calista.arasaka.io.FileIO;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.HostAccess;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

public final class AIKernel {
    private final FileIO io;
    private final ObjectMapper mapper;
    private final AIConfig cfg;

    private final KnowledgeBase kb;
    private final EventStore events;
    private final KnowledgeSnapshotStore snapshots;
    private final Context JS_CONTEXT;

    private AIKernel(FileIO io, ObjectMapper mapper, AIConfig cfg, KnowledgeBase kb, EventStore events, KnowledgeSnapshotStore snapshots) {
        this.io = io;
        this.mapper = mapper;
        this.cfg = cfg;
        this.kb = kb;
        this.events = events;
        this.snapshots = snapshots;
        JS_CONTEXT = Context.newBuilder("js").allowHostAccess(HostAccess.NONE).option("engine.WarnInterpreterOnly", "false").allowIO(false).build();
    }

    public static AIKernel boot(Path configFile) throws IOException {
        ObjectMapper mapper = new ObjectMapper()
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        FileIO external = new FileIO(Path.of("."), StandardCharsets.UTF_8, true);
        AIConfig cfg = AIConfig.load(external, configFile, mapper);

        FileIO io = new FileIO(Path.of(cfg.baseDir), StandardCharsets.UTF_8, true);
        io.ensureBaseDir();

        KnowledgeBase kb = new InMemoryKnowledgeBase();

        Path corporaDir = io.resolveExternal(Path.of(cfg.corpora.dir));
        CorpusBootstrapper bs = new CorpusBootstrapper(io, mapper);
        for (String name : cfg.corpora.bootstrap) {
            bs.loadInto(kb, corporaDir.resolve(name), cfg.corpora.failFast);
        }

        EventStore events = new EventStore(io, mapper, io.resolve(cfg.events.logFile));
        KnowledgeSnapshotStore snapshots = new KnowledgeSnapshotStore(io, mapper, io.resolve(cfg.knowledge.snapshotFile));

        return new AIKernel(io, mapper, cfg, kb, events, snapshots);
    }

    public FileIO io() { return io; }
    public ObjectMapper mapper() { return mapper; }
    public AIConfig config() { return cfg; }
    public KnowledgeBase knowledge() { return kb; }
    public EventStore eventStore() { return events; }
    public KnowledgeSnapshotStore snapshotStore() { return snapshots; }

    public Context getJS_CONTEXT() {
        return JS_CONTEXT;
    }
}