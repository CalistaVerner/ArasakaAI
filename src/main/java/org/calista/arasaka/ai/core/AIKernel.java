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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Objects;

/**
 * AIKernel — instance-owned runtime container.
 *
 * Enterprise lifecycle:
 *   1) build(config)  -> create container + loadOrCreate config + init stores/JS (NO bootstrap)
 *   2) bootstrap()    -> load corpora into KB (explicit, controllable)
 *   3) use            -> runtime
 *   4) close()        -> close resources
 *
 * No statics singletons: lifecycle is explicit.
 */
public final class AIKernel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(AIKernel.class);

    private final FileIO io;
    private final ObjectMapper mapper;
    private final AIConfig cfg;

    private final KnowledgeBase kb;
    private final EventStore events;
    private final KnowledgeSnapshotStore snapshots;

    private final Context jsContext;

    // bootstrap state
    private volatile boolean bootstrapped = false;

    private AIKernel(FileIO io,
                     ObjectMapper mapper,
                     AIConfig cfg,
                     KnowledgeBase kb,
                     EventStore events,
                     KnowledgeSnapshotStore snapshots,
                     Context jsContext) {
        this.io = Objects.requireNonNull(io, "io");
        this.mapper = Objects.requireNonNull(mapper, "mapper");
        this.cfg = Objects.requireNonNull(cfg, "cfg");
        this.kb = Objects.requireNonNull(kb, "kb");
        this.events = Objects.requireNonNull(events, "events");
        this.snapshots = Objects.requireNonNull(snapshots, "snapshots");
        this.jsContext = jsContext; // nullable allowed
    }

    // ---------------------------------------------------------------------
    // Builder
    // ---------------------------------------------------------------------

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {

        private Charset charset = StandardCharsets.UTF_8;

        /**
         * Root directory where config lives.
         * Config is read BEFORE baseDir is known (baseDir is inside config).
         */
        private Path configRoot = Path.of(".");

        private ObjectMapper mapper;
        private KnowledgeBase knowledgeBase;

        // GraalVM policy
        private boolean enableJs = true;
        private boolean allowHostAccess = false;
        private boolean allowIO = false;
        private String warnInterpreterOnly = "false";

        public Builder charset(Charset charset) {
            this.charset = Objects.requireNonNull(charset, "charset");
            return this;
        }

        public Builder configRoot(Path configRoot) {
            this.configRoot = Objects.requireNonNull(configRoot, "configRoot");
            return this;
        }

        public Builder mapper(ObjectMapper mapper) {
            this.mapper = Objects.requireNonNull(mapper, "mapper");
            return this;
        }

        public Builder knowledgeBase(KnowledgeBase kb) {
            this.knowledgeBase = Objects.requireNonNull(kb, "knowledgeBase");
            return this;
        }

        public Builder enableJs(boolean enableJs) {
            this.enableJs = enableJs;
            return this;
        }

        public Builder allowHostAccess(boolean allowHostAccess) {
            this.allowHostAccess = allowHostAccess;
            return this;
        }

        public Builder allowIO(boolean allowIO) {
            this.allowIO = allowIO;
            return this;
        }

        public Builder warnInterpreterOnly(String value) {
            this.warnInterpreterOnly = (value == null ? "false" : value);
            return this;
        }

        /**
         * Creates kernel container: loads/creates config, initializes IO and stores.
         * Does NOT bootstrap corpora.
         */
        public AIKernel build(Path configFile) throws IOException {
            Objects.requireNonNull(configFile, "configFile");

            ObjectMapper om = (this.mapper != null) ? this.mapper : defaultMapper();

            // Config IO (outside baseDir)
            FileIO external = new FileIO(configRoot, charset, true);

            // IMPORTANT: respect configFile argument (no hardcode)
            Path cfgPath = configFile.isAbsolute() ? configFile : configRoot.resolve(configFile);

            AIConfig cfg = AIConfig.loadOrCreate(external, cfgPath, om);

            // Base IO bound to cfg.baseDir (runtime data dir)
            FileIO io = new FileIO(Path.of(cfg.baseDir), charset, true);
            io.ensureBaseDir();

            KnowledgeBase kb = (this.knowledgeBase != null) ? this.knowledgeBase : new InMemoryKnowledgeBase();

            EventStore events = new EventStore(io, om, io.resolve(cfg.events.logFile));
            KnowledgeSnapshotStore snapshots = new KnowledgeSnapshotStore(io, om, io.resolve(cfg.knowledge.snapshotFile));

            Context js = enableJs ? buildJsContext(allowHostAccess, allowIO, warnInterpreterOnly) : null;

            AIKernel k = new AIKernel(io, om, cfg, kb, events, snapshots, js);

            k.logCreated(cfgPath);
            return k;
        }

        private void logCreated(Path cfgPath) {
            if (!log.isInfoEnabled()) return;
            log.info("AIKernel created (no bootstrap yet): config={}, baseDir={}, js={}",
                    cfgPath, "<loaded>", enableJs);
        }

        private static ObjectMapper defaultMapper() {
            ObjectMapper om = new ObjectMapper();
            om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
            return om;
        }

        private static Context buildJsContext(boolean allowHostAccess, boolean allowIO, String warnInterpreterOnly) {
            HostAccess ha = allowHostAccess ? HostAccess.ALL : HostAccess.NONE;
            return Context.newBuilder("js")
                    .allowHostAccess(ha)
                    .option("engine.WarnInterpreterOnly", warnInterpreterOnly == null ? "false" : warnInterpreterOnly)
                    .allowIO(allowIO)
                    .build();
        }
    }

    // ---------------------------------------------------------------------
    // Bootstrap (explicit)
    // ---------------------------------------------------------------------

    /**
     * Loads corpora into KB according to config.
     * Can be called once; repeated calls become no-op.
     */
    public synchronized void bootstrap() throws IOException {
        if (bootstrapped) return;

        Path corporaDir = io.resolveExternal(Path.of(cfg.corpora.dir));
        CorpusBootstrapper bs = new CorpusBootstrapper(io, mapper);

        int loaded = 0;
        for (String name : cfg.corpora.bootstrap) {
            if (name == null || name.isBlank()) continue;
            bs.loadInto(kb, corporaDir.resolve(name), cfg.corpora.failFast);
            loaded++;
        }

        bootstrapped = true;
        log.info("AIKernel bootstrap done: corporaDir={}, filesLoaded={}, failFast={}",
                corporaDir, loaded, cfg.corpora.failFast);
    }

    /**
     * Bootstrap only if not bootstrapped.
     * Useful when some runtime paths may postpone bootstrap.
     */
    public void bootstrapIfNeeded() throws IOException {
        if (!bootstrapped) bootstrap();
    }

    public boolean isBootstrapped() {
        return bootstrapped;
    }

    // ---------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------

    public FileIO io() { return io; }
    public ObjectMapper mapper() { return mapper; }
    public AIConfig config() { return cfg; }
    public KnowledgeBase knowledge() { return kb; }
    public EventStore eventStore() { return events; }
    public KnowledgeSnapshotStore snapshotStore() { return snapshots; }

    /** JS context can be disabled. */
    public Context jsContext() { return jsContext; }

    // ---------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------

    @Override
    public void close() {
        try {
            if (jsContext != null) jsContext.close(true);
        } catch (Throwable ignored) {
            // keep shutdown robust
        }
    }

    private void logCreated(Path cfgPath) {
        if (!log.isInfoEnabled()) return;
        log.info("AIKernel created: config={}, baseDir={}, jsEnabled={}",
                cfgPath, cfg.baseDir, (jsContext != null));
    }
}