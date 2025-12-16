package org.calista.arasaka.io;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.FileTime;
import java.time.Duration;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * FileIO — единая точка I/O в проекте.
 *
 * Фокус под AI/корпуса:
 * - стриминг больших файлов (lines/jsonl) без загрузки в память
 * - атомарные коммиты + (опционально) fsync
 * - безопасный resolve (anti path traversal)
 * - блокировки для конкурентной записи (jsonl лог/датасет)
 *
 * Без внешних зависимостей.
 */
public final class FileIO {
    private static final Logger log = LogManager.getLogger(FileIO.class);

    // ----------------------------
    // Options / Builder
    // ----------------------------

    public static final class Options {
        public final Charset charset;
        public final boolean atomicWrites;
        public final boolean fsyncOnCommit;
        public final boolean lockWrites;
        public final Duration lockTimeout;
        public final int ioBufferSize;

        private Options(Builder b) {
            this.charset = b.charset;
            this.atomicWrites = b.atomicWrites;
            this.fsyncOnCommit = b.fsyncOnCommit;
            this.lockWrites = b.lockWrites;
            this.lockTimeout = b.lockTimeout;
            this.ioBufferSize = b.ioBufferSize;
        }

        public static Builder builder() { return new Builder(); }

        public static final class Builder {
            private Charset charset = StandardCharsets.UTF_8;
            private boolean atomicWrites = true;
            private boolean fsyncOnCommit = true; // важнее для моделей/артефактов
            private boolean lockWrites = false;   // включай для jsonl/логов с конкуренцией
            private Duration lockTimeout = Duration.ofSeconds(3);
            private int ioBufferSize = 64 * 1024;

            public Builder charset(Charset v) { this.charset = Objects.requireNonNull(v); return this; }
            public Builder atomicWrites(boolean v) { this.atomicWrites = v; return this; }
            public Builder fsyncOnCommit(boolean v) { this.fsyncOnCommit = v; return this; }
            public Builder lockWrites(boolean v) { this.lockWrites = v; return this; }
            public Builder lockTimeout(Duration v) { this.lockTimeout = Objects.requireNonNull(v); return this; }
            public Builder ioBufferSize(int v) {
                if (v < 4 * 1024) throw new IllegalArgumentException("ioBufferSize too small: " + v);
                this.ioBufferSize = v;
                return this;
            }

            public Options build() { return new Options(this); }
        }
    }

    private final Path baseDir;
    private final Options opt;

    public FileIO(Path baseDir) {
        this(baseDir, Options.builder().build());
    }

    /** Backward-compatible constructor: старый стиль создания (Path, Charset, atomicWrites). */
    public FileIO(Path baseDir, Charset charset, boolean atomicWrites) {
        this(baseDir, Options.builder()
                .charset(Objects.requireNonNull(charset, "charset"))
                .atomicWrites(atomicWrites)
                .build());
    }

    public FileIO(Path baseDir, Options options) {
        this.baseDir = Objects.requireNonNull(baseDir, "baseDir").toAbsolutePath().normalize();
        this.opt = Objects.requireNonNull(options, "options");
        log.info("FileIO init: baseDir={}, charset={}, atomicWrites={}, fsyncOnCommit={}, lockWrites={}, lockTimeout={}, ioBufferSize={}",
                this.baseDir, opt.charset, opt.atomicWrites, opt.fsyncOnCommit, opt.lockWrites, opt.lockTimeout, opt.ioBufferSize);
    }

    // ----------------------------
    // Base dir / Resolve
    // ----------------------------

    public Path baseDir() { return baseDir; }
    public Options options() { return opt; }

    public void ensureBaseDir() throws IOException {
        Files.createDirectories(baseDir);
        log.debug("Base dir ensured: {}", baseDir);
    }

    /**
     * Резолвит относительный путь внутри baseDir и защищает от выхода через "..".
     * Абсолютные пути запрещены.
     */
    public Path resolve(String relative) {
        Objects.requireNonNull(relative, "relative");
        Path rel = Paths.get(relative);
        if (rel.isAbsolute()) throw new IllegalArgumentException("resolve(relative) does not accept absolute paths: " + relative);

        Path p = baseDir.resolve(rel).normalize().toAbsolutePath();
        if (!p.startsWith(baseDir)) throw new IllegalArgumentException("Path traversal detected: " + relative);
        return p;
    }

    public Path resolve(Path relative) {
        Objects.requireNonNull(relative, "relative");
        if (relative.isAbsolute()) throw new IllegalArgumentException("resolve(relative) does not accept absolute paths: " + relative);
        return resolve(relative.toString());
    }

    /** Для внешних путей (корпуса вне sandbox) — только нормализация. */
    public Path resolveExternal(String anyPath) {
        Objects.requireNonNull(anyPath, "anyPath");
        return Paths.get(anyPath).toAbsolutePath().normalize();
    }

    public Path resolveExternal(Path anyPath) {
        Objects.requireNonNull(anyPath, "anyPath");
        return anyPath.toAbsolutePath().normalize();
    }

    public void createDirectories(Path dir) throws IOException {
        if (dir == null) return;
        Files.createDirectories(dir);
        log.trace("Directories ensured: {}", dir.toAbsolutePath().normalize());
    }

    public void ensureParentDir(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        Path parent = file.getParent();
        if (parent != null) Files.createDirectories(parent);
    }

    // ----------------------------
    // Existence / Stat
    // ----------------------------

    public boolean exists(Path file) {
        Objects.requireNonNull(file, "file");
        return Files.exists(file);
    }

    public long size(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        return Files.size(file);
    }

    public FileTime lastModified(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        return Files.getLastModifiedTime(file);
    }

    public void touch(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        ensureParentDir(file);
        if (!Files.exists(file)) {
            Files.createFile(file);
        }
        Files.setLastModifiedTime(file, FileTime.fromMillis(System.currentTimeMillis()));
    }

    public void deleteIfExists(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        boolean deleted = Files.deleteIfExists(file);
        log.debug("deleteIfExists: {} -> {}", file, deleted);
    }

    // ----------------------------
    // Text (small/medium)
    // ----------------------------

    public String readString(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        return Files.readString(file, opt.charset);
    }

    public String readString(Path file, Charset cs) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(cs, "cs");
        return Files.readString(file, cs);
    }

    public Optional<String> readStringIfExists(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        if (!Files.exists(file)) return Optional.empty();
        return Optional.of(readString(file));
    }

    public List<String> readLines(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        return Files.readAllLines(file, opt.charset);
    }

    public List<String> readLines(Path file, Charset cs) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(cs, "cs");
        return Files.readAllLines(file, cs);
    }

    public void writeString(Path file, String content) throws IOException {
        writeString(file, content, opt.charset);
    }

    public void writeString(Path file, String content, Charset cs) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(content, "content");
        Objects.requireNonNull(cs, "cs");
        ensureParentDir(file);

        if (!opt.atomicWrites) {
            writeStringDirect(file, content, cs);
            return;
        }

        Path tmp = tempSibling(file);
        writeStringDirect(tmp, content, cs);
        atomicCommit(tmp, file);
    }

    public boolean writeStringIfMissing(Path file, String content) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(content, "content");
        if (exists(file)) return false;
        writeString(file, content);
        return true;
    }

    public void appendLine(Path file, String line) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(line, "line");
        ensureParentDir(file);

        // Для JSONL/логов конкуренция бывает — по желанию включаем lockWrites
        if (!opt.lockWrites) {
            Files.writeString(file, line + System.lineSeparator(), opt.charset,
                    StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
            return;
        }

        withWriteLock(file, () -> {
            Files.writeString(file, line + System.lineSeparator(), opt.charset,
                    StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
        });
    }

    // ----------------------------
    // Streaming (важно для корпусов)
    // ----------------------------

    /** Стрим строк. ВАЖНО: stream надо закрыть (try-with-resources). */
    public Stream<String> lines(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        return Files.lines(file, opt.charset);
    }

    /** Обработать строки без риска забыть close. */
    public void forEachLine(Path file, Consumer<String> consumer) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(consumer, "consumer");
        try (Stream<String> s = Files.lines(file, opt.charset)) {
            s.forEach(consumer);
        }
    }

    // ----------------------------
    // JSONL helpers
    // ----------------------------

    public void appendJsonl(Path file, String jsonLine) throws IOException {
        Objects.requireNonNull(jsonLine, "jsonLine");
        String s = jsonLine.trim();
        if (s.isEmpty()) return;
        appendLine(file, s);
    }

    /** Небольшие JSONL (в память). Для больших используй jsonlStream/jsonlForEach. */
    public List<String> readJsonl(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        try (Stream<String> s = Files.lines(file, opt.charset)) {
            List<String> out = s.map(x -> x == null ? "" : x.trim())
                    .filter(x -> !x.isEmpty())
                    .collect(Collectors.toList());
            log.debug("readJsonl: {} ({} records)", file, out.size());
            return out;
        }
    }

    /** Стрим JSONL записей (trim + skip empty). Stream надо закрыть. */
    public Stream<String> jsonlStream(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        return Files.lines(file, opt.charset)
                .map(x -> x == null ? "" : x.trim())
                .filter(x -> !x.isEmpty());
    }

    public void jsonlForEach(Path file, Consumer<String> consumer) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(consumer, "consumer");
        try (Stream<String> s = jsonlStream(file)) {
            s.forEach(consumer);
        }
    }

    // ----------------------------
    // Bytes / Channels (артефакты моделей)
    // ----------------------------

    public byte[] readBytes(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        return Files.readAllBytes(file);
    }

    public void writeBytes(Path file, byte[] bytes) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(bytes, "bytes");
        ensureParentDir(file);

        if (!opt.atomicWrites) {
            writeBytesDirect(file, bytes);
            return;
        }

        Path tmp = tempSibling(file);
        writeBytesDirect(tmp, bytes);
        atomicCommit(tmp, file);
    }

    /** Быстрый write через канал (можно использовать для больших бинарников). */
    public void writeBytes(Path file, ByteBuffer buf) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(buf, "buf");
        ensureParentDir(file);

        if (!opt.atomicWrites) {
            writeBufferDirect(file, buf);
            return;
        }

        Path tmp = tempSibling(file);
        writeBufferDirect(tmp, buf);
        atomicCommit(tmp, file);
    }

    // ----------------------------
    // Copy (НЕ закрываем чужие потоки по умолчанию)
    // ----------------------------

    public long copy(InputStream in, Path target) throws IOException {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(target, "target");
        ensureParentDir(target);
        long n = Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
        log.debug("copy(InputStream -> file): {} ({} bytes)", target, n);
        return n;
    }

    public long copy(Path source, OutputStream out) throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(out, "out");
        long n = Files.copy(source, out);
        log.debug("copy(file -> OutputStream): {} ({} bytes)", source, n);
        return n;
    }

    // ----------------------------
    // Safe Writer API (большие тексты/корпуса)
    // ----------------------------

    /**
     * Открывает BufferedWriter.
     * atomicWrites=true: пишет во временный файл, затем commit(handle)
     * atomicWrites=false: пишет прямо в target.
     */
    public WriterHandle openWriter(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        ensureParentDir(file);

        if (!opt.atomicWrites) {
            BufferedWriter w = Files.newBufferedWriter(file, opt.charset,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            return new WriterHandle(file, null, w);
        }

        Path tmp = tempSibling(file);
        BufferedWriter w = Files.newBufferedWriter(tmp, opt.charset,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        return new WriterHandle(file, tmp, w);
    }

    /** Новый API: commit(handle). */
    public void commit(WriterHandle h) throws IOException {
        Objects.requireNonNull(h, "handle");

        IOException closeEx = null;
        try { h.writer.close(); } catch (IOException e) { closeEx = e; }

        if (closeEx != null) {
            log.error("commit: failed to close writer for {}", h.targetFile, closeEx);
            throw closeEx;
        }

        if (h.tmpFile != null) atomicCommit(h.tmpFile, h.targetFile);
    }

    /** Новый API: rollback(handle). */
    public void rollback(WriterHandle h) {
        if (h == null) return;
        try { h.writer.close(); } catch (Exception ignored) {}
        if (h.tmpFile != null) {
            try { Files.deleteIfExists(h.tmpFile); } catch (Exception e) {
                log.warn("rollback: failed to delete tmp {}", h.tmpFile, e);
            }
        }
    }

    // ---- Backward compatible aliases (старые имена методов) ----

    /** @deprecated use commit(handle) */
    @Deprecated
    public void commitWriter(WriterHandle h) throws IOException {
        commit(h);
    }

    /** @deprecated use rollback(handle) */
    @Deprecated
    public void rollbackWriter(WriterHandle h) {
        rollback(h);
    }

    public static final class WriterHandle {
        public final Path targetFile;
        public final Path tmpFile; // null если не atomicWrites
        public final BufferedWriter writer;

        private WriterHandle(Path targetFile, Path tmpFile, BufferedWriter writer) {
            this.targetFile = targetFile;
            this.tmpFile = tmpFile;
            this.writer = writer;
        }
    }

    // ----------------------------
    // Listing / glob / rotate
    // ----------------------------

    public List<Path> list(Path dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (!Files.isDirectory(dir)) return List.of();
        try (Stream<Path> s = Files.list(dir)) {
            return s.sorted().collect(Collectors.toList());
        }
    }

    public List<Path> glob(Path dir, String pattern) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(pattern, "pattern");
        if (!Files.isDirectory(dir)) return List.of();

        PathMatcher matcher = dir.getFileSystem().getPathMatcher("glob:" + pattern);
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(p -> matcher.matches(p.getFileName()))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    /**
     * Простейшая ротация: file -> file.1 -> file.2 ... (maxBackups)
     * Полезно для "корпус-логов" / jsonl трейсов.
     */
    public void rotate(Path file, int maxBackups) throws IOException {
        Objects.requireNonNull(file, "file");
        if (maxBackups < 1) throw new IllegalArgumentException("maxBackups must be >= 1");

        if (!Files.exists(file)) return;

        // file.(max) удаляем
        Path last = file.resolveSibling(file.getFileName() + "." + maxBackups);
        Files.deleteIfExists(last);

        // сдвиг
        for (int i = maxBackups - 1; i >= 1; i--) {
            Path from = file.resolveSibling(file.getFileName() + "." + i);
            if (Files.exists(from)) {
                Path to = file.resolveSibling(file.getFileName() + "." + (i + 1));
                Files.move(from, to, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        // file -> file.1
        Path first = file.resolveSibling(file.getFileName() + ".1");
        Files.move(file, first, StandardCopyOption.REPLACE_EXISTING);
        log.info("rotate: {} -> {} (maxBackups={})", file, first, maxBackups);
    }

    // ----------------------------
    // Internals
    // ----------------------------

    private Path tempSibling(Path target) {
        String name = target.getFileName().toString();
        return target.resolveSibling(name + ".tmp");
    }

    private void writeStringDirect(Path file, String content, Charset cs) throws IOException {
        if (!opt.lockWrites) {
            Files.writeString(file, content, cs,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            return;
        }
        withWriteLock(file, () -> Files.writeString(file, content, cs,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING));
    }

    private void writeBytesDirect(Path file, byte[] bytes) throws IOException {
        if (!opt.lockWrites) {
            Files.write(file, bytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            return;
        }
        withWriteLock(file, () -> Files.write(file, bytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING));
    }

    private void writeBufferDirect(Path file, ByteBuffer buf) throws IOException {
        // гарантируем запись от позиции до лимита
        ByteBuffer src = buf.slice();
        if (!opt.lockWrites) {
            try (FileChannel ch = FileChannel.open(file,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
                while (src.hasRemaining()) ch.write(src);
                if (opt.fsyncOnCommit) ch.force(true);
            }
            return;
        }

        withWriteLock(file, () -> {
            try (FileChannel ch = FileChannel.open(file,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
                while (src.hasRemaining()) ch.write(src);
                if (opt.fsyncOnCommit) ch.force(true);
            }
        });
    }

    private void atomicCommit(Path tmp, Path target) throws IOException {
        // ВАЖНО: fsync tmp до move, чтобы “артефакты” не были пустыми после падения/выключения
        if (opt.fsyncOnCommit) fsyncFile(tmp);

        try {
            Files.move(tmp, target,
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
            if (opt.fsyncOnCommit) fsyncParentDir(target); // best-effort, важно на некоторых FS
            log.trace("atomicCommit: {} -> {} (ATOMIC)", tmp, target);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
            if (opt.fsyncOnCommit) fsyncParentDir(target);
            log.trace("atomicCommit: {} -> {} (NON-ATOMIC fallback)", tmp, target);
        } finally {
            // если move не удался — tmp может остаться, убираем best-effort
            try { Files.deleteIfExists(tmp); } catch (Exception ignored) {}
        }
    }

    private void fsyncFile(Path file) {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            ch.force(true);
        } catch (Exception e) {
            log.debug("fsyncFile ignored for {}: {}", file, e.toString());
        }
    }

    private void fsyncParentDir(Path file) {
        Path parent = file.getParent();
        if (parent == null) return;
        // на Windows и части FS это может быть невозможно — делаем best-effort
        try (FileChannel ch = FileChannel.open(parent, StandardOpenOption.READ)) {
            ch.force(true);
        } catch (Exception e) {
            log.debug("fsyncParentDir ignored for {}: {}", parent, e.toString());
        }
    }

    private void withWriteLock(Path file, IoRunnable action) throws IOException {
        // lock на target-файле: если его нет — создаём пустой (для lock)
        ensureParentDir(file);
        if (!Files.exists(file)) {
            try { Files.createFile(file); } catch (FileAlreadyExistsException ignored) {}
        }

        long deadlineNs = System.nanoTime() + opt.lockTimeout.toNanos();

        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.WRITE)) {
            while (true) {
                try {
                    FileLock lock = ch.tryLock();
                    if (lock != null) {
                        try (lock) {
                            action.run();
                            if (opt.fsyncOnCommit) ch.force(true);
                            return;
                        }
                    }
                } catch (OverlappingFileLockException ignored) {
                    // внутри одного процесса — подождём
                }

                if (System.nanoTime() >= deadlineNs) {
                    throw new IOException("Write lock timeout for " + file);
                }
                sleepQuiet(10);
            }
        }
    }

    private static void sleepQuiet(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
    }

    @FunctionalInterface
    private interface IoRunnable { void run() throws IOException; }
}