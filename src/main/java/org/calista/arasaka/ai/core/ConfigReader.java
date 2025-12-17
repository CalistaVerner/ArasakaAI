package org.calista.arasaka.ai.core;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.calista.arasaka.io.FileIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * ConfigReader reads config as JsonNode to detect missing fields,
 * logs warnings for missing values and fallback to defaults,
 * then binds to AIConfig and validates.
 *
 * No magic: defaults come from AIConfig field initializers + AIConfig.validate() normalization.
 */
@Deprecated
public final class ConfigReader {

    private static final Logger log = LoggerFactory.getLogger(ConfigReader.class);

    private ConfigReader() {}

    public static AIConfig load(FileIO io, Path configFile, ObjectMapper mapper) throws IOException {
        Objects.requireNonNull(io, "io");
        Objects.requireNonNull(configFile, "configFile");
        Objects.requireNonNull(mapper, "mapper");

        String json = io.readString(configFile);

        JsonNode root = mapper.readTree(json);
        if (root == null || root.isNull()) {
            log.warn("Config file {} is empty/null. Falling back to defaults.", configFile);
            AIConfig cfg = new AIConfig();
            cfg.validate();
            return cfg;
        }
        if (!root.isObject()) {
            throw new IllegalStateException("Config root must be JSON object: " + configFile);
        }

        // bind to POJO (unknown props ignored by annotation)
        AIConfig cfg = mapper.treeToValue(root, AIConfig.class);
        if (cfg == null) cfg = new AIConfig();

        // warn about missing fields and fallback (dynamic, reflection-based)
        warnMissing(root);

        // validate + normalize
        cfg.validate();
        return cfg;
    }

    /**
     * Warn only for fields that are ABSENT in JSON (dynamic).
     *
     * Design:
     * - We build a "schema" from AIConfig defaults (a probe instance) via reflection.
     * - We warn on missing OBJECT nodes once, and then skip all children under that prefix
     *   to avoid log spam.
     * - We respect @JsonProperty names where present.
     */
    private static void warnMissing(JsonNode root) {
        AIConfig defaultsProbe = new AIConfig();
        // normalize defaults so warnings reflect effective defaults, not raw unvalidated ones
        try { defaultsProbe.validate(); } catch (Throwable ignored) {}

        final List<FieldSpec> specs = SchemaBuilder.build(defaultsProbe);

        // If an object is missing, log it once and skip all nested fields
        final ArrayList<String> missingPrefixes = new ArrayList<>(16);

        for (FieldSpec s : specs) {
            // skip if parent object already missing
            if (isUnderAnyPrefix(s.pointer, missingPrefixes)) continue;

            JsonNode node = root.at(s.pointer);
            if (!node.isMissingNode()) continue;

            if (s.kind == Kind.OBJECT) {
                log.warn("Config missing section {} -> fallback to default: {}", s.pointer, "{...}");
                missingPrefixes.add(s.pointer);
            } else {
                log.warn("Config missing field {} -> fallback to default: {}", s.pointer, pretty(s.defaultValue));
            }
        }
    }

    private static boolean isUnderAnyPrefix(String pointer, List<String> prefixes) {
        for (String p : prefixes) {
            if (pointer.equals(p)) return true;
            if (pointer.startsWith(p.endsWith("/") ? p : (p + "/"))) return true;
        }
        return false;
    }

    // -------------------- Schema building (dynamic) --------------------

    private enum Kind { OBJECT, SCALAR, COLLECTION, MAP }

    private static final class FieldSpec {
        final String pointer;
        final Kind kind;
        final Object defaultValue;

        FieldSpec(String pointer, Kind kind, Object defaultValue) {
            this.pointer = pointer;
            this.kind = kind;
            this.defaultValue = defaultValue;
        }
    }

    /**
     * Builds a field schema using reflection:
     * - adds a spec for each object section (Kind.OBJECT)
     * - adds a spec for each scalar leaf (Kind.SCALAR)
     * - adds a spec for collections/maps as a section (Kind.COLLECTION/MAP)
     *
     * NOTE: This does NOT require hardcoding JSON pointers.
     */
    private static final class SchemaBuilder {

        static List<FieldSpec> build(Object rootProbe) {
            ArrayList<FieldSpec> out = new ArrayList<>(256);
            IdentityHashMap<Object, Boolean> visited = new IdentityHashMap<>();
            walkObject(rootProbe, "", out, visited, 0);
            return out;
        }

        private static void walkObject(Object probe, String basePointer, List<FieldSpec> out,
                                       IdentityHashMap<Object, Boolean> visited, int depth) {
            if (probe == null) return;
            if (depth > 32) return; // safety
            if (visited.put(probe, Boolean.TRUE) != null) return;

            Class<?> type = probe.getClass();

            // For root, basePointer == "" (JsonPointer expects leading "/field")
            // Add object spec for non-root objects (root is implied)
            if (!basePointer.isEmpty()) {
                out.add(new FieldSpec(basePointer, Kind.OBJECT, "{...}"));
            }

            for (Field f : allFields(type)) {
                if (!isConfigField(f)) continue;

                String name = jsonNameOf(f);
                String pointer = basePointer + "/" + name;

                Object v = getFieldValueQuiet(probe, f);

                Class<?> ft = f.getType();

                if (isScalarType(ft, v)) {
                    out.add(new FieldSpec(pointer, Kind.SCALAR, v));
                    continue;
                }

                if (Map.class.isAssignableFrom(ft)) {
                    out.add(new FieldSpec(pointer, Kind.MAP, "{...}"));
                    continue;
                }

                if (Collection.class.isAssignableFrom(ft)) {
                    out.add(new FieldSpec(pointer, Kind.COLLECTION, (v instanceof Collection<?> c) ? c : List.of()));
                    continue;
                }

                // Nested POJO section
                out.add(new FieldSpec(pointer, Kind.OBJECT, "{...}"));
                walkObject(v, pointer, out, visited, depth + 1);
            }
        }

        private static List<Field> allFields(Class<?> type) {
            ArrayList<Field> fs = new ArrayList<>();
            Class<?> c = type;
            while (c != null && c != Object.class) {
                Field[] declared = c.getDeclaredFields();
                Collections.addAll(fs, declared);
                c = c.getSuperclass();
            }
            return fs;
        }

        private static boolean isConfigField(Field f) {
            int m = f.getModifiers();
            if (Modifier.isStatic(m)) return false;
            if (Modifier.isTransient(m)) return false;
            if (f.isSynthetic()) return false;
            // usually you don't want these
            String n = f.getName();
            if ("this$0".equals(n)) return false;
            return true;
        }

        private static String jsonNameOf(Field f) {
            JsonProperty jp = f.getAnnotation(JsonProperty.class);
            if (jp != null && jp.value() != null && !jp.value().isBlank()) return jp.value();
            return f.getName();
        }

        private static Object getFieldValueQuiet(Object obj, Field f) {
            try {
                if (!f.canAccess(obj)) f.setAccessible(true);
                return f.get(obj);
            } catch (Throwable t) {
                return null;
            }
        }

        private static boolean isScalarType(Class<?> ft, Object v) {
            if (ft.isPrimitive()) return true;
            if (ft.isEnum()) return true;

            // common scalar-ish types
            if (ft == String.class) return true;
            if (Number.class.isAssignableFrom(ft)) return true;
            if (ft == Boolean.class || ft == Character.class) return true;

            // Path is semantically scalar in config
            if (Path.class.isAssignableFrom(ft)) return true;

            // time-ish types often appear in configs
            if (ft == Duration.class || ft == Instant.class) return true;

            // treat boxed primitives as scalar
            if (ft == Integer.class || ft == Long.class || ft == Double.class || ft == Float.class
                    || ft == Short.class || ft == Byte.class) return true;

            // if value is null, we still decide by type (non-scalar => object section)
            return false;
        }
    }

    // -------------------- Formatting --------------------

    private static String pretty(Object v) {
        if (v == null) return "null";
        if (v instanceof String s) return "\"" + s + "\"";
        if (v instanceof Path p) return "\"" + p.toString() + "\"";
        if (v instanceof Collection<?> xs) return xs.toString();
        if (v instanceof Map<?, ?> m) return "{...}"; // avoid dumping huge structures
        return String.valueOf(v);
    }
}