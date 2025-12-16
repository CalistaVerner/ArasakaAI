package org.calista.arasaka.ai.retrieve.exploration.impl;

import org.calista.arasaka.ai.retrieve.exploration.ExplorationConfig;
import org.calista.arasaka.ai.retrieve.Scored;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationStrategy;

import java.util.*;

public final class SoftmaxSampler implements ExplorationStrategy {
    @Override
    public <T> List<T> select(List<Scored<T>> ranked, int k, ExplorationConfig cfg, long seed) {
        if (ranked == null || ranked.isEmpty() || k <= 0) return List.of();

        // Deterministic: no randomness, seed used only for stable tie-break.

        int hardTop = Math.min(cfg.topK * Math.max(1, cfg.candidateMultiplier), ranked.size());
        int need = Math.min(k, Math.min(cfg.topK, ranked.size()));
        List<Scored<T>> pool = ranked.subList(0, hardTop);

        // Precompute softmax log-weights (numerically stable): logw = score/temperature - max
        double max = Double.NEGATIVE_INFINITY;
        for (Scored<T> s : pool) max = Math.max(max, s.score / cfg.temperature);
        if (!Double.isFinite(max)) max = 0.0;

        final double[] logw = new double[pool.size()];
        for (int i = 0; i < pool.size(); i++) {
            double lw = (pool.get(i).score / cfg.temperature) - max;
            if (!Double.isFinite(lw)) lw = -50.0;
            logw[i] = lw;
        }

        // Cache token sets once per candidate to avoid repeated regex-split.
        final HashMap<Integer, Set<String>> tokenCache = new HashMap<>(pool.size() * 2);

        ArrayList<Integer> remaining = new ArrayList<>(pool.size());
        for (int i = 0; i < pool.size(); i++) remaining.add(i);

        ArrayList<T> out = new ArrayList<>(need);
        ArrayList<Integer> chosenIdx = new ArrayList<>(need);
        HashSet<Integer> used = new HashSet<>(need * 2);

        for (int pick = 0; pick < need && !remaining.isEmpty(); pick++) {
            int bestIdx = -1;
            double best = Double.NEGATIVE_INFINITY;

            for (int idx : remaining) {
                if (used.contains(idx)) continue;

                Scored<T> cand = pool.get(idx);
                double base = logw[idx];

                // Diversity penalty: subtract similarity to already chosen items.
                if (cfg.diversity > 0.0 && !chosenIdx.isEmpty()) {
                    Set<String> cTok = tokenCache.computeIfAbsent(idx, i -> tokenSetOf(textOf(cand.item)));
                    double maxSim = 0.0;
                    for (int j : chosenIdx) {
                        Set<String> sTok = tokenCache.computeIfAbsent(j, ii -> tokenSetOf(textOf(pool.get(ii).item)));
                        maxSim = Math.max(maxSim, jaccard(cTok, sTok));
                    }
                    base -= cfg.diversity * maxSim;
                }

                // Deterministic tie-break using stable hash mixed with seed.
                double tie = (mix64(seed, stableHash(cand.item)) & 0xFFFF) / 65535.0;
                double total = base + (tie * 1e-9);

                if (total > best) {
                    best = total;
                    bestIdx = idx;
                }
            }

            if (bestIdx < 0) break;

            used.add(bestIdx);
            out.add(pool.get(bestIdx).item);
            chosenIdx.add(bestIdx);

            int finalBestIdx = bestIdx;
            remaining.removeIf(i -> i == finalBestIdx);
        }

        return out;
    }

    private static String textOf(Object o) {
        if (o == null) return "";
        // If Statement exists on classpath, use reflection to avoid hard dependency.
        try {
            var cl = o.getClass();
            var f = cl.getField("text");
            Object v = f.get(o);
            if (v instanceof String s) return s;
        } catch (Throwable ignored) {
        }
        return String.valueOf(o);
    }

    private static Set<String> tokenSetOf(String s) {
        if (s == null || s.isBlank()) return Set.of();
        String low = s.toLowerCase(Locale.ROOT);
        String[] parts = low.split("[^\\p{L}\\p{Nd}_]+");
        if (parts.length == 0) return Set.of();

        HashSet<String> out = new HashSet<>(Math.min(64, parts.length * 2));
        for (String p : parts) {
            if (p == null) continue;
            String t = p.trim();
            if (t.length() < 3) continue;
            out.add(t);
        }
        return out.isEmpty() ? Set.of() : out;
    }

    private static double jaccard(Set<String> a, Set<String> b) {
        if (a == null || b == null || a.isEmpty() || b.isEmpty()) return 0.0;
        if (a == b) return 1.0;

        Set<String> small = a.size() <= b.size() ? a : b;
        Set<String> big = a.size() <= b.size() ? b : a;

        int inter = 0;
        for (String t : small) if (big.contains(t)) inter++;
        int union = a.size() + b.size() - inter;
        return union <= 0 ? 0.0 : (double) inter / (double) union;
    }

    private static long stableHash(Object o) {
        if (o == null) return 0L;
        // Use toString() for stability across JVM runs (Object.hashCode is identity-based).
        return (long) String.valueOf(o).hashCode();
    }

    private static long mix64(long a, long b) {
        long x = a ^ b;
        x ^= (x >>> 33);
        x *= 0xff51afd7ed558ccdL;
        x ^= (x >>> 33);
        x *= 0xc4ceb9fe1a85ec53L;
        x ^= (x >>> 33);
        return x;
    }
}