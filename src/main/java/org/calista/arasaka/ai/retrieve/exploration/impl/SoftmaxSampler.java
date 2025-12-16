package org.calista.arasaka.ai.retrieve.exploration.impl;

import org.calista.arasaka.ai.retrieve.exploration.ExplorationConfig;
import org.calista.arasaka.ai.retrieve.Scored;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.HashSet;

public final class SoftmaxSampler implements ExplorationStrategy {
    @Override
    public <T> List<T> select(List<Scored<T>> ranked, int k, ExplorationConfig cfg, long seed) {
        if (ranked == null || ranked.isEmpty() || k <= 0) return List.of();

        // NOTE: "no randomness" requirement.
        // We keep the softmax idea (temperature) but selection is deterministic and diversity-aware.
        // Seed is used only for deterministic tie-breaking.

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

        ArrayList<Integer> remaining = new ArrayList<>(pool.size());
        for (int i = 0; i < pool.size(); i++) remaining.add(i);

        ArrayList<T> out = new ArrayList<>(need);
        ArrayList<String> selectedTexts = new ArrayList<>(need);
        Set<Integer> used = new HashSet<>(need * 2);

        for (int pick = 0; pick < need && !remaining.isEmpty(); pick++) {
            int bestIdx = -1;
            double best = Double.NEGATIVE_INFINITY;

            for (int idx : remaining) {
                if (used.contains(idx)) continue;
                Scored<T> cand = pool.get(idx);
                double base = logw[idx];

                // Diversity penalty: subtract similarity to already chosen items.
                double penalty = 0.0;
                if (cfg.diversity > 0.0 && !selectedTexts.isEmpty()) {
                    String ct = textOf(cand.item);
                    for (String st : selectedTexts) penalty = Math.max(penalty, tokenJaccard(ct, st));
                    base -= cfg.diversity * penalty;
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
            Scored<T> chosen = pool.get(bestIdx);
            out.add(chosen.item);
            selectedTexts.add(textOf(chosen.item));

            int finalBestIdx = bestIdx;
            remaining.removeIf(i -> Objects.equals(i, finalBestIdx));
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

    private static double tokenJaccard(String a, String b) {
        if (a == null || b == null) return 0.0;
        String aa = a.toLowerCase(Locale.ROOT);
        String bb = b.toLowerCase(Locale.ROOT);
        String[] ta = aa.split("[^\\p{L}\\p{Nd}_]+");
        String[] tb = bb.split("[^\\p{L}\\p{Nd}_]+");
        if (ta.length == 0 || tb.length == 0) return 0.0;

        HashSet<String> sa = new HashSet<>();
        for (String t : ta) if (t != null && t.length() >= 3) sa.add(t);
        if (sa.isEmpty()) return 0.0;
        int inter = 0;
        HashSet<String> sb = new HashSet<>();
        for (String t : tb) if (t != null && t.length() >= 3) sb.add(t);
        if (sb.isEmpty()) return 0.0;

        for (String t : sa) if (sb.contains(t)) inter++;
        int union = sa.size() + sb.size() - inter;
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