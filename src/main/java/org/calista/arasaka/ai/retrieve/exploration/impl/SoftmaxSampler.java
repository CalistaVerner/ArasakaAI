package org.calista.arasaka.ai.retrieve.exploration.impl;

import org.calista.arasaka.ai.retrieve.Scored;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationConfig;
import org.calista.arasaka.ai.retrieve.exploration.ExplorationStrategy;

import java.util.*;

/**
 * Deterministic "softmax exploration" selector.
 *
 * Key idea:
 *  - still deterministic (no RNG), but behaves like sampling by adding deterministic Gumbel noise
 *    derived from (seed, stableKey).
 *  - supports diversity penalty using Jaccard similarity over cheap token sets.
 *
 * This gives you exploration-like behavior without randomness or nondeterminism.
 */
public final class SoftmaxSampler implements ExplorationStrategy {
    @Override
    public <T> List<T> select(List<Scored<T>> ranked, int k, ExplorationConfig cfg, long seed) {
        if (ranked == null || ranked.isEmpty() || k <= 0) return List.of();

        final int hardTop = Math.min(cfg.topK * Math.max(1, cfg.candidateMultiplier), ranked.size());
        final int need = Math.min(k, ranked.size());
        final List<Scored<T>> pool = ranked.subList(0, hardTop);

        final HashMap<Integer, Set<String>> tokenCache = cfg.diversity > 0.0
                ? new HashMap<>(pool.size() * 2)
                : null;

        final ArrayList<Integer> remaining = new ArrayList<>(pool.size());
        for (int i = 0; i < pool.size(); i++) remaining.add(i);

        final ArrayList<T> out = new ArrayList<>(need);
        final ArrayList<Integer> chosenIdx = new ArrayList<>(need);

        for (int pick = 0; pick < need && !remaining.isEmpty(); pick++) {
            int bestIdx = -1;
            double best = Double.NEGATIVE_INFINITY;

            for (int idx : remaining) {
                Scored<T> cand = pool.get(idx);

                double base = cand.score / cfg.temperature;
                if (!Double.isFinite(base)) base = -1e9;

                if (cfg.diversity > 0.0 && !chosenIdx.isEmpty()) {
                    Set<String> cTok = tokenCache.computeIfAbsent(idx, i -> tokenSetOf(textOf(cand.item)));
                    double maxSim = 0.0;
                    for (int j : chosenIdx) {
                        Set<String> sTok = tokenCache.computeIfAbsent(j, ii -> tokenSetOf(textOf(pool.get(ii).item)));
                        maxSim = Math.max(maxSim, jaccard(cTok, sTok));
                    }
                    base -= cfg.diversity * maxSim;
                }

                double u = toUnitInterval(mix64(seed, stableHash(cand.stableKey())));
                double g = gumbel(u);
                double total = base + g;

                if (total > best) {
                    best = total;
                    bestIdx = idx;
                }
            }

            if (bestIdx < 0) break;

            out.add(pool.get(bestIdx).item);
            chosenIdx.add(bestIdx);

            int finalBestIdx = bestIdx;
            remaining.removeIf(i -> i == finalBestIdx);
        }

        return out;
    }

    private static double gumbel(double u) {
        double x = Math.min(1.0 - 1e-12, Math.max(1e-12, u));
        return -Math.log(-Math.log(x));
    }

    private static double toUnitInterval(long x) {
        long v = (x >>> 11) & ((1L << 53) - 1);
        return (v + 1.0) / ((double) (1L << 53) + 2.0);
    }

    private static String textOf(Object o) {
        if (o == null) return "";
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

    private static long stableHash(String s) {
        return s == null ? 0L : (long) s.hashCode();
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