package org.calista.arasaka.ai.retrieve;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class SoftmaxSampler implements ExplorationStrategy {
    @Override
    public <T> List<T> select(List<Scored<T>> ranked, int k, ExplorationConfig cfg, long seed) {
        if (ranked.isEmpty() || k <= 0) return List.of();
        int limit = Math.min(cfg.topK, ranked.size());
        int need = Math.min(k, limit);

        List<Scored<T>> pool = ranked.subList(0, limit);

        double[] w = new double[pool.size()];
        double sum = 0.0;
        for (int i = 0; i < pool.size(); i++) {
            double val = Math.exp(pool.get(i).score / cfg.temperature);
            if (!Double.isFinite(val) || val <= 0) val = 1e-9;
            w[i] = val;
            sum += val;
        }
        if (sum <= 0) {
            ArrayList<T> out = new ArrayList<>(need);
            for (int i = 0; i < need; i++) out.add(pool.get(i).item);
            return out;
        }

        Random rnd = new Random(seed);

        ArrayList<Scored<T>> mutable = new ArrayList<>(pool);
        ArrayList<Double> weights = new ArrayList<>(w.length);
        for (double x : w) weights.add(x);

        ArrayList<T> out = new ArrayList<>(need);
        for (int pick = 0; pick < need && !mutable.isEmpty(); pick++) {
            double total = 0.0;
            for (double x : weights) total += x;

            double r = rnd.nextDouble() * total;
            int idx = 0;
            double acc = 0.0;
            for (; idx < weights.size(); idx++) {
                acc += weights.get(idx);
                if (acc >= r) break;
            }
            if (idx >= mutable.size()) idx = mutable.size() - 1;

            out.add(mutable.get(idx).item);
            mutable.remove(idx);
            weights.remove(idx);
        }
        return out;
    }
}