package org.calista.arasaka.ai.knowledge;

import java.util.*;

public final class InMemoryKnowledgeBase implements KnowledgeBase {
    private final Map<String, Statement> byId = new HashMap<>();

    @Override
    public synchronized boolean upsert(Statement st) {
        Objects.requireNonNull(st, "st");
        st.validate();

        Statement prev = byId.put(st.id, st);
        if (prev == null) return true;

        // “изменилось ли” — полезно для аналитики/логов
        if (!Objects.equals(prev.text, st.text)) return true;
        if (prev.weight != st.weight) return true;
        return !Objects.equals(prev.tags, st.tags);
    }

    @Override
    public synchronized Optional<Statement> get(String id) {
        Objects.requireNonNull(id, "id");
        return Optional.ofNullable(byId.get(id));
    }

    @Override
    public synchronized List<Statement> snapshotSorted() {
        ArrayList<Statement> out = new ArrayList<>(byId.values());
        out.sort(Comparator.comparing(a -> a.id));
        return Collections.unmodifiableList(out);
    }

    @Override
    public synchronized int size() {
        return byId.size();
    }
}