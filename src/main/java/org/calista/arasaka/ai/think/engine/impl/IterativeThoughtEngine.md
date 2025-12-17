# IterativeThoughtEngine — обновлённый MD гайд

> Package: `org.calista.arasaka.ai.think.engine.impl`
> Класс: `IterativeThoughtEngine implements ThoughtCycleEngine`

## Зачем он нужен

**IterativeThoughtEngine** — enterprise-реализация детерминированного think-loop’а: движок, который **в несколько итераций** строит лучший ответ, используя пайплайн RAG + генерацию + оценку + самокоррекцию.

Он создан как “оркестратор” вокруг модулей:

* `Retriever` (RAG),
* `IntentDetector` (интент как сигнал, не как токсичный токен),
* `ResponseStrategy` / `TextGenerator` (генерация черновиков),
* `CandidateEvaluator` (качество и валидность),
* (опционально) лёгкая LTM-память без магии.

Движок **не хардкодит** доменные ответы: он управляет процессом.

---

## Ключевой контракт (что делает движок)

Пайплайн каждой итерации:

1. **Query build** → 2) **Retrieve** → 3) **Rerank** → 4) **Compress**
2. **Draft** (N вариантов) → 6) **Evaluate** → 7) **Self-correct hints** → 8) (опц.) **LTM write**

### Важные фиксы в этой версии

* **НЕ** добавлять `UNKNOWN` в запрос ретривера (не “травить” retrieval).
* **НЕ** выводить refine-запросы из плохого/невалидного `bestSoFar`.
* **НЕ** допускать коллапс draft’ов в 1 штуку при детерминированной генерации.
* Для очень коротких запросов (≤2 токена) — **только `summary`** секция (без поведения “нужен контекст”).
* Executor для оценки **инжектится извне** (владелец — Think оркестратор), **нет статических пулов**.

---

## Высокоуровневая схема

```
think(userText)
  ├─ normalize + intent
  ├─ initResponseSchema(tags)
  ├─ repeat iterations:
  │    ├─ buildQueries(userText, state)
  │    ├─ retrieveRerankCompress(...)
  │    ├─ (optional) recallFromLtm + merge + rerank + compress
  │    ├─ tuneGenerationTags(state, userText, context)
  │    ├─ generateDrafts(N)
  │    ├─ Drafts.sanitize(...)
  │    ├─ evaluateDrafts(parallel)
  │    ├─ pickBest + update bestSoFar
  │    ├─ (optional) maybeWritebackToLtm
  │    └─ stop rules
  └─ return ThoughtResult(answer, trace, intent, bestCandidate, evaluation)
```

---

## Вход/выход

### Вход

* `think(String userText, long seed)`

### Выход

* `ThoughtResult`:

    * `answerText` (лучший ответ),
    * `trace` (итерационный трейс — важен для отладки/метрик),
    * `intent`,
    * `bestCandidate` + `bestEvaluation`.

---

## Детерминизм

Детерминизм обеспечивается:

* детерминированным `mix(seed, …)` для разных стадий,
* стабильными сортировками (stable key на Statement),
* `Drafts.sanitize()` (дедуп + детерминированный padding),
* отсутствием случайных выборок внутри движка.

> Важно: если сам `TextGenerator` не детерминирован — движок не сможет гарантировать полный детерминизм, но сохранит стабильные правила оркестрации.

---

## Контекст: retrieve → rerank → compress

### Retrieve

Движок поддерживает два пути:

1. **Оптимизированный** (если `retriever instanceof KnowledgeRetriever`):

    * использует `retrieveTrace(joinedQueries, retrieveK, seed)`
    * может получить готовый `rerankedTop` и `compressedContext` + `quality`.

2. **Fallback** (любой другой `Retriever`):

    * `retrieveAndMerge()` по каждому запросу
    * refine rounds: `deriveQueriesFromContext()` и повторный retrieval

### Merge

Контекст собирается в `Map<stableKey, Statement>` чтобы:

* исключать дубликаты,
* сохранять детерминизм,
* позволять мерджить LTM и raw-evidence.

### Rerank

`rerankContext()` сортирует по overlap с токенами запроса (`overlapScore`) и стабильно tie-break’ит по `stableStmtKey()`.

### Compress

`compressContextToStatements()` превращает каждый statement в короткую синтетическую версию:

* берёт `COMPRESS_SENTENCES` предложений (по дефолту 2),
* не превышает `COMPRESS_MAX_CHARS`.

---

## Генерация: Strategy vs TextGenerator

### Если есть `TextGenerator`

Используется:

* `generator.generateN(userText, context, state, draftsPerIteration)`

Это “правильный путь” под BeamSearch/TensorFlow.

### Если генератора нет

Fallback:

* `strategy.generate(userText, context, state)` повторяется `draftsPerIteration` раз.

---

## Контроль генерации: tags + hints

Движок не диктует фразы, он задаёт **параметры** генерации через `state.tags`:

* `response.sections` — `summary` или `summary,evidence,actions`
* `response.style` — `md`
* `gen.minTok`, `gen.maxTok`

### Схема секций (важный фикс)

* если запрос короткий (≤2 токена) → `response.sections=summary`
* иначе → `summary,evidence,actions`

### Repair-теги (самокоррекция)

Если `lastEvaluation.valid == false`, движок выставляет флаги:

* `repair.addEvidence`
* `repair.reduceNovelty`
* `repair.fixStructure`
* `repair.avoidEcho`

И добавляет их в `generationHint` как:

`repair=addEvidence,reduceNovelty,fixStructure,avoidEcho`

> Это **контрольные сигналы**, а не хардкод ответов.

---

## Drafts.sanitize — анти-коллапс

Встроенный helper `Drafts.sanitize()`:

* удаляет null/blank,
* тримит,
* дедуплицирует (exact string),
* гарантирует минимум draft’ов: **никогда не опускается до 1**, целится в `2..12`.

Это критично для детерминированных генераторов, которые иногда возвращают один и тот же текст.

---

## Оценка кандидатов (parallel)

`evaluateDrafts()`:

* при `drafts <= 2` или `evalParallelism <= 1` — синхронно,
* иначе — параллельно через `CompletableFuture` на **инжектнутом** `evalPool`.

### MDC / ThreadContext

Движок копирует `ThreadContext` в каждую async-задачу:

* сохраняются `reqId`, `seed`, `iter`, `intent` и любые другие MDC-поля,
* после выполнения чистится контекст, чтобы не было утечек.

---

## Stop rules

Остановка итераций выполняется через `StopRule`:

* `targetScore(targetScore)` — достигли качества
* `patience(patience)` — слишком долго без улучшений

Это заменяет “магические if-ы” на расширяемую модель правил.

---

## LTM (Long-Term Memory) без магии

LTM — это просто детерминированный буфер `Statement`-ов:

* `ltmByKey`: key → statement
* `ltmPriorityByKey`: key → priority
* `ltmTokensByKey`: key → token cache

### Recall

`recallFromLtm(queries)`:

* собирает токены запросов,
* ранжирует память по overlap + priority,
* возвращает top-K.

### Writeback

`maybeWritebackToLtm(best, ctx)`:

* пишет **только evidence** (statements), а не “ответ модели”,
* только если `best.evaluation.valid` и `groundedness >= ltmWriteMinGroundedness`,
* ограничивает размер `ltmCapacity`.

Это важно: LTM хранит **факты/источники**, а не “галлюцинации”.

---

## Логирование и трассировка

### MDC поля

Движок выставляет:

* `reqId` (генерируется от seed + time),
* `seed`,
* `engine=IterativeThoughtEngine`,
* `intent`,
* `iter`.

### INFO

* `think.start` — параметры запуска
* `think.end` — финальные итоги

### DEBUG

* `think.iter` — разрез по стадиям, длительности, размерам контекста

### Trace list

Возвращаемый `trace` содержит компактные строки:

* `iter=... raw=... rerank=... compress=... ltm=... drafts=... bestIter=... global=... t...`.

Это удобно для:

* профилирования,
* регресс-тестов,
* построения метрик качества.

---

## Конфиг/параметры

| Параметр                  | Значение | Что делает                                    |
| ------------------------- | -------: | --------------------------------------------- |
| `iterations`              |       ≥1 | число think-итераций                          |
| `retrieveK`               |       ≥1 | сколько брать из retriever                    |
| `draftsPerIteration`      |       ≥6 | сколько черновиков на итерацию (анти-коллапс) |
| `patience`                |       ≥0 | сколько итераций терпеть без улучшения        |
| `targetScore`             |   double | цель по качеству                              |
| `refineRounds`            |       ≥0 | сколько раундов refine retrieval              |
| `refineQueryBudget`       |       ≥1 | бюджет derive-запросов                        |
| `ltmEnabled`              |     bool | включить LTM                                  |
| `ltmCapacity`             |       ≥0 | максимум памяти                               |
| `ltmRecallK`              |       ≥0 | сколько вспоминать                            |
| `ltmWriteMinGroundedness` |     0..1 | порог записи в LTM                            |
| `evalPool`                | injected | пул потоков под оценку                        |
| `evalParallelism`         |       ≥1 | сколько параллельно оценивать                 |

---

## Известные ограничения

1. **Токен-оверлап ранжирование**
   Rerank и derive-queries используют простые токены. Без эмбеддингов:

* синонимы и парафразы хуже работают.

2. **Compression примитивный**
   Сжимает по предложениям. Для кода/таблиц иногда плохо.

3. **LTM хранит только evidence**
   Это плюс по надёжности, но минус по “богатству знаний”: нужны внешние корпуса/ретривер.

4. **Refine derive-queries DF-эвристика**
   `DERIVE_DF_CUT` — общая эвристика; на некоторых доменах может давать шум.

---

## Типовые баги (и как эта версия их чинит)

### 1) Poison retrieval: “UNKNOWN”

**Было:** запросы вида `"Привет UNKNOWN"` ломали retrieval.
**Стало:** `QueryProvider.intentName()` пропускает `UNKNOWN`.

### 2) Refine от плохого bestSoFar

**Было:** derive-запросы строились из мусорного ответа.
**Стало:** `bestTerms()` работает только если `bestEvaluation.valid == true`.

### 3) Коллапс draft’ов до 1

**Было:** детерминированный генератор возвращал одинаковое → один draft.
**Стало:** `Drafts.sanitize()` держит минимум 2..4 (и больше) draft’ов.

### 4) Короткие запросы требовали evidence/actions

**Было:** “Привет” → система просит контекст.
**Стало:** ≤2 токена → `sections=summary`.

---

## Точки роста (что улучшать дальше)

### Высокий приоритет

1. **Explainable retrieval quality**

* нормировать `traceQuality`,
* логировать причины низкого качества.

2. **Better query derivation**

* вместо DF-cut добавить PMI/классические IR сигналы,
* учитывать intent.

3. **Adaptive section contract**

* завязать количество секций на сложность вопроса/intent.

### Средний приоритет

4. **Семантический rerank**

* подключить эмбеддинги или TF similarity,
* гибрид: overlap + embedding.

5. **Repair feedback loop**

* на основе `evaluation.telemetry` обновлять генераторные параметры более точно.

### Низкий приоритет

6. **LTM decay/aging**

* затухание priority,
* защита от “перекоса” памяти.

---

## Рекомендации по интеграции

* Всегда задавайте `evalPool` сверху (оркестратор Think владеет ресурсами).
* Для prod: фиксируйте `seed` в тестах для воспроизводимости.
* Включайте `DEBUG` на этапах тюнинга, затем переходите к sampling.
* Подключайте `TextGenerator` (Beam/TensorFlow) — это главный путь к качеству.

---

## Быстрый чеклист

* [ ] `IntentDetector` не возвращает `UNKNOWN` по умолчанию без весов (или включён bootstrap).
* [ ] `draftsPerIteration >= 6`
* [ ] `CandidateEvaluator` настроен под noctx/ctx режимы.
* [ ] `evalPool` ограничен и не создаётся внутри движка.
* [ ] LTM включать только после стабилизации groundedness.