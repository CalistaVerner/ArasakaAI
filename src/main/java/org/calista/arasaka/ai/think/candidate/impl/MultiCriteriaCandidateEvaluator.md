# MultiCriteriaCandidateEvaluator — документация

## Назначение

**MultiCriteriaCandidateEvaluator** — продвинутый детерминированный оценщик ответов (candidate evaluator), предназначенный для использования в think-цикле (например, `IterativeThoughtEngine`) и ориентированный на **многофакторную (квази-квантовую) оценку качества ответа**.

Он расширяет классический `AdvancedCandidateEvaluator`, добавляя:

* **многоканальную (superposition) модель качества**,
* оценку **согласованности (coherence)** между сигналами,
* **энтропийный штраф** за “хаотичный” ответ,
* строгий **valid-gate** на базе когерентности,
* расширенную телеметрию для анализа и обучения.

Дизайн полностью **детерминирован**, без случайности, и подходит для enterprise-использования и воспроизводимых экспериментов.

---

## Архитектурная роль в системе

```
TextGenerator
  ↓
Candidate (text)
  ↓
AdvancedCandidateEvaluator   ← базовые метрики
  ↓
MultiCriteriaCandidateEvaluator
  ↓
Candidate(score, valid, telemetry)
```

`MultiCriteriaCandidateEvaluator`:

* **не заменяет** `AdvancedCandidateEvaluator`,
* а использует его как “классический слой” (base),
* поверх которого применяется квантово-инспирированная агрегация сигналов.

---

## Основные возможности

### 1) Многоканальная модель качества (superposition)

Ответ оценивается не одной метрикой, а **набором каналов**:

| Канал           | Источник            | Смысл                            |
| --------------- | ------------------- | -------------------------------- |
| `factual`       | `groundedness`      | фактическая привязка к контексту |
| `structure`     | `structureScore`    | структурированность ответа       |
| `coverage`      | `coverage`          | покрытие вопроса                 |
| `actionability` | derived             | практическая применимость        |
| `risk`          | `contradictionRisk` | риск противоречий                |

Каждый канал нормализован в диапазон **[0..1]**.

---

### 2) Entropy penalty (штраф за хаос)

**Энтропия** отражает, насколько “разбалансированы” каналы.

* высокая энтропия → сигналы противоречат друг другу
* низкая энтропия → сигналы согласованы

Используется нормализованная энтропия Шеннона:

```
entropy = H(channels) / log(N)
```

Влияние на итоговый score:

```
- entropyPenaltyWeight * entropy
```

---

### 3) Coherence (согласованность сигналов)

**Coherence** — мера согласованности каналов, вычисляемая как:

```
coherence = 1 - sqrt(variance(channels))
```

* высокая coherence → ответ “цельный”
* низкая coherence → ответ внутренне конфликтный

Используется:

* как **прибавка к score**,
* и как **валидатор** (`coherence >= minCoherence`).

---

### 4) Quantum-adjusted score

Итоговая формула (как в коде):

```
quantumScore =
    base.score
  + coherenceWeight * coherence
  - entropyPenaltyWeight * entropy
  - 0.30 * contradictionRisk
```

Это делает итоговую оценку:

* чувствительной к структуре качества,
* устойчивой к “случайно хорошим” метрикам,
* более строгой к логическим конфликтам.

---

### 5) Жёсткий valid-gate

Даже если базовый evaluator пометил ответ как `valid=true`,
он может быть **отфильтрован**:

```
valid = base.valid && coherence >= minCoherence
```

Это предотвращает:

* “формально корректные, но распадающиеся” ответы,
* ответы с перекосом (например: хорошая структура, но нулевая фактичность).

---

### 6) Расширенная телеметрия

В `Evaluation.validationNotes` добавляются квантовые сигналы:

* `q_f` — factual
* `q_s` — structure
* `q_c` — coverage
* `q_a` — actionability
* `q_r` — risk
* `q_ent` — entropy
* `q_coh` — coherence
* `q_minCoh` — threshold
* `q_v` — final valid flag

Это делает evaluator:

* пригодным для анализа качества,
* удобным для обучения / адаптации весов,
* совместимым с логированием и трассировкой.

---

## Принцип работы (pipeline)

### Вход

```java
evaluate(String userText, String candidateText, List<Statement> context)
```

### Шаги

1. **Base evaluation**

    * Делегирование в `AdvancedCandidateEvaluator`
    * Получение классических метрик и `base.score`

2. **Channel extraction**

    * Нормализация сигналов в диапазон `[0..1]`

3. **Entropy calculation**

    * Оценка распределённости каналов

4. **Coherence calculation**

    * Оценка согласованности сигналов

5. **Quantum score aggregation**

    * Применение весов и штрафов

6. **Validity decision**

    * `valid = base.valid && coherence >= minCoherence`

7. **Telemetry assembly**

    * Добавление квантовых метрик в notes

8. **Debug logging (опционально)**

---

## Конфигурация

### Конструкторы

```java
new MultiCriteriaCandidateEvaluator(tokenizer)
```

Использует:

* стандартный `AdvancedCandidateEvaluator`
* дефолтные квантовые параметры

```java
new MultiCriteriaCandidateEvaluator(
    baseEvaluator,
    minCoherence,
    entropyPenaltyWeight,
    coherenceWeight
)
```

### Основные параметры

| Параметр               | Диапазон | Назначение                            |
| ---------------------- | -------: | ------------------------------------- |
| `minCoherence`         |     0..1 | минимальная согласованность для valid |
| `entropyPenaltyWeight` |     0..1 | штраф за хаос                         |
| `coherenceWeight`      |     0..1 | бонус за согласованность              |
| `debugEnabled`         |  boolean | расширенные debug-логи                |
| `debugSnippetChars`    |     ≥ 40 | длина snippet в логах                 |

---

## Логирование и отладка

### DEBUG (по умолчанию)

Логируется, если:

* ответ стал invalid после квантового слоя,
* coherence ниже порога,
* или `debugEnabled=true`.

Формат:

```
QuantumEval | ctx=... tok=... baseScore=... qScore=... eff=...
| g=... st=... cov=... act=... risk=... coh=... ent=...
```

### TRACE-подобный режим

Если `debugEnabled=true`, дополнительно логируется:

```
QuantumEvalTelemetry | q_f=...;q_s=...;...
```

---

## Ограничения

1. **Квантовость — концептуальная**

Это квантово-инспирированная модель, а не реальная квантовая математика.
Все операции классические и детерминированные.

2. **Actionability — эвристика**

`estimateActionability()` переиспользует:

* `structureScore`
* `coverage`

Без семантического анализа шагов/действий.

3. **Зависимость от base evaluator**

Если `AdvancedCandidateEvaluator` плохо настроен:

* квантовый слой не спасёт,
* он лишь переагрегирует сигналы.

4. **Фиксированный штраф риска**

Коэффициент `0.30 * risk` зашит.
Для более гибкой системы его стоит вынести в конфиг.

---

## Типовые проблемы

### Ответ стал invalid “неожиданно”

Причина:

* `coherence < minCoherence`

Решение:

* снизить `minCoherence`,
* проверить дисбаланс каналов (entropy).

### Ответы слишком жёстко отбрасываются

Причина:

* высокая `entropyPenaltyWeight`
* высокий `minCoherence`

Решение:

* уменьшить штрафы,
* включить debug-логи и посмотреть каналы.

### Много логов

Причина:

* `debugEnabled=true`

Решение:

* включать только в tuning / dev,
* использовать sampling логов.

---

## Точки роста

1. **Обучаемые веса каналов**

* хранить веса в config/LTM,
* адаптировать под домен.

2. **Настоящая actionability**

* поиск шагов, imperative форм,
* проверка наличия инструкций.

3. **Многоуровневая coherence**

* coherence между группами сигналов,
* отдельная логическая и структурная coherence.

4. **Pareto-оценка**

* не сводить всё в один score,
* использовать доминирование (Pareto front).

5. **Интеграция с verify-pass**

* второй проход валидации лучшего кандидата,
* repair-итерация при низкой coherence.

---

## Идеи улучшений

### Высокий приоритет

* Вынести коэффициенты (`0.30`, веса) в конфиг.
* Логировать распределения entropy/coherence.
* Связать coherence с repair-сигналами генератора.

### Средний приоритет

* Улучшить actionability-оценку.
* Добавить “domain-aware” каналы (например, code-correctness).

### Низкий приоритет

* Визуализация каналов (radar-chart).
* A/B-калибровка квантовых порогов.

---

## Рекомендации по использованию

* Используйте `MultiCriteriaCandidateEvaluator` после стабилизации `AdvancedCandidateEvaluator`.
* Начинайте с умеренных порогов (`minCoherence ~ 0.40–0.50`).
* Включайте `debugEnabled` только на тестовых прогонах.
* Для enterprise-продакшена храните параметры как данные, а не код.

---

## Ключевые места кода

* `evaluate()` — основной квантовый pipeline
* `entropy()` — мера хаоса
* `coherence()` — мера согласованности
* `estimateActionability()` — эвристика применимости
* `QuantumEvalTelemetry` — диагностическая строка
