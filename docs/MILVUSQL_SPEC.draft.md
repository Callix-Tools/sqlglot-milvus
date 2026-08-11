# MilvusQL — спецификация языка (ЧЕРНОВИК, Фаза 0)

> Статус: **черновик**. Разделы, помеченные `⚠ ОТКРЫТО`, ждут результатов API-разведки sqlglot 30.16
> (см. `SQLGLOT_API_GROUND_TRUTH.md`). Не реализовывать до снятия пометок.

## 0. Область действия MVP

Первый релиз `sqlglot-milvus` покрывает **только** то, что перечислено в §2 как `MVP`. Конструкции,
помеченные `POST-MVP`, должны в MVP-парсере давать понятный `ParseError`, а не молча искажать дерево.

| Конструкция | Статус |
|---|---|
| `CREATE TABLE` + `WITH (...)` | MVP |
| `CREATE INDEX ... USING <method> WITH (...)` | MVP |
| `LOAD TABLE` / `RELEASE TABLE` | MVP |
| `INSERT INTO ... VALUES (...)` | MVP |
| `DELETE FROM ... WHERE` | MVP |
| `SELECT ... ORDER BY <col> <dist-op> <param> LIMIT n` (ANN-поиск) | MVP |
| `SEARCH PARAMS (...)` | MVP |
| `DROP TABLE` / `DROP INDEX` | MVP |
| `HYBRID SEARCH (...) RERANK ...` | POST-MVP |
| `MATCH(...) AGAINST (...)` / `BM25_SCORE(...)` | POST-MVP |
| `ALTER TABLE ... ADD FIELD` | POST-MVP |

Обоснование: гибридный поиск и BM25 — это две самые синтаксически спорные конструкции (см. §5), и обе
требуют решений, которые дешевле принять, имея на руках работающий MVP, чем до него.

## 1. Лексика

### 1.1 Операторы дистанции

Наследуются от pgvector один в один — это делает транспиляцию `postgres → milvus` почти бесплатной
и снимает с пользователя, мигрирующего с pgvector, необходимость переучиваться.

| Оператор | Метрика | Milvus `metric_type` |
|---|---|---|
| `<->` | L2 / евклидова | `L2` |
| `<#>` | inner product (отрицательное) | `IP` |
| `<=>` | косинусная дистанция | `COSINE` |
| `<+>` | L1 / манхэттенская | `L1` (если поддерживается бэкендом) |

⚠ ОТКРЫТО (лексика): в базовом токенайзере sqlglot 30.16 из четырёх операторов «из коробки» одним
токеном разбираются только два — `<->` → `LR_ARROW` и `<=>` → `NULLSAFE_EQ`. `<#>` распадается на
`LT` + `HASH_ARROW`, `<+>` — на `LT` + `PLUS` + `GT`. Требуется расширение токенайзера; механизм
уточняется разведкой.

⚠ ОТКРЫТО (семантика `<=>`): токен `NULLSAFE_EQ` в базовом парсере уже связан с null-safe равенством
(MySQL-семантика). Нужно решить, перепривязываем ли мы его в нашем диалекте к дистанции — и что при
этом происходит с транспиляцией в другие диалекты.

### 1.2 Bind-параметры

Единственная поддерживаемая форма в MVP — **именованная**: `:name`.

Это соответствует `paramstyle = "named"` из трека B и означает, что вектор **никогда не попадает в
текст запроса**: в AST на его месте стоит placeholder-узел, а само значение едет отдельным словарём
в `Cursor.execute(sql, params)`.

⚠ ОТКРЫТО: точный класс AST-узла (`exp.Placeholder` / `exp.Parameter`) и необходимость opt-in
на уровне диалекта.

### 1.3 Идентификаторы и регистр

- Кавычки идентификаторов: двойные (`"my table"`), как в стандартном SQL.
- Нормализация регистра: **не применяется** — Milvus чувствителен к регистру имён коллекций и полей,
  поэтому `items` и `Items` — разные объекты. Стратегия нормализации диалекта должна это отражать.
- Строковые литералы — одинарные кавычки.

## 2. Грамматика конструкций

### 2.1 CREATE TABLE

```
CREATE TABLE [IF NOT EXISTS] <name> (
    <column> <type> [PRIMARY KEY] [AUTO_INCREMENT] [NOT NULL] ...
    [, ...]
) [WITH ( <property> [, ...] )]
```

Типы полей:

| MilvusQL | Milvus |
|---|---|
| `BIGINT` | `Int64` |
| `INT` / `INTEGER` | `Int32` |
| `SMALLINT` / `TINYINT` | `Int16` / `Int8` |
| `FLOAT` / `DOUBLE` | `Float` / `Double` |
| `BOOLEAN` | `Bool` |
| `VARCHAR(n)` | `VarChar(max_length=n)` |
| `JSON` | `JSON` |
| `ARRAY<T>(n)` | `Array` |
| `VECTOR(d)` | `FloatVector(dim=d)` |
| `SPARSEVEC` | `SparseFloatVector` |
| `BINARYVEC(d)` | `BinaryVector(dim=d)` |
| `FLOAT16VEC(d)` / `BFLOAT16VEC(d)` | `Float16Vector` / `BFloat16Vector` |

Свойства `WITH (...)`: `shards`, `consistency_level`, `partition_key`, `enable_dynamic_field`, `ttl_seconds`.

⚠ ОТКРЫТО: как представить `VECTOR(768)` — как пользовательский тип, как `exp.DataType` с
неизвестным типом, или зарегистрировать собственное имя типа. Влияет на round-trip.

### 2.2 CREATE INDEX

```
CREATE INDEX [IF NOT EXISTS] <name> ON <table> ( <column> )
    USING <method>
    [WITH ( <property> [, ...] )]
```

`<method>`: `HNSW`, `IVF_FLAT`, `IVF_SQ8`, `IVF_PQ`, `DISKANN`, `AUTOINDEX`, `SPARSE_INVERTED_INDEX`,
`BIN_FLAT`, `GPU_CAGRA`, `FLAT`.

Свойства: `metric_type` (`L2`/`IP`/`COSINE`/`HAMMING`/`JACCARD`), `M`, `ef_construction`, `nlist`, `m`, `nbits`.

Совместимость: постгресовый `USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64)`
разбирается штатным парсером `postgres` (это стандартный `USING <access method>`, существовавший
задолго до pgvector), поэтому транспиляция pgvector-индексов достаётся почти бесплатно —
нужно лишь смапить класс операторов (`vector_l2_ops` → `metric_type='L2'`).

### 2.3 LOAD / RELEASE

```
LOAD TABLE <name> [WITH ( replicas = <n> )]
RELEASE TABLE <name>
```

Семантика (загрузка коллекции в память перед поиском) живёт **не здесь**, а в треке B — грамматика
только описывает синтаксис.

### 2.4 ANN-поиск

```
SELECT <projection>
FROM <table>
[WHERE <filter>]
ORDER BY <vector_column> <dist-op> <param> [ASC]
LIMIT <n>
[OFFSET <n>]
[SEARCH PARAMS ( <property> [, ...] )]
```

⚠ ОТКРЫТО (позиция `SEARCH PARAMS`): предложенное размещение — **после** `LIMIT`. Нужно проверить,
допускает ли цикл модификаторов запроса в sqlglot модификатор после `LIMIT`; если нет — либо
переносим блок перед `LIMIT`, либо пишем собственный цикл.

Свойства: `ef_search` (он же `ef`), `nprobe`, `search_k`, `radius`, `range_filter`, `consistency_level`.

### 2.5 DROP

```
DROP TABLE [IF EXISTS] <name>
DROP INDEX [IF EXISTS] <name> ON <table>
```

## 3. POST-MVP конструкции

### 3.1 Гибридный поиск

```
SELECT <projection>
FROM <table>
HYBRID SEARCH ( <arm> [, ...] )
RERANK <strategy>
LIMIT <n>
```

где `<arm>` ::= `<vector_column> <dist-op> <param> [WEIGHT <number>]`,
`<strategy>` ::= `RRF(<n>)` | `WEIGHTED` | имя стратегии.

⚠ ОТКРЫТО (ловушка `WEIGHT`): голое слово после выражения в SQL — это неявный алиас
(`expr alias` без `AS`). Есть риск, что `embedding <=> :dv WEIGHT 0.7` разберётся как
«выражение с алиасом WEIGHT», а `0.7` останется висеть. Требуется проверка и, возможно,
смена синтаксиса на скобочный (`WEIGHT(0.7)`) или на `WITH (weight=0.7)`.

⚠ ОТКРЫТО (ловушка `RRF(k=60)`): `k=60` внутри вызова функции — не валидный SQL-синтаксис
аргументов. Кандидаты на замену: `RRF(60)` (позиционный), `RRF(k => 60)` (kwargs-стрелка, как в
Snowflake/BigQuery) или `RERANK RRF WITH (k=60)` (переиспользование общего механизма свойств —
предпочтительно, т.к. не требует нового синтаксиса вообще).

### 3.2 Полнотекстовый поиск

```
SELECT <projection> FROM <table>
WHERE MATCH ( <column> ) AGAINST ( <param> )
ORDER BY BM25_SCORE ( <column>, <param> ) DESC
LIMIT <n>
```

⚠ ОТКРЫТО: возможно, переиспользуем готовый узел `MATCH ... AGAINST` из диалекта MySQL вместо
собственного.

### 3.3 ALTER TABLE

```
ALTER TABLE <name> ADD FIELD <column> <type>
```

Только добавление поля. Смена типа, размерности вектора и удаление поля Milvus не поддерживает —
парсер обязан отвергать их явно и с понятным сообщением, а не притворяться, что операция состоялась.

## 4. Требования к round-trip

Для каждой конструкции: `parse_one(sql, read="milvus").sql(dialect="milvus")` должен давать SQL,
который при повторном разборе даёт **идентичное** дерево (сравнение по AST, не по строке — различия
в пробелах и регистре ключевых слов допустимы).

## 5. Принятые принципы

1. **`TABLE`, а не `COLLECTION`.** Стандартное SQL-слово даёт бесплатный парсинг `CREATE TABLE`,
   `DROP TABLE` и т.д. Термин «коллекция» остаётся уровнем ниже — в вызовах `pymilvus` (трек B).
2. **Векторы — только через bind-параметры.** Ни один эмбеддинг не должен оказаться литералом в тексте.
3. **Минимум новых AST-узлов.** Всё, что выражается через `exp.Properties`, `exp.Create`, штатные
   механизмы — выражается через них.
4. **Никакой тихой деградации.** Неподдерживаемая конструкция — это `ParseError` с позицией либо
   явный `exp.Command`, но не молча испорченное дерево.

## 6. Открытые вопросы к разрешению перед реализацией

Сводка всех `⚠ ОТКРЫТО` выше:

1. Механизм добавления токенов `<#>` и `<+>`.
2. Перепривязка `<=>` с `NULLSAFE_EQ` на дистанцию — безопасно ли.
3. Класс AST-узла для `:name`.
4. Представление типа `VECTOR(768)`.
5. Допустимая позиция `SEARCH PARAMS` относительно `LIMIT`.
6. Синтаксис `WEIGHT` (ловушка неявного алиаса).
7. Синтаксис параметров реранкера.
8. Переиспользование `MATCH ... AGAINST` из MySQL.
