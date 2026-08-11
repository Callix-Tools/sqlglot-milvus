# sqlglot-milvus

**MilvusQL** — a SQL dialect for [Milvus](https://milvus.io), implemented as a
[sqlglot](https://github.com/tobymao/sqlglot) plugin.

Milvus speaks gRPC, not SQL. This package gives it a SQL *surface*: it parses and generates a
SQL-like language covering collections, indexes, vector search, hybrid search and full-text search.
It performs no I/O and holds no connection — it is a pure parser/generator. Executing MilvusQL
against a real server is the job of the companion package `sqlalchemy-milvus`, which uses this one
as its parser and translates the resulting AST into `pymilvus` calls.

```python
import sqlglot

sqlglot.parse_one(
    "SELECT id, category FROM items "
    "WHERE category = :cat "
    "ORDER BY embedding <-> :q "
    "LIMIT 10 SEARCH PARAMS (ef_search=64)",
    read="milvus",
)
```

Installing the package registers the dialect; no explicit import is needed.

## Installation

```bash
pip install sqlglot-milvus
```

> [!IMPORTANT]
> **Do not install `sqlglot[c]` or `sqlglot[rs]`.** Those builds are mypyc-compiled, and mypyc
> disables runtime subclassing of the compiled classes — which every third-party sqlglot dialect
> is built on. `sqlglot-milvus` detects this at import time and raises `ImportError` with
> remediation steps, because the alternative is a `TypeError` from deep inside sqlglot on your
> first `parse_one`.

## Language

| Construct | Syntax |
|---|---|
| Create collection | `CREATE TABLE items (id BIGINT PRIMARY KEY AUTO_INCREMENT, embedding VECTOR(768), category VARCHAR(64)) WITH (shards=2, consistency_level='Bounded', partition_key=category)` |
| Create index | `CREATE INDEX idx_emb ON items (embedding) USING HNSW WITH (metric_type='COSINE', M=16, ef_construction=200)` |
| Load / release | `LOAD TABLE items WITH (replicas=2)` · `RELEASE TABLE items` |
| Insert | `INSERT INTO items (embedding, category) VALUES (:emb, :cat)` |
| Delete | `DELETE FROM items WHERE category = :cat` |
| Vector search | `SELECT id FROM items WHERE category = :cat ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)` |
| Hybrid search | `SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7, sparse_emb <#> :sv WEIGHT 0.3) RERANK RRF(k=60) LIMIT 10` |
| Full-text | `SELECT id FROM items WHERE MATCH(text) AGAINST (:q) ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10` |
| Consistency level | `SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10 CONSISTENCY LEVEL Bounded` |
| Add field | `ALTER TABLE items ADD FIELD tag VARCHAR(32)` |

The entity is called `TABLE`, not `COLLECTION`. On the SQL surface that is the ordinary SQL word,
so `CREATE TABLE` / `DROP TABLE` parse for free; "collection" remains the term one level down, in
the `pymilvus` calls the AST is translated into.

Full grammar, including the constructs above, canonical clause ordering and every documented
tradeoff, is in [`docs/MILVUSQL_SPEC.md`](docs/MILVUSQL_SPEC.md).

`ADD FIELD` and `RENAME TO` are the only supported `ALTER TABLE` actions. Milvus cannot change a
field's type, change a vector's dimension or drop a field, so `DROP`/`ALTER COLUMN ... TYPE`/
`MODIFY` are rejected with a `ParseError` explaining that the collection needs recreating — never
silently accepted or degraded into an opaque, unreadable node.

### Distance operators

Spelled exactly as pgvector spells them, so a pgvector query needs no rewriting:

| Operator | Metric | Milvus `metric_type` |
|---|---|---|
| `<->` | L2 / Euclidean | `L2` |
| `<#>` | inner product | `IP` |
| `<=>` | cosine | `COSINE` |
| `<+>` | L1 / Manhattan | `L1` |

### Bind parameters

MilvusQL is written with named parameters: `:name`. Vectors travel as bind parameters and are
**never** interpolated into the query text — a 768-float embedding is tens of kilobytes of SQL, and
round-tripping the numbers through text risks precision loss.

`:name` is the only spelling the language defines, but it is not the only one the *parser* accepts:
sqlglot's anonymous `?` (`exp.Placeholder` with no name) and `@name` (`exp.Parameter`) both parse
and are re-emitted verbatim, because refusing them would make several `read="postgres"` migrations
unparseable one hop before anyone would look. Track B's cursor uses `paramstyle="named"`, so **a `?`
in generated MilvusQL can never be bound** — see the rewrite step under *Migrating from pgvector*.

### Clause order is strict

Canonical order is `HYBRID SEARCH ... LIMIT [OFFSET] SEARCH PARAMS ... CONSISTENCY LEVEL ...`.
`HYBRID SEARCH` must precede `LIMIT`/`OFFSET`; `SEARCH PARAMS` and `CONSISTENCY LEVEL` must follow
them; `SEARCH PARAMS` must precede `CONSISTENCY LEVEL`. Non-canonical order is a `ParseError`, not a
silent reordering:

```
SELECT id FROM items SEARCH PARAMS (ef=64) LIMIT 10
                     ^ ParseError: SEARCH PARAMS must follow LIMIT
```

sqlglot's modifier loop has no positional state and would otherwise accept the above and emit it
back reordered — accepted input, different output text. The rule applies inside a parenthesized
subquery or a `UNION` branch too, not only at the top of a statement.

## Migrating from pgvector

```python
sqlglot.transpile(
    "SELECT id FROM items ORDER BY embedding <-> %(q)s LIMIT 5",
    read="postgres", write="milvus",
)
# ['SELECT id FROM items ORDER BY embedding <-> :q LIMIT 5']
```

Use psycopg's **`pyformat`** (`%(name)s`) source queries where you can: those carry a name and land
directly on MilvusQL's `:name`. The positional `format` style (`%s`) transpiles to `?`, which is
valid syntax and permanently unbindable under `paramstyle="named"` — give each parameter a name
before or after the transpile:

```python
sqlglot.transpile(
    "SELECT id FROM items ORDER BY embedding <-> %s LIMIT 5",
    read="postgres", write="milvus",
)
# ['SELECT id FROM items ORDER BY embedding <-> ? LIMIT 5']   <- rewrite the ? to :q
```

`CREATE INDEX` is accepted in **both** word orders — MilvusQL's `ON items (embedding) USING HNSW`
and Postgres/pgvector's `ON items USING hnsw (embedding)` — so index DDL transpiles without
rewriting.

> [!WARNING]
> **`<=>` collides with MySQL.** MySQL spells null-safe equality `<=>`; MilvusQL spells cosine
> distance the same way. The collision is at the level of the characters themselves and cannot be
> resolved — pgvector parity was judged the more valuable property. It is contained rather than
> hidden: the dialect does not inherit from MySQL, and generating MilvusQL from a MySQL
> `exp.NullSafeEQ` reports an unsupported-operation error instead of emitting `<=>`. Transpile with
> `unsupported_level=ErrorLevel.RAISE` and `mysql -> milvus` fails loudly rather than turning a
> comparison into a vector search.

## Reserved words

`RELEASE`, `HYBRID SEARCH`, `SEARCH PARAMS` and `CONSISTENCY LEVEL` are the only additions to
sqlglot's base keyword set. The three multi-word ones are registered as *pairs*, so `hybrid`,
`search`, `params`, `consistency` and `level` all remain usable as ordinary identifiers, aliases and
placeholders. `RELEASE` is single-word and therefore genuinely reserved by the tokenizer, but the
parser adds it back everywhere identifiers are accepted — `SELECT release FROM t` works.

Known casualty: a table literally named `search` with an alias `params` (`FROM search params`) is
lexed as the `SEARCH PARAMS` keyword. Quote it (`FROM "search" params`) to recover.

## Only generate with `dialect="milvus"`

`ast.sql()` and `ast.sql(dialect="postgres")` on a query that carries `HYBRID SEARCH` or
`SEARCH PARAMS` **silently drop the clause** and return a syntactically valid, semantically wrong
full-table-scan query — with no error, not even under `unsupported_level=ErrorLevel.RAISE`. This
is not a bug we can fix from this package: those clauses live as extra keys on `Select.args`, and a
foreign `Generator` (postgres's, the default one) simply never looks at keys it doesn't know about,
so there is no unknown node to raise on — the modifier is just never visited. Track B (and any other
caller) must always pass `dialect="milvus"` explicitly when calling `.sql()`.

## Development

```bash
uv venv && uv pip install -e '.[dev]'
pytest
```

## License

MIT
