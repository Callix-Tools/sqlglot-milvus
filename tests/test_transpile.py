"""Cross-dialect transpilation: ``postgres <-> milvus``, ``mysql -> milvus``, ``milvus -> *``.

The headline use case for this dialect is migrating pgvector users, so the ``postgres -> milvus``
cases below are lifted from real pgvector-python query shapes rather than invented. Every case
asserts the **exact output text** and then re-parses that output under ``milvus``, because a
transpile that produces plausible-looking garbage is indistinguishable from a working one until
someone reads it.

Three asymmetries are pinned here deliberately, each with its own test:

* pgvector's ``<#>`` and ``<+>`` do not parse under ``read="postgres"`` at all, and ``<=>`` parses
  as MySQL-style null-safe equality. Only ``<->`` survives the postgres front end.
* ``milvus -> postgres`` silently **drops** ``HYBRID SEARCH`` / ``SEARCH PARAMS``.
* MilvusQL-only statements raise a bare ``ValueError`` on foreign generators -- not a
  ``SqlglotError``, so the obvious ``except`` clause misses it.

Pinning them is the point: they are the things that would otherwise be discovered in production.
"""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, SqlglotError, UnsupportedError

from sqlglot_milvus import CosineDistance, InnerProduct, L1Distance

# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------
#
# Both error levels are pinned to RAISE everywhere. sqlglot's defaults are WARN for generation and
# a lenient level for parsing, under which a botched transpile returns a string instead of failing.


def to_milvus(sql: str, read: str) -> str:
    ast = sqlglot.parse_one(sql, read=read, error_level=ErrorLevel.RAISE)
    return ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)


def assert_is_real_milvus(sql: str, expected_type: type) -> exp.Expression:
    """Re-parse ``sql`` as MilvusQL and prove it is a structured AST, not an opaque blob.

    ``exp.Command`` is the trap: sqlglot degrades anything it cannot parse into a node that stores
    the raw source text, so it re-renders byte-identically while containing no columns, no tables
    and no operators. Asserting the node type is what separates "we parsed it" from "we kept it".
    """
    ast = sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)
    assert not isinstance(ast, exp.Command), f"degraded to Command: {sql!r}"
    assert isinstance(ast, expected_type)
    assert ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE) == sql
    return ast


# ------------------------------------------------------------------------------------------
# 1. postgres -> milvus: real pgvector query shapes
# ------------------------------------------------------------------------------------------

PGVECTOR_CASES = [
    # --- the canonical pgvector-python nearest-neighbour queries -----------------------------
    (
        "l2-literal-vector",
        "SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5",
        "SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5",
        exp.Select,
    ),
    (
        "l2-psycopg-placeholder",
        "SELECT * FROM items ORDER BY embedding <-> %s LIMIT 5",
        "SELECT * FROM items ORDER BY embedding <-> ? LIMIT 5",
        exp.Select,
    ),
    (
        "distance-in-select-list",
        "SELECT id, embedding <-> %s AS distance FROM items ORDER BY distance LIMIT 5",
        "SELECT id, embedding <-> ? AS distance FROM items ORDER BY distance LIMIT 5",
        exp.Select,
    ),
    (
        "filtered-search",
        "SELECT * FROM items WHERE category_id = 123 ORDER BY embedding <-> %s LIMIT 5",
        "SELECT * FROM items WHERE category_id = 123 ORDER BY embedding <-> ? LIMIT 5",
        exp.Select,
    ),
    (
        # pgvector's "distance threshold" recipe. Interesting because it pins operator precedence:
        # <-> must bind tighter than <, i.e. (embedding <-> ?) < 5, in both dialects.
        "distance-threshold",
        "SELECT * FROM items WHERE embedding <-> %s < 5",
        "SELECT * FROM items WHERE embedding <-> ? < 5",
        exp.Select,
    ),
    (
        "paginated-search",
        "SELECT * FROM items ORDER BY embedding <-> %s LIMIT 5 OFFSET 10",
        "SELECT * FROM items ORDER BY embedding <-> ? LIMIT 5 OFFSET 10",
        exp.Select,
    ),
    (
        # "find items similar to item 1" -- the query vector is itself a scalar subquery.
        "subquery-as-query-vector",
        "SELECT id FROM items WHERE id != 1 "
        "ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5",
        "SELECT id FROM items WHERE id <> 1 "
        "ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5",
        exp.Select,
    ),
    (
        "derived-table",
        "SELECT * FROM (SELECT * FROM items ORDER BY embedding <-> %s LIMIT 5) AS sub "
        "WHERE category = 'a'",
        "SELECT * FROM (SELECT * FROM items ORDER BY embedding <-> ? LIMIT 5) AS sub "
        "WHERE category = 'a'",
        exp.Select,
    ),
    (
        "cte",
        "WITH nearest AS (SELECT id, embedding <-> %s AS d FROM items ORDER BY d LIMIT 5) "
        "SELECT * FROM nearest",
        "WITH nearest AS (SELECT id, embedding <-> ? AS d FROM items ORDER BY d LIMIT 5) "
        "SELECT * FROM nearest",
        exp.Select,
    ),
    (
        "aggregate-over-vectors",
        "SELECT category_id, AVG(embedding) FROM items GROUP BY category_id",
        "SELECT category_id, AVG(embedding) FROM items GROUP BY category_id",
        exp.Select,
    ),
    # --- DDL ---------------------------------------------------------------------------------
    (
        # D6 in action: pgvector writes USING before the column list, MilvusQL after it.
        "hnsw-index-unnamed",
        "CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)",
        "CREATE INDEX ON items (embedding vector_l2_ops) USING HNSW",
        exp.Create,
    ),
    (
        "hnsw-index-with-build-params",
        "CREATE INDEX idx_emb ON items USING hnsw (embedding vector_l2_ops) "
        "WITH (m = 16, ef_construction = 64)",
        "CREATE INDEX idx_emb ON items (embedding vector_l2_ops) USING HNSW "
        "WITH (m=16, ef_construction=64)",
        exp.Create,
    ),
    (
        "ivfflat-index",
        "CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)",
        "CREATE INDEX ON items (embedding vector_l2_ops) USING IVFFLAT WITH (lists=100)",
        exp.Create,
    ),
    (
        "create-table-with-vector-column",
        "CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(1536))",
        "CREATE TABLE items (id BIGSERIAL PRIMARY KEY, embedding VECTOR(1536))",
        exp.Create,
    ),
    # --- DML ---------------------------------------------------------------------------------
    (
        "insert",
        "INSERT INTO items (embedding, category) VALUES (%s, %s)",
        "INSERT INTO items (embedding, category) VALUES (?, ?)",
        exp.Insert,
    ),
    (
        "delete",
        "DELETE FROM items WHERE id = %s",
        "DELETE FROM items WHERE id = ?",
        exp.Delete,
    ),
    (
        # pgvector tunes recall with a GUC. It survives as a Set statement rather than turning into
        # MilvusQL's SEARCH PARAMS (ef_search=...) -- structurally the same knob, different clause,
        # and nothing in sqlglot bridges the two. Migrations must rewrite this by hand.
        "session-level-ef-search",
        "SET hnsw.ef_search = 100",
        "SET hnsw.ef_search = 100",
        exp.Set,
    ),
]


@pytest.mark.parametrize(
    "postgres_sql, milvus_sql, node_type",
    [case[1:] for case in PGVECTOR_CASES],
    ids=[case[0] for case in PGVECTOR_CASES],
)
def test_pgvector_transpiles_to_milvus(postgres_sql, milvus_sql, node_type):
    assert to_milvus(postgres_sql, read="postgres") == milvus_sql
    assert_is_real_milvus(milvus_sql, node_type)


@pytest.mark.parametrize(
    "postgres_sql, milvus_sql",
    [
        # psycopg "format" paramstyle -- by far the most common in pgvector-python examples.
        # It carries no name, so it lands on the anonymous exp.Placeholder and comes out as "?".
        ("SELECT x FROM t WHERE c = %s", "SELECT x FROM t WHERE c = ?"),
        # psycopg "pyformat" paramstyle: named, and maps straight onto MilvusQL's :name.
        ("SELECT x FROM t WHERE c = %(emb)s", "SELECT x FROM t WHERE c = :emb"),
        # asyncpg's numeric style is NOT a Placeholder in sqlglot -- postgres parses $1 as
        # exp.Parameter, which the milvus generator renders with the @ sigil. It re-parses as a
        # Parameter under milvus, so the round trip is lossless, but the *spelling* changes.
        ("SELECT x FROM t WHERE c = $1", "SELECT x FROM t WHERE c = @1"),
    ],
    ids=["format-%s", "pyformat-%(name)s", "numeric-$1"],
)
def test_pgvector_placeholder_styles(postgres_sql, milvus_sql):
    assert to_milvus(postgres_sql, read="postgres") == milvus_sql
    assert_is_real_milvus(milvus_sql, exp.Select)
    # ...and back again, unchanged: the paramstyle mapping is a bijection in both directions.
    ast = sqlglot.parse_one(milvus_sql, read="milvus", error_level=ErrorLevel.RAISE)
    assert ast.sql(dialect="postgres", unsupported_level=ErrorLevel.RAISE) == postgres_sql


def test_transpiled_index_keeps_a_structured_ast():
    """The index cases above are exactly where a Command blob would hide.

    ``CREATE INDEX ... USING HNSW`` is unusual enough that a dialect which never implemented it
    would still "pass" the text assertion. Look inside instead.
    """
    out = to_milvus(
        "CREATE INDEX idx_emb ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16)",
        read="postgres",
    )
    ast = assert_is_real_milvus(out, exp.Create)

    params = ast.find(exp.IndexParameters)
    assert params is not None
    assert params.args["using"].name == "HNSW"
    assert [c.name for c in ast.find_all(exp.Column)] == ["embedding"]


def test_pgvector_opclass_is_carried_through_verbatim():
    """FINDING: ``vector_l2_ops`` survives into MilvusQL output as a meaningless operator class.

    pgvector encodes the metric in the opclass; Milvus encodes it in ``metric_type`` inside the
    ``WITH`` list. sqlglot preserves the ``exp.Opclass`` node faithfully, which means the transpiled
    DDL is *syntactically* fine and *semantically* silent about the metric. Nothing in this package
    maps ``vector_l2_ops -> metric_type='L2'``, so a migration tool has to do it.
    """
    out = to_milvus(
        "CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)", read="postgres"
    )
    assert out == "CREATE INDEX ON items (embedding vector_cosine_ops) USING HNSW"

    ast = assert_is_real_milvus(out, exp.Create)
    assert ast.find(exp.Opclass) is not None
    assert "metric_type" not in out


# ------------------------------------------------------------------------------------------
# 2. D6 -- both index word orders parse, output normalizes to MilvusQL order
# ------------------------------------------------------------------------------------------

MILVUS_ORDER = (
    "CREATE INDEX idx_emb ON items (embedding) USING HNSW WITH (metric_type='COSINE', M=16)"
)
PGVECTOR_ORDER = (
    "CREATE INDEX idx_emb ON items USING HNSW (embedding) WITH (metric_type='COSINE', M=16)"
)


@pytest.mark.parametrize(
    "sql", [MILVUS_ORDER, PGVECTOR_ORDER], ids=["milvus-order", "pgvector-order"]
)
def test_both_index_orders_normalize_to_milvus_order(sql):
    # Without D6 the pgvector order degrades to exp.Command -- which re-renders byte-identically,
    # so the text assertion below would pass on its own. The type assertions are what test D6.
    ast = sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)
    assert not isinstance(ast, exp.Command)
    assert isinstance(ast, exp.Create)
    assert [c.name for c in ast.find_all(exp.Column)] == ["embedding"]
    assert ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE) == MILVUS_ORDER


def test_both_index_orders_produce_the_same_ast():
    # Not just the same text: the same tree. Accepting pgvector's order is only useful if the
    # result is indistinguishable from natively-written MilvusQL downstream.
    assert repr(sqlglot.parse_one(MILVUS_ORDER, read="milvus")) == repr(
        sqlglot.parse_one(PGVECTOR_ORDER, read="milvus")
    )


def test_index_method_is_uppercased_on_output_not_on_parse():
    # The generator upper()s the method name, so lowercase "hnsw" input still yields "USING HNSW".
    # The parsed AST keeps the source spelling, which is why AST comparisons must use one casing.
    lower = sqlglot.parse_one("CREATE INDEX i ON t USING hnsw (e)", read="milvus")
    upper = sqlglot.parse_one("CREATE INDEX i ON t USING HNSW (e)", read="milvus")
    normalized = "CREATE INDEX i ON t (e) USING HNSW"
    assert lower.sql(dialect="milvus") == upper.sql(dialect="milvus") == normalized
    assert repr(lower) != repr(upper)


# ------------------------------------------------------------------------------------------
# 3. D3 -- the <=> collision between MySQL null-safe equality and cosine distance
# ------------------------------------------------------------------------------------------


def test_mysql_null_safe_eq_raises_when_written_to_milvus():
    with pytest.raises(UnsupportedError, match="cosine distance"):
        sqlglot.transpile(
            "SELECT * FROM t WHERE a <=> b",
            read="mysql",
            write="milvus",
            unsupported_level=ErrorLevel.RAISE,
        )


def test_mysql_null_safe_eq_never_emits_the_operator_at_the_default_level():
    # transpile() defaults to ErrorLevel.WARN, i.e. it logs and keeps going. What matters is that
    # the fallback is unambiguous: emitting "<=>" here would turn a null check into a vector search
    # with no diagnostic anywhere.
    (out,) = sqlglot.transpile("SELECT * FROM t WHERE a <=> b", read="mysql", write="milvus")
    assert out == "SELECT * FROM t WHERE a IS NOT DISTINCT FROM b"
    assert "<=>" not in out


def test_the_two_dialects_genuinely_disagree_about_the_operator():
    """The accepted cost of D3, stated as an assertion so nobody "fixes" it by accident."""
    assert isinstance(sqlglot.parse_one("SELECT a <=> b", read="milvus").selects[0], CosineDistance)
    assert isinstance(sqlglot.parse_one("SELECT a <=> b", read="mysql").selects[0], exp.NullSafeEQ)


def test_cosine_distance_has_no_rendering_in_foreign_dialects():
    """The reverse direction of the same collision, and it fails *loudly* -- but with a ValueError.

    sqlglot raises ``ValueError`` for a node with no TRANSFORMS entry, and ``ValueError`` is not a
    ``SqlglotError``: ``except SqlglotError`` around a transpile will not catch this.
    """
    ast = sqlglot.parse_one("SELECT a <=> b", read="milvus")
    with pytest.raises(ValueError) as excinfo:
        ast.sql(dialect="mysql", unsupported_level=ErrorLevel.RAISE)
    assert not isinstance(excinfo.value, SqlglotError)


# ------------------------------------------------------------------------------------------
# 4. FINDING -- the postgres front end cannot read most of pgvector's operators
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operator",
    ["<#>", "<+>"],
    ids=["inner-product", "l1"],
)
def test_postgres_cannot_parse_pgvector_ip_and_l1_operators(operator):
    """FINDING: only ``<->`` makes it through ``read="postgres"``.

    sqlglot's postgres tokenizer has no pgvector support: ``<#>`` collides with the JSONB path
    operator ``#>`` and ``<+>`` shatters into ``<`` ``+`` ``>``. So the two operators are not a
    transpilation problem, they are a *parse* problem, one hop earlier than anyone would look.
    """
    with pytest.raises(ParseError):
        sqlglot.parse_one(
            f"SELECT * FROM items ORDER BY embedding {operator} %s LIMIT 5",
            read="postgres",
            error_level=ErrorLevel.RAISE,
        )


def test_postgres_reads_pgvector_cosine_distance_as_null_safe_equality():
    """FINDING: the single most common pgvector query cannot be transpiled from postgres.

    ``ORDER BY embedding <=> %s`` is *the* cosine-similarity query (it is what every OpenAI
    embedding tutorial uses). sqlglot's postgres dialect parses ``<=>`` as MySQL-style null-safe
    equality, so by the time the milvus generator sees it the vector search is already gone. The
    D3 guard turns that into a hard error under RAISE -- but at the default WARN level it emits
    plausible, silently wrong SQL.
    """
    ast = sqlglot.parse_one(
        "SELECT * FROM items ORDER BY embedding <=> %s LIMIT 5",
        read="postgres",
        error_level=ErrorLevel.RAISE,
    )
    assert isinstance(ast.args["order"].expressions[0].this, exp.NullSafeEQ)

    with pytest.raises(UnsupportedError):
        ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)

    assert ast.sql(dialect="milvus") == (
        "SELECT * FROM items ORDER BY embedding IS NOT DISTINCT FROM ? LIMIT 5"
    )


@pytest.mark.parametrize(
    "operator, node_type",
    [("<->", exp.Distance), ("<=>", CosineDistance), ("<#>", InnerProduct), ("<+>", L1Distance)],
    ids=["l2", "cosine", "inner-product", "l1"],
)
def test_milvus_reader_handles_raw_pgvector_text_for_all_four_operators(operator, node_type):
    """The workaround for the two findings above: read pgvector SQL with ``read="milvus"``.

    MilvusQL borrows pgvector's operator spellings exactly, so pgvector source text parses directly
    -- and correctly, including the two operators postgres chokes on and the one it mis-reads.
    """
    sql = f"SELECT * FROM items ORDER BY embedding {operator} '[3,1,2]' LIMIT 5"
    ast = assert_is_real_milvus(sql, exp.Select)
    assert isinstance(ast.args["order"].expressions[0].this, node_type)


# ------------------------------------------------------------------------------------------
# 5. milvus -> postgres / duckdb, shared subset
# ------------------------------------------------------------------------------------------

SHARED_SUBSET = [
    (
        "select-where-limit",
        "SELECT id, category FROM items WHERE category = :cat LIMIT 10",
        "SELECT id, category FROM items WHERE category = %(cat)s LIMIT 10",
        "SELECT id, category FROM items WHERE category = $cat LIMIT 10",
    ),
    (
        "group-by-having",
        "SELECT category, COUNT(*) AS n FROM items WHERE price > 10 "
        "GROUP BY category HAVING COUNT(*) > 2 LIMIT 5",
        "SELECT category, COUNT(*) AS n FROM items WHERE price > 10 "
        "GROUP BY category HAVING COUNT(*) > 2 LIMIT 5",
        "SELECT category, COUNT(*) AS n FROM items WHERE price > 10 "
        "GROUP BY category HAVING COUNT(*) > 2 LIMIT 5",
    ),
    (
        "join",
        "SELECT a.id, b.name FROM items AS a JOIN cats AS b ON a.cat_id = b.id WHERE b.name = :n",
        "SELECT a.id, b.name FROM items AS a JOIN cats AS b ON a.cat_id = b.id "
        "WHERE b.name = %(n)s",
        "SELECT a.id, b.name FROM items AS a JOIN cats AS b ON a.cat_id = b.id WHERE b.name = $n",
    ),
    (
        # exp.Distance is shared vocabulary, so an L2 search survives the trip to postgres intact.
        "l2-search",
        "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10",
        "SELECT id FROM items ORDER BY embedding <-> %(q)s LIMIT 10",
        "SELECT id FROM items ORDER BY embedding <-> $q LIMIT 10",
    ),
    (
        # Milvus declares NULL_ORDERING = "nulls_are_last"; postgres sorts NULLs first on DESC. The
        # explicit NULLS LAST is sqlglot preserving our semantics, not noise.
        "order-by-desc-null-ordering",
        "SELECT id FROM items ORDER BY score DESC LIMIT 5",
        "SELECT id FROM items ORDER BY score DESC NULLS LAST LIMIT 5",
        "SELECT id FROM items ORDER BY score DESC LIMIT 5",
    ),
]


@pytest.mark.parametrize(
    "milvus_sql, postgres_sql, duckdb_sql",
    [case[1:] for case in SHARED_SUBSET],
    ids=[case[0] for case in SHARED_SUBSET],
)
def test_milvus_transpiles_to_postgres_and_duckdb(milvus_sql, postgres_sql, duckdb_sql):
    ast = sqlglot.parse_one(milvus_sql, read="milvus", error_level=ErrorLevel.RAISE)
    assert ast.sql(dialect="postgres", unsupported_level=ErrorLevel.RAISE) == postgres_sql
    assert ast.sql(dialect="duckdb", unsupported_level=ErrorLevel.RAISE) == duckdb_sql

    # Sanity: the emitted SQL is readable by the dialect it claims to be written in.
    for sql, dialect in ((postgres_sql, "postgres"), (duckdb_sql, "duckdb")):
        out = sqlglot.parse_one(sql, read=dialect, error_level=ErrorLevel.RAISE)
        assert not isinstance(out, exp.Command)


# ------------------------------------------------------------------------------------------
# 6. FINDING -- MilvusQL-only constructs leaving the dialect
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("write", ["postgres", "duckdb"])
@pytest.mark.parametrize(
    "milvus_sql, expected",
    [
        (
            "SELECT id FROM items HYBRID SEARCH "
            "(embedding <=> :dv WEIGHT 0.7, sparse_emb <#> :sv WEIGHT 0.3) "
            "RERANK RRF(k=60) LIMIT 10",
            "SELECT id FROM items LIMIT 10",
        ),
        (
            "SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef_search=64)",
            "SELECT id FROM items LIMIT 10",
        ),
    ],
    ids=["hybrid-search", "search-params"],
)
def test_milvus_only_select_clauses_are_silently_dropped(milvus_sql, expected, write):
    """FINDING (data loss): ``HYBRID SEARCH`` and ``SEARCH PARAMS`` vanish without a diagnostic.

    Both clauses live as extra keys in ``exp.Select.args``, rendered by the *milvus* generator's
    ``query_modifiers`` / ``after_limit_modifiers`` overrides. A foreign generator has no such
    override, so it simply never visits those keys -- there is no ``TRANSFORMS`` lookup to miss and
    therefore no ``self.unsupported(...)`` call and no ``ValueError``. ``ErrorLevel.RAISE`` does not
    help; ``generator.unsupported_messages`` stays empty. The whole vector search is dropped and a
    plain, valid, completely different query comes out.

    This is worth stating loudly: it is silent semantic corruption in the one direction where a
    user is least likely to re-read the output (exporting MilvusQL to a "normal" database).
    """
    ast = sqlglot.parse_one(milvus_sql, read="milvus", error_level=ErrorLevel.RAISE)
    assert ast.args.get("hybrid") or ast.args.get("search_params")

    generator = sqlglot.Dialect.get_or_raise(write).generator(unsupported_level=ErrorLevel.WARN)
    assert generator.generate(ast) == expected
    assert generator.unsupported_messages == []  # nothing was reported. that is the finding.

    # Even the strictest setting stays quiet.
    assert ast.sql(dialect=write, unsupported_level=ErrorLevel.RAISE) == expected


@pytest.mark.parametrize("write", ["postgres", "duckdb"])
@pytest.mark.parametrize(
    "milvus_sql",
    [
        "LOAD TABLE items WITH (replicas=2)",
        "RELEASE TABLE items",
        "ALTER TABLE items ADD FIELD tag VARCHAR(32)",
        "SELECT id FROM items ORDER BY embedding <=> :q LIMIT 10",
        "SELECT id FROM items ORDER BY embedding <#> :q LIMIT 10",
        "SELECT id FROM items ORDER BY embedding <+> :q LIMIT 10",
    ],
    ids=["load-table", "release-table", "add-field", "cosine", "inner-product", "l1"],
)
def test_milvus_only_nodes_fail_loudly_but_with_a_plain_valueerror(milvus_sql, write):
    """Custom *nodes* do fail on foreign generators -- unlike the custom Select args above.

    The exception is ``ValueError``, which is deliberately pinned here because it is **not** a
    ``SqlglotError``: code that wraps transpilation in ``except SqlglotError`` will let it through,
    and ``unsupported_level=IGNORE`` will not suppress it.
    """
    ast = sqlglot.parse_one(milvus_sql, read="milvus", error_level=ErrorLevel.RAISE)
    with pytest.raises(ValueError) as excinfo:
        ast.sql(dialect=write, unsupported_level=ErrorLevel.IGNORE)
    assert not isinstance(excinfo.value, SqlglotError)


def test_milvus_full_text_search_degrades_differently_per_dialect():
    """MATCH ... AGAINST has a postgres analogue; BM25_SCORE does not and passes through unchecked.

    Pinned rather than endorsed: ``@@`` against a bare placeholder is not valid postgres full-text
    search (it needs a tsquery), and ``BM25_SCORE`` is emitted as a function call to something that
    does not exist there. Neither produces a warning.
    """
    sql = (
        "SELECT id FROM items WHERE MATCH(text) AGAINST (:q) "
        "ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10"
    )
    ast = sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)

    assert ast.sql(dialect="postgres", unsupported_level=ErrorLevel.RAISE) == (
        "SELECT id FROM items WHERE text @@ %(q)s "
        "ORDER BY BM25_SCORE(text, %(q)s) DESC NULLS LAST LIMIT 10"
    )
    assert ast.sql(dialect="duckdb", unsupported_level=ErrorLevel.RAISE) == (
        "SELECT id FROM items WHERE MATCH(text) AGAINST($q) "
        "ORDER BY BM25_SCORE(text, $q) DESC LIMIT 10"
    )


def test_create_table_properties_survive_to_postgres_but_not_to_duckdb():
    """MilvusQL's table properties are the one place a foreign generator *does* complain."""
    sql = (
        "CREATE TABLE items (id BIGINT PRIMARY KEY AUTO_INCREMENT, embedding VECTOR(768)) "
        "WITH (shards=2, consistency_level='Bounded')"
    )
    ast = sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)

    # postgres has a WITH storage-parameter clause, so the properties come through verbatim --
    # syntactically valid, semantically nonsense to postgres.
    assert ast.sql(dialect="postgres", unsupported_level=ErrorLevel.RAISE) == (
        "CREATE TABLE items (id BIGINT GENERATED BY DEFAULT AS IDENTITY NOT NULL PRIMARY KEY, "
        "embedding VECTOR(768)) WITH (shards=2, consistency_level='Bounded')"
    )

    with pytest.raises(UnsupportedError):
        ast.sql(dialect="duckdb", unsupported_level=ErrorLevel.RAISE)


# ------------------------------------------------------------------------------------------
# 7. Round trips through a foreign dialect
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "milvus_sql",
    [
        "SELECT id, category FROM items WHERE category = :cat LIMIT 10",
        "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10",
        "SELECT category, COUNT(*) AS n FROM items GROUP BY category",
    ],
    ids=["filtered", "l2-search", "group-by"],
)
def test_milvus_postgres_milvus_round_trip_is_lossless(milvus_sql):
    ast = sqlglot.parse_one(milvus_sql, read="milvus", error_level=ErrorLevel.RAISE)
    postgres_sql = ast.sql(dialect="postgres", unsupported_level=ErrorLevel.RAISE)

    # The intermediate really is postgres, not our own text handed back unexamined.
    assert not isinstance(
        sqlglot.parse_one(postgres_sql, read="postgres", error_level=ErrorLevel.RAISE), exp.Command
    )
    assert to_milvus(postgres_sql, read="postgres") == milvus_sql


@pytest.mark.parametrize(
    "postgres_sql",
    [
        "SELECT * FROM items ORDER BY embedding <-> %s LIMIT 5",
        "SELECT * FROM items WHERE category = %(cat)s ORDER BY embedding <-> %(emb)s LIMIT 5",
        "SELECT id, embedding <-> %s AS distance FROM items ORDER BY distance LIMIT 5",
    ],
    ids=["format-param", "named-param", "aliased-distance"],
)
def test_postgres_milvus_postgres_round_trip_is_lossless(postgres_sql):
    milvus_sql = to_milvus(postgres_sql, read="postgres")
    back = sqlglot.parse_one(milvus_sql, read="milvus", error_level=ErrorLevel.RAISE)
    assert back.sql(dialect="postgres", unsupported_level=ErrorLevel.RAISE) == postgres_sql
