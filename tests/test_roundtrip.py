"""The golden corpus: every MilvusQL construct, parsed and regenerated verbatim.

Each case is checked against the full four-property contract in :func:`assert_golden`. The first
property -- "is not an ``exp.Command``" -- is the one that makes this file worth anything.

sqlglot never fails a statement outright: what it cannot parse it degrades to ``exp.Command``, a
node that stores the raw source text and re-emits it character for character. Such a node therefore
satisfies a naive ``parse(sql).sql() == sql`` check *perfectly* while containing no structure at
all (``find_all(exp.Column)`` returns nothing, the optimizer sees an opaque string, and Track B's
translator has nothing to translate). ``CREATE INDEX ... USING HNSW``, ``LOAD TABLE`` and
``ALTER TABLE ... ADD FIELD`` all "round-trip" this way against a stock sqlglot with zero lines of
dialect code written. Asserting the node type is what separates a parsed statement from a quoted
one.
"""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel

from sqlglot_milvus.expressions import (
    HYBRID_ARG,
    SEARCH_PARAMS_ARG,
    AddField,
    BM25Score,
    CosineDistance,
    HybridSearch,
    InnerProduct,
    L1Distance,
    LoadTable,
    ReleaseTable,
    Rerank,
    SearchArm,
    SearchParams,
)

DIALECT = "milvus"


def assert_golden(sql: str, node_type: type[exp.Expression]) -> None:
    """Assert all four properties of a golden case.

    ``error_level=RAISE`` on the way in and ``unsupported_level=RAISE`` on the way out: both
    default to logging a warning, which would let a half-parsed statement or a
    ``self.unsupported()`` call (``exp.NullSafeEQ``) slip through as a green test.
    """
    ast = sqlglot.parse_one(sql, read=DIALECT, error_level=ErrorLevel.RAISE)

    assert not isinstance(ast, exp.Command), (
        f"degraded to exp.Command (parsed as an opaque string, not a statement): {sql!r}"
    )
    assert isinstance(ast, node_type), (
        f"expected {node_type.__name__}, got {type(ast).__name__}"
    )

    out = ast.sql(dialect=DIALECT, unsupported_level=ErrorLevel.RAISE)
    assert out == sql

    # Text equality alone would still permit a generator that emits a *different* statement which
    # happens to spell the same; re-parsing pins the meaning, not just the characters.
    assert repr(sqlglot.parse_one(out, read=DIALECT)) == repr(ast)


# ------------------------------------------------------------------------------------------
# The canonical ten -- one statement per construct, exactly as the spec and README print them.
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "node_type"),
    [
        pytest.param(
            "CREATE TABLE items (id BIGINT PRIMARY KEY AUTO_INCREMENT, embedding VECTOR(768),"
            " category VARCHAR(64)) WITH (shards=2, consistency_level='Bounded',"
            " partition_key=category)",
            exp.Create,
            id="create_table",
        ),
        pytest.param(
            "CREATE INDEX idx_emb ON items (embedding) USING HNSW WITH (metric_type='COSINE',"
            " M=16, ef_construction=200)",
            exp.Create,
            id="create_index",
        ),
        pytest.param(
            "LOAD TABLE items WITH (replicas=2)", LoadTable, id="load_table"
        ),
        pytest.param("RELEASE TABLE items", ReleaseTable, id="release_table"),
        pytest.param(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            exp.Insert,
            id="insert",
        ),
        pytest.param(
            "DELETE FROM items WHERE category = :cat", exp.Delete, id="delete"
        ),
        pytest.param(
            "SELECT id, category FROM items WHERE category = :cat ORDER BY embedding <-> :q"
            " LIMIT 10 SEARCH PARAMS (ef_search=64)",
            exp.Select,
            id="ann_search",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7,"
            " sparse_emb <#> :sv WEIGHT 0.3) RERANK RRF(k=60) LIMIT 10",
            exp.Select,
            id="hybrid_search",
        ),
        pytest.param(
            "SELECT id FROM items WHERE MATCH(text) AGAINST (:q)"
            " ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10",
            exp.Select,
            id="fulltext_search",
        ),
        pytest.param(
            "ALTER TABLE items ADD FIELD tag VARCHAR(32)",
            exp.Alter,
            id="alter_add_field",
        ),
    ],
)
def test_canonical_statements(sql, node_type) -> None:
    assert_golden(sql, node_type)


# ------------------------------------------------------------------------------------------
# CREATE TABLE
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "CREATE TABLE t (a BIGINT, b INT, c SMALLINT, d TINYINT)",
            id="int_types",
        ),
        pytest.param(
            "CREATE TABLE t (a FLOAT, b DOUBLE, c BOOLEAN)",
            id="float_and_bool_types",
        ),
        pytest.param(
            "CREATE TABLE t (a VARCHAR(64), b JSON)",
            id="varchar_and_json_types",
        ),
        # The four vector types are the whole reason this dialect exists, and each is a bare
        # identifier as far as sqlglot's type parser is concerned -- SUPPORTS_USER_DEFINED_TYPES
        # is what keeps them from becoming columns named "VECTOR" with a stray "(768)".
        pytest.param("CREATE TABLE t (v VECTOR(768))", id="vector_type"),
        pytest.param(
            "CREATE TABLE t (v SPARSEVEC)", id="sparsevec_type_no_dimension"
        ),
        pytest.param("CREATE TABLE t (v BINARYVEC(128))", id="binaryvec_type"),
        pytest.param(
            "CREATE TABLE t (v FLOAT16VEC(256))", id="float16vec_type"
        ),
        pytest.param(
            "CREATE TABLE t (v BFLOAT16VEC(256))", id="bfloat16vec_type"
        ),
        pytest.param(
            "CREATE TABLE t (a ARRAY<INT>)", id="array_type_untyped_length"
        ),
        pytest.param(
            "CREATE TABLE t (a ARRAY<VARCHAR(64)>)", id="array_of_varchar"
        ),
        pytest.param(
            "CREATE TABLE t (id BIGINT PRIMARY KEY AUTO_INCREMENT, v VECTOR(8), s SPARSEVEC,"
            " b BINARYVEC(16), f FLOAT16VEC(8), bf BFLOAT16VEC(8), j JSON, txt VARCHAR(128),"
            " ok BOOLEAN, n DOUBLE)",
            id="every_scalar_and_vector_type_at_once",
        ),
        pytest.param(
            "CREATE TABLE IF NOT EXISTS items (id BIGINT PRIMARY KEY)",
            id="if_not_exists",
        ),
        pytest.param(
            "CREATE TABLE items (id BIGINT NOT NULL, v VECTOR(4))",
            id="not_null",
        ),
        pytest.param(
            "CREATE TABLE t (id BIGINT PRIMARY KEY AUTO_INCREMENT)",
            id="primary_key_auto_increment",
        ),
        pytest.param(
            "CREATE TABLE db.items (id BIGINT)", id="qualified_table_name"
        ),
        # Milvus is case-sensitive about collection and field names, so quoting has to survive.
        pytest.param(
            'CREATE TABLE "my table" ("my col" BIGINT)',
            id="quoted_identifiers",
        ),
        pytest.param("CREATE TABLE t (id BIGINT)", id="no_with_clause"),
        pytest.param(
            "CREATE TABLE t (id BIGINT) WITH (shards=2, consistency_level='Strong',"
            " partition_key=id, enable_dynamic_field=TRUE, ttl_seconds=3600)",
            id="every_with_property",
        ),
    ],
)
def test_create_table(sql) -> None:
    assert_golden(sql, exp.Create)


@pytest.mark.xfail(
    strict=True,
    reason="ARRAY<T>(n) from spec 2.1 misparses: the (n) is taken as a call, yielding a ColumnDef "
    "whose kind is Cast(Array([8]) AS ARRAY<INT>). It regenerates as "
    "'CREATE TABLE t (a CAST(ARRAY(8) AS ARRAY<INT>))', which then fails to re-parse.",
)
def test_create_table_array_with_capacity() -> None:
    assert_golden("CREATE TABLE t (a ARRAY<INT>(8))", exp.Create)


# ------------------------------------------------------------------------------------------
# CREATE INDEX
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "CREATE INDEX i ON t (v) USING HNSW", id="hnsw_no_properties"
        ),
        pytest.param(
            "CREATE INDEX i ON t (v) USING IVF_FLAT WITH (nlist=1024)",
            id="ivf_flat",
        ),
        pytest.param(
            "CREATE INDEX i ON t (v) USING IVF_SQ8 WITH (nlist=128)",
            id="ivf_sq8",
        ),
        pytest.param(
            "CREATE INDEX i ON t (v) USING IVF_PQ WITH (nlist=128, m=8, nbits=8)",
            id="ivf_pq",
        ),
        pytest.param("CREATE INDEX i ON t (v) USING DISKANN", id="diskann"),
        pytest.param(
            "CREATE INDEX i ON t (v) USING AUTOINDEX WITH (metric_type='L2')",
            id="autoindex",
        ),
        pytest.param(
            "CREATE INDEX i ON t (v) USING SPARSE_INVERTED_INDEX",
            id="sparse_inverted_index",
        ),
        pytest.param(
            "CREATE INDEX i ON t (v) USING BIN_FLAT WITH (metric_type='HAMMING')",
            id="bin_flat",
        ),
        pytest.param(
            "CREATE INDEX i ON t (v) USING GPU_CAGRA", id="gpu_cagra"
        ),
        pytest.param(
            "CREATE INDEX i ON t (v) USING FLAT WITH (metric_type='JACCARD')",
            id="flat",
        ),
        pytest.param(
            "CREATE INDEX IF NOT EXISTS i ON t (v) USING HNSW",
            id="if_not_exists",
        ),
        # No USING at all: a scalar index. The custom _parse_index_params branch must not assume
        # the method is present.
        pytest.param("CREATE INDEX i ON t (v)", id="no_using_scalar_index"),
        pytest.param(
            "CREATE INDEX i ON t (a, b) USING HNSW", id="multiple_columns"
        ),
        pytest.param(
            "CREATE INDEX i ON db.t (v) USING HNSW", id="qualified_table_name"
        ),
        pytest.param(
            'CREATE INDEX "idx emb" ON "my table" ("embedding") USING HNSW',
            id="quoted_identifiers",
        ),
    ],
)
def test_create_index(sql) -> None:
    assert_golden(sql, exp.Create)


# ------------------------------------------------------------------------------------------
# LOAD / RELEASE
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "node_type"),
    [
        pytest.param("LOAD TABLE items", LoadTable, id="load_no_properties"),
        pytest.param(
            "LOAD TABLE items WITH (replicas=2)",
            LoadTable,
            id="load_with_replicas",
        ),
        pytest.param(
            "LOAD TABLE items WITH (replicas=2, resource_groups='rg1')",
            LoadTable,
            id="load_multiple_properties",
        ),
        pytest.param("LOAD TABLE db.items", LoadTable, id="load_qualified"),
        pytest.param('LOAD TABLE "my table"', LoadTable, id="load_quoted"),
        pytest.param("RELEASE TABLE items", ReleaseTable, id="release"),
        pytest.param(
            "RELEASE TABLE db.items", ReleaseTable, id="release_qualified"
        ),
        pytest.param(
            'RELEASE TABLE "my table"', ReleaseTable, id="release_quoted"
        ),
    ],
)
def test_load_release(sql, node_type) -> None:
    assert_golden(sql, node_type)


# ------------------------------------------------------------------------------------------
# INSERT / DELETE
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "INSERT INTO items (id) VALUES (:id)", id="single_placeholder"
        ),
        pytest.param(
            "INSERT INTO items (id, category) VALUES (1, 'a')", id="literals"
        ),
        pytest.param(
            "INSERT INTO items (id, embedding, category) VALUES (1, :emb, 'a')",
            id="literals_and_placeholders_mixed",
        ),
        pytest.param(
            "INSERT INTO items (id, category) VALUES (1, 'a'), (2, 'b')",
            id="two_rows",
        ),
        pytest.param(
            "INSERT INTO items (id, v) VALUES (1, :v), (2, :w), (3, :z)",
            id="three_rows_of_placeholders",
        ),
        pytest.param("INSERT INTO items VALUES (1, 'a')", id="no_column_list"),
    ],
)
def test_insert(sql) -> None:
    assert_golden(sql, exp.Insert)


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("DELETE FROM items", id="unfiltered"),
        pytest.param(
            "DELETE FROM items WHERE id > 10 AND category = :cat", id="and"
        ),
        pytest.param(
            "DELETE FROM items WHERE id > 1 OR category = :cat", id="or"
        ),
        pytest.param("DELETE FROM items WHERE id IN (1, 2, 3)", id="in"),
        pytest.param("DELETE FROM items WHERE NOT category = :cat", id="not"),
        pytest.param(
            "DELETE FROM items WHERE id BETWEEN 1 AND 100", id="between"
        ),
        pytest.param("DELETE FROM items WHERE category LIKE 'a%'", id="like"),
        pytest.param(
            "DELETE FROM items WHERE category NOT LIKE 'a%'", id="not_like"
        ),
        pytest.param("DELETE FROM items WHERE category IS NULL", id="is_null"),
        pytest.param(
            "DELETE FROM items WHERE (category = :cat OR category IS NULL) AND id > 1"
            " AND NOT id BETWEEN 5 AND 9",
            id="compound_with_parentheses",
        ),
    ],
)
def test_delete(sql) -> None:
    assert_golden(sql, exp.Delete)


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "DELETE FROM items WHERE category IS NOT NULL", id="is_not_null"
        ),
        pytest.param("DELETE FROM items WHERE id NOT IN (1, 2)", id="not_in"),
        pytest.param(
            "DELETE FROM items WHERE id NOT BETWEEN 1 AND 2", id="not_between"
        ),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason="Negated predicates lose their infix spelling: the bare Dialect generator renders "
    "exp.Not(exp.Is/In/Between) as a prefix 'NOT x IS NULL' / 'NOT x IN (...)'. Semantically "
    "equivalent and AST-stable, but not text-identical. Postgres overrides this; Milvus does not.",
)
def test_delete_negated_predicates(sql) -> None:
    assert_golden(sql, exp.Delete)


# ------------------------------------------------------------------------------------------
# SELECT: ANN search
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        # All four distance operators. <#> and <+> exist only because the tokenizer registers
        # them (D1); without that they shatter into LT + HASH_ARROW and LT + PLUS + GT.
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10",
            id="l2_operator",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <#> :q LIMIT 10",
            id="inner_product_operator",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <=> :q LIMIT 10",
            id="cosine_operator",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <+> :q LIMIT 10",
            id="l1_operator",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q ASC LIMIT 10",
            id="explicit_asc",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q DESC LIMIT 10",
            id="explicit_desc",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q, id DESC LIMIT 10",
            id="distance_then_tiebreak",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10 OFFSET 20",
            id="offset",
        ),
        pytest.param("SELECT id FROM items LIMIT :n", id="placeholder_limit"),
        pytest.param(
            "SELECT id FROM items AS i ORDER BY i.embedding <-> :q LIMIT 3",
            id="table_qualified_vector_column",
        ),
        pytest.param(
            "SELECT embedding <-> :q AS dist FROM items LIMIT 5",
            id="distance_in_projection",
        ),
        pytest.param(
            "SELECT id FROM items WHERE embedding <-> :q < 0.5 LIMIT 5",
            id="distance_in_where_range_search",
        ),
        # Distance operators sit at FACTOR precedence, so this is (embedding <=> :q) + 1 rather
        # than embedding <=> (:q + 1). Regenerating without parentheses proves the grouping.
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <=> :q + 1 LIMIT 5",
            id="factor_precedence_binds_tighter_than_plus",
        ),
        # SEARCH PARAMS: one param, many params, and a placeholder value.
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)",
            id="search_params_one",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10"
            " SEARCH PARAMS (ef_search=64, nprobe=16, search_k=100, radius=0.5,"
            " range_filter=0.1, consistency_level='Strong')",
            id="search_params_all_six",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10"
            " SEARCH PARAMS (ef_search=:ef)",
            id="search_params_placeholder_value",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10 OFFSET 5"
            " SEARCH PARAMS (ef_search=64)",
            id="search_params_after_limit_offset",
        ),
        pytest.param(
            "SELECT id FROM items WHERE category = :cat AND id > 1 ORDER BY embedding <=> :q"
            " LIMIT 10 SEARCH PARAMS (ef_search=64)",
            id="filtered_ann_search",
        ),
        # Projections.
        pytest.param("SELECT * FROM items LIMIT 10", id="star"),
        pytest.param("SELECT COUNT(*) FROM items", id="count_star"),
        pytest.param("SELECT DISTINCT category FROM items", id="distinct"),
        pytest.param(
            "SELECT id AS pk, category AS cat FROM items LIMIT 5", id="aliases"
        ),
        pytest.param(
            "SELECT id + 1 AS x FROM items LIMIT 5", id="expression_projection"
        ),
        pytest.param(
            'SELECT "id" FROM "items" LIMIT 1', id="quoted_identifiers"
        ),
    ],
)
def test_select_ann(sql) -> None:
    assert_golden(sql, exp.Select)


# ------------------------------------------------------------------------------------------
# SELECT: hybrid search
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x) LIMIT 10",
            id="one_arm",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv, sparse_emb <#> :sv) LIMIT 10",
            id="two_arms_no_weights",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7,"
            " sparse_emb <#> :sv WEIGHT 0.3) LIMIT 10",
            id="two_arms_weighted",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x WEIGHT 0.2, b <#> :y WEIGHT 0.3,"
            " c <=> :z WEIGHT 0.5) RERANK WEIGHTED LIMIT 10",
            id="three_arms_weighted",
        ),
        # A weight on some arms but not others: _parse_search_arm has to treat WEIGHT as optional
        # per arm, not per clause.
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x WEIGHT 0.7, b <#> :y)"
            " RERANK RRF(k=60) LIMIT 10",
            id="mixed_weighted_and_unweighted_arms",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x WEIGHT 1) LIMIT 10",
            id="integer_weight",
        ),
        # Nothing constrains an arm to be a distance expression, so a BM25 arm is free.
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x WEIGHT 0.5,"
            " BM25_SCORE(text, :q) WEIGHT 0.5) LIMIT 10",
            id="dense_plus_fulltext_arm",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x, b <#> :y) LIMIT 10",
            id="no_rerank",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x, b <#> :y) RERANK RRF LIMIT 10",
            id="rerank_without_params",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x, b <#> :y) RERANK RRF(k=60) LIMIT 10",
            id="rerank_one_param",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x, b <#> :y)"
            " RERANK RRF(k=60, strategy='sum') LIMIT 10",
            id="rerank_two_params",
        ),
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x, b <#> :y)",
            id="no_limit",
        ),
        # D5: HYBRID SEARCH before LIMIT, SEARCH PARAMS after it, in one statement.
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (a <-> :x, b <#> :y) RERANK RRF(k=60)"
            " LIMIT 10 SEARCH PARAMS (ef_search=64)",
            id="hybrid_and_search_params_together",
        ),
        # HYBRID SEARCH renders immediately before LIMIT, so it sits after every clause that
        # precedes LIMIT in ordinary SQL rather than being hoisted up next to FROM.
        pytest.param(
            "SELECT id FROM items WHERE category = :cat HYBRID SEARCH (a <-> :x) LIMIT 10",
            id="filtered_hybrid_search",
        ),
        pytest.param(
            "SELECT id FROM items JOIN o ON items.id = o.id HYBRID SEARCH (a <-> :x) LIMIT 10",
            id="joined_hybrid_search",
        ),
        pytest.param(
            "SELECT id FROM items WHERE category = :cat ORDER BY id HYBRID SEARCH (a <-> :x)"
            " LIMIT 10 SEARCH PARAMS (ef_search=64)",
            id="hybrid_search_between_order_by_and_limit",
        ),
    ],
)
def test_hybrid_search(sql) -> None:
    assert_golden(sql, exp.Select)


@pytest.mark.xfail(
    strict=True,
    reason="SEARCH PARAMS written before OFFSET is silently reordered to 'OFFSET ... SEARCH "
    "PARAMS ...': after_limit_modifiers appends it last. _validate_clause_order checks position "
    "against LIMIT only, never OFFSET.",
)
def test_search_params_before_offset() -> None:
    assert_golden(
        "SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef_search=64) OFFSET 5",
        exp.Select,
    )


# ------------------------------------------------------------------------------------------
# SELECT: full-text search
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "SELECT id FROM items WHERE MATCH(text) AGAINST (:q) LIMIT 10",
            id="match_against",
        ),
        pytest.param(
            "SELECT id FROM items WHERE MATCH(title, body) AGAINST (:q) LIMIT 10",
            id="match_multiple_columns",
        ),
        pytest.param(
            "SELECT id FROM items WHERE MATCH(text) AGAINST ('hello') LIMIT 10",
            id="match_string_literal",
        ),
        pytest.param(
            "SELECT id FROM items WHERE MATCH(text) AGAINST (:q) AND category = :cat LIMIT 10",
            id="match_combined_with_scalar_filter",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10",
            id="bm25_score_in_order_by",
        ),
        pytest.param(
            "SELECT id, BM25_SCORE(text, :q) AS score FROM items WHERE MATCH(text) AGAINST (:q)"
            " LIMIT 10",
            id="bm25_score_in_projection",
        ),
    ],
)
def test_fulltext_search(sql) -> None:
    assert_golden(sql, exp.Select)


# ------------------------------------------------------------------------------------------
# ALTER / DROP
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "ALTER TABLE items ADD FIELD tag VARCHAR(32)",
            id="add_field_varchar",
        ),
        pytest.param(
            "ALTER TABLE items ADD FIELD score DOUBLE", id="add_field_double"
        ),
        pytest.param(
            "ALTER TABLE items ADD FIELD n BIGINT", id="add_field_bigint"
        ),
        pytest.param(
            "ALTER TABLE items ADD FIELD meta JSON", id="add_field_json"
        ),
        pytest.param(
            "ALTER TABLE items ADD FIELD v VECTOR(128)", id="add_field_vector"
        ),
        pytest.param(
            "ALTER TABLE items ADD FIELD tag VARCHAR(32) NOT NULL",
            id="add_field_constrained",
        ),
        pytest.param(
            'ALTER TABLE "my table" ADD FIELD "my col" VARCHAR(32)',
            id="add_field_quoted",
        ),
        # ADD FIELD hooks the ALTER_PARSERS "ADD" entry, so plain ADD COLUMN must still fall
        # through to the stock parser rather than being swallowed.
        pytest.param(
            "ALTER TABLE items ADD COLUMN tag VARCHAR(32)",
            id="add_column_still_works",
        ),
        # IF NOT EXISTS is what ADD COLUMN accepts, so ADD FIELD has to accept it as well. Without
        # its own _parse_exists the three words reached _parse_field_def, which read them as the
        # expression `CASE WHEN NOT EXISTS THEN tag END` -- the field name gone from any name-based
        # lookup, and the statement re-emitted in that shape without a word of warning.
        pytest.param(
            "ALTER TABLE items ADD FIELD IF NOT EXISTS tag VARCHAR(32)",
            id="add_field_if_not_exists",
        ),
        pytest.param(
            "ALTER TABLE items ADD COLUMN IF NOT EXISTS tag VARCHAR(32)",
            id="add_column_if_not_exists",
        ),
    ],
)
def test_alter(sql) -> None:
    assert_golden(sql, exp.Alter)


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("DROP TABLE items", id="drop_table"),
        pytest.param("DROP TABLE IF EXISTS items", id="drop_table_if_exists"),
        pytest.param("DROP TABLE db.items", id="drop_table_qualified"),
        pytest.param("DROP INDEX idx_emb ON items", id="drop_index"),
        pytest.param(
            "DROP INDEX IF EXISTS idx_emb ON items", id="drop_index_if_exists"
        ),
    ],
)
def test_drop(sql) -> None:
    assert_golden(sql, exp.Drop)


# ------------------------------------------------------------------------------------------
# Plain SQL that the vector additions must not break
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "node_type"),
    [
        pytest.param(
            "SELECT category, COUNT(*) AS n FROM items GROUP BY category",
            exp.Select,
            id="group_by",
        ),
        pytest.param(
            "SELECT category, COUNT(*) AS n FROM items GROUP BY category HAVING COUNT(*) > 5",
            exp.Select,
            id="having",
        ),
        pytest.param(
            "SELECT id FROM items WHERE id IN (SELECT id FROM other)",
            exp.Select,
            id="subquery_in_where",
        ),
        pytest.param(
            "SELECT id FROM (SELECT id FROM items) AS sub",
            exp.Select,
            id="derived_table",
        ),
        pytest.param(
            "WITH recent AS (SELECT id FROM items WHERE id > 100) SELECT id FROM recent",
            exp.Select,
            id="cte",
        ),
        pytest.param(
            "WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT * FROM a, b",
            exp.Select,
            id="two_ctes",
        ),
        pytest.param(
            "SELECT id FROM a UNION SELECT id FROM b", exp.Union, id="union"
        ),
        pytest.param(
            "SELECT id FROM a UNION ALL SELECT id FROM b",
            exp.Union,
            id="union_all",
        ),
        pytest.param(
            "SELECT a.id FROM a JOIN b ON a.id = b.id", exp.Select, id="join"
        ),
        pytest.param(
            "SELECT a.id FROM a LEFT JOIN b ON a.id = b.id",
            exp.Select,
            id="left_join",
        ),
        pytest.param(
            "SELECT CASE WHEN id > 1 THEN 'big' ELSE 'small' END AS bucket FROM items",
            exp.Select,
            id="case_when",
        ),
        pytest.param(
            "SELECT COALESCE(NULLIF(category, ''), 'none') AS c FROM items",
            exp.Select,
            id="nested_functions",
        ),
        pytest.param(
            "SELECT UPPER(TRIM(category)) AS c FROM items",
            exp.Select,
            id="functions",
        ),
        pytest.param(
            "SELECT id FROM items WHERE id > 1 AND (category = 'a' OR category = 'b')",
            exp.Select,
            id="parenthesised_boolean",
        ),
        pytest.param(
            "UPDATE items SET category = 'x' WHERE id = 1",
            exp.Update,
            id="update",
        ),
        pytest.param(
            "SELECT id FROM items ORDER BY id LIMIT 10",
            exp.Select,
            id="plain_order_by",
        ),
    ],
)
def test_plain_sql(sql, node_type) -> None:
    assert_golden(sql, node_type)


@pytest.mark.parametrize(
    "sql",
    [
        # D2: registering "HYBRID SEARCH"/"SEARCH PARAMS" as multi-word keywords is what keeps the
        # bare words ordinary identifiers. RELEASE has to be single-word, hence reserved by the
        # tokenizer, and is added back to ID_VAR_TOKENS/TABLE_ALIAS_TOKENS by hand.
        pytest.param(
            "SELECT release, hybrid, search, params FROM t",
            id="as_column_names",
        ),
        pytest.param(
            "SELECT release AS r FROM t", id="release_as_projection_with_alias"
        ),
        pytest.param(
            "SELECT id FROM t AS release", id="release_as_table_alias"
        ),
        pytest.param("SELECT * FROM search", id="search_as_table_name"),
        pytest.param(
            "SELECT id FROM t WHERE search = 1 AND params = 2 AND hybrid = 3",
            id="in_where_predicates",
        ),
        # The README's documented casualty: a table "search" aliased "params" lexes as the
        # SEARCH PARAMS keyword. Quoting is the stated recovery, so it has to actually work.
        pytest.param(
            'SELECT * FROM "search" AS params',
            id="quoted_search_aliased_params",
        ),
    ],
)
def test_reserved_words_stay_identifiers(sql) -> None:
    assert_golden(sql, exp.Select)


# ------------------------------------------------------------------------------------------
# Structure behind the text
# ------------------------------------------------------------------------------------------
#
# The corpus above pins only the *top-level* node type, which for every SELECT is exp.Select --
# enough to catch an exp.Command degradation, not enough to catch a search clause that parsed into
# the wrong shape. These pin the shapes Track B's translator actually reads.


@pytest.mark.parametrize(
    ("operator", "node_type"),
    [
        pytest.param("<->", exp.Distance, id="l2"),
        pytest.param("<#>", InnerProduct, id="inner_product"),
        pytest.param("<=>", CosineDistance, id="cosine"),
        pytest.param("<+>", L1Distance, id="l1"),
    ],
)
def test_distance_operators_produce_their_own_nodes(
    operator, node_type
) -> None:
    # <=> is the load-bearing one (D3): it arrives as TokenType.NULLSAFE_EQ, and if the FACTOR
    # rebinding failed it would parse as exp.NullSafeEQ -- which also regenerates as "<=>", so the
    # text round-trip would still pass while the query silently meant null-safe equality.
    ast = sqlglot.parse_one(
        f"SELECT id FROM t ORDER BY v {operator} :q LIMIT 1", read=DIALECT
    )
    distance = ast.args["order"].expressions[0].this
    assert type(distance) is node_type
    assert isinstance(distance.this, exp.Column)
    assert isinstance(distance.expression, exp.Placeholder)


def test_hybrid_search_arms_and_rerank_shape() -> None:
    ast = sqlglot.parse_one(
        "SELECT id FROM items HYBRID SEARCH (a <-> :x WEIGHT 0.7, b <#> :y)"
        " RERANK RRF(k=60) LIMIT 10",
        read=DIALECT,
    )
    hybrid = ast.args[HYBRID_ARG]
    assert isinstance(hybrid, HybridSearch)

    dense, sparse = hybrid.expressions
    assert isinstance(dense, SearchArm)
    assert isinstance(sparse, SearchArm)
    assert isinstance(dense.this, exp.Distance)
    assert dense.args["weight"].this == "0.7"
    assert isinstance(sparse.this, InnerProduct)
    # An unweighted arm must carry no weight rather than a default one.
    assert sparse.args.get("weight") is None

    rerank = hybrid.args["rerank"]
    assert isinstance(rerank, Rerank)
    assert rerank.this.name == "RRF"
    # Properties, not EQ(Column(k), 60): a phantom column here would send qualify() and lineage
    # analysis looking for a column named "k" in the schema.
    (param,) = rerank.expressions
    assert isinstance(param, exp.Property)
    assert param.this.name == "k"


def test_search_params_shape() -> None:
    ast = sqlglot.parse_one(
        "SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef_search=64, nprobe=16)",
        read=DIALECT,
    )
    params = ast.args[SEARCH_PARAMS_ARG]
    assert isinstance(params, SearchParams)
    assert [p.this.name for p in params.expressions] == ["ef_search", "nprobe"]


def test_fulltext_nodes() -> None:
    ast = sqlglot.parse_one(
        "SELECT id FROM items WHERE MATCH(text) AGAINST (:q) ORDER BY BM25_SCORE(text, :q) DESC"
        " LIMIT 10",
        read=DIALECT,
    )
    assert isinstance(ast.find(exp.MatchAgainst), exp.MatchAgainst)
    assert isinstance(ast.find(BM25Score), BM25Score)


def test_load_release_and_add_field_are_structured() -> None:
    load = sqlglot.parse_one(
        "LOAD TABLE items WITH (replicas=2)", read=DIALECT
    )
    assert isinstance(load, LoadTable)
    assert load.this.name == "items"
    assert isinstance(load.args["properties"], exp.Properties)

    release = sqlglot.parse_one("RELEASE TABLE items", read=DIALECT)
    assert isinstance(release, ReleaseTable)
    assert release.this.name == "items"

    alter = sqlglot.parse_one(
        "ALTER TABLE items ADD FIELD tag VARCHAR(32)", read=DIALECT
    )
    (action,) = alter.args["actions"]
    assert isinstance(action, AddField)
    assert isinstance(action.this, exp.ColumnDef)
    assert action.this.name == "tag"


def test_add_field_if_not_exists_keeps_the_field_name_recoverable() -> None:
    # The failure this pins was not "wrong text": the field name ended up inside an exp.If, so
    # `action.this.name` -- how Track B finds the column to add -- returned something else entirely.
    alter = sqlglot.parse_one(
        "ALTER TABLE items ADD FIELD IF NOT EXISTS tag VARCHAR(32)",
        read=DIALECT,
    )
    (action,) = alter.args["actions"]
    assert isinstance(action, AddField)
    assert action.args["exists"] is True
    assert isinstance(action.this, exp.ColumnDef)
    assert action.this.name == "tag"
    assert action.this.args["kind"].sql(DIALECT) == "VARCHAR(32)"
    assert not list(alter.find_all(exp.If))


# ------------------------------------------------------------------------------------------
# Accepted-but-normalised spellings
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "canonical"),
    [
        # D6: pgvector puts the method before the column list. Both parse; MilvusQL order is what
        # comes out, which is what makes postgres -> milvus index DDL free.
        pytest.param(
            "CREATE INDEX i ON t USING hnsw (v)",
            "CREATE INDEX i ON t (v) USING HNSW",
            id="pgvector_index_word_order",
        ),
        pytest.param(
            "CREATE INDEX i ON t USING ivf_flat (v) WITH (nlist=128)",
            "CREATE INDEX i ON t (v) USING IVF_FLAT WITH (nlist=128)",
            id="pgvector_index_word_order_with_properties",
        ),
        pytest.param(
            "CREATE TABLE t (a INTEGER)",
            "CREATE TABLE t (a INT)",
            id="integer_alias_for_int",
        ),
        pytest.param(
            'SELECT * FROM "search" params',
            'SELECT * FROM "search" AS params',
            id="implicit_alias_made_explicit",
        ),
    ],
)
def test_accepted_spellings_normalise(sql, canonical) -> None:
    ast = sqlglot.parse_one(sql, read=DIALECT, error_level=ErrorLevel.RAISE)
    assert not isinstance(ast, exp.Command)
    assert (
        ast.sql(dialect=DIALECT, unsupported_level=ErrorLevel.RAISE)
        == canonical
    )
    # The canonical form must be a fixpoint, or the normalisation just moved the problem.
    assert_golden(canonical, type(ast))


# ------------------------------------------------------------------------------------------
# Pretty printing
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7,"
            " sparse_emb <#> :sv WEIGHT 0.3) RERANK RRF(k=60) LIMIT 10",
            id="hybrid_search",
        ),
        pytest.param(
            "SELECT id, category FROM items WHERE category = :cat ORDER BY embedding <-> :q"
            " LIMIT 10 SEARCH PARAMS (ef_search=64)",
            id="search_params",
        ),
    ],
)
def test_pretty_output_reparses_identically(sql) -> None:
    # Both clauses are emitted through self.seg(), which switches between " " and "\n" depending on
    # pretty mode. A missing seg() would still look right on one line and produce a run-on
    # "...FROM itemsHYBRID SEARCH..." on the other.
    ast = sqlglot.parse_one(sql, read=DIALECT, error_level=ErrorLevel.RAISE)
    pretty = ast.sql(
        dialect=DIALECT, pretty=True, unsupported_level=ErrorLevel.RAISE
    )
    assert "\n" in pretty
    assert repr(
        sqlglot.parse_one(pretty, read=DIALECT, error_level=ErrorLevel.RAISE)
    ) == repr(ast)
