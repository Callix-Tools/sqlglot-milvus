"""The AST contract that Track B (``sqlalchemy-milvus``) compiles against.

``test_dialect.py`` proves the text round-trips. That is not enough for Track B, which never looks
at text: it walks the AST and emits ``pymilvus`` calls. This file pins the *shapes* it walks --
which node class appears where, which ``args`` key holds it, and which attribute holds the value --
so that an upstream sqlglot release or a local refactor that silently re-shapes the tree fails here
rather than in the other repo.

The extraction helpers below (``ann_search_plan`` and friends) are deliberately written the way
Track B's translator would write them, and the tests assert on their *output dicts*. That is the
real claim being made: not "the AST contains a Placeholder somewhere", but "everything pymilvus
needs is mechanically recoverable from the AST alone".

Every parse here goes through :func:`_parse`, which raises on the first parse error *and* rejects
``exp.Command``. sqlglot degrades anything it cannot parse into ``Command``, which stores the raw
text and therefore round-trips byte-identically while being an opaque blob -- ``find_all`` over a
``Command`` yields nothing, so an unguarded shape test on one would trivially "pass" by asserting
nothing at all.
"""

from __future__ import annotations

import decimal
import typing as t

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import build_scope

from sqlglot_milvus import expressions as milvus_exp
from sqlglot_milvus.expressions import (
    HYBRID_ARG,
    METRIC_TYPES,
    SEARCH_PARAMS_ARG,
    BM25Score,
    CosineDistance,
    HybridSearch,
    InnerProduct,
    L1Distance,
    Rerank,
    SearchArm,
    SearchParams,
)

# The canonical statements from the spec. Everything in this file is a claim about these.
ANN_SEARCH = (
    "SELECT id, category FROM items WHERE category = :cat "
    "ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)"
)
HYBRID_SEARCH = (
    "SELECT id FROM items HYBRID SEARCH "
    "(embedding <=> :dv WEIGHT 0.7, sparse_emb <#> :sv WEIGHT 0.3) RERANK RRF(k=60) LIMIT 10"
)
FULL_TEXT_SEARCH = (
    "SELECT id FROM items WHERE MATCH(text) AGAINST (:q) "
    "ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10"
)
CREATE_TABLE = (
    "CREATE TABLE items (id BIGINT PRIMARY KEY AUTO_INCREMENT, embedding VECTOR(768), "
    "category VARCHAR(64)) "
    "WITH (shards=2, consistency_level='Bounded', partition_key=category)"
)
CREATE_INDEX = (
    "CREATE INDEX idx_emb ON items (embedding) USING HNSW "
    "WITH (metric_type='COSINE', M=16, ef_construction=200)"
)
INSERT = "INSERT INTO items (embedding, category) VALUES (:emb, :cat)"


def _parse(sql: str) -> exp.Expression:
    ast = sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)
    # See module docstring: Command is the opaque-blob failure mode, and it is silent.
    assert not isinstance(ast, exp.Command), f"degraded to Command: {sql}"
    return ast


# ------------------------------------------------------------------------------------------------
# Extraction helpers -- a sketch of Track B's translator, used as the assertion subject
# ------------------------------------------------------------------------------------------------


def _py(node: exp.Expression) -> t.Any:
    """Python value of a property value node.

    ``exp.Var`` needs its own branch: an unquoted right-hand side (``partition_key=category``)
    parses as ``Var``, whose ``to_py()`` raises ``ValueError``. Its text *is* the value.
    """
    if isinstance(node, exp.Var):
        return node.name
    return node.to_py()


def _props(properties: t.Iterable[exp.Expression]) -> dict[str, t.Any]:
    """``[Property(Var(k), Literal(v)), ...]`` -> ``{"k": v}``."""
    return {p.name: _py(p.args["value"]) for p in properties}


def _weight(node: exp.Expression | None) -> float | None:
    """A hybrid arm's WEIGHT as a float.

    ``Literal.to_py()`` returns an ``int`` for an integral literal but a ``decimal.Decimal`` for a
    fractional one, and ``Decimal("0.7") != 0.7``. pymilvus wants a float, so coerce at the boundary
    rather than letting a Decimal leak into the request. (A bind-parameter weight -- ``WEIGHT :w``
    -- parses to a Placeholder and is resolved at execution time, not here.)
    """
    return None if node is None else float(node.to_py())


def _distance_parts(node: exp.Binary) -> dict[str, t.Any]:
    """``(column, metric_type, placeholder)`` for one distance expression.

    Operand order is deliberately *not* assumed: ``:q <-> embedding`` is legal MilvusQL and parses
    with the operands swapped, so the arms are identified by node type. ``METRIC_TYPES`` is keyed by
    exact class, not by ``isinstance``, because the four distance nodes are siblings -- an
    ``isinstance`` chain would need a hand-maintained ordering to stay correct.
    """
    operands = (node.this, node.expression)
    column = next(o for o in operands if isinstance(o, exp.Column))
    placeholder = next(o for o in operands if isinstance(o, exp.Placeholder))
    return {
        "column": column.name,
        "metric": METRIC_TYPES[type(node)],
        "placeholder": placeholder.name,
    }


def ann_search_plan(select: exp.Select) -> dict[str, t.Any]:
    """Everything ``pymilvus`` ``Collection.search()`` needs, taken from the AST alone."""
    # A vector search has exactly one ORDER BY key. The parser does not enforce that (see
    # test_parser_accepts_a_mixed_order_by), so the unpack is where a translator would notice.
    (ordered,) = select.args["order"].expressions
    where = select.args.get("where")
    limit = select.args.get("limit")
    offset = select.args.get("offset")
    search_params = select.args.get(SEARCH_PARAMS_ARG)
    return {
        # NOTE: the key is "from_", not "from" -- sqlglot 30 renamed it.
        "table": select.args["from_"].this.name,
        "output_fields": select.named_selects,
        "filter": where.this.sql(dialect="milvus") if where else None,
        **_distance_parts(ordered.this),
        "limit": limit.expression.to_py() if limit else None,
        "offset": offset.expression.to_py() if offset else None,
        "search_params": _props(search_params.expressions)
        if search_params
        else {},
    }


def hybrid_search_plan(select: exp.Select) -> dict[str, t.Any]:
    """Everything ``pymilvus`` ``Collection.hybrid_search()`` needs."""
    hybrid = select.args[HYBRID_ARG]
    rerank = hybrid.args.get("rerank")
    return {
        "table": select.args["from_"].this.name,
        "arms": [
            {
                **_distance_parts(arm.this),
                "weight": _weight(arm.args.get("weight")),
            }
            for arm in hybrid.expressions
        ],
        "rerank": rerank.name if rerank else None,
        "rerank_params": _props(rerank.args.get("expressions") or [])
        if rerank
        else {},
        "limit": select.args["limit"].expression.to_py(),
    }


def create_table_plan(create: exp.Create) -> dict[str, t.Any]:
    """Everything ``pymilvus`` ``CollectionSchema``/``FieldSchema`` needs."""
    schema = create.this
    fields = []
    for column_def in schema.expressions:
        kind = column_def.kind
        # DataType.expressions holds DataTypeParam wrappers; VECTOR(768) has exactly one.
        dim = kind.expressions[0].this.to_py() if kind.expressions else None
        constraints = {type(c.kind) for c in column_def.constraints}
        fields.append(
            {
                "name": column_def.name,
                "type": kind.this.name,
                "dim": dim,
                "is_primary": exp.PrimaryKeyColumnConstraint in constraints,
                "auto_id": exp.AutoIncrementColumnConstraint in constraints,
            }
        )
    properties = create.args.get("properties")
    return {
        "table": schema.this.name,
        "fields": fields,
        "properties": _props(properties.expressions) if properties else {},
    }


def create_index_plan(create: exp.Create) -> dict[str, t.Any]:
    """Everything ``pymilvus`` ``Collection.create_index()`` needs."""
    index = create.this
    params = index.args["params"]
    using = params.args.get("using")
    return {
        "index": index.name,
        "table": index.args["table"].name,
        # Index columns are wrapped in exp.Ordered even without ASC/DESC -- sqlglot models an index
        # column list as an ordered one.
        "columns": [c.this.name for c in params.args["columns"]],
        "method": using.name.upper() if using else None,
        "params": _props(params.args.get("with_storage") or []),
    }


# ------------------------------------------------------------------------------------------------
# 1. Placeholders -- a vector must never become text
# ------------------------------------------------------------------------------------------------


def test_placeholder_node_class_and_name_attribute() -> None:
    """The single most load-bearing fact in this file, spelled out for Track B.

    ``:emb`` -> ``exp.Placeholder``; the name lives in ``args["this"]`` as a **plain ``str``**, not
    as an ``Identifier``. (Postgres' ``%(emb)s`` form does wrap it in an ``Identifier``; ours does
    not, so ``node.this.name`` would raise ``AttributeError``. Always read ``node.name``.)
    """
    placeholder = _parse(INSERT).find(exp.Placeholder)

    assert type(placeholder) is exp.Placeholder
    assert isinstance(placeholder.args["this"], str)
    assert placeholder.args["this"] == "emb"
    assert placeholder.name == "emb"


# Each case pins the *structural position* of the placeholder, not merely its presence somewhere in
# the tree: `locate` is the exact path Track B will walk to reach the bind parameter.
PLACEHOLDER_POSITIONS = [
    pytest.param(
        ANN_SEARCH,
        lambda a: a.args["order"].expressions[0].this.expression,
        "q",
        id="order-by-distance-operand",
    ),
    pytest.param(
        ANN_SEARCH,
        lambda a: a.args["where"].this.expression,
        "cat",
        id="where-comparison",
    ),
    pytest.param(
        INSERT,
        lambda a: a.expression.expressions[0].expressions[0],
        "emb",
        id="insert-values",
    ),
    pytest.param(
        HYBRID_SEARCH,
        lambda a: a.args[HYBRID_ARG].expressions[0].this.expression,
        "dv",
        id="hybrid-search-arm",
    ),
    pytest.param(
        HYBRID_SEARCH,
        lambda a: a.args[HYBRID_ARG].expressions[1].this.expression,
        "sv",
        id="hybrid-search-second-arm",
    ),
    pytest.param(
        FULL_TEXT_SEARCH,
        lambda a: a.args["order"].expressions[0].this.expression,
        "q",
        id="bm25-score-argument",
    ),
    pytest.param(
        FULL_TEXT_SEARCH,
        lambda a: a.args["where"].this.this,
        "q",
        id="match-against-argument",
    ),
    pytest.param(
        "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=:ef)",
        lambda a: a.args[SEARCH_PARAMS_ARG].expressions[0].args["value"],
        "ef",
        id="search-params-value",
    ),
    pytest.param(
        "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT :w) LIMIT 10",
        lambda a: a.args[HYBRID_ARG].expressions[0].args["weight"],
        "w",
        id="hybrid-arm-weight",
    ),
]


@pytest.mark.parametrize(("sql", "locate", "name"), PLACEHOLDER_POSITIONS)
def test_placeholder_is_a_bind_parameter_in_every_position(
    sql, locate, name
) -> None:
    node = locate(_parse(sql))

    assert isinstance(node, exp.Placeholder), f"got {type(node).__name__}"
    assert node.name == name


@pytest.mark.parametrize(
    ("sql", "names"),
    [
        pytest.param(ANN_SEARCH, {"cat", "q"}, id="ann-search"),
        pytest.param(HYBRID_SEARCH, {"dv", "sv"}, id="hybrid-search"),
        pytest.param(FULL_TEXT_SEARCH, {"q"}, id="full-text-search"),
        pytest.param(INSERT, {"emb", "cat"}, id="insert"),
        pytest.param(
            "DELETE FROM items WHERE category = :cat", {"cat"}, id="delete"
        ),
    ],
)
def test_every_bind_parameter_is_reachable_by_find_all(sql, names) -> None:
    """``find_all`` is how Track B collects the parameter set for a statement.

    ``find_all`` walks breadth-first, so the *order* of the results is not source order -- compare
    as a set, and never index into it.
    """
    ast = _parse(sql)

    assert {p.name for p in ast.find_all(exp.Placeholder)} == names
    # The whole point: no bind parameter was inlined into the tree as a literal or an identifier.
    assert not any(l.this in names for l in ast.find_all(exp.Literal))  # noqa: E741


def test_vector_parameter_never_degrades_to_text() -> None:
    """The core architectural invariant, stated negatively.

    If ``:q`` ever parsed as anything but a Placeholder -- a Column named "q", a string literal, an
    ``exp.Var`` -- Track B would have to embed the vector in the query text, which is the failure
    this whole design exists to prevent.
    """
    ast = _parse(ANN_SEARCH)
    distance = ast.args["order"].expressions[0].this

    assert isinstance(distance.expression, exp.Placeholder)
    assert not [c for c in ast.find_all(exp.Column) if c.name == "q"]
    assert not [v for v in ast.find_all(exp.Var) if v.name == "q"]


# ------------------------------------------------------------------------------------------------
# 2. Distance operators
# ------------------------------------------------------------------------------------------------

DISTANCE_OPERATORS = [
    pytest.param("<->", exp.Distance, "L2", id="l2"),
    pytest.param("<#>", InnerProduct, "IP", id="inner-product"),
    pytest.param("<=>", CosineDistance, "COSINE", id="cosine"),
    pytest.param("<+>", L1Distance, "L1", id="l1"),
]


@pytest.mark.parametrize(
    ("operator", "node_type", "metric"), DISTANCE_OPERATORS
)
def test_distance_operator_maps_to_node_and_metric(
    operator, node_type, metric
) -> None:
    ast = _parse(
        f"SELECT id FROM items ORDER BY embedding {operator} :q LIMIT 3"
    )
    distance = ast.args["order"].expressions[0].this

    assert type(distance) is node_type
    assert METRIC_TYPES[type(distance)] == metric
    # A distance node is a Binary: column on the left, bind parameter on the right.
    assert isinstance(distance, exp.Binary)
    assert distance.this.name == "embedding"
    assert distance.expression.name == "q"


@pytest.mark.parametrize(
    ("operator", "node_type", "metric"), DISTANCE_OPERATORS
)
def test_distance_operator_inside_hybrid_arm(
    operator, node_type, metric
) -> None:
    """Same node model inside HYBRID SEARCH as in ORDER BY -- SearchArm.this is the binary node."""
    ast = _parse(
        f"SELECT id FROM items HYBRID SEARCH (embedding {operator} :dv) LIMIT 3"
    )
    (arm,) = ast.args[HYBRID_ARG].expressions

    assert isinstance(arm, SearchArm)
    assert type(arm.this) is node_type
    assert METRIC_TYPES[type(arm.this)] == metric


def test_metric_types_covers_every_distance_node_and_nothing_else() -> None:
    """A new distance operator that forgets its METRIC_TYPES entry must fail here, not in Track B.

    The second assertion is the one with teeth: it derives the set of distance nodes from the module
    (every Binary we define) instead of restating the literal four.
    """
    assert set(METRIC_TYPES) == {
        exp.Distance,
        InnerProduct,
        CosineDistance,
        L1Distance,
    }

    defined_binaries = {
        obj
        for obj in vars(milvus_exp).values()
        if isinstance(obj, type)
        and issubclass(obj, exp.Binary)
        and obj.__module__ == milvus_exp.__name__
    }
    assert defined_binaries <= set(METRIC_TYPES)
    assert set(METRIC_TYPES.values()) == {"L2", "IP", "COSINE", "L1"}


def test_cosine_operator_is_not_null_safe_equality() -> None:
    """D3's guard, from the AST side: ``<=>`` must never yield exp.NullSafeEQ in this dialect."""
    ast = _parse("SELECT id FROM items ORDER BY embedding <=> :q LIMIT 3")

    assert not list(ast.find_all(exp.NullSafeEQ))
    assert list(ast.find_all(CosineDistance))


# ------------------------------------------------------------------------------------------------
# 3. ANN search -- the full translator plan
# ------------------------------------------------------------------------------------------------


def test_ann_search_plan() -> None:
    plan = ann_search_plan(_parse(ANN_SEARCH))

    assert plan == {
        "table": "items",
        "output_fields": ["id", "category"],
        "filter": "category = :cat",
        "column": "embedding",
        "metric": "L2",
        "placeholder": "q",
        "limit": 10,
        "offset": None,
        "search_params": {"ef_search": 64},
    }


def test_ann_search_plan_with_offset_and_multiple_search_params() -> None:
    sql = (
        "SELECT id FROM items ORDER BY embedding <=> :q LIMIT 10 OFFSET 5 "
        "SEARCH PARAMS (ef_search=64, nprobe=16)"
    )
    plan = ann_search_plan(_parse(sql))

    assert plan["limit"] == 10
    assert plan["offset"] == 5
    assert plan["metric"] == "COSINE"
    assert plan["search_params"] == {"ef_search": 64, "nprobe": 16}
    assert plan["filter"] is None


def test_ann_search_filter_is_a_live_expression_not_a_string() -> None:
    """Track B compiles the filter into a Milvus boolean expression, so it needs the subtree."""
    where = _parse(ANN_SEARCH).args["where"]

    assert isinstance(where, exp.Where)
    assert isinstance(where.this, exp.EQ)
    assert where.this.this.name == "category"
    assert isinstance(where.this.expression, exp.Placeholder)


def test_ann_search_plan_survives_reversed_operands() -> None:
    """``:q <-> embedding`` is legal and parses swapped -- position-based extraction would break."""
    sql = "SELECT id FROM items ORDER BY :q <-> embedding LIMIT 10"
    plan = ann_search_plan(_parse(sql))

    assert (plan["column"], plan["metric"], plan["placeholder"]) == (
        "embedding",
        "L2",
        "q",
    )


def test_a_vector_search_is_recognised_by_a_distance_node_in_order_by() -> (
    None
):
    """How Track B routes a SELECT: search() vs hybrid_search() vs a plain query().

    Nothing marks a statement as "a vector search" other than the presence of a distance node in
    ORDER BY (or a HYBRID SEARCH clause), so this is the discriminator -- and it is the reason
    METRIC_TYPES is worth having as a public mapping rather than a private lookup.
    """
    distance_nodes = tuple(METRIC_TYPES)

    def is_ann_search(select: exp.Select) -> bool:
        order = select.args.get("order")
        return bool(order and order.find(*distance_nodes))

    assert is_ann_search(_parse(ANN_SEARCH))
    assert not is_ann_search(
        _parse("SELECT id FROM items ORDER BY id LIMIT 10")
    )
    assert not is_ann_search(
        _parse("SELECT id FROM items WHERE category = :cat LIMIT 10")
    )
    # A hybrid search has no ORDER BY at all -- it is routed on the HYBRID_ARG key instead.
    hybrid = _parse(HYBRID_SEARCH)
    assert not is_ann_search(hybrid)
    assert hybrid.args.get(HYBRID_ARG)


def test_parser_accepts_a_mixed_order_by() -> None:
    """Documented gap, not a passing grade: ``ORDER BY <distance>, id`` parses.

    Milvus cannot honour a secondary sort key on a vector search, so a translator has to reject
    this itself; the AST gives it what it needs to (an Order with two expressions, only one of
    which carries a distance node).
    """
    ast = _parse("SELECT id FROM items ORDER BY embedding <-> :q, id LIMIT 10")

    assert [type(o.this).__name__ for o in ast.args["order"].expressions] == [
        "Distance",
        "Column",
    ]


def test_search_params_node_shape() -> None:
    search_params = _parse(ANN_SEARCH).args[SEARCH_PARAMS_ARG]

    assert isinstance(search_params, SearchParams)
    assert [type(p) for p in search_params.expressions] == [exp.Property]
    # exp.Property, not EQ(Column, Literal): "ef_search" is a knob name, not a column reference, and
    # a phantom Column here would be picked up by qualify() and lineage analysis.
    assert search_params.expressions[0].name == "ef_search"


def test_select_arg_types_declare_the_custom_modifier_keys() -> None:
    """Import-time mutation of ``Select.arg_types``; without it validate_expression rejects both."""
    assert HYBRID_ARG in exp.Select.arg_types
    assert SEARCH_PARAMS_ARG in exp.Select.arg_types


# ------------------------------------------------------------------------------------------------
# 4. Hybrid search
# ------------------------------------------------------------------------------------------------


def test_hybrid_search_plan() -> None:
    plan = hybrid_search_plan(_parse(HYBRID_SEARCH))

    assert plan == {
        "table": "items",
        "arms": [
            {
                "column": "embedding",
                "metric": "COSINE",
                "placeholder": "dv",
                "weight": 0.7,
            },
            {
                "column": "sparse_emb",
                "metric": "IP",
                "placeholder": "sv",
                "weight": 0.3,
            },
        ],
        "rerank": "RRF",
        "rerank_params": {"k": 60},
        "limit": 10,
    }


def test_hybrid_search_node_shape() -> None:
    hybrid = _parse(HYBRID_SEARCH).args[HYBRID_ARG]

    assert isinstance(hybrid, HybridSearch)
    assert [type(a) for a in hybrid.expressions] == [SearchArm, SearchArm]
    # SearchArm.this is the whole binary distance node, not a (column, op, param) triple -- one
    # operator model everywhere means _distance_parts works on ORDER BY and on arms alike.
    assert isinstance(hybrid.expressions[0].this, exp.Binary)
    assert isinstance(hybrid.args["rerank"], Rerank)
    assert [type(p) for p in hybrid.args["rerank"].args["expressions"]] == [
        exp.Property
    ]


def test_hybrid_search_arm_without_weight() -> None:
    """Weight is optional; the plan must report None rather than inventing a default."""
    sql = "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv, sparse_emb <#> :sv) LIMIT 10"
    plan = hybrid_search_plan(_parse(sql))

    assert [a["weight"] for a in plan["arms"]] == [None, None]
    assert plan["rerank"] is None
    assert plan["rerank_params"] == {}


def test_fractional_literals_arrive_as_decimal_not_float() -> None:
    """A sharp edge worth pinning: ``to_py()`` is not float-typed for fractional numbers.

    Anything Track B forwards to pymilvus (arm weights, ``radius``, ``range_filter``) has to be
    coerced, because ``Decimal("0.7") == 0.7`` is False and json/grpc encoders reject Decimal.
    """
    arm = _parse(HYBRID_SEARCH).args[HYBRID_ARG].expressions[0]
    assert isinstance(arm.args["weight"].to_py(), decimal.Decimal)

    sql = "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 5 SEARCH PARAMS (radius=0.8)"
    radius = (
        _parse(sql)
        .args[SEARCH_PARAMS_ARG]
        .expressions[0]
        .args["value"]
        .to_py()
    )
    assert isinstance(radius, decimal.Decimal)
    assert float(radius) == 0.8


def test_hybrid_search_rerank_without_params() -> None:
    sql = "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv) RERANK WEIGHTED LIMIT 10"
    plan = hybrid_search_plan(_parse(sql))

    assert plan["rerank"] == "WEIGHTED"
    assert plan["rerank_params"] == {}


# ------------------------------------------------------------------------------------------------
# 5. CREATE TABLE
# ------------------------------------------------------------------------------------------------


def test_create_table_plan() -> None:
    plan = create_table_plan(_parse(CREATE_TABLE))

    assert plan == {
        "table": "items",
        "fields": [
            {
                "name": "id",
                "type": "BIGINT",
                "dim": None,
                "is_primary": True,
                "auto_id": True,
            },
            {
                "name": "embedding",
                "type": "VECTOR",
                "dim": 768,
                "is_primary": False,
                "auto_id": False,
            },
            {
                "name": "category",
                "type": "VARCHAR",
                "dim": 64,
                "is_primary": False,
                "auto_id": False,
            },
        ],
        "properties": {
            "shards": 2,
            "consistency_level": "Bounded",
            "partition_key": "category",
        },
    }


def test_vector_dimension_is_recoverable_as_an_int() -> None:
    """FieldSchema(dim=...) takes an int, so the 768 must not arrive as the string "768"."""
    schema = _parse(CREATE_TABLE).this
    embedding = next(c for c in schema.expressions if c.name == "embedding")
    kind = embedding.kind

    assert isinstance(kind, exp.DataType)
    assert kind.this is exp.DataType.Type.VECTOR
    (param,) = kind.expressions
    assert isinstance(param, exp.DataTypeParam)

    dim = param.this.to_py()
    assert dim == 768
    assert isinstance(dim, int)


def test_create_table_property_value_kinds() -> None:
    """Three different value node types in one WITH clause -- the extractor must handle all three.

    ``partition_key=category`` is the awkward one: an unquoted value is an ``exp.Var``, and
    ``Var.to_py()`` raises ``ValueError`` rather than returning its text.
    """
    properties = _parse(CREATE_TABLE).args["properties"]
    by_name = {p.name: p.args["value"] for p in properties.expressions}

    assert isinstance(by_name["shards"], exp.Literal)
    assert by_name["shards"].to_py() == 2
    assert by_name["consistency_level"].to_py() == "Bounded"
    assert isinstance(by_name["partition_key"], exp.Var)
    with pytest.raises(ValueError):
        by_name["partition_key"].to_py()
    assert by_name["partition_key"].name == "category"


def test_add_field_carries_a_columndef() -> None:
    """ALTER TABLE ... ADD FIELD reuses ColumnDef, so the CREATE TABLE field extractor fits it."""
    ast = _parse("ALTER TABLE items ADD FIELD tag VARCHAR(32)")
    (action,) = ast.args["actions"]

    assert isinstance(action, milvus_exp.AddField)
    assert isinstance(action.this, exp.ColumnDef)
    assert action.this.name == "tag"
    assert action.this.kind.this is exp.DataType.Type.VARCHAR
    assert ast.this.name == "items"


@pytest.mark.parametrize(
    ("sql", "node_type", "replicas"),
    [
        pytest.param(
            "LOAD TABLE items WITH (replicas=2)",
            milvus_exp.LoadTable,
            2,
            id="load",
        ),
        pytest.param(
            "RELEASE TABLE items", milvus_exp.ReleaseTable, None, id="release"
        ),
    ],
)
def test_load_release_shape(sql, node_type, replicas) -> None:
    ast = _parse(sql)

    assert isinstance(ast, node_type)
    assert isinstance(ast.this, exp.Table)
    assert ast.this.name == "items"
    properties = ast.args.get("properties")
    assert (_props(properties.expressions) if properties else {}) == (
        {"replicas": replicas} if replicas else {}
    )


# ------------------------------------------------------------------------------------------------
# 6. CREATE INDEX
# ------------------------------------------------------------------------------------------------

EXPECTED_INDEX_PLAN = {
    "index": "idx_emb",
    "table": "items",
    "columns": ["embedding"],
    "method": "HNSW",
    "params": {"metric_type": "COSINE", "M": 16, "ef_construction": 200},
}


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(CREATE_INDEX, id="milvusql-order"),
        pytest.param(
            "CREATE INDEX idx_emb ON items USING hnsw (embedding) "
            "WITH (metric_type='COSINE', M=16, ef_construction=200)",
            id="pgvector-order",
        ),
    ],
)
def test_create_index_plan_is_identical_for_both_word_orders(sql) -> None:
    """D6: the two accepted spellings must produce the same plan, not merely the same text."""
    assert create_index_plan(_parse(sql)) == EXPECTED_INDEX_PLAN


def test_create_index_node_shape() -> None:
    ast = _parse(CREATE_INDEX)
    index = ast.this

    assert ast.args["kind"] == "INDEX"
    assert isinstance(index, exp.Index)
    assert isinstance(index.args["params"], exp.IndexParameters)
    # Index columns arrive wrapped in exp.Ordered even with no ASC/DESC in the source.
    (column,) = index.args["params"].args["columns"]
    assert isinstance(column, exp.Ordered)
    assert isinstance(column.this, exp.Column)
    assert isinstance(index.args["params"].args["using"], exp.Var)


# ------------------------------------------------------------------------------------------------
# 7. Traversal -- our custom nodes must not be opaque
# ------------------------------------------------------------------------------------------------
#
# find_all() walks args generically, so nodes nested under a custom expression are only reachable if
# that expression stores them as real child Expressions. Anything stored as a pre-rendered string,
# or under an args key the generator reads but the walker cannot, becomes invisible to the
# optimizer, to qualify(), and to Track B.


@pytest.mark.parametrize(
    ("sql", "columns"),
    [
        pytest.param(
            HYBRID_SEARCH, ["embedding", "id", "sparse_emb"], id="hybrid-arms"
        ),
        pytest.param(
            ANN_SEARCH,
            ["category", "category", "embedding", "id"],
            id="ann-order-by",
        ),
        pytest.param(
            FULL_TEXT_SEARCH, ["id", "text", "text"], id="match-and-bm25"
        ),
        pytest.param(CREATE_INDEX, ["embedding"], id="index-columns"),
    ],
)
def test_find_all_columns_reaches_into_custom_nodes(sql, columns) -> None:
    assert sorted(c.name for c in _parse(sql).find_all(exp.Column)) == columns


def test_find_all_reaches_placeholders_under_every_custom_node() -> None:
    sql = (
        "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT :w) RERANK RRF(k=:k) "
        "LIMIT 10 SEARCH PARAMS (ef_search=:ef)"
    )
    ast = _parse(sql)

    # :dv is under SearchArm -> CosineDistance, :w under SearchArm.weight, :k under Rerank ->
    # Property, :ef under SearchParams -> Property. All four must surface from the root.
    assert {p.name for p in ast.find_all(exp.Placeholder)} == {
        "dv",
        "w",
        "k",
        "ef",
    }


def test_custom_nodes_are_findable_by_their_own_class() -> None:
    ast = _parse(HYBRID_SEARCH)

    assert len(list(ast.find_all(HybridSearch))) == 1
    assert len(list(ast.find_all(SearchArm))) == 2
    assert len(list(ast.find_all(Rerank))) == 1
    assert len(list(_parse(ANN_SEARCH).find_all(SearchParams))) == 1
    assert len(list(_parse(FULL_TEXT_SEARCH).find_all(BM25Score))) == 1


def test_parent_links_are_wired_through_custom_nodes() -> None:
    """``parent``/``arg_key``/``parent_select`` drive replace(), scope building and lineage."""
    ast = _parse(HYBRID_SEARCH)
    hybrid = ast.args[HYBRID_ARG]

    assert hybrid.parent is ast
    assert hybrid.arg_key == HYBRID_ARG

    arm = hybrid.expressions[0]
    assert arm.parent is hybrid
    assert (arm.arg_key, arm.index) == ("expressions", 0)

    column = arm.this.this
    assert isinstance(column, exp.Column)
    assert [
        type(n).__name__ for n in (column.parent, column.parent.parent)
    ] == [
        "CosineDistance",
        "SearchArm",
    ]
    # Without this, any rule that asks "which SELECT does this column belong to?" silently skips
    # every column inside HYBRID SEARCH.
    assert column.parent_select is ast


def test_nodes_inside_custom_expressions_can_be_replaced() -> None:
    ast = _parse(HYBRID_SEARCH)
    column = next(c for c in ast.find_all(exp.Column) if c.name == "embedding")
    column.replace(exp.column("emb_v2"))

    assert "emb_v2 <=> :dv" in ast.sql(dialect="milvus")


def test_copy_deep_copies_custom_nodes() -> None:
    ast = _parse(HYBRID_SEARCH)
    clone = ast.copy()
    clone.args[HYBRID_ARG].expressions[0].set(
        "weight", exp.Literal.number("0.9")
    )

    assert ast.sql(dialect="milvus") == HYBRID_SEARCH
    assert "WEIGHT 0.9" in clone.sql(dialect="milvus")


def test_transform_rewrites_bind_parameters_inside_custom_nodes() -> None:
    """A whole-tree rewrite must see into the custom nodes; the optimizer works exactly this way."""
    ast = _parse(HYBRID_SEARCH).transform(
        lambda node: (
            exp.Literal.number(1)
            if isinstance(node, exp.Placeholder)
            else node
        )
    )

    assert not list(ast.find_all(exp.Placeholder))
    assert "embedding <=> 1 WEIGHT 0.7" in ast.sql(dialect="milvus")


def test_scope_sees_columns_inside_hybrid_search() -> None:
    """The optimizer does not use ``find_all``: ``walk_in_scope`` iterates ``node.args`` itself.

    So this is a genuinely separate traversal path, and it only reaches our arms because they hold
    real ``exp.Expression`` children rather than, say, a pre-rendered clause string.
    """
    scope = build_scope(_parse(HYBRID_SEARCH))

    assert sorted(c.name for c in scope.columns) == [
        "embedding",
        "id",
        "sparse_emb",
    ]


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(ANN_SEARCH, id="ann"),
        pytest.param(HYBRID_SEARCH, id="hybrid"),
    ],
)
def test_qualify_resolves_columns_inside_custom_nodes_without_touching_placeholders(
    sql,
) -> None:
    """The strongest traversal evidence: a real optimizer rule rewriting inside our nodes.

    It also guards the invariant from the other direction -- qualify must not decide that ``:dv``
    is an unknown column and try to resolve it.
    """
    schema = {
        "items": {
            "id": "BIGINT",
            "category": "VARCHAR",
            "embedding": "VECTOR",
            "sparse_emb": "VECTOR",
        }
    }
    qualified = qualify(_parse(sql), schema=schema, dialect="milvus")
    out = qualified.sql(dialect="milvus")

    assert all(c.table == "items" for c in qualified.find_all(exp.Column))
    assert {p.name for p in qualified.find_all(exp.Placeholder)} == {
        p.name for p in _parse(sql).find_all(exp.Placeholder)
    }
    assert '"items"."embedding"' in out
    # Knob names must stay Vars: had SEARCH PARAMS/RERANK parsed "ef_search=64" as EQ(Column, 64),
    # qualify would now be trying to resolve a column that does not exist.
    assert not [
        c
        for c in qualified.find_all(exp.Column)
        if c.name in {"ef_search", "k"}
    ]


# ------------------------------------------------------------------------------------------------
# Known gap
# ------------------------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="known gap: a foreign generator drops HYBRID SEARCH / SEARCH PARAMS silently",
)
@pytest.mark.parametrize(
    "dialect",
    [
        pytest.param(None, id="default"),
        pytest.param("postgres", id="postgres"),
    ],
)
def test_foreign_dialect_generation_must_not_silently_drop_search_clauses(
    dialect,
) -> None:
    """Generating a MilvusQL AST with any other dialect throws the vector search away.

    ``SELECT ... HYBRID SEARCH (...) RERANK ... LIMIT 10`` comes back as ``SELECT ... LIMIT 10``:
    a full scan with the same shape as the intended query and no error, not even under
    ``unsupported_level=RAISE`` -- nothing calls ``self.unsupported`` because the base generator
    simply never looks at the extra ``Select.args`` keys. The invariant that a vector never becomes
    text is intact, but the invariant that a query never quietly changes meaning is not.

    Fixing it means teaching the base generator to complain about unknown modifier args, which is an
    upstream change; the realistic mitigation is Track B never calling ``.sql()`` without
    ``dialect="milvus"``. Pinned strict-xfail so that a fix flips this test to a failure and forces
    the decision to be revisited.
    """
    ast = _parse(HYBRID_SEARCH)

    assert "HYBRID SEARCH" in ast.sql(
        dialect=dialect, unsupported_level=ErrorLevel.RAISE
    )
