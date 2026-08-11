"""AST nodes added by the MilvusQL dialect.

Two sqlglot-30 specifics govern everything here.

First, ``exp.Func``, ``exp.Binary``, ``exp.Condition`` and ``exp.Expr`` are ``@trait`` stubs, not
concrete classes: ``class BM25Score(exp.Func)`` raises a message-less ``NotImplementedError`` the
moment you construct it. Every node must therefore inherit ``exp.Expression`` *first* and mix the
trait in second.

Second, custom modifiers on ``SELECT`` live as extra keys in ``exp.Select.args`` -- the same place
sqlglot keeps ``LIMIT``, ``QUALIFY`` and ``WINDOW``. Those keys must be declared in
``Select.arg_types`` at import time, otherwise ``validate_expression`` rejects them whenever
sqlglot's ``UNITTEST`` flag is set (i.e. under pytest, where our own suite runs).
"""

from __future__ import annotations

from sqlglot import exp

# --------------------------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------------------------


class LoadTable(exp.Expression):
    """``LOAD TABLE items WITH (replicas=2)`` -- loads a collection into memory."""

    arg_types = {"this": True, "properties": False}


class ReleaseTable(exp.Expression):
    """``RELEASE TABLE items`` -- evicts a collection from memory."""

    arg_types = {"this": True}


class AddField(exp.Expression):
    """``ALTER TABLE items ADD FIELD tag VARCHAR(32)``.

    Milvus supports adding a field but not dropping one or changing a type, so this is
    deliberately the only ``ALTER`` action MilvusQL models structurally.
    """

    arg_types = {"this": True}


# --------------------------------------------------------------------------------------------
# Distance operators
# --------------------------------------------------------------------------------------------
#
# Spelled exactly as pgvector spells them, so that a pgvector query transpiles to MilvusQL
# unchanged. ``<->`` reuses sqlglot's built-in ``exp.Distance`` (the base tokenizer already emits
# a single ``LR_ARROW`` token for it); the other three need nodes of their own.


class InnerProduct(exp.Expression, exp.Binary):
    """``<#>`` -- negative inner product. Milvus ``metric_type='IP'``."""


class L1Distance(exp.Expression, exp.Binary):
    """``<+>`` -- L1 / Manhattan distance. Milvus ``metric_type='L1'``."""


class CosineDistance(exp.Expression, exp.Binary):
    """``<=>`` -- cosine distance. Milvus ``metric_type='COSINE'``.

    Note the collision documented in the README: MySQL spells null-safe equality ``<=>``. Inside
    the MilvusQL dialect these six characters mean cosine distance and nothing else; the generator
    refuses to emit MySQL's meaning rather than silently reinterpreting it.
    """


#: Maps a distance node to the Milvus ``metric_type`` it corresponds to. Track B
#: (``sqlalchemy-milvus``) consumes this when translating an AST into ``pymilvus`` calls.
METRIC_TYPES = {
    exp.Distance: "L2",
    InnerProduct: "IP",
    CosineDistance: "COSINE",
    L1Distance: "L1",
}


# --------------------------------------------------------------------------------------------
# Search clauses
# --------------------------------------------------------------------------------------------


class SearchParams(exp.Expression):
    """``SEARCH PARAMS (ef_search=64)`` -- index-time tuning knobs for one query.

    A dedicated node rather than a bare ``exp.Properties``: the ``WITH`` prefix that ``Properties``
    renders is a class-level constant shared with ``CREATE TABLE``, so reusing it would emit
    ``WITH (...)`` here.
    """

    arg_types = {"expressions": True}


class SearchArm(exp.Expression):
    """One arm of a hybrid search: ``embedding <=> :dv WEIGHT 0.7``.

    ``this`` holds the *binary distance node*, not a loose (column, operator, parameter) triple.
    Keeping a single operator model everywhere means ``find_all``, the optimizer and Track B's
    translator all see the same shapes here as they do in ``ORDER BY embedding <-> :q``.
    """

    arg_types = {"this": True, "weight": False}


class Rerank(exp.Expression):
    """``RERANK RRF(k=60)``.

    ``expressions`` holds plain ``exp.Property`` nodes. Parsing ``k=60`` with the generic function
    parser would instead yield ``EQ(Column(k), Literal(60))`` -- a phantom column reference that
    ``qualify`` and lineage analysis would try to resolve against the schema.
    """

    arg_types = {"this": True, "expressions": False}


class HybridSearch(exp.Expression):
    """``HYBRID SEARCH (<arm>, ...) RERANK <strategy>``."""

    arg_types = {"expressions": True, "rerank": False}


# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------


class BM25Score(exp.Expression, exp.Func):
    """``BM25_SCORE(text, :q)`` -- full-text relevance score."""

    _sql_names = ["BM25_SCORE"]
    arg_types = {"this": True, "expression": True}


# --------------------------------------------------------------------------------------------
# Select modifiers
# --------------------------------------------------------------------------------------------

#: ``args`` key holding a :class:`HybridSearch`.
HYBRID_ARG = "hybrid"

#: ``args`` key holding a :class:`SearchParams`.
SEARCH_PARAMS_ARG = "search_params"

exp.Select.arg_types[HYBRID_ARG] = False
exp.Select.arg_types[SEARCH_PARAMS_ARG] = False


__all__ = [
    "AddField",
    "BM25Score",
    "CosineDistance",
    "HybridSearch",
    "HYBRID_ARG",
    "InnerProduct",
    "L1Distance",
    "LoadTable",
    "METRIC_TYPES",
    "ReleaseTable",
    "Rerank",
    "SearchArm",
    "SearchParams",
    "SEARCH_PARAMS_ARG",
]
