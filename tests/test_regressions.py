"""Regression suite: proof that the MilvusQL extensions did not break base SQL.

The dialect does five invasive things, each of which is a chance to have broken something that
has nothing to do with vector search:

    * recycles five ``TokenType`` members (``SPACE``, ``BREAK``, ``LANGUAGE``, ``PROPERTIES``,
      ``SOUNDS_LIKE``) as ``<#>``, ``<+>``, ``HYBRID SEARCH``, ``SEARCH PARAMS`` and ``RELEASE``;
    * rebinds ``<=>`` from null-safe equality to cosine distance;
    * deletes ``TokenType.NULLSAFE_EQ`` from ``EQUALITY``;
    * registers two multi-word keywords and one genuinely reserved single word;
    * injects two ``QUERY_MODIFIER_PARSERS`` and a hand-written post-parse clause-order validator.

The dominant technique here is **differential testing against sqlglot's own dialect-neutral
parser** (``read=None``). Asserting "milvus behaves exactly as stock sqlglot does" is far stronger
than asserting a hand-written expected string: it fails on any behaviour *we* changed while
staying green through upstream's own quirks and cosmetic reformatting, which are not ours to fix.

Anything genuinely broken is recorded as ``xfail(strict=True)`` asserting the *correct* behaviour,
so the finding is documented here and the test flips to a failure the moment it is fixed.
"""

from __future__ import annotations

import copy as copy_module
import pickle

import pytest
import sqlglot
from sqlglot import exp, serde
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ErrorLevel, ParseError, UnsupportedError
from sqlglot.parsers.base import BaseParser
from sqlglot.tokens import Tokenizer, TokenType

from sqlglot_milvus import (
    CosineDistance,
    HybridSearch,
    InnerProduct,
    L1Distance,
    LoadTable,
    ReleaseTable,
    SearchArm,
)
from sqlglot_milvus._tokens import (
    EXPECTED_TOKEN_TYPE_COUNT,
    TT_HYBRID_SEARCH,
    TT_INNER_PRODUCT,
    TT_L1,
    TT_RELEASE,
    TT_SEARCH_PARAMS,
)
from sqlglot_milvus.dialect import Milvus
from sqlglot_milvus.expressions import HYBRID_ARG, SEARCH_PARAMS_ARG

RECYCLED = {TT_INNER_PRODUCT, TT_L1, TT_HYBRID_SEARCH, TT_SEARCH_PARAMS, TT_RELEASE}


def parse(sql: str, dialect: str | None = "milvus"):
    return sqlglot.parse_one(sql, read=dialect, error_level=ErrorLevel.RAISE)


def behaviour(sql: str, dialect: str | None):
    """A dialect's full observable reaction to ``sql``, as a comparable tuple.

    Both halves of every differential assertion are computed at run time, so an upstream change to
    a node name or an error message moves both sides together and the test stays green. Only a
    divergence *between* the two dialects -- i.e. something we caused -- can fail it.
    """
    try:
        ast = sqlglot.parse_one(sql, read=dialect, error_level=ErrorLevel.RAISE)
    except Exception as error:  # noqa: BLE001 - the exception itself is the observation
        return ("raised", type(error).__name__, str(error).splitlines()[0])
    return (type(ast).__name__, ast.sql(dialect=dialect), isinstance(ast, exp.Command))


def _clauses(select: exp.Select) -> dict:
    """Every populated arg of a SELECT, keyed by name -- an order-insensitive stand-in for repr()."""
    return {key: repr(value) for key, value in select.args.items() if value}


def assert_stable(sql: str, node_type: type[exp.Expression]) -> exp.Expression:
    """The D12 four-property contract, reused for the MilvusQL statements this file exercises."""
    ast = parse(sql)
    assert not isinstance(ast, exp.Command), "degraded to an opaque exp.Command blob"
    assert isinstance(ast, node_type)
    out = ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
    assert out == sql
    assert repr(parse(out)) == repr(ast)
    return ast


# =================================================================================================
# 1. Identifiers that collide with the new keywords
# =================================================================================================

KEYWORD_WORDS = [
    "release",
    "search",
    "params",
    "hybrid",
    "weight",
    "rerank",
    "field",
    "text",
    "k",
    "load",
    "table",
]

IDENTIFIER_POSITIONS = {
    "column": "SELECT {w} FROM t",
    "qualified-column": "SELECT {w}.a FROM {w}",
    "column-alias": "SELECT a AS {w} FROM t",
    "table": "SELECT a FROM {w}",
    "qualified-table": "SELECT a FROM db.{w}",
    "table-alias": "SELECT a FROM t AS {w}",
    "bare-table-alias": "SELECT a FROM t {w}",
    "cte": "WITH {w} AS (SELECT 1 AS a) SELECT a FROM {w}",
    "placeholder": "SELECT a FROM t WHERE a = :{w}",
    "create-table-name": "CREATE TABLE {w} (id BIGINT)",
    "create-column-name": "CREATE TABLE t ({w} BIGINT)",
    "insert-column": "INSERT INTO t ({w}) VALUES (1)",
    "update-target": "UPDATE t SET {w} = 1",
    "group-by": "SELECT a FROM t GROUP BY {w}",
    "order-by": "SELECT a FROM t ORDER BY {w}",
    "join-on": "SELECT a FROM t JOIN u ON t.{w} = u.{w}",
    "function-1-arg": "SELECT {w}(a) FROM t",
    "function-2-args": "SELECT {w}(a, b) FROM t",
}

# FINDING 1. `RELEASE` owns a TokenType so that it can open a statement, and the parser adds it
# back to ID_VAR_TOKENS / TABLE_ALIAS_TOKENS -- but not to FUNC_TOKENS, which is the set
# _parse_function consults. So `release(a)` is not recognised as a call, falls through to the
# postfix-alias rule and becomes exp.Aliases, which regenerates as the syntactically invalid
# `release AS (a)`. Silent, and only in this one position. Fix: FUNC_TOKENS |= {TT_RELEASE}.
_RELEASE_FUNC_BUG = pytest.mark.xfail(
    strict=True,
    reason="TT_RELEASE missing from Parser.FUNC_TOKENS; see test_release_as_function_name_is_mangled",
)


def _identifier_cases():
    for label, template in IDENTIFIER_POSITIONS.items():
        for word in KEYWORD_WORDS:
            broken = word == "release" and label.startswith("function")
            yield pytest.param(
                template.format(w=word),
                marks=[_RELEASE_FUNC_BUG] if broken else [],
                id=f"{label}-{word}",
            )


@pytest.mark.parametrize("sql", list(_identifier_cases()))
def test_new_keywords_remain_usable_as_identifiers(sql):
    assert behaviour(sql, "milvus") == behaviour(sql, None)


def test_release_as_function_name_is_mangled():
    """Pins FINDING 1's *actual* output, because "wrong" understates it: the emitted text is not
    valid SQL in any dialect, and nothing warns."""
    ast = parse("SELECT release(a) FROM t")
    assert isinstance(ast.expressions[0], exp.Aliases)
    assert ast.sql("milvus") == "SELECT release AS (a) FROM t"


@pytest.mark.xfail(strict=True, reason="FINDING 1: TT_RELEASE missing from Parser.FUNC_TOKENS")
def test_release_should_be_usable_as_a_function_name():
    assert isinstance(parse("SELECT release(a) FROM t").expressions[0], exp.Anonymous)


# FINDING 2. The multi-word keywords are lexed by longest match on the *token stream*, so two bare
# identifiers that happen to sit next to each other in that order are swallowed. This only bites
# the implicit-alias spelling; `AS` or quoting both restore it. Documented limitation of D2.
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT search params FROM t",
        "SELECT hybrid search FROM t",
        "SELECT a FROM search params",
        "SELECT a FROM hybrid search",
    ],
)
def test_adjacent_bare_words_forming_a_multiword_keyword_are_rejected(sql):
    assert behaviour(sql, None)[0] == "Select"  # stock sqlglot reads these as implicit aliases
    with pytest.raises(ParseError):
        parse(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT search AS params FROM t",
        "SELECT hybrid AS search FROM t",
        'SELECT "search" "params" FROM t',
        "SELECT search, params FROM t",
        "SELECT hybrid, search FROM t",
        "SELECT a FROM t WHERE search = params",
    ],
)
def test_multiword_collision_has_a_workaround(sql):
    assert behaviour(sql, "milvus") == behaviour(sql, None)


# =================================================================================================
# 2. TokenType recycling
# =================================================================================================
#
# The whole scheme rests on an empirical claim about a *third-party* enum: that sqlglot never looks
# at these five members. These tests are the tripwire for the release that changes that.
#
# tests/test_packaging.py pins the alias -> host mapping itself; what is here is the wider sweep it
# does not do -- every registered dialect, every class-level table on BaseParser, and the English
# words behind the recycled members.


def test_token_type_member_count_is_pinned():
    assert len(list(TokenType)) == EXPECTED_TOKEN_TYPE_COUNT


@pytest.mark.parametrize(
    "name,alias",
    [
        ("SPACE", TT_INNER_PRODUCT),
        ("BREAK", TT_L1),
        ("LANGUAGE", TT_HYBRID_SEARCH),
        ("PROPERTIES", TT_SEARCH_PARAMS),
        ("SOUNDS_LIKE", TT_RELEASE),
    ],
)
def test_recycled_members_still_exist_under_the_expected_names(name, alias):
    # Looked up by *string*, so an upstream rename fails here rather than at import time with a
    # traceback that says nothing about why the name mattered.
    assert getattr(TokenType, name, None) is alias


def test_recycled_members_are_five_distinct_members():
    assert len(RECYCLED) == 5
    assert not RECYCLED & {TokenType.LIMIT, TokenType.NULLSAFE_EQ, TokenType.LOAD, TokenType.TABLE}


def test_base_tokenizer_assigns_no_keyword_to_a_recycled_member():
    assert [kw for kw, tt in Tokenizer.KEYWORDS.items() if tt in RECYCLED] == []


def test_no_builtin_dialect_assigns_a_keyword_to_a_recycled_member():
    # Dialect.classes[""] is the bare Dialect, which has no Tokenizer of its own.
    offenders = {
        name: [
            kw
            for kw, tt in getattr(dialect, "Tokenizer", Tokenizer).KEYWORDS.items()
            if tt in RECYCLED
        ]
        for name, dialect in Dialect.classes.items()
        if name != "milvus"
    }
    assert {n: kws for n, kws in offenders.items() if kws} == {}
    assert len(Dialect.classes) > 20  # guard against the loop above silently iterating nothing


def test_base_parser_gives_no_behaviour_to_a_recycled_member():
    """Recycling is only safe while these members are inert everywhere in sqlglot -- not merely
    absent from KEYWORDS, but absent from every token set and token-keyed dispatch table."""
    hits = {}
    for name, value in vars(BaseParser).items():
        if isinstance(value, (set, frozenset)):
            found = RECYCLED & set(value)
        elif isinstance(value, dict):
            found = RECYCLED & {k for k in value if isinstance(k, TokenType)}
        else:
            continue
        if found:
            hits[name] = found
    assert hits == {}


@pytest.mark.parametrize(
    "sql",
    [
        # The English words behind the recycled members must be unaffected: none of them is a
        # keyword of ours, and all of them are ordinary identifiers or base-dialect constructs.
        "SELECT space, break, language, properties FROM t",
        "SELECT SPACE(3) FROM t",
        "CREATE FUNCTION f() RETURNS INT LANGUAGE js AS 'x'",
        "CREATE TABLE t (a INT) WITH (properties=1)",
    ],
)
def test_words_behind_recycled_members_are_unaffected(sql):
    assert behaviour(sql, "milvus") == behaviour(sql, None)


# =================================================================================================
# 3. EQUALITY surgery
# =================================================================================================


def test_only_nullsafe_eq_was_removed_from_equality():
    assert set(BaseParser.EQUALITY) - set(Milvus.Parser.EQUALITY) == {TokenType.NULLSAFE_EQ}
    assert set(Milvus.Parser.EQUALITY) - set(BaseParser.EQUALITY) == set()
    # ... and it landed in FACTOR, which is where the rebinding actually happens.
    assert Milvus.Parser.FACTOR[TokenType.NULLSAFE_EQ] is CosineDistance


@pytest.mark.parametrize(
    "sql,node",
    [
        ("SELECT a FROM t WHERE a = 1", exp.EQ),
        ("SELECT a FROM t WHERE a != 1", exp.NEQ),
        ("SELECT a FROM t WHERE a <> 1", exp.NEQ),
        ("SELECT a FROM t WHERE a < 1", exp.LT),
        ("SELECT a FROM t WHERE a <= 1", exp.LTE),
        ("SELECT a FROM t WHERE a > 1", exp.GT),
        ("SELECT a FROM t WHERE a >= 1", exp.GTE),
        ("SELECT a FROM t WHERE a IS NULL", exp.Is),
        ("SELECT a FROM t WHERE a IS NOT NULL", exp.Not),
        ("SELECT a FROM t WHERE a IN (1, 2)", exp.In),
        ("SELECT a FROM t WHERE NOT a IN (1, 2)", exp.Not),
        ("SELECT a FROM t WHERE a LIKE '%x%'", exp.Like),
        ("SELECT a FROM t WHERE a ILIKE '%x%'", exp.ILike),
        ("SELECT a FROM t WHERE a NOT LIKE '%x%'", exp.Like),  # exp.Like(negate=True)
        ("SELECT a FROM t WHERE a BETWEEN 1 AND 2", exp.Between),
        ("SELECT a FROM t WHERE a IS DISTINCT FROM b", exp.NullSafeNEQ),
    ],
    ids=lambda v: v.split("WHERE ")[-1] if isinstance(v, str) else v.__name__,
)
def test_every_other_comparison_operator_survived(sql, node):
    assert behaviour(sql, "milvus") == behaviour(sql, None)
    assert isinstance(parse(sql).args["where"].this, node)


def test_nullsafe_eq_is_cosine_distance_here():
    ast = assert_stable("SELECT a <=> b FROM t", exp.Select)
    assert isinstance(ast.expressions[0], CosineDistance)


def test_mysql_nullsafe_eq_never_becomes_a_vector_search():
    # The whole point of D3's fourth guard: the six characters must not survive a transpile.
    with pytest.raises(UnsupportedError):
        sqlglot.transpile(
            "SELECT a <=> b FROM t", read="mysql", write="milvus", unsupported_level=ErrorLevel.RAISE
        )
    assert sqlglot.transpile("SELECT a <=> b FROM t", read="mysql", write="milvus") == [
        "SELECT a IS NOT DISTINCT FROM b FROM t"
    ]


def test_is_not_distinct_from_parses_but_cannot_be_re_emitted_strictly():
    """FINDING 3. The refusal to emit exp.NullSafeEQ is aimed at mysql -> milvus, but the milvus
    *parser* still accepts the ANSI spelling and produces the very node the generator refuses. So
    a query MilvusQL happily reads cannot be regenerated under unsupported_level=RAISE -- the
    node type, not its provenance, is what the guard keys on."""
    ast = parse("SELECT a FROM t WHERE a IS NOT DISTINCT FROM b")
    assert isinstance(ast.args["where"].this, exp.NullSafeEQ)
    assert ast.sql("milvus") == "SELECT a FROM t WHERE a IS NOT DISTINCT FROM b"
    with pytest.raises(UnsupportedError):
        ast.sql("milvus", unsupported_level=ErrorLevel.RAISE)


# =================================================================================================
# 4. JSON operators
# =================================================================================================
#
# `<#>` starts with `<`, so the tokenizer's longest-match trie only reaches it after `<`; `#>` at
# the start of an operator is untouched. Verified empirically below rather than argued: milvus and
# stock sqlglot agree on all four JSON operators, including the no-whitespace spellings.


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a -> 'b' FROM t",
        "SELECT a ->> 'b' FROM t",
        "SELECT a #> 'b' FROM t",
        "SELECT a #>> 'b' FROM t",
        "SELECT a#>'b' FROM t",
        "SELECT JSON_EXTRACT(a, '$.b') FROM t",
    ],
)
def test_json_operators_are_untouched_by_the_inner_product_operator(sql):
    assert behaviour(sql, "milvus") == behaviour(sql, None)


def test_json_operators_still_produce_json_nodes():
    # Pinned explicitly: an opaque `behaviour()` match would also be satisfied if *both* dialects
    # regressed to something meaningless.
    assert isinstance(parse("SELECT a -> 'b' FROM t").expressions[0], exp.JSONExtract)
    assert isinstance(parse("SELECT a #> 'b' FROM t").expressions[0], exp.JSONBExtract)


@pytest.mark.parametrize(
    "sql,node",
    [
        ("SELECT a <-> b FROM t", exp.Distance),
        ("SELECT a <#> b FROM t", InnerProduct),
        ("SELECT a <=> b FROM t", CosineDistance),
        ("SELECT a <+> b FROM t", L1Distance),
    ],
    ids=["l2", "ip", "cosine", "l1"],
)
def test_distance_operators(sql, node):
    assert isinstance(assert_stable(sql, exp.Select).expressions[0], node)


@pytest.mark.parametrize("sql", ["SELECT a <+ 1 FROM t", "SELECT a < +1 FROM t", "SELECT a <@ b FROM t"])
def test_operators_that_merely_start_like_ours_are_unaffected(sql):
    assert behaviour(sql, "milvus") == behaviour(sql, None)


# =================================================================================================
# 5. LOAD DATA still delegates
# =================================================================================================


@pytest.mark.parametrize(
    "sql",
    [
        "LOAD DATA INPATH 'x' INTO TABLE t",
        "LOAD DATA LOCAL INPATH 'x' OVERWRITE INTO TABLE t",
        "LOAD DATA INPATH 'x' INTO TABLE t PARTITION (a=1)",
    ],
)
def test_load_data_is_unchanged(sql):
    assert behaviour(sql, "milvus") == behaviour(sql, None)
    # _parse_milvus_load retreats and calls super() only when the next token is not TABLE; assert
    # the delegation actually happened rather than trusting the round-trip.
    assert isinstance(parse(sql), exp.LoadData)


@pytest.mark.parametrize("sql", ["LOAD TABLE items", "LOAD TABLE items WITH (replicas=2)"])
def test_load_table_is_ours(sql):
    assert_stable(sql, LoadTable)


# =================================================================================================
# 6. Query modifiers we did not touch
# =================================================================================================


def test_query_modifier_tokens_is_keyed_only_on_token_types_we_own():
    # The documented trap (ground truth §3.5): recomputing QUERY_MODIFIER_TOKENS locally breaks
    # every `GROUP BY <column>` *if* a modifier is keyed on TokenType.VAR. We key on dedicated
    # (recycled) members instead, which is why the recomputation below is safe.
    assert TokenType.VAR not in Milvus.Parser.QUERY_MODIFIER_TOKENS
    assert set(BaseParser.QUERY_MODIFIER_PARSERS) < Milvus.Parser.QUERY_MODIFIER_TOKENS
    assert Milvus.Parser.QUERY_MODIFIER_TOKENS - set(BaseParser.QUERY_MODIFIER_PARSERS) == {
        TT_HYBRID_SEARCH,
        TT_SEARCH_PARAMS,
    }


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t WHERE b = 1",
        "SELECT category, COUNT(*) AS c FROM items GROUP BY category",
        "SELECT category FROM items GROUP BY category HAVING COUNT(*) > 1",
        "SELECT category FROM items GROUP BY category LIMIT 5",
        "SELECT a FROM t GROUP BY a, b HAVING SUM(c) > 0 ORDER BY a LIMIT 3 OFFSET 1",
        "SELECT a FROM t GROUP BY ROLLUP(a, b)",
        "SELECT a FROM t GROUP BY CUBE(a, b)",
        "SELECT a FROM t GROUP BY GROUPING SETS ((a), (b))",
        "SELECT a FROM t GROUP BY ALL",
        "SELECT a FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY b ORDER BY c) = 1",
        "SELECT a FROM t WINDOW w AS (PARTITION BY b)",
        "SELECT a FROM t ORDER BY a DESC",
        "SELECT a FROM t LIMIT 5 OFFSET 10",
        "SELECT a FROM t OFFSET 5 ROWS FETCH FIRST 10 ROWS ONLY",
        "SELECT a FROM t FOR UPDATE",
        "SELECT a FROM t CLUSTER BY a",
        "SELECT a FROM t DISTRIBUTE BY a",
        "SELECT a FROM t SORT BY a",
        "SELECT a FROM t CONNECT BY PRIOR a = b START WITH a = 1",
        "SELECT a FROM t TABLESAMPLE (10 PERCENT)",
        "SELECT a FROM t JOIN u ON t.a = u.a",
        "SELECT a FROM t LEFT JOIN u USING (a)",
        "SELECT a FROM t, LATERAL (SELECT 1) AS x",
        "SELECT a FROM t UNION ALL SELECT b FROM u",
        "SELECT a FROM t INTERSECT SELECT b FROM u",
        "SELECT a FROM t EXCEPT SELECT b FROM u",
        "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r) SELECT n FROM r",
        "SELECT DISTINCT a FROM t",
        "SELECT a FROM t WHERE a IN (SELECT b FROM u)",
        "SELECT a FROM t WHERE EXISTS(SELECT 1 FROM u)",
    ],
)
def test_untouched_query_modifiers_behave_exactly_as_stock_sqlglot(sql):
    assert behaviour(sql, "milvus") == behaviour(sql, None)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT category FROM items GROUP BY category LIMIT 10 SEARCH PARAMS (ef=1)",
        "SELECT category FROM items GROUP BY category HAVING COUNT(*) > 1 LIMIT 10 SEARCH PARAMS (ef=1)",
    ],
)
def test_group_by_column_survives_next_to_our_modifiers(sql):
    # The precise shape the ground truth warns about: a non-empty GROUP BY list whose terminator
    # is one of our modifier tokens.
    ast = assert_stable(sql, exp.Select)
    assert [e.sql() for e in ast.args["group"].expressions] == ["category"]


@pytest.mark.parametrize(
    "sql",
    ["SELECT a FROM t ORDER BY a NULLS FIRST", "SELECT a FROM t ORDER BY a ASC NULLS LAST"],
)
def test_null_ordering_differs_from_stock_and_that_is_deliberate(sql):
    """NULL_ORDERING = "nulls_are_last" is what suppresses a spurious " NULLS LAST" in generated
    index column lists. The visible side effect is that an explicit, redundant NULLS LAST is
    dropped -- semantics preserved, text not. Pinned so a change to NULL_ORDERING shows up here."""
    assert behaviour(sql, "milvus") != behaviour(sql, None)
    assert parse(sql).sql("milvus") in (
        "SELECT a FROM t ORDER BY a NULLS FIRST",
        "SELECT a FROM t ORDER BY a ASC",
    )


# =================================================================================================
# 7. Nested contexts vs the clause-order validator
# =================================================================================================
#
# _validate_clause_order rescans a raw token range with hand-rolled paren-depth tracking. Anything
# it mistakes for a top-level LIMIT / HYBRID SEARCH / SEARCH PARAMS becomes a spurious ParseError
# on legal SQL. Every case below has an inner LIMIT (or an inner copy of our own clauses) that
# must be invisible to the outer query's check, and vice versa.

NESTED = [
    "SELECT id FROM items WHERE id IN (SELECT id FROM other LIMIT 5) LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT id FROM (SELECT id FROM other LIMIT 5) AS s LIMIT 10 SEARCH PARAMS (ef=1)",
    "WITH c AS (SELECT id FROM other LIMIT 5) SELECT id FROM c LIMIT 10 SEARCH PARAMS (ef=1)",
    "WITH a AS (SELECT 1 AS x LIMIT 1), b AS (SELECT 2 AS x LIMIT 2) SELECT x FROM a LIMIT 3 SEARCH PARAMS (ef=1)",
    "SELECT (SELECT MAX(x) FROM o LIMIT 1) AS m FROM items LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT id FROM items WHERE EXISTS(SELECT 1 FROM o LIMIT 1) LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT id FROM items WHERE a > ANY (SELECT b FROM o LIMIT 2) LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT id FROM a JOIN (SELECT id FROM b LIMIT 5) AS c ON a.id = c.id LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT id FROM a JOIN LATERAL (SELECT 1 LIMIT 1) AS x ON TRUE LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT id FROM items ORDER BY (SELECT 1 FROM x LIMIT 1) LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT CASE WHEN (SELECT COUNT(*) FROM o LIMIT 1) > 0 THEN 1 ELSE 0 END AS c FROM items LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT id FROM items GROUP BY id HAVING COUNT(*) > (SELECT COUNT(*) FROM o LIMIT 1) LIMIT 10 SEARCH PARAMS (ef=1)",
    # Three levels of derived table, each with its own LIMIT, under both of our clauses at once.
    # (HYBRID SEARCH precedes WHERE here because that is the order the generator emits -- see
    # test_clause_order_strictness_is_one_sided.)
    "SELECT id FROM items HYBRID SEARCH (e <=> :v WEIGHT 0.5, f <#> :w WEIGHT 0.5) RERANK RRF(k=60)"
    " WHERE id IN (SELECT id FROM (SELECT id FROM (SELECT id FROM w LIMIT 1) AS x LIMIT 2) AS y LIMIT 3)"
    " LIMIT 10 SEARCH PARAMS (ef=1)",
    # Inner queries carrying *our* clauses, which must not be attributed to the outer one.
    "SELECT id FROM (SELECT id FROM items LIMIT 5 SEARCH PARAMS (ef=1)) AS s LIMIT 10",
    "SELECT id FROM (SELECT id FROM items LIMIT 5 SEARCH PARAMS (ef=1)) AS s LIMIT 10 SEARCH PARAMS (ef=2)",
    "WITH c AS (SELECT id FROM items LIMIT 5 SEARCH PARAMS (ef=1)) SELECT id FROM c LIMIT 10 SEARCH PARAMS (ef=2)",
    "SELECT (SELECT id FROM o LIMIT 5 SEARCH PARAMS (ef=2)) AS x FROM items LIMIT 10 SEARCH PARAMS (ef=1)",
    "SELECT id FROM items WHERE id IN (SELECT id FROM o LIMIT 5 SEARCH PARAMS (ef=9)) LIMIT 10",
    "SELECT id FROM (SELECT id FROM o HYBRID SEARCH (e <=> :v) LIMIT 5) AS s LIMIT 10 SEARCH PARAMS (ef=1)",
    # Extra parens around the whole thing shift every token index the validator scans.
    "(SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1))",
    "((SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1)))",
    "SELECT id FROM ((SELECT id FROM items LIMIT 5 SEARCH PARAMS (ef=1))) AS s",
    # A string literal and a comment that spell a clause name must not be seen as tokens.
    "SELECT id FROM items WHERE s = 'LIMIT 1' LIMIT 10 SEARCH PARAMS (ef=1)",
    # Our clauses with no LIMIT at all, so `positions` is missing an entry the checks index into.
    "SELECT id FROM items SEARCH PARAMS (ef=1)",
    "SELECT id FROM items HYBRID SEARCH (e <=> :v)",
    "SELECT id FROM items HYBRID SEARCH (e <=> :v) SEARCH PARAMS (ef=1)",
    "SELECT id FROM items WHERE id IN (SELECT id FROM o LIMIT 1) SEARCH PARAMS (ef=1)",
]


@pytest.mark.parametrize("sql", NESTED, ids=range(len(NESTED)))
def test_nesting_never_triggers_a_spurious_clause_order_error(sql):
    assert_stable(sql, (exp.Select, exp.Subquery))


@pytest.mark.parametrize(
    "sql,message",
    [
        (
            "SELECT id FROM items SEARCH PARAMS (ef=1) LIMIT 10",
            "SEARCH PARAMS must follow LIMIT",
        ),
        (
            "SELECT id FROM items ORDER BY id SEARCH PARAMS (ef=1) LIMIT 10",
            "SEARCH PARAMS must follow LIMIT",
        ),
        (
            "SELECT id FROM items LIMIT 10 HYBRID SEARCH (e <=> :v)",
            "HYBRID SEARCH must precede LIMIT",
        ),
        (
            "SELECT id FROM items SEARCH PARAMS (ef=1) HYBRID SEARCH (e <=> :v) LIMIT 3",
            "SEARCH PARAMS must follow LIMIT",
        ),
    ],
)
def test_top_level_clause_order_violations_are_still_rejected(sql, message):
    with pytest.raises(ParseError, match=message):
        parse(sql)


@pytest.mark.parametrize(
    "sql,emitted",
    [
        (
            "SELECT id FROM items WHERE a = 1 HYBRID SEARCH (e <=> :v) LIMIT 10",
            "SELECT id FROM items HYBRID SEARCH (e <=> :v) WHERE a = 1 LIMIT 10",
        ),
        (
            "SELECT id FROM items ORDER BY id HYBRID SEARCH (e <=> :v) LIMIT 10",
            "SELECT id FROM items HYBRID SEARCH (e <=> :v) ORDER BY id LIMIT 10",
        ),
        (
            "SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1) OFFSET 5",
            "SELECT id FROM items LIMIT 10 OFFSET 5 SEARCH PARAMS (ef=1)",
        ),
    ],
)
def test_clause_order_strictness_is_one_sided(sql, emitted):
    """FINDING 4. D5 says a violation must raise rather than be silently reordered, but the
    validator only compares HYBRID SEARCH / SEARCH PARAMS / LIMIT against each other. Their order
    relative to WHERE, ORDER BY and OFFSET is unchecked, so exactly the accept-then-rewrite
    behaviour D5 forbids survives for those pairs. The AST is right; only the text moves."""
    assert parse(sql).sql("milvus") == emitted
    assert parse(emitted).sql("milvus") == emitted  # the rewritten form is a fixed point
    # repr() would differ purely because args are keyed in parse order, so compare per-clause.
    assert _clauses(parse(sql)) == _clauses(parse(emitted))


def test_search_params_survives_a_set_operation_round_trip():
    """FINDING 5. On a set operation the LIMIT binds to the exp.Union while SEARCH PARAMS binds to
    the right-hand Select, and the Union generator emits the operand (with its after-limit
    modifiers) before its own LIMIT. The result is text in the order the dialect itself rejects:
    parse -> generate -> parse raises. This is the one place a legal input does not survive a
    round trip."""
    sql = "SELECT id FROM a UNION SELECT id FROM b LIMIT 10 SEARCH PARAMS (ef=1)"
    emitted = parse(sql).sql("milvus")
    assert emitted == "SELECT id FROM a UNION SELECT id FROM b SEARCH PARAMS (ef=1) LIMIT 10"
    with pytest.raises(ParseError, match="SEARCH PARAMS must follow LIMIT"):
        parse(emitted)


@pytest.mark.xfail(
    strict=True, reason="FINDING 5: union LIMIT and SEARCH PARAMS bind to different nodes"
)
def test_search_params_should_survive_a_set_operation_round_trip():
    sql = "SELECT id FROM a UNION SELECT id FROM b LIMIT 10 SEARCH PARAMS (ef=1)"
    out = parse(sql).sql("milvus")
    assert repr(parse(out)) == repr(parse(sql))


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1) UNION SELECT id FROM b LIMIT 5",
        "(SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1)) UNION (SELECT id FROM b LIMIT 5)",
        "INSERT INTO t SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1)",
        "CREATE VIEW v AS SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1)",
        "CREATE TABLE x AS SELECT id FROM items HYBRID SEARCH (e <=> :v) LIMIT 10",
    ],
)
def test_our_modifiers_inside_other_statements(sql):
    assert_stable(sql, (exp.Union, exp.Insert, exp.Create))


# =================================================================================================
# 8. Expression copy / serialization / optimizer
# =================================================================================================

HYBRID_SQL = (
    "SELECT id FROM items"
    " HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7, sparse_emb <#> :sv WEIGHT 0.3)"
    " RERANK RRF(k=60) LIMIT 10"
)
PARAMS_SQL = (
    "SELECT id, category FROM items WHERE category = :cat"
    " ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)"
)


def test_the_arg_types_patch_leaves_ordinary_selects_alone():
    # exp.Select.arg_types is patched *globally*, so the two keys exist for every dialect. They
    # must stay optional and absent: a Select nobody asked to carry them must be byte-identical to
    # what stock sqlglot builds. (test_packaging.py pins the patch itself.)
    assert exp.Select.arg_types[HYBRID_ARG] is False
    assert exp.Select.arg_types[SEARCH_PARAMS_ARG] is False
    plain = exp.select("a").from_("t")
    assert HYBRID_ARG not in plain.args and SEARCH_PARAMS_ARG not in plain.args
    assert plain.sql("postgres") == "SELECT a FROM t"


@pytest.mark.parametrize("sql", [HYBRID_SQL, PARAMS_SQL], ids=["hybrid", "search-params"])
@pytest.mark.parametrize(
    "clone",
    [
        lambda a: a.copy(),
        copy_module.deepcopy,
        lambda a: pickle.loads(pickle.dumps(a)),
        lambda a: serde.load(serde.dump(a)),
    ],
    ids=["copy", "deepcopy", "pickle", "serde"],
)
def test_custom_select_args_survive_cloning(sql, clone):
    ast = parse(sql)
    clone_ = clone(ast)
    assert repr(clone_) == repr(ast)
    assert clone_.sql("milvus") == sql


def test_copy_is_deep_for_custom_args():
    ast = parse(HYBRID_SQL)
    clone = ast.copy()
    assert clone.args[HYBRID_ARG] is not ast.args[HYBRID_ARG]
    clone.args[HYBRID_ARG].expressions.pop()
    assert ast.sql("milvus") == HYBRID_SQL  # the original is untouched


def test_walk_and_find_all_reach_inside_custom_args():
    # If the custom args were stored anywhere other than exp.Select.args, traversal would silently
    # skip them and every optimizer rule, lineage query and Track B translator would miss them.
    ast = parse(HYBRID_SQL)
    assert len(list(ast.find_all(SearchArm))) == 2
    assert len(list(ast.find_all(CosineDistance))) == 1
    assert len(list(ast.find_all(InnerProduct))) == 1
    assert [c.sql() for c in ast.find_all(exp.Column)] == ["id", "embedding", "sparse_emb"]
    assert [p.sql() for p in ast.find_all(exp.Placeholder)] == [":dv", ":sv"]
    assert all(node.parent is not None for node in ast.find_all(SearchArm))
    assert set(ast.walk()) >= set(ast.find_all(HybridSearch))


SCHEMA = {"items": {"id": "BIGINT", "category": "VARCHAR", "embedding": "TEXT", "sparse_emb": "TEXT"}}


@pytest.mark.parametrize(
    "sql,clause,qualified",
    [
        (HYBRID_SQL, HYBRID_ARG, '"items"."embedding" <=> :dv'),
        (PARAMS_SQL, SEARCH_PARAMS_ARG, "SEARCH PARAMS (ef_search=64)"),
    ],
    ids=["hybrid", "search-params"],
)
@pytest.mark.parametrize("rule", ["qualify", "optimize"], ids=["qualify", "optimize"])
def test_optimizer_preserves_custom_args(sql, clause, qualified, rule):
    from sqlglot.optimizer import optimize
    from sqlglot.optimizer.qualify import qualify

    rewritten = (qualify if rule == "qualify" else optimize)(
        parse(sql), schema=SCHEMA, dialect="milvus"
    )
    out = rewritten.sql("milvus")
    assert clause in rewritten.args
    # The columns *inside* our clauses get qualified like any other: they are ordinary AST
    # members, not opaque payloads the optimizer has to walk past.
    assert qualified in out
    assert parse(out).sql("milvus") == out


@pytest.mark.parametrize(
    "write,expected",
    [
        ("postgres", "SELECT id FROM items LIMIT 10"),
        ("duckdb", "SELECT id FROM items LIMIT 10"),
    ],
)
def test_search_params_is_dropped_without_warning_when_writing_other_dialects(write, expected):
    """FINDING 6. Foreign generators never call after_limit_modifiers, so SEARCH PARAMS vanishes
    silently -- while a CosineDistance in the same query raises ValueError. Nothing here is a
    regression in *base* SQL, but the asymmetry means milvus -> postgres can quietly return the
    wrong rows."""
    sql = "SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef_search=64)"
    assert sqlglot.transpile(sql, read="milvus", write=write) == [expected]
    assert sqlglot.transpile(
        sql, read="milvus", write=write, unsupported_level=ErrorLevel.RAISE
    ) == [expected]  # not even under RAISE


# =================================================================================================
# 9. CREATE INDEX: the pgvector-order branch (D6)
# =================================================================================================


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE INDEX i ON items (a)",
        "CREATE UNIQUE INDEX i ON items (a)",
        "CREATE INDEX idx_emb ON items (embedding) USING HNSW WITH (metric_type='COSINE', M=16)",
    ],
)
def test_create_index_still_round_trips(sql):
    assert_stable(sql, exp.Create)


def test_create_index_accepts_pgvector_word_order():
    ast = parse("CREATE INDEX i ON items USING hnsw (embedding)")
    assert ast.sql("milvus") == "CREATE INDEX i ON items (embedding) USING HNSW"
    assert not isinstance(ast, exp.Command)


def test_partial_index_predicate_is_parsed_then_dropped():
    """FINDING 7. _parse_index_params captures `where`, but the Milvus indexparameters_sql only
    renders columns/using/with_storage. A partial index therefore silently widens to the whole
    table -- the AST holds the predicate, the SQL does not. Same override also loses `include`,
    `partition_by` and `tablespace`."""
    ast = parse("CREATE INDEX i ON items (a) WHERE a > 1")
    assert ast.find(exp.IndexParameters).args["where"].sql() == "WHERE a > 1"
    assert ast.sql("milvus") == "CREATE INDEX i ON items (a)"
    assert ast.sql("postgres") == "CREATE INDEX i ON items(a) WHERE a > 1"


@pytest.mark.xfail(strict=True, reason="FINDING 7: indexparameters_sql drops IndexParameters.where")
def test_partial_index_predicate_should_round_trip():
    assert_stable("CREATE INDEX i ON items (a) WHERE a > 1", exp.Create)


def test_index_include_degrades_to_a_command():
    """FINDING 8. The D6 branch fires on any `(` after the table name and then only understands
    USING/WITH/WHERE, so INCLUDE -- which stock sqlglot parses fine -- falls off the end and the
    whole statement degrades to an opaque exp.Command."""
    sql = "CREATE INDEX i ON items (a) INCLUDE (b)"
    assert isinstance(parse(sql, dialect=None), exp.Create)
    assert isinstance(sqlglot.parse_one(sql, read="milvus"), exp.Command)


@pytest.mark.xfail(strict=True, reason="FINDING 8: the D6 branch does not handle INCLUDE")
def test_index_include_should_parse():
    assert_stable("CREATE INDEX i ON items (a) INCLUDE (b)", exp.Create)


# =================================================================================================
# 10. The canonical statements, as an end-to-end backstop
# =================================================================================================

SPEC_STATEMENTS = [
    pytest.param(sql, node, columns, id=name)
    for name, sql, node, columns in [
        (
            "create-table",
            "CREATE TABLE items (id BIGINT PRIMARY KEY AUTO_INCREMENT, embedding VECTOR(768),"
            " category VARCHAR(64)) WITH (shards=2, consistency_level='Bounded',"
            " partition_key=category)",
            exp.Create,
            0,
        ),
        (
            "create-index",
            "CREATE INDEX idx_emb ON items (embedding) USING HNSW"
            " WITH (metric_type='COSINE', M=16, ef_construction=200)",
            exp.Create,
            1,
        ),
        ("load-table", "LOAD TABLE items WITH (replicas=2)", LoadTable, 0),
        ("release-table", "RELEASE TABLE items", ReleaseTable, 0),
        ("insert", "INSERT INTO items (embedding, category) VALUES (:emb, :cat)", exp.Insert, 0),
        ("delete", "DELETE FROM items WHERE category = :cat", exp.Delete, 1),
        (
            "vector-search",
            "SELECT id, category FROM items WHERE category = :cat"
            " ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)",
            exp.Select,
            4,
        ),
        (
            "hybrid-search",
            "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7,"
            " sparse_emb <#> :sv WEIGHT 0.3) RERANK RRF(k=60) LIMIT 10",
            exp.Select,
            3,
        ),
        (
            "full-text",
            "SELECT id FROM items WHERE MATCH(text) AGAINST (:q)"
            " ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10",
            exp.Select,
            3,
        ),
        ("alter-add-field", "ALTER TABLE items ADD FIELD tag VARCHAR(32)", exp.Alter, 0),
    ]
]


@pytest.mark.parametrize("sql,node,columns", SPEC_STATEMENTS)
def test_spec_statements_are_not_opaque_blobs(sql, node, columns):
    # The trap in one test: an exp.Command round-trips byte-identically while containing nothing,
    # so the round trip alone proves nothing. Pin the node type and the column count too.
    ast = assert_stable(sql, node)
    assert len(list(ast.find_all(exp.Column))) == columns


def test_statements_can_be_parsed_in_a_single_script():
    # Each statement gets its own token slice, so the validator's absolute token indices must be
    # slice-relative. A script is the cheapest way to prove they are.
    statements = sqlglot.parse(
        "SELECT 1; RELEASE TABLE t; LOAD TABLE t WITH (replicas=1);"
        " SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1)",
        read="milvus",
    )
    assert [type(s).__name__ for s in statements] == [
        "Select",
        "ReleaseTable",
        "LoadTable",
        "Select",
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE items ADD COLUMN tag VARCHAR(32)",
        "ALTER TABLE items ADD CONSTRAINT c UNIQUE (a)",
        "ALTER TABLE items DROP COLUMN a",
        "ALTER TABLE items RENAME TO x",
        "ALTER TABLE t ALTER COLUMN a SET DEFAULT 1",
    ],
)
def test_alter_parsers_override_still_delegates(sql):
    # ALTER_PARSERS["ADD"] is replaced wholesale; everything that is not `ADD FIELD` must reach
    # the base implementation unchanged.
    assert behaviour(sql, "milvus") == behaviour(sql, None)
