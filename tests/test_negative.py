"""Everything MilvusQL must reject -- and the exact shape of what it wrongly accepts.

The governing principle is **no silent degradation**: invalid MilvusQL must raise a ``ParseError``
that points at the offending token, or fail loudly at generation time. A quietly wrong AST is the
worst possible outcome, because every downstream consumer (Track B's compiler, ``qualify``,
lineage) then works from a query the user never wrote.

Three sqlglot behaviours make that principle hard to test, and shape this whole file:

* **``exp.Command``** -- sqlglot's fallback for syntax it cannot parse. It stores the raw text, so
  it regenerates *byte-identically* while being completely opaque (``find_all(exp.Column) == []``).
  A round-trip assertion therefore proves nothing on its own; see :func:`assert_accepted` and
  :func:`test_no_milvusql_construct_degrades_to_command`.
* **``ErrorLevel``** -- ``WARN`` and ``IGNORE`` swallow errors and hand back a truncated or silently
  *reordered* AST. Every parse in this suite goes through :func:`parse`, which pins ``RAISE``.
* **implicit aliases** -- a single bare word at the end of a statement is a legal column/table
  alias, so it is absorbed before the trailing-token check ever runs (§P9, pinned below).

The ``test_known_defect_*`` tests and the ``xfail(strict=True)`` markers below are not decoration:
each one is a case where the current parser violates the principle above. They are written as
executable bug reports so that fixing one turns the suite red and forces the marker to be removed.
"""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError

from sqlglot_milvus import (
    HybridSearch,
    LoadTable,
    ReleaseTable,
    SearchArm,
    SearchParams,
)

# --------------------------------------------------------------------------------------------
# Contract helpers
# --------------------------------------------------------------------------------------------

#: Every canonical MilvusQL statement as ``(id, sql, node_type)``, where ``node_type`` is what the
#: parse *must* produce. Kept as one list so the Command guard below cannot drift out of sync with
#: the language surface: a new construct is added here or it is not tested at all.
CANONICAL: list[tuple[str, str, type]] = [
    (
        "create-table",
        "CREATE TABLE items (id BIGINT PRIMARY KEY AUTO_INCREMENT, embedding VECTOR(768), "
        "category VARCHAR(64)) WITH (shards=2, consistency_level='Bounded', "
        "partition_key=category)",
        exp.Create,
    ),
    (
        "create-index",
        "CREATE INDEX idx_emb ON items (embedding) USING HNSW "
        "WITH (metric_type='COSINE', M=16, ef_construction=200)",
        exp.Create,
    ),
    ("load-table", "LOAD TABLE items WITH (replicas=2)", LoadTable),
    ("release-table", "RELEASE TABLE items", ReleaseTable),
    (
        "insert",
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        exp.Insert,
    ),
    ("delete", "DELETE FROM items WHERE category = :cat", exp.Delete),
    (
        "ann-search",
        "SELECT id, category FROM items WHERE category = :cat ORDER BY embedding <-> :q "
        "LIMIT 10 SEARCH PARAMS (ef_search=64)",
        exp.Select,
    ),
    (
        "hybrid-search",
        "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7, "
        "sparse_emb <#> :sv WEIGHT 0.3) RERANK RRF(k=60) LIMIT 10",
        exp.Select,
    ),
    (
        "full-text-search",
        "SELECT id FROM items WHERE MATCH(text) AGAINST (:q) "
        "ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10",
        exp.Select,
    ),
    (
        "alter-add-field",
        "ALTER TABLE items ADD FIELD tag VARCHAR(32)",
        exp.Alter,
    ),
    (
        "l1-distance",
        "SELECT id FROM items ORDER BY embedding <+> :q LIMIT 5",
        exp.Select,
    ),
    (
        "inner-product",
        "SELECT id FROM items ORDER BY embedding <#> :q LIMIT 5",
        exp.Select,
    ),
]


def parse(sql: str) -> exp.Expression:
    """The only parse entry point in this file: strictest error level, milvus dialect."""
    return sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)


def assert_accepted(sql: str, node_type: type) -> exp.Expression:
    """Assert all four properties of the D12 test contract.

    Round-tripping alone is worthless -- ``exp.Command`` round-trips perfectly -- so the node type
    and the Command guard carry the actual signal. The final re-parse catches generators that emit
    text this parser cannot read back, which is a live bug in UNION (pinned at the bottom of this
    file), not a hypothetical.
    """
    ast = parse(sql)
    assert not isinstance(ast, exp.Command), (
        f"degraded to an opaque Command: {sql}"
    )
    assert isinstance(ast, node_type), (
        f"expected {node_type.__name__}, got {type(ast).__name__}"
    )
    out = ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
    assert out == sql
    assert repr(parse(out)) == repr(ast)
    return ast


def assert_rejected(
    sql: str, description: str = "", *, highlight: str | None = None
) -> ParseError:
    """Assert ``sql`` raises, optionally with a given message and the token it points at.

    ``highlight`` is the substring sqlglot underlines in its rendered error. Asserting it is how we
    test "with a useful position" -- an error that points at the wrong token is nearly as bad as no
    error, and P5 showed that is a real failure mode for new keywords. ``description`` is omitted
    only by the ``xfail`` tests below, where the defect is that nothing is raised at all.
    """
    with pytest.raises(ParseError) as excinfo:
        parse(sql)

    error = excinfo.value
    # At ErrorLevel.RAISE sqlglot *accumulates* errors and raises them merged at the end of the
    # parse, so one bad statement can carry several descriptions; a substring check is correct.
    assert description in str(error), f"expected {description!r} in:\n{error}"
    if highlight is not None:
        assert highlight in [d.get("highlight") for d in error.errors]
    return error


# --------------------------------------------------------------------------------------------
# D5 -- clause order
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)",
        "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv) RERANK RRF(k=60) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (e <=> :d) LIMIT 10 SEARCH PARAMS (ef_search=64)",
    ],
    ids=[
        "params-after-limit",
        "hybrid-before-limit",
        "hybrid-rerank-limit",
        "both-clauses",
    ],
)
def test_canonical_clause_order_is_accepted(sql) -> None:
    # The validator is a token scan, which is exactly the kind of code that starts rejecting valid
    # input. Pinning the canonical orders is what makes the rejection tests below trustworthy.
    assert_accepted(sql, exp.Select)


@pytest.mark.parametrize(
    ("sql", "description", "highlight"),
    [
        (
            "SELECT id FROM items SEARCH PARAMS (ef_search=64) LIMIT 10",
            "SEARCH PARAMS must follow LIMIT",
            "SEARCH PARAMS",
        ),
        (
            "SELECT id FROM items LIMIT 10 HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7)",
            "HYBRID SEARCH must precede LIMIT",
            "HYBRID SEARCH",
        ),
        (
            "SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef=1) HYBRID SEARCH (e <=> :d)",
            "HYBRID SEARCH must precede SEARCH PARAMS",
            "HYBRID SEARCH",
        ),
    ],
    ids=["params-before-limit", "hybrid-after-limit", "hybrid-after-params"],
)
def test_non_canonical_clause_order_is_rejected(
    sql, description, highlight
) -> None:
    assert_rejected(sql, description, highlight=highlight)


def test_clause_order_is_enforced_at_the_default_error_level() -> None:
    # Users calling plain parse_one() get ErrorLevel.IMMEDIATE, not RAISE. D5 would be worthless if
    # it only fired under the strict level this suite happens to use.
    with pytest.raises(ParseError, match="SEARCH PARAMS must follow LIMIT"):
        sqlglot.parse_one(
            "SELECT id FROM items SEARCH PARAMS (a=1) LIMIT 10", read="milvus"
        )


@pytest.mark.parametrize(
    "level", [ErrorLevel.WARN, ErrorLevel.IGNORE], ids=["warn", "ignore"]
)
def test_lenient_error_levels_silently_reorder_the_query(level) -> None:
    # This is the whole reason `parse()` above pins RAISE, and it is worth a test rather than a
    # comment: under WARN/IGNORE the D5 violation is swallowed *and the clause is moved*, so the
    # user gets back a different query than they wrote, with a log line as the only trace.
    ast = sqlglot.parse_one(
        "SELECT id FROM items SEARCH PARAMS (a=1) LIMIT 10",
        read="milvus",
        error_level=level,
    )
    assert (
        ast.sql(dialect="milvus")
        == "SELECT id FROM items LIMIT 10 SEARCH PARAMS (a=1)"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM items SEARCH PARAMS (ef_search=64)",
        "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv)",
        "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv) SEARCH PARAMS (ef_search=64)",
    ],
    ids=["params-only", "hybrid-only", "both-no-limit"],
)
def test_search_clauses_without_limit_are_accepted(sql) -> None:
    """A missing LIMIT is a *semantic* error, not a syntactic one -- deliberately not our job.

    The spec's ANN grammar writes ``LIMIT <n>`` unbracketed, so Milvus will reject these queries:
    there is no top-k to send. But D5 is a rule about clause *order*, and with no LIMIT token there
    is no order to violate -- nothing is silently reordered and nothing is silently dropped, so the
    "no silent degradation" principle is not engaged. Requiring LIMIT here would also make the
    dialect unable to parse the fragments and sub-selects that sqlglot callers routinely hand it.
    The arity check belongs in Track B, where an actual top-k value is needed to build the call.
    """
    assert_accepted(sql, exp.Select)


def test_limit_inside_a_subquery_does_not_trigger_the_order_check() -> None:
    # The validator scans raw tokens, so it has to track paren depth by hand. A nested LIMIT is the
    # obvious way for that to produce a false positive on a perfectly legal query.
    assert_accepted(
        "SELECT id FROM items WHERE x IN (SELECT y FROM u LIMIT 3) "
        "LIMIT 10 SEARCH PARAMS (ef=1)",
        exp.Select,
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM (SELECT id FROM items HYBRID SEARCH (e <=> :d) LIMIT 5) AS x LIMIT 3",
        "SELECT * FROM (SELECT id FROM items LIMIT 5 SEARCH PARAMS (a=1)) AS x LIMIT 3",
    ],
    ids=["hybrid-in-subquery", "params-in-subquery"],
)
def test_search_clauses_in_a_subquery_are_scoped_to_that_subquery(sql) -> None:
    # The inner clause is correctly ordered relative to the *inner* LIMIT; the outer LIMIT must not
    # be dragged into the comparison.
    assert_accepted(sql, exp.Select)


# --------------------------------------------------------------------------------------------
# Duplicate clauses
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "description"),
    [
        (
            "SELECT id FROM items HYBRID SEARCH (a <=> :x) HYBRID SEARCH (b <=> :y) LIMIT 10",
            "Found multiple 'HYBRID SEARCH' clauses",
        ),
        (
            "SELECT id FROM items LIMIT 10 SEARCH PARAMS (a=1) SEARCH PARAMS (b=2)",
            "Found multiple 'SEARCH PARAMS' clauses",
        ),
        # Control: the same machinery that catches ours already catches a built-in modifier. If
        # this ever starts passing, the two above are passing for the wrong reason.
        (
            "SELECT id FROM items LIMIT 10 LIMIT 5",
            "Found multiple 'LIMIT' clauses",
        ),
    ],
    ids=["two-hybrid", "two-search-params", "two-limit-control"],
)
def test_duplicate_clauses_are_rejected(sql, description) -> None:
    # Free, but only because the clauses are keyed on dedicated TokenTypes: sqlglot's modifier loop
    # reports the *token text*, so "HYBRID SEARCH" in the message is evidence D2's multi-word
    # keyword really is a single token.
    assert_rejected(sql, description)


def test_second_rerank_is_rejected() -> None:
    # RERANK is parsed by hand, outside the modifier loop, so it gets no duplicate check for free.
    assert_rejected(
        "SELECT id FROM items HYBRID SEARCH (e <=> :d) RERANK RRF(k=60) RERANK WEIGHTED LIMIT 10",
        "Invalid expression / Unexpected token",
    )


# --------------------------------------------------------------------------------------------
# Malformed constructs
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "description"),
    [
        # Unclosed parens.
        (
            "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7 LIMIT 10",
            "Expecting )",
        ),
        (
            "SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef_search=64",
            "Expecting )",
        ),
        (
            "CREATE INDEX i ON items (embedding USING HNSW",
            "Expected table name",
        ),
        # Missing operands. Each of the four distance operators has to fail on its own: <-> is
        # sqlglot's built-in exp.Distance, <=> is our FACTOR rebinding of NULLSAFE_EQ, and <#>/<+>
        # ride on recycled TokenTypes -- three different code paths.
        (
            "SELECT id FROM items ORDER BY embedding <-> LIMIT 10",
            "Invalid expression",
        ),
        (
            "SELECT id FROM items ORDER BY embedding <=> LIMIT 10",
            "Invalid expression",
        ),
        (
            "SELECT id FROM items ORDER BY embedding <#> LIMIT 10",
            "Invalid expression",
        ),
        (
            "SELECT id FROM items ORDER BY embedding <+> LIMIT 10",
            "Invalid expression",
        ),
        (
            "SELECT id FROM items ORDER BY embedding <->",
            "Required keyword: 'expression' missing",
        ),
        (
            "SELECT id FROM items ORDER BY <-> :q LIMIT 10",
            "Required keyword: 'this' missing",
        ),
        # WEIGHT with a non-numeric operand. (A *missing* operand is a known defect -- see below.)
        (
            "SELECT id FROM items HYBRID SEARCH (e <=> :d WEIGHT 'x') LIMIT 10",
            "Expecting )",
        ),
        (
            "SELECT id FROM items HYBRID SEARCH (e <=> :d WEIGHT 0.7 0.3) LIMIT 10",
            "Expecting )",
        ),
        # RERANK with no strategy.
        (
            "SELECT id FROM items HYBRID SEARCH (e <=> :d) RERANK LIMIT 10",
            "Invalid expression / Unexpected token",
        ),
        (
            "SELECT id FROM items HYBRID SEARCH (e <=> :d) RERANK",
            "Required keyword: 'this' missing",
        ),
        # Empty or absent clause bodies.
        (
            "SELECT id FROM items LIMIT 10 SEARCH PARAMS ()",
            "Required keyword: 'expressions'",
        ),
        (
            "SELECT id FROM items HYBRID SEARCH () LIMIT 10",
            "Required keyword: 'this' missing",
        ),
        ("SELECT id FROM items HYBRID SEARCH LIMIT 10", "Expecting ("),
        (
            "SELECT id FROM items HYBRID SEARCH (e <=> :d,) LIMIT 10",
            "Required keyword: 'this'",
        ),
        # SEARCH PARAMS bodies that are not properties.
        ("SELECT id FROM items LIMIT 10 SEARCH PARAMS (a)", "Expecting )"),
        (
            "SELECT id FROM items LIMIT 10 SEARCH PARAMS (a=)",
            "Required keyword: 'value'",
        ),
        # Statement keywords with nothing to operate on.
        ("RELEASE TABLE", "Expected table name"),
    ],
    ids=[
        "unclosed-hybrid-paren",
        "unclosed-search-params-paren",
        "unclosed-index-paren",
        "l2-missing-rhs",
        "cosine-missing-rhs",
        "ip-missing-rhs",
        "l1-missing-rhs",
        "l2-at-eof",
        "l2-missing-lhs",
        "weight-string",
        "weight-two-numbers",
        "rerank-no-strategy",
        "rerank-at-eof",
        "empty-search-params",
        "empty-hybrid-search",
        "hybrid-search-no-parens",
        "hybrid-trailing-comma",
        "search-params-bare-word",
        "search-params-no-value",
        "release-table-no-name",
    ],
)
def test_malformed_constructs_are_rejected(sql, description) -> None:
    assert_rejected(sql, description)


# A HYBRID SEARCH arm that is not a scoring expression is the worst shape in this file: it parses
# into a non-Command exp.Select and round-trips text-identically, so the whole D12 contract passes
# on nonsense and the failure surfaces in Track B, at runtime, as a KeyError on METRIC_TYPES.
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM items HYBRID SEARCH (embedding) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (1 + 2) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH ('x') LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (embedding = :dv) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (a <-> :x, b) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (embedding WEIGHT 0.5) LIMIT 10",
    ],
    ids=[
        "bare-column",
        "arithmetic",
        "literal",
        "equality",
        "second-arm-bare",
        "weighted-column",
    ],
)
def test_a_search_arm_must_be_a_scoring_expression(sql) -> None:
    assert_rejected(
        sql, "a HYBRID SEARCH arm must be <column> <distance-operator> <param>"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM items HYBRID SEARCH (a <-> :x) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (a <#> :x) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (a <=> :x) LIMIT 10",
        "SELECT id FROM items HYBRID SEARCH (a <+> :x) LIMIT 10",
        # BM25_SCORE is admitted alongside the four distance operators: dense + full-text is a
        # first-class Milvus hybrid, not a loophole. Anything *else* is rejected above.
        "SELECT id FROM items HYBRID SEARCH (a <-> :x, BM25_SCORE(text, :q)) LIMIT 10",
    ],
    ids=["l2", "ip", "cosine", "l1", "bm25"],
)
def test_every_legal_arm_shape_is_still_accepted(sql) -> None:
    assert_accepted(sql, exp.Select)


def test_rerank_strategy_must_be_a_name() -> None:
    # _parse_var(any_token=True) accepted anything at all, so `RERANK 60` produced Rerank(Var(60)):
    # a strategy name that is not a name, round-tripping perfectly all the way to Track B.
    assert_rejected(
        "SELECT id FROM items HYBRID SEARCH (e <=> :d) RERANK 60 LIMIT 10",
        "RERANK expects a strategy name",
    )
    assert_rejected(
        "SELECT id FROM items HYBRID SEARCH (e <=> :d) RERANK 'rrf' LIMIT 10",
        "RERANK expects a strategy name",
    )


def test_a_negative_weight_names_weight_in_the_error() -> None:
    # _parse_number does not consume a unary minus, so this used to fail with a bare "Expecting )"
    # pointing at the minus sign -- an error naming neither WEIGHT nor the reason.
    assert_rejected(
        "SELECT id FROM items HYBRID SEARCH (e <=> :d WEIGHT -0.3) LIMIT 10",
        "WEIGHT expects a number",
    )


def test_weight_accepts_a_placeholder() -> None:
    # Not a defect, and pinned so nobody "fixes" it: _parse_number falls through to
    # _parse_placeholder (P13), which is what makes `WEIGHT :w` work for prepared statements.
    ast = assert_accepted(
        "SELECT id FROM items HYBRID SEARCH (e <=> :d WEIGHT :w) LIMIT 10",
        exp.Select,
    )
    arm = next(ast.find_all(SearchArm))
    assert isinstance(arm.args["weight"], exp.Placeholder)


# --------------------------------------------------------------------------------------------
# Trailing garbage -- the P9 boundary
# --------------------------------------------------------------------------------------------


def test_known_limitation_a_single_trailing_word_is_absorbed_as_an_alias() -> (
    None
):
    """KNOWN LIMITATION (§P9). One trailing word is silently eaten; two or more raise.

    ``parser.py`` checks for unconsumed tokens *after* statement parsing, but a bare word in a
    position where an alias is legal has already been consumed as that alias. This is upstream
    behaviour common to every sqlglot dialect and not fixable from a dialect, so it is pinned here
    rather than fought: callers that need strictness must compare the generated SQL to their input.
    """
    ast = parse("SELECT a FROM t GARBAGE")
    assert ast.sql(dialect="milvus") == "SELECT a FROM t AS GARBAGE"
    assert ast.args["from_"].this.alias == "GARBAGE"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t GARBAGE HERE",
        "SELECT a FROM t GARBAGE TOKENS HERE",
    ],
    ids=["two-words", "three-words"],
)
def test_two_or_more_trailing_words_are_rejected(sql) -> None:
    # The boundary is exactly one word: the first is taken as the alias, the second has nowhere to
    # go and trips "Invalid expression / Unexpected token".
    assert_rejected(sql, "Invalid expression / Unexpected token")


@pytest.mark.parametrize(
    "sql",
    [
        # Alias slot already filled -> even one trailing word raises.
        "SELECT a FROM t AS GARBAGE HERE",
        # No alias slot at all after these clauses, so the single-word hole does not exist.
        "SELECT id FROM items LIMIT 10 GARBAGE",
        "SELECT id FROM items LIMIT 10 SEARCH PARAMS (a=1) GARBAGE",
        "RELEASE TABLE items GARBAGE",
        "LOAD TABLE items GARBAGE",
    ],
    ids=[
        "alias-taken",
        "after-limit",
        "after-search-params",
        "after-release",
        "after-load",
    ],
)
def test_a_single_trailing_word_is_rejected_when_no_alias_slot_is_open(
    sql,
) -> None:
    # Pins the *scope* of the P9 hole: it only exists where an implicit alias is grammatical. In
    # particular our own statements (LOAD/RELEASE) and clauses (SEARCH PARAMS) are not exposed.
    assert_rejected(sql, "Invalid expression / Unexpected token")


# --------------------------------------------------------------------------------------------
# The opaque-blob guard
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "node_type"),
    [(sql, node_type) for _, sql, node_type in CANONICAL],
    ids=[name for name, _, _ in CANONICAL],
)
def test_no_milvusql_construct_degrades_to_command(sql, node_type) -> None:
    """The single most important test in the suite.

    ``CREATE INDEX ... USING HNSW``, ``LOAD TABLE`` and ``ALTER TABLE ... ADD FIELD`` all degrade
    to ``exp.Command`` in the base dialect -- and ``exp.Command`` regenerates byte-identically, so
    all three pass a naive round-trip test with zero dialect code written. ``find_all`` catches the
    nastier variant too: ``ALTER TABLE ... DROP FIELD`` yields a perfectly ordinary ``exp.Alter``
    with an opaque ``Command`` hidden in its actions list, which a top-level isinstance check
    would wave straight through.
    """
    ast = assert_accepted(sql, node_type)
    assert not list(ast.find_all(exp.Command)), (
        f"opaque Command inside the AST: {ast!r}"
    )


def test_search_clauses_survive_generation() -> None:
    # P10: an unwired custom Select arg is dropped by query_modifiers with zero diagnostics, and
    # the resulting SQL still parses -- so this loss would be invisible to every other test here.
    ast = parse(
        "SELECT id FROM items HYBRID SEARCH (e <=> :d WEIGHT 0.7) LIMIT 10 SEARCH PARAMS (ef=1)"
    )
    assert isinstance(ast.args["hybrid"], HybridSearch)
    assert isinstance(ast.args["search_params"], SearchParams)
    out = ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
    assert "HYBRID SEARCH" in out
    assert "SEARCH PARAMS" in out


# --------------------------------------------------------------------------------------------
# D2 -- the accepted casualty of multi-word keywords
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM search params",
        "SELECT * FROM hybrid search",
        "SELECT search params FROM t",
        "SELECT * FROM t search params",
    ],
    ids=[
        "table-then-alias",
        "table-then-alias-hybrid",
        "projection",
        "alias-after-table",
    ],
)
def test_d2_casualty_adjacent_word_pairs_are_destroyed(sql) -> None:
    """D2's accepted cost: ``search`` immediately followed by ``params`` is one token.

    The tokenizer matches the *pair* before either word is offered to the parser, so a table named
    ``search`` aliased ``params`` becomes unspellable unquoted. This is the price of keeping the
    bare words usable as identifiers (the test below), and it is pinned so the trade is visible
    rather than folklore.
    """
    with pytest.raises(ParseError):
        parse(sql)


@pytest.mark.parametrize(
    "sql",
    ['SELECT * FROM "search" "params"', 'SELECT * FROM "hybrid" "search"'],
    ids=["search-params", "hybrid-search"],
)
def test_quoting_rescues_the_d2_casualty(sql) -> None:
    # The escape hatch: quoted identifiers are lexed as one token each, so the multi-word keyword
    # never matches. Anyone actually hitting the casualty above has a fix.
    ast = parse(sql)
    table = ast.args["from_"].this
    assert table.name in ("search", "hybrid")
    assert table.alias in ("params", "search")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT hybrid, search, params, rerank, weight FROM t",
        "SELECT * FROM search",
        "SELECT * FROM params",
        "SELECT release FROM t",
        "SELECT * FROM t AS release",
    ],
    ids=[
        "as-columns",
        "search-table",
        "params-table",
        "release-column",
        "release-alias",
    ],
)
def test_the_bare_words_remain_ordinary_identifiers(sql) -> None:
    # What D2 bought. RELEASE is the interesting one: it *is* reserved by the tokenizer (it has to
    # open a statement) and is handed back via ID_VAR_TOKENS/TABLE_ALIAS_TOKENS.
    assert_accepted(sql, exp.Select)


def test_known_defect_multiword_keyword_fabricates_a_spaced_identifier() -> (
    None
):
    """KNOWN DEFECT: ``AS search params`` yields ``Identifier("SEARCH PARAMS")``.

    In explicit-alias position the parser takes any token, so it accepts the merged keyword token
    and stores its *text* -- uppercased, with an embedded space, matching neither word the user
    typed. The statement is then regenerated as ``... AS SEARCH PARAMS``, which round-trips
    stably and is therefore invisible to every other check.

    Correct behaviour is to reject this: ``AS`` must be followed by exactly one identifier, and a
    merged multi-word keyword is not one. Pinned as-is because it is a two-word alias in a language
    with no two-word identifiers -- a quietly wrong AST, exactly what this file exists to prevent.
    """
    ast = parse("SELECT * FROM t AS search params")
    assert ast.args["from_"].this.alias == "SEARCH PARAMS"
    assert ast.sql(dialect="milvus") == "SELECT * FROM t AS SEARCH PARAMS"


# --------------------------------------------------------------------------------------------
# ALTER -- operations Milvus cannot perform
# --------------------------------------------------------------------------------------------
#
# The spec (§3.3) is explicit: only ADD FIELD (and RENAME) exist; dropping a field, changing a
# type and changing a vector dimension must be rejected explicitly and with a clear message, not
# pretended to have succeeded. ALTER_PARSERS overrides DROP/ALTER/MODIFY to do exactly that.


def test_alter_add_field_is_structured() -> None:
    # The control: the one supported ALTER really does produce a typed node, not a Command.
    ast = assert_accepted(
        "ALTER TABLE items ADD FIELD tag VARCHAR(32)", exp.Alter
    )
    assert isinstance(ast.args["actions"][0].this, exp.ColumnDef)


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE items DROP FIELD tag",
        "ALTER TABLE items DROP COLUMN tag",
        "ALTER TABLE items ALTER COLUMN tag TYPE VARCHAR(64)",
        "ALTER TABLE items MODIFY COLUMN tag VARCHAR(64)",
        "ALTER TABLE items ALTER FIELD embedding VECTOR(1024)",
        "ALTER TABLE items ALTER COLUMN tag SET DEFAULT 1",
    ],
    ids=[
        "drop-field",
        "drop-column",
        "alter-column-type",
        "modify-column",
        "alter-vector-dim",
        "alter-set-default",
    ],
)
def test_unsupported_alter_operations_must_be_rejected(sql) -> None:
    assert_rejected(sql)


# --------------------------------------------------------------------------------------------
# Remaining silent degradations
# --------------------------------------------------------------------------------------------


def test_weight_without_a_number_must_be_rejected() -> None:
    assert_rejected(
        "SELECT id FROM items HYBRID SEARCH (e <=> :d WEIGHT) LIMIT 10"
    )


@pytest.mark.parametrize(
    "sql", ["LOAD items", "LOAD"], ids=["load-name", "load-bare"]
)
def test_load_without_table_or_data_must_be_rejected(sql) -> None:
    assert_rejected(sql)


def test_known_defect_release_without_table_becomes_an_aliased_expression() -> (
    None
):
    """``RELEASE items`` -- a plausible typo -- parses as the expression ``RELEASE AS items``.

    This is the deliberate D2/P12 fallback: RELEASE must stay usable as an identifier, and
    re-dispatching into _parse_statement would recurse forever, so the parser rewinds and parses an
    expression. Base sqlglot does exactly the same for ``foo bar``, so it is not a Milvus
    regression -- but it does mean a mistyped RELEASE is accepted as a no-op statement.
    """
    ast = parse("RELEASE items")
    assert isinstance(ast, exp.Alias)
    assert ast.sql(dialect="milvus") == "RELEASE AS items"


def test_create_index_with_a_dangling_using_must_be_rejected() -> None:
    assert_rejected("CREATE INDEX i ON items (embedding) USING")


def test_union_search_params_binds_to_the_same_node_as_the_union_limit() -> (
    None
):
    """Was a round-trip corruption: in a UNION the trailing LIMIT is hoisted onto the ``Union`` by
    ``SET_OP_MODIFIERS`` while SEARCH PARAMS stayed on the right-hand ``Select``, so the generator
    emitted the branch's clause before the union's LIMIT -- the D5-illegal order -- and its own
    output no longer parsed. Any parse/rewrite/re-parse pipeline blew up on the second pass with an
    error pointing at SQL the user never wrote.

    Both clauses now hoist together, which is the only arrangement in which the emitted text is
    readable by the dialect that emitted it.
    """
    sql = "SELECT a FROM t LIMIT 1 UNION SELECT b FROM u LIMIT 2 SEARCH PARAMS (x=1)"
    ast = assert_accepted(sql, exp.Union)
    assert isinstance(
        ast.args["limit"], exp.Limit
    )  # LIMIT 2 is the union's, not the select's
    assert isinstance(
        ast.args["search_params"], SearchParams
    )  # ... and so is SEARCH PARAMS
    assert not ast.expression.args.get("search_params")
    # The left branch keeps its own LIMIT 1: only the *last* branch's modifiers are hoisted.
    assert isinstance(ast.this.args["limit"], exp.Limit)


@pytest.mark.parametrize(
    ("sql", "normalized"),
    [
        (
            "SELECT id FROM items HYBRID SEARCH (e <=> :d) RERANK RRF() LIMIT 10",
            "SELECT id FROM items HYBRID SEARCH (e <=> :d) RERANK RRF LIMIT 10",
        ),
        (
            "CREATE INDEX i ON items USING hnsw (embedding)",
            "CREATE INDEX i ON items (embedding) USING HNSW",
        ),
    ],
    ids=["empty-rrf-args", "pgvector-index-order"],
)
def test_accepted_normalizations_reach_a_fixpoint(sql, normalized) -> None:
    # Not defects: D6 deliberately accepts pgvector's index word order and always emits MilvusQL's.
    # What matters is that the rewrite is *idempotent* -- the text changes exactly once, and a
    # second pass changes nothing. A rewrite that kept moving would corrupt any parse/emit loop.
    out = parse(sql).sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
    assert out == normalized
    assert (
        parse(out).sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
        == normalized
    )


def test_known_wart_index_method_case_is_normalized_at_generation_not_at_parse() -> (
    None
):
    """The index method survives parsing verbatim and is uppercased only on the way out.

    So the *text* reaches a fixpoint (test above) but the *AST* does not: parse -> generate ->
    parse turns ``Var(hnsw)`` into ``Var(HNSW)``, breaking the fourth property of the D12 contract
    for any lower-case input. Harmless for equality against ``METRIC_TYPES``-style lookups only
    because nothing keys off this Var today; it would bite the moment something did. The clean fix
    is to fold the case in ``_parse_index_params`` so parse and generate agree.
    """
    ast = parse("CREATE INDEX i ON items USING hnsw (embedding)")
    index_params = ast.find(exp.IndexParameters)
    assert index_params is not None
    assert index_params.args["using"].name == "hnsw"

    out = ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
    out_index_params = parse(out).find(exp.IndexParameters)
    assert out_index_params is not None
    assert out_index_params.args["using"].name == "HNSW"
    assert repr(parse(out)) != repr(ast)


# --------------------------------------------------------------------------------------------
# D3 -- <=> must never mean null-safe equality
# --------------------------------------------------------------------------------------------


def test_null_safe_equality_cannot_round_trip_into_a_distance_operator() -> (
    None
):
    # The silent-corruption channel D3 exists to close: MySQL emits `<=>` for NullSafeEQ, which
    # this parser reads as cosine distance. Same six characters, entirely different query.
    with pytest.raises(Exception, match="NULL-safe equality"):
        sqlglot.transpile(
            "SELECT a <=> b FROM t",
            read="mysql",
            write="milvus",
            unsupported_level=ErrorLevel.RAISE,
        )


def test_null_safe_equality_falls_back_to_unambiguous_spelling() -> None:
    # Below RAISE the transpile still must not produce `<=>`; the standard spelling is unambiguous
    # in both directions.
    assert sqlglot.transpile(
        "SELECT a <=> b FROM t", read="mysql", write="milvus"
    ) == ["SELECT a IS NOT DISTINCT FROM b FROM t"]


def test_milvus_never_parses_null_safe_equality() -> None:
    ast = assert_accepted("SELECT * FROM t WHERE a <=> b", exp.Select)
    assert not list(ast.find_all(exp.NullSafeEQ))


# --------------------------------------------------------------------------------------------
# Generation from hand-built nodes
# --------------------------------------------------------------------------------------------
#
# Track B builds ASTs rather than parsing text, so the generator has to be correct for nodes no
# parse can produce. Both cases below are shapes `arg_types` happily accepts at construction time.


@pytest.mark.parametrize("clause", ["hybrid", "search_params"])
def test_an_empty_clause_body_renders_as_nothing_rather_than_unreadable_sql(
    clause,
) -> None:
    # An empty `expressions` list satisfies arg_types["expressions"] = True, and the transforms
    # rendered the parentheses unconditionally -- so the generator emitted "SEARCH PARAMS ()",
    # which this very dialect then refuses to parse.
    node = (
        HybridSearch(expressions=[])
        if clause == "hybrid"
        else SearchParams(expressions=[])
    )
    select = exp.select("id").from_("items")
    select.set(clause, node)

    out = select.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
    assert out == "SELECT id FROM items"
    assert_accepted(out, exp.Select)


def test_a_zero_weight_is_never_dropped_in_silence() -> None:
    """The transform guarded on truthiness, and ``self.sql(node, key)`` returns "" for any falsy
    arg -- so a weight of 0 (a real weight, and the one that means "ignore this arm") vanished
    while a weight of 1 raised ``ValueError``. Same input class, opposite failure modes."""
    distance = parse("SELECT a <=> :b FROM t").expressions[0]

    # A properly built literal renders, whatever its value.
    assert SearchArm(this=distance.copy(), weight=exp.Literal.number(0)).sql(
        "milvus"
    ) == ("a <=> :b WEIGHT 0")
    # A raw Python number is not a valid arg at all -- and 0 now says so, exactly like 1.
    for raw in (0, 1):
        with pytest.raises(
            ValueError, match="Unsupported expression type int"
        ):
            SearchArm(this=distance.copy(), weight=raw).sql("milvus")

    assert_accepted(
        "SELECT id FROM items HYBRID SEARCH (a <=> :b WEIGHT 0) LIMIT 1",
        exp.Select,
    )
