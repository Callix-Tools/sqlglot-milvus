"""The ``milvus`` sqlglot dialect: tokenizer, parser and generator for MilvusQL."""

from __future__ import annotations

import typing as t

from sqlglot import exp, generator, tokens
from sqlglot.dialects.dialect import Dialect, NormalizationStrategy
from sqlglot.helper import ensure_list
from sqlglot.parsers.base import BaseParser
from sqlglot.tokens import TokenType

from ._tokens import (
    TT_HYBRID_SEARCH,
    TT_INNER_PRODUCT,
    TT_L1,
    TT_RELEASE,
    TT_SEARCH_PARAMS,
)
from .expressions import (
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

#: What a HYBRID SEARCH arm is allowed to score by. The four distance operators are the spec's
#: ``<vector_column> <dist-op> <param>``; ``BM25_SCORE`` is admitted because a dense+full-text
#: hybrid is a first-class Milvus query and the golden corpus has always carried one.
_SEARCH_ARM_NODES = (
    exp.Distance,
    InnerProduct,
    CosineDistance,
    L1Distance,
    BM25Score,
)

#: What may legally follow a search arm's expression when there is no WEIGHT value.
_ARM_TERMINATORS = frozenset({TokenType.COMMA, TokenType.R_PAREN})

#: ``IndexParameters`` args MilvusQL has no syntax for. The parser can populate ``where`` itself,
#: and a ``postgres -> milvus`` transpile can hand us any of the rest.
_UNSUPPORTED_INDEX_ARGS = (
    "where",
    "include",
    "partition_by",
    "tablespace",
    "on",
)


def _null_safe_eq_unsupported(
    self: generator.Generator, expression: exp.NullSafeEQ
) -> str:
    """Refuse to emit MySQL's ``<=>`` rather than silently reinterpreting it.

    In MilvusQL ``<=>`` is cosine distance. If we rendered ``exp.NullSafeEQ`` as ``<=>`` here, a
    ``mysql -> milvus`` transpile would quietly turn a null-safe comparison into a vector search:
    same six characters, completely different query, no error anywhere. So we register the
    unsupported message -- which raises outright under ``unsupported_level=RAISE`` -- and otherwise
    fall back to the unambiguous standard spelling.
    """
    self.unsupported(
        "NULL-safe equality (<=>) has no MilvusQL equivalent; in MilvusQL <=> is cosine distance"
    )
    return self.binary(expression, "IS NOT DISTINCT FROM")


def _search_arm_sql(self: generator.Generator, e: SearchArm) -> str:
    """Render one hybrid-search arm.

    The weight is tested for *presence*, not truth. ``self.sql(node, key)`` returns ``""`` for any
    falsy arg, so guarding on truthiness dropped a weight of ``0`` -- a real weight, and the one
    that means "ignore this arm" -- without a word, while a raw ``1`` raised ``ValueError``.
    Non-Expression weights are rejected explicitly so ``0`` fails exactly as every other raw number
    already does instead of emitting a dangling ``WEIGHT``.
    """
    this = self.sql(e, "this")
    weight = e.args.get("weight")
    if weight is None:
        return this
    if not isinstance(weight, exp.Expression):
        msg = f"Unsupported expression type {type(weight).__name__}"
        raise ValueError(msg)
    return f"{this} WEIGHT {self.sql(weight)}"


def _recording_modifier(parser: t.Callable) -> t.Callable:
    """Wrap a ``QUERY_MODIFIER_PARSERS`` entry so it records *where* it fired.

    D5's clause-order rule needs the source order of LIMIT, HYBRID SEARCH and SEARCH PARAMS. The
    obvious implementation -- re-scan the raw token range afterwards looking for those token types
    -- cannot tell a clause from an identifier that merely shares its name, so
    ``WHERE limit = 1``, ``ORDER BY limit`` and ``:limit`` all became spurious ParseErrors, while
    ``FETCH FIRST n ROWS ONLY`` (same ``limit`` arg, different token) was invisible. Recording the
    index at the moment the modifier parser actually runs is exact by construction.
    """

    def wrapped(self: BaseParser) -> tuple[str, t.Any]:
        token, index = self._curr, self._index
        result = parser(self)
        positions = getattr(self, "_milvus_modifier_positions", None)
        # `positions is None` only when a caller invoked a modifier parser outside the modifier
        # loop; there is no query to validate in that case, so recording would be meaningless.
        if positions is not None and result and result[1] is not None:
            positions.append((result[0], index, token))
        return result

    return wrapped


class Milvus(Dialect):
    """MilvusQL -- a SQL surface over Milvus collections, indexes and vector search.

    Deliberately derived from the bare :class:`Dialect` rather than from ``Postgres`` or ``MySQL``.
    ``MySQL`` is the only built-in dialect that *emits* ``<=>``, so inheriting from it would open a
    silent round-trip corruption channel for the very operator we rebind; ``Postgres`` would drag in
    ``%(name)s`` placeholder output, which conflicts with the ``named`` paramstyle Track B relies on.
    """

    SUPPORTS_USER_DEFINED_TYPES = True

    # Spec §1.3: Milvus is case-sensitive about collection and field names, so `items` and `Items`
    # are different objects. The inherited default is LOWERCASE, under which normalize_identifiers()
    # and qualify() would silently retarget a query at a collection that does not exist.
    NORMALIZATION_STRATEGY = NormalizationStrategy.CASE_SENSITIVE

    # Milvus has no NULLS FIRST/LAST concept; declaring the default keeps the *parser* from
    # inventing a nulls_first flag the generator would then have to refuse (see ordered_sql).
    NULL_ORDERING = "nulls_are_last"

    class Tokenizer(tokens.Tokenizer):
        KEYWORDS = {
            **tokens.Tokenizer.KEYWORDS,
            # Distance operators. "<->" and "<=>" are already single tokens in the base tokenizer
            # (LR_ARROW and NULLSAFE_EQ); the other two would otherwise shatter into "<" + "#>"
            # and "<" + "+" + ">".
            "<#>": TT_INNER_PRODUCT,
            "<+>": TT_L1,
            # Multi-word keywords. Registering the *pair* means the individual words stay ordinary
            # identifiers -- "SELECT hybrid, search, params FROM t" keeps working -- while the
            # clause opener is a single token that no alias rule will swallow.
            "HYBRID SEARCH": TT_HYBRID_SEARCH,
            "SEARCH PARAMS": TT_SEARCH_PARAMS,
            # Necessarily single-word, so genuinely reserved here; the parser un-reserves it.
            "RELEASE": TT_RELEASE,
        }

    class Parser(BaseParser):
        # Distance operators bind at FACTOR precedence, so "a <=> b + c" groups as
        # "(a <=> b) + c" -- consistent with how pgvector's operators behave in Postgres.
        FACTOR = {
            **BaseParser.FACTOR,
            TT_INNER_PRODUCT: InnerProduct,
            TT_L1: L1Distance,
            TokenType.NULLSAFE_EQ: CosineDistance,
        }

        # Removing NULLSAFE_EQ from EQUALITY is not optional: comparison parsing runs before
        # FACTOR, so leaving it in place means "embedding <=> :q" silently yields exp.NullSafeEQ
        # and the FACTOR entry above never fires.
        EQUALITY = {
            k: v
            for k, v in BaseParser.EQUALITY.items()
            if k is not TokenType.NULLSAFE_EQ
        }

        FUNCTIONS = {
            **BaseParser.FUNCTIONS,
            **BM25Score.default_parser_mappings(),
        }

        # RELEASE owns a TokenType so it can open a statement, which would otherwise make it a
        # reserved word. Add it back everywhere identifiers are accepted: STATEMENT_PARSERS is
        # consulted before any expression parsing, so "RELEASE TABLE x" still wins at statement
        # position while "SELECT release FROM t" keeps working.
        #
        # Each of these is a *separate snapshot* of the base ID_VAR_TOKENS taken at class-creation
        # time, not a view of it -- patching ID_VAR_TOKENS alone left `SELECT a release FROM t`,
        # `UPDATE t release SET ...`, `SELECT release(a) FROM t` and the two comment/window alias
        # positions still rejecting a word the README promises works everywhere.
        ID_VAR_TOKENS = BaseParser.ID_VAR_TOKENS | {TT_RELEASE}
        TABLE_ALIAS_TOKENS = BaseParser.TABLE_ALIAS_TOKENS | {TT_RELEASE}
        ALIAS_TOKENS = BaseParser.ALIAS_TOKENS | {TT_RELEASE}
        UPDATE_ALIAS_TOKENS = BaseParser.UPDATE_ALIAS_TOKENS | {TT_RELEASE}
        UNNEST_OFFSET_ALIAS_TOKENS = BaseParser.UNNEST_OFFSET_ALIAS_TOKENS | {
            TT_RELEASE
        }
        FUNC_TOKENS = BaseParser.FUNC_TOKENS | {TT_RELEASE}
        COMMENT_TABLE_ALIAS_TOKENS = BaseParser.COMMENT_TABLE_ALIAS_TOKENS | {
            TT_RELEASE
        }
        WINDOW_ALIAS_TOKENS = BaseParser.WINDOW_ALIAS_TOKENS | {TT_RELEASE}
        # Derived from the *upstream* set rather than from our ID_VAR_TOKENS: upstream's value is
        # ID_VAR_TOKENS | {STRAIGHT_JOIN}, so assigning ID_VAR_TOKENS here would have dropped
        # ":straight_join" -- a legal named placeholder in the paramstyle Track B depends on.
        COLON_PLACEHOLDER_TOKENS = BaseParser.COLON_PLACEHOLDER_TOKENS | {
            TT_RELEASE
        }

        # Our two clauses bind to the last branch of a set operation exactly as LIMIT does, so they
        # have to be hoisted onto the exp.SetOperation with it. Otherwise the union's LIMIT and the
        # branch's SEARCH PARAMS end up on different nodes and the generator emits them in the
        # D5-illegal order -- output this very dialect refuses to re-read.
        SET_OP_MODIFIERS = BaseParser.SET_OP_MODIFIERS | {
            HYBRID_ARG,
            SEARCH_PARAMS_ARG,
        }

        STATEMENT_PARSERS = {
            **BaseParser.STATEMENT_PARSERS,
            TokenType.LOAD: lambda self: self._parse_milvus_load(),
            TT_RELEASE: lambda self: self._parse_milvus_release(),
        }

        # Every entry is wrapped, ours and upstream's alike, so that LIMIT and FETCH are recorded
        # by the same mechanism as HYBRID SEARCH and SEARCH PARAMS.
        QUERY_MODIFIER_PARSERS = {
            key: _recording_modifier(parser)
            for key, parser in {
                **BaseParser.QUERY_MODIFIER_PARSERS,
                TT_HYBRID_SEARCH: lambda self: (
                    HYBRID_ARG,
                    self._parse_hybrid_search(),
                ),
                TT_SEARCH_PARAMS: lambda self: (
                    SEARCH_PARAMS_ARG,
                    self._parse_search_params(),
                ),
            }.items()
        }
        QUERY_MODIFIER_TOKENS = set(QUERY_MODIFIER_PARSERS)

        ALTER_PARSERS = {
            **BaseParser.ALTER_PARSERS,
            "ADD": lambda self: self._parse_milvus_alter_add(),
        }

        # ---------------------------------------------------------------------------------
        # Statements
        # ---------------------------------------------------------------------------------

        def _parse_milvus_load(self):
            """``LOAD TABLE x [WITH (...)]``, falling through to ``LOAD DATA ...``."""
            index = self._index
            if not self._match(TokenType.TABLE):
                self._retreat(index)
                return super()._parse_load()
            this = self._parse_table_parts()
            return self.expression(
                LoadTable(this=this, properties=self._parse_properties())
            )

        def _parse_milvus_release(self):
            """``RELEASE TABLE x``, falling through to RELEASE-as-identifier."""
            index = self._index
            if not self._match(TokenType.TABLE):
                # Not our statement. Rewind *past* the RELEASE token itself and parse from there
                # as an ordinary expression -- retreating to the token and re-dispatching would
                # land back in this method and recurse forever.
                self._retreat(index - 1)
                return self._parse_expression()
            return self.expression(
                ReleaseTable(this=self._parse_table_parts())
            )

        def _parse_milvus_alter_add(self):
            if self._match_text_seq("FIELD"):
                # _parse_exists first, exactly as _parse_alter_table_add does for ADD COLUMN.
                # Without it "IF NOT EXISTS" is fed to _parse_field_def, which reads it as the
                # expression `CASE WHEN NOT EXISTS THEN tag END` and loses the field name.
                exists = self._parse_exists(not_=True)
                return ensure_list(
                    self.expression(
                        AddField(this=self._parse_field_def(), exists=exists)
                    )
                )
            return self._parse_alter_table_add()

        # ---------------------------------------------------------------------------------
        # Search clauses
        # ---------------------------------------------------------------------------------

        def _parse_search_params(self):
            self._advance()
            return self.expression(
                SearchParams(expressions=self._parse_wrapped_properties())
            )

        def _parse_hybrid_search(self):
            self._advance()
            arms = self._parse_wrapped_csv(self._parse_search_arm)
            rerank = None
            if self._match_text_seq("RERANK"):
                # any_token=True would happily accept `RERANK 60`, producing Var(60) -- a strategy
                # name that is not a name. Check the token *before* consuming it so the existing
                # "unexpected token" diagnostics for `RERANK LIMIT 10` still surface too.
                curr = self._curr
                if not curr or curr.token_type not in self.ID_VAR_TOKENS:
                    self.raise_error("RERANK expects a strategy name")
                kind = self._parse_var(any_token=True)
                params = (
                    self._parse_wrapped_properties()
                    if self._match(TokenType.L_PAREN, advance=False)
                    else None
                )
                rerank = self.expression(Rerank(this=kind, expressions=params))
            return self.expression(
                HybridSearch(expressions=arms, rerank=rerank)
            )

        def _parse_search_arm(self):
            # _parse_assignment, never _parse_expression: the latter consumes a trailing bare word
            # as an implicit alias, so "embedding <=> :dv WEIGHT 0.7" would parse as an expression
            # aliased "WEIGHT" and then choke on the number.
            this = self._parse_assignment()
            # An arm is a *scoring* expression. Without this check `HYBRID SEARCH (embedding)`,
            # `(1 + 2)` and `('x')` all produce a non-Command Select that round-trips perfectly --
            # the whole D12 contract passing on nonsense -- and Track B then fails at runtime on a
            # METRIC_TYPES lookup instead of at parse time. `this is None` is left to the missing
            # -keyword check in SearchArm, which already names the right thing.
            if this is not None and not isinstance(this, _SEARCH_ARM_NODES):
                self.raise_error(
                    "a HYBRID SEARCH arm must be <column> <distance-operator> <param>"
                )

            weight = None
            if self._match_text_seq("WEIGHT"):
                weight = self._parse_number()
                if weight is None and not self._match_set(
                    _ARM_TERMINATORS, advance=False
                ):
                    # `WEIGHT -0.3` otherwise reports a bare "Expecting )" pointing at the minus,
                    # naming neither WEIGHT nor the reason. A WEIGHT with *nothing* after it is a
                    # separate, pinned defect and is deliberately left to that report.
                    self.raise_error("WEIGHT expects a number")
            return self.expression(SearchArm(this=this, weight=weight))

        # ---------------------------------------------------------------------------------
        # Indexes
        # ---------------------------------------------------------------------------------

        def _parse_index_params(self):
            """Accept both MilvusQL's and pgvector's word order.

            MilvusQL:  ``ON items (embedding) USING HNSW WITH (...)``
            pgvector:  ``ON items USING hnsw (embedding) WITH (...)``

            Supporting both costs one branch and makes ``postgres -> milvus`` index transpilation
            work without any rewriting.
            """
            if not self._match(TokenType.L_PAREN, advance=False):
                return super()._parse_index_params()

            columns = self._parse_wrapped_csv(self._parse_with_operator)
            using = (
                self._parse_var(any_token=True)
                if self._match(TokenType.USING)
                else None
            )
            with_storage = (
                self._match(TokenType.WITH)
                and self._parse_wrapped_properties()
            )
            where = self._parse_where()
            return self.expression(
                exp.IndexParameters(
                    using=using,
                    columns=columns,
                    with_storage=with_storage,
                    where=where,
                )
            )

        # ---------------------------------------------------------------------------------
        # Clause-order enforcement
        # ---------------------------------------------------------------------------------

        def _parse_query_modifiers(self, this):
            # Saved and restored rather than simply reset: a modifier may itself contain a
            # subquery, whose own modifiers must be validated against each other and not against
            # the enclosing query's.
            outer = getattr(self, "_milvus_modifier_positions", None)
            self._milvus_modifier_positions = []
            try:
                this = super()._parse_query_modifiers(this)
                self._validate_clause_order(self._milvus_modifier_positions)
            finally:
                self._milvus_modifier_positions = outer
            return this

        def _validate_clause_order(
            self, positions: list[tuple[str, int, t.Any]]
        ) -> None:
            """Reject non-canonical clause order.

            sqlglot's modifier loop is a bare ``while True`` with no positional state, so it
            happily accepts ``SEARCH PARAMS (...) LIMIT 10`` and then regenerates it as
            ``LIMIT 10 SEARCH PARAMS (...)`` -- input accepted, different text out. MilvusQL is a
            language we define and (via Track B's compiler) mostly generate ourselves, so there is
            no legacy corpus to be lenient toward, and silently reordering a user's SQL is worse
            than refusing it.

            ``positions`` is whatever :func:`_recording_modifier` collected for *this* query, so
            "the LIMIT clause" here means the clause and never a column, alias or placeholder that
            happens to be spelled ``limit``. Both ``LIMIT n`` and ``FETCH FIRST n ROWS ONLY`` fill
            the same ``limit`` slot and are therefore recorded under the same key.
            """
            first: dict[str, tuple[int, t.Any]] = {}
            for key, index, token in positions:
                first.setdefault(key, (index, token))

            hybrid = first.get(HYBRID_ARG)
            params = first.get(SEARCH_PARAMS_ARG)
            limit = first.get("limit")

            if hybrid and limit and hybrid[0] > limit[0]:
                self.raise_error("HYBRID SEARCH must precede LIMIT", hybrid[1])
            if params and limit and params[0] < limit[0]:
                self.raise_error("SEARCH PARAMS must follow LIMIT", params[1])
            if hybrid and params and hybrid[0] > params[0]:
                self.raise_error(
                    "HYBRID SEARCH must precede SEARCH PARAMS", hybrid[1]
                )

    class Generator(generator.Generator):
        TRANSFORMS = {
            **generator.Generator.TRANSFORMS,
            InnerProduct: lambda self, e: self.binary(e, "<#>"),
            L1Distance: lambda self, e: self.binary(e, "<+>"),
            CosineDistance: lambda self, e: self.binary(e, "<=>"),
            exp.NullSafeEQ: _null_safe_eq_unsupported,
            LoadTable: lambda self, e: (
                f"LOAD TABLE {self.sql(e, 'this')}"
                + (
                    f" {self.sql(e, 'properties')}"
                    if e.args.get("properties")
                    else ""
                )
            ),
            ReleaseTable: lambda self, e: (
                f"RELEASE TABLE {self.sql(e, 'this')}"
            ),
            AddField: lambda self, e: (
                "ADD FIELD "
                + ("IF NOT EXISTS " if e.args.get("exists") else "")
                + self.sql(e, "this")
            ),
            # Both clause bodies are guarded on the rendered text: an empty `expressions` list
            # satisfies arg_types at construction time but renders as "SEARCH PARAMS ()", which
            # this dialect's own parser rejects.
            SearchParams: lambda self, e: (
                f"{self.seg('SEARCH PARAMS')} ({body})"
                if (body := self.expressions(e, flat=True))
                else ""
            ),
            SearchArm: _search_arm_sql,
            Rerank: lambda self, e: (
                self.sql(e, "this")
                + (
                    f"({self.expressions(e, flat=True)})"
                    if e.args.get("expressions")
                    else ""
                )
            ),
            HybridSearch: lambda self, e: (
                f"{self.seg('HYBRID SEARCH')} ({body})"
                + (
                    f" RERANK {self.sql(e, 'rerank')}"
                    if e.args.get("rerank")
                    else ""
                )
                if (body := self.expressions(e, flat=True))
                else ""
            ),
        }

        def offset_limit_modifiers(self, expression, fetch, limit):
            # HYBRID SEARCH belongs immediately before LIMIT (D5). Appending it to
            # query_modifiers' *sqls instead -- the obvious spelling -- splices it in ahead of
            # JOIN/WHERE/GROUP BY/HAVING, i.e. directly after FROM, which is not a valid clause
            # position in any SQL surface and does not re-parse to the same AST.
            return [
                self.sql(expression, HYBRID_ARG),
                *super().offset_limit_modifiers(expression, fetch, limit),
            ]

        def after_limit_modifiers(self, expression):
            return [
                *super().after_limit_modifiers(expression),
                self.sql(expression, SEARCH_PARAMS_ARG),
            ]

        def ordered_sql(self, e: exp.Ordered) -> str:
            # MilvusQL has no NULLS FIRST/LAST grammar, but sqlglot's own builders and the MySQL
            # front end set nulls_first=True for a plain `ORDER BY id`. Emitting it would invent a
            # clause the user never wrote and this parser would have to read back.
            if e.args.get("nulls_first"):
                self.unsupported(
                    "MilvusQL has no NULLS FIRST/LAST; null ordering is fixed"
                )
                e = e.copy()
                e.set("nulls_first", False)
            return super().ordered_sql(e)

        def matchagainst_sql(self, e: exp.MatchAgainst) -> str:
            # Identical to the base rendering except for the space before "(", which the MilvusQL
            # spec requires: MATCH(text) AGAINST (:q).
            modifier = e.args.get("modifier")
            modifier = f" {modifier}" if modifier else ""
            return f"{self.func('MATCH', *e.expressions)} AGAINST ({self.sql(e, 'this')}{modifier})"

        def indexparameters_sql(self, e: exp.IndexParameters) -> str:
            # This override renders only what MilvusQL has syntax for, so anything else present on
            # the node is *dropped*. Silently widening a partial index to the whole table is the
            # same class of failure the <=> guard exists to prevent, so say so.
            for key in _UNSUPPORTED_INDEX_ARGS:
                if e.args.get(key):
                    self.unsupported(
                        f"MilvusQL indexes do not support {key.upper()}"
                    )

            columns = self.expressions(e, key="columns", flat=True)
            columns = f" ({columns})" if columns else ""
            using = self.sql(e, "using")
            using = f" USING {using.upper()}" if using else ""
            with_storage = self.expressions(e, key="with_storage", flat=True)
            with_storage = f" WITH ({with_storage})" if with_storage else ""
            return f"{columns}{using}{with_storage}"
