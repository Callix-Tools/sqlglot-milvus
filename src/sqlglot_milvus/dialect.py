"""The ``milvus`` sqlglot dialect: tokenizer, parser and generator for
MilvusQL."""

from __future__ import annotations

import typing as t

from sqlglot import exp, generator, tokens
from sqlglot.dialects.dialect import Dialect, NormalizationStrategy
from sqlglot.helper import ensure_list
from sqlglot.parsers.base import BaseParser
from sqlglot.tokens import TokenType

from ._tokens import (
    TT_CONSISTENCY_LEVEL,
    TT_HYBRID_SEARCH,
    TT_INNER_PRODUCT,
    TT_L1,
    TT_RELEASE,
    TT_SEARCH_PARAMS,
)
from .expressions import (
    CONSISTENCY_ARG,
    HYBRID_ARG,
    SEARCH_PARAMS_ARG,
    AddField,
    BM25Score,
    ConsistencyLevel,
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

#: What a HYBRID SEARCH arm is allowed to score by. The four distance
#: operators are the spec's ``<vector_column> <dist-op> <param>``;
#: ``BM25_SCORE`` is admitted because a dense+full-text hybrid is a
#: first-class Milvus query and the golden corpus has always carried one.
_SEARCH_ARM_NODES = (
    exp.Distance,
    InnerProduct,
    CosineDistance,
    L1Distance,
    BM25Score,
)

#: ``IndexParameters`` args MilvusQL has no syntax for. The parser can
#: populate ``where`` itself, and a ``postgres -> milvus`` transpile can
#: hand us any of the rest.
_UNSUPPORTED_INDEX_ARGS = (
    "where",
    "include",
    "partition_by",
    "tablespace",
    "on",
)


def _rewrite_array_capacity(node: exp.Expr | None) -> exp.Expr | None:
    """Reinterpret ``ARRAY<T>(n)`` as a capacity, not a cast.

    The base grammar already claims ``TYPE<...>(v1, v2, ...)`` as shorthand
    for ``CAST(ARRAY(v1, v2, ...) AS TYPE<...>)`` -- ``_parse_types``'s
    branch for the closing ``>`` immediately followed by ``(`` or ``[``
    (``parser.py``, right after the nested-type ``LT`` match). MilvusQL has
    no array-literal-cast syntax of its own, and the spec's ``ARRAY<T>(n)``
    has exactly that shape: one integer, immediately after the element
    type. Reusing sqlglot's own fixed-size-array ``values`` slot (the same
    one ``INT[3]`` populates) rather than inventing a new arg keeps every
    other type consumer -- the optimizer, ``find_all``, a foreign generator
    -- working unchanged; only ``datatype_sql`` needs to know the capacity
    renders in parens here instead of brackets.
    """
    if (
        isinstance(node, exp.Cast)
        and isinstance(node.this, exp.Array)
        and len(node.this.expressions) == 1
        and isinstance(node.this.expressions[0], exp.Literal)
        and node.this.expressions[0].is_int
        and isinstance(node.to, exp.DataType)
        and node.to.is_type(exp.DataType.Type.ARRAY)
    ):
        capacity = node.this.expressions[0]
        data_type = node.to.copy()
        data_type.set("values", [capacity])
        return data_type
    return node


def _null_safe_eq_unsupported(
    self: generator.Generator, expression: exp.NullSafeEQ
) -> str:
    """Refuse to emit MySQL's ``<=>`` rather than silently reinterpreting
    it.

    In MilvusQL ``<=>`` is cosine distance. If we rendered
    ``exp.NullSafeEQ`` as ``<=>`` here, a ``mysql -> milvus`` transpile
    would quietly turn a null-safe comparison into a vector search: same
    six characters, completely different query, no error anywhere. So we
    register the unsupported message -- which raises outright under
    ``unsupported_level=RAISE`` -- and otherwise fall back to the
    unambiguous standard spelling.
    """
    self.unsupported(
        "NULL-safe equality (<=>) has no MilvusQL equivalent; in MilvusQL "
        "<=> is cosine distance"
    )
    return self.binary(expression, "IS NOT DISTINCT FROM")


def _search_arm_sql(self: generator.Generator, e: SearchArm) -> str:
    """Render one hybrid-search arm.

    The weight is tested for *presence*, not truth. ``self.sql(node,
    key)`` returns ``""`` for any falsy arg, so guarding on truthiness
    dropped a weight of ``0`` -- a real weight, and the one that means
    "ignore this arm" -- without a word, while a raw ``1`` raised
    ``ValueError``. Non-Expression weights are rejected explicitly so
    ``0`` fails exactly as every other raw number already does instead of
    emitting a dangling ``WEIGHT``.
    """
    this = self.sql(e, "this")
    weight = e.args.get("weight")
    if weight is None:
        return this
    if not isinstance(weight, exp.Expression):
        msg = f"Unsupported expression type {type(weight).__name__}"
        raise ValueError(msg)
    return f"{this} WEIGHT {self.sql(weight)}"


def _recording_modifier(
    parser: t.Callable[[BaseParser], tuple[str, t.Any]],
) -> t.Callable[[BaseParser], tuple[str, t.Any]]:
    """Wrap a ``QUERY_MODIFIER_PARSERS`` entry so it records *where* it
    fired.

    D5's clause-order rule needs the source order of LIMIT, HYBRID SEARCH
    and SEARCH PARAMS. The obvious implementation -- re-scan the raw token
    range afterwards looking for those token types -- cannot tell a clause
    from an identifier that merely shares its name, so ``WHERE limit =
    1``, ``ORDER BY limit`` and ``:limit`` all became spurious
    ParseErrors, while ``FETCH FIRST n ROWS ONLY`` (same ``limit`` arg,
    different token) was invisible. Recording the index at the moment the
    modifier parser actually runs is exact by construction.
    """

    def wrapped(self: BaseParser) -> tuple[str, t.Any]:
        token, index = self._curr, self._index
        result = parser(self)
        positions = getattr(self, "_milvus_modifier_positions", None)
        # `positions is None` only when a caller invoked a modifier parser
        # outside the modifier loop; there is no query to validate in
        # that case, so recording would be meaningless.
        if positions is not None and result and result[1] is not None:
            positions.append((result[0], index, token))
        return result

    return wrapped


class Milvus(Dialect):
    """MilvusQL -- a SQL surface over Milvus collections, indexes and
    vector search.

    Deliberately derived from the bare :class:`Dialect` rather than from
    ``Postgres`` or ``MySQL``. ``MySQL`` is the only built-in dialect that
    *emits* ``<=>``, so inheriting from it would open a silent round-trip
    corruption channel for the very operator we rebind; ``Postgres`` would
    drag in ``%(name)s`` placeholder output, which conflicts with the
    ``named`` paramstyle Track B relies on.
    """

    SUPPORTS_USER_DEFINED_TYPES = True

    # Spec §1.3: Milvus is case-sensitive about collection and field
    # names, so `items` and `Items` are different objects. The inherited
    # default is LOWERCASE, under which normalize_identifiers() and
    # qualify() would silently retarget a query at a collection that does
    # not exist.
    NORMALIZATION_STRATEGY = NormalizationStrategy.CASE_SENSITIVE

    # Milvus has no NULLS FIRST/LAST concept; declaring the default keeps
    # the *parser* from inventing a nulls_first flag the generator would
    # then have to refuse (see ordered_sql).
    NULL_ORDERING = "nulls_are_last"

    class Tokenizer(tokens.Tokenizer):
        KEYWORDS: t.ClassVar[dict[str, TokenType]] = {
            **tokens.Tokenizer.KEYWORDS,
            # Distance operators. "<->" and "<=>" are already single
            # tokens in the base tokenizer (LR_ARROW and NULLSAFE_EQ); the
            # other two would otherwise shatter into "<" + "#>" and
            # "<" + "+" + ">".
            "<#>": TT_INNER_PRODUCT,
            "<+>": TT_L1,
            # Multi-word keywords. Registering the *pair* means the
            # individual words stay ordinary identifiers -- "SELECT
            # hybrid, search, params FROM t" keeps working -- while the
            # clause opener is a single token that no alias rule will
            # swallow.
            "HYBRID SEARCH": TT_HYBRID_SEARCH,
            "SEARCH PARAMS": TT_SEARCH_PARAMS,
            "CONSISTENCY LEVEL": TT_CONSISTENCY_LEVEL,
            # Necessarily single-word, so genuinely reserved here; the
            # parser un-reserves it.
            "RELEASE": TT_RELEASE,
        }

    class Parser(BaseParser):
        # Distance operators bind at FACTOR precedence, so "a <=> b + c"
        # groups as "(a <=> b) + c" -- consistent with how pgvector's
        # operators behave in Postgres.
        FACTOR: t.ClassVar = {
            **BaseParser.FACTOR,
            TT_INNER_PRODUCT: InnerProduct,
            TT_L1: L1Distance,
            TokenType.NULLSAFE_EQ: CosineDistance,
        }

        # Removing NULLSAFE_EQ from EQUALITY is not optional: comparison
        # parsing runs before FACTOR, so leaving it in place means
        # "embedding <=> :q" silently yields exp.NullSafeEQ and the
        # FACTOR entry above never fires.
        EQUALITY: t.ClassVar = {
            k: v
            for k, v in BaseParser.EQUALITY.items()
            if k is not TokenType.NULLSAFE_EQ
        }

        FUNCTIONS: t.ClassVar[dict[str, t.Callable]] = {
            **BaseParser.FUNCTIONS,
            **BM25Score.default_parser_mappings(),
        }

        # RELEASE owns a TokenType so it can open a statement, which
        # would otherwise make it a reserved word. Add it back everywhere
        # identifiers are accepted: STATEMENT_PARSERS is consulted before
        # any expression parsing, so "RELEASE TABLE x" still wins at
        # statement position while "SELECT release FROM t" keeps working.
        #
        # Each of these is a *separate snapshot* of the base
        # ID_VAR_TOKENS taken at class-creation time, not a view of it --
        # patching ID_VAR_TOKENS alone left `SELECT a release FROM t`,
        # `UPDATE t release SET ...`, `SELECT release(a) FROM t` and the
        # two comment/window alias positions still rejecting a word the
        # README promises works everywhere.
        ID_VAR_TOKENS: t.ClassVar[set] = BaseParser.ID_VAR_TOKENS | {
            TT_RELEASE
        }
        TABLE_ALIAS_TOKENS: t.ClassVar[set] = BaseParser.TABLE_ALIAS_TOKENS | {
            TT_RELEASE
        }
        ALIAS_TOKENS: t.ClassVar = BaseParser.ALIAS_TOKENS | {TT_RELEASE}
        UPDATE_ALIAS_TOKENS: t.ClassVar = BaseParser.UPDATE_ALIAS_TOKENS | {
            TT_RELEASE
        }
        UNNEST_OFFSET_ALIAS_TOKENS: t.ClassVar = (
            BaseParser.UNNEST_OFFSET_ALIAS_TOKENS | {TT_RELEASE}
        )
        FUNC_TOKENS: t.ClassVar = BaseParser.FUNC_TOKENS | {TT_RELEASE}
        COMMENT_TABLE_ALIAS_TOKENS: t.ClassVar = (
            BaseParser.COMMENT_TABLE_ALIAS_TOKENS | {TT_RELEASE}
        )
        WINDOW_ALIAS_TOKENS: t.ClassVar = BaseParser.WINDOW_ALIAS_TOKENS | {
            TT_RELEASE
        }
        # Derived from the *upstream* set rather than from our
        # ID_VAR_TOKENS: upstream's value is ID_VAR_TOKENS |
        # {STRAIGHT_JOIN}, so assigning ID_VAR_TOKENS here would have
        # dropped ":straight_join" -- a legal named placeholder in the
        # paramstyle Track B depends on.
        COLON_PLACEHOLDER_TOKENS: t.ClassVar = (
            BaseParser.COLON_PLACEHOLDER_TOKENS | {TT_RELEASE}
        )

        # Our two clauses bind to the last branch of a set operation
        # exactly as LIMIT does, so they have to be hoisted onto the
        # exp.SetOperation with it. Otherwise the union's LIMIT and the
        # branch's SEARCH PARAMS end up on different nodes and the
        # generator emits them in the D5-illegal order -- output this
        # very dialect refuses to re-read.
        SET_OP_MODIFIERS: t.ClassVar = BaseParser.SET_OP_MODIFIERS | {
            HYBRID_ARG,
            SEARCH_PARAMS_ARG,
        }

        STATEMENT_PARSERS: t.ClassVar = {
            **BaseParser.STATEMENT_PARSERS,
            TokenType.LOAD: lambda self: self._parse_milvus_load(),
            TT_RELEASE: lambda self: self._parse_milvus_release(),
        }

        # Every entry is wrapped, ours and upstream's alike, so that
        # LIMIT and FETCH are recorded by the same mechanism as HYBRID
        # SEARCH and SEARCH PARAMS.
        QUERY_MODIFIER_PARSERS: t.ClassVar = {
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
                TT_CONSISTENCY_LEVEL: lambda self: (
                    CONSISTENCY_ARG,
                    self._parse_consistency_level(),
                ),
            }.items()
        }
        QUERY_MODIFIER_TOKENS: t.ClassVar = set(QUERY_MODIFIER_PARSERS)

        # Milvus can add a field to a live collection and rename one, and
        # that is the whole list. Changing a field's type, changing a
        # vector's dimension and dropping a field are not supported by
        # the engine at all, so accepting them here -- which the
        # inherited parsers do, turning ALTER COLUMN tag TYPE VARCHAR(64)
        # into a SET DATA TYPE that reads as if something happened --
        # would promise the user an operation that can never run.
        ALTER_PARSERS: t.ClassVar = {
            **BaseParser.ALTER_PARSERS,
            "ADD": lambda self: self._parse_milvus_alter_add(),
            "DROP": lambda self: self._reject_alter("DROP"),
            "ALTER": lambda self: self._reject_alter("ALTER"),
            "MODIFY": lambda self: self._reject_alter("MODIFY"),
        }

        # ---------------------------------------------------------------
        # Statements
        # ---------------------------------------------------------------

        def _parse_milvus_load(
            self,
        ) -> LoadTable | exp.LoadData | exp.Command:
            """``LOAD TABLE x [WITH (...)]``, falling through to ``LOAD
            DATA ...``."""
            index = self._index
            if not self._match(TokenType.TABLE):
                # Anything that is not LOAD TABLE and not LOAD DATA would
                # otherwise be swallowed by the inherited fallback into
                # an opaque exp.Command that round-trips perfectly while
                # meaning nothing -- the exact failure mode the D12
                # contract exists to catch. DATA is matched by text, not
                # a TokenType (see the base _parse_load), so we do the
                # same here rather than inventing a check the base parser
                # doesn't have either.
                if self._curr is None or self._curr.text.upper() != "DATA":
                    self.raise_error("Expected TABLE or DATA after LOAD")
                self._retreat(index)
                return super()._parse_load()
            this = self._parse_table_parts()
            return self.expression(
                LoadTable(this=this, properties=self._parse_properties())
            )

        def _parse_milvus_release(self) -> exp.Expr | None:
            """``RELEASE TABLE x``, falling through to
            RELEASE-as-identifier."""
            index = self._index
            if not self._match(TokenType.TABLE):
                # Not our statement. Rewind *past* the RELEASE token
                # itself and parse from there as an ordinary expression
                # -- retreating to the token and re-dispatching would
                # land back in this method and recurse forever.
                self._retreat(index - 1)
                return self._parse_expression()
            return self.expression(
                ReleaseTable(this=self._parse_table_parts())
            )

        def _parse_milvus_alter_add(self) -> list[exp.Expr]:
            if self._match_text_seq("FIELD"):
                # _parse_exists first, exactly as _parse_alter_table_add
                # does for ADD COLUMN. Without it "IF NOT EXISTS" is fed
                # to _parse_field_def, which reads it as the expression
                # `CASE WHEN NOT EXISTS THEN tag END` and loses the field
                # name.
                exists = self._parse_exists(not_=True)
                return ensure_list(
                    self.expression(
                        AddField(this=self._parse_field_def(), exists=exists)
                    )
                )
            return self._parse_alter_table_add()

        def _parse_types(
            self,
            check_func: bool = False,
            schema: bool = False,
            allow_identifiers: bool = True,
            with_collation: bool = False,
        ) -> exp.Expr | None:
            return _rewrite_array_capacity(
                super()._parse_types(
                    check_func=check_func,
                    schema=schema,
                    allow_identifiers=allow_identifiers,
                    with_collation=with_collation,
                )
            )

        def _reject_alter(self, action: str) -> None:
            self.raise_error(
                f"Milvus cannot {action} a field: only ADD FIELD and "
                "RENAME are supported. Changing a field's type, changing "
                "a vector's dimension and dropping a field require "
                "recreating the collection."
            )

        # ---------------------------------------------------------------
        # Alias guard
        # ---------------------------------------------------------------

        #: The three multi-word keywords are tokenized as one token each
        #: (see the Tokenizer.KEYWORDS pair registration above), so a
        #: matched token's ``.text`` is two words joined by a space --
        #: something no MilvusQL identifier can ever spell (the escape
        #: hatch is quoting, which lexes each word as its own token and
        #: never produces one of these three token types). None of the
        #: three are in ``ID_VAR_TOKENS``/``ALIAS_TOKENS``, so they are
        #: only ever reachable through the ``any_token=True`` catch-all
        #: every "accept literally anything here" call site uses --
        #: explicit alias (``AS x``), implicit table alias, and
        #: everywhere else ``_parse_id_var`` is asked to be permissive.
        #: Left unguarded, ``AS search params`` parses: ``_parse_alias``
        #: matches ``AS`` and ``_parse_id_var(any_token=True, ...)``
        #: happily takes the merged token's raw text, producing
        #: ``Identifier("SEARCH PARAMS")`` that regenerates as ``AS
        #: SEARCH PARAMS`` -- a merged keyword masquerading as a
        #: one-word alias, invisible to every other check because it is
        #: a stable round-trip. ``SELECT a AS search`` still works:
        #: ``search`` alone tokenizes as an ordinary identifier, this
        #: only fires once the tokenizer has already merged two adjacent
        #: words into one of these three tokens.
        _MULTIWORD_KEYWORD_TOKENS: t.ClassVar = {
            TT_HYBRID_SEARCH,
            TT_SEARCH_PARAMS,
            TT_CONSISTENCY_LEVEL,
        }

        def _parse_id_var(
            self,
            any_token: bool = True,
            tokens: t.Collection[TokenType] | None = None,
        ) -> exp.Expr | None:
            if (
                any_token
                and self._curr
                and self._curr.token_type in self._MULTIWORD_KEYWORD_TOKENS
            ):
                return None
            return super()._parse_id_var(any_token=any_token, tokens=tokens)

        # ---------------------------------------------------------------
        # Search clauses
        # ---------------------------------------------------------------

        def _parse_search_params(self) -> SearchParams:
            self._advance()
            return self.expression(
                SearchParams(expressions=self._parse_wrapped_properties())
            )

        def _parse_consistency_level(self) -> ConsistencyLevel:
            self._advance()
            # A bare word rather than a quoted string: the levels are an
            # enum, not free text, and `CONSISTENCY LEVEL Bounded` reads
            # the way the rest of the language does. Any word is
            # accepted -- validating the set belongs to Track B, which
            # talks to a specific server version and can say what that
            # version actually supports.
            level = self._parse_var(any_token=True)
            if level is None:
                self.raise_error("CONSISTENCY LEVEL expects a level name")
            return self.expression(ConsistencyLevel(this=level))

        def _parse_hybrid_search(self) -> HybridSearch:
            self._advance()
            arms = self._parse_wrapped_csv(self._parse_search_arm)
            rerank = None
            if self._match_text_seq("RERANK"):
                # any_token=True would happily accept `RERANK 60`,
                # producing Var(60) -- a strategy name that is not a
                # name. Check the token *before* consuming it so the
                # existing "unexpected token" diagnostics for `RERANK
                # LIMIT 10` still surface too.
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

        def _parse_search_arm(self) -> SearchArm:
            # _parse_assignment, never _parse_expression: the latter
            # consumes a trailing bare word as an implicit alias, so
            # "embedding <=> :dv WEIGHT 0.7" would parse as an expression
            # aliased "WEIGHT" and then choke on the number.
            this = self._parse_assignment()
            # An arm is a *scoring* expression. Without this check
            # `HYBRID SEARCH (embedding)`, `(1 + 2)` and `('x')` all
            # produce a non-Command Select that round-trips perfectly --
            # the whole D12 contract passing on nonsense -- and Track B
            # then fails at runtime on a METRIC_TYPES lookup instead of
            # at parse time. `this is None` is left to the missing
            # -keyword check in SearchArm, which already names the right
            # thing.
            if this is not None and not isinstance(this, _SEARCH_ARM_NODES):
                self.raise_error(
                    "a HYBRID SEARCH arm must be "
                    "<column> <distance-operator> <param>"
                )

            weight = None
            if self._match_text_seq("WEIGHT"):
                weight = self._parse_number()
                if weight is None:
                    # Covers both `WEIGHT -0.3` (which otherwise reported
                    # a bare "Expecting )" pointing at the minus) and
                    # `WEIGHT` followed by a comma or the closing paren
                    # (which was silently dropped, leaving an unweighted
                    # arm the user never wrote).
                    self.raise_error("WEIGHT expects a number")
            return self.expression(SearchArm(this=this, weight=weight))

        # ---------------------------------------------------------------
        # Indexes
        # ---------------------------------------------------------------

        def _parse_index_params(self) -> exp.IndexParameters:
            """Accept both MilvusQL's and pgvector's word order.

            MilvusQL:  ``ON items (embedding) USING HNSW WITH (...)``
            pgvector:  ``ON items USING hnsw (embedding) WITH (...)``

            Supporting both costs one branch and makes ``postgres ->
            milvus`` index transpilation work without any rewriting.

            FINDING 8: this branch used to stop after ``WHERE``, so any
            trailing clause the base grammar knows about but MilvusQL
            has no syntax for (``INCLUDE``, ``PARTITION BY``, ``USING
            INDEX TABLESPACE``, a trailing ``ON``) was left unconsumed.
            An unconsumed suffix is not a parse error at this level --
            it just makes the *statement* fail to fully parse, and
            ``CREATE INDEX`` falls back whole to an opaque ``exp.Command``
            that round-trips byte-identical while carrying no structure
            at all (``find_all`` sees nothing). That is a strictly worse
            outcome than the one ``WHERE`` already gets: parsed into a
            real ``IndexParameters`` node and reported as unsupported at
            generation time (see ``indexparameters_sql``), not silently
            dropped. Parsing the same trailing clauses the base grammar
            supports -- in the base parser's own order -- gets every one
            of them the same honest treatment as ``WHERE``, instead of
            three different failure modes depending on which clause a
            caller happens to write.
            """
            if not self._match(TokenType.L_PAREN, advance=False):
                return super()._parse_index_params()

            columns = self._parse_wrapped_csv(self._parse_with_operator)
            using = None
            if self._match(TokenType.USING):
                using = self._parse_var(any_token=True)
                if using is None:
                    # A dangling USING was consumed and dropped, so
                    # `... (embedding) USING` built an index with no
                    # method at all and regenerated as if USING had
                    # never been typed.
                    self.raise_error("USING expects an index method")
            include = (
                self._parse_wrapped_id_vars()
                if self._match_text_seq("INCLUDE")
                else None
            )
            partition_by = self._parse_partition_by()
            with_storage = (
                self._match(TokenType.WITH)
                and self._parse_wrapped_properties()
            )
            tablespace = (
                self._parse_var(any_token=True)
                if self._match_text_seq("USING", "INDEX", "TABLESPACE")
                else None
            )
            where = self._parse_where()
            on = self._parse_field() if self._match(TokenType.ON) else None
            return self.expression(
                exp.IndexParameters(
                    using=using,
                    columns=columns,
                    include=include,
                    partition_by=partition_by,
                    with_storage=with_storage,
                    tablespace=tablespace,
                    where=where,
                    on=on,
                )
            )

        # ---------------------------------------------------------------
        # Clause-order enforcement
        # ---------------------------------------------------------------

        def _parse_query_modifiers(
            self, this: exp.Expr | None
        ) -> exp.Expr | None:
            # Saved and restored rather than simply reset: a modifier may
            # itself contain a subquery, whose own modifiers must be
            # validated against each other and not against the enclosing
            # query's.
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

            sqlglot's modifier loop is a bare ``while True`` with no
            positional state, so it happily accepts ``SEARCH PARAMS
            (...) LIMIT 10`` and then regenerates it as ``LIMIT 10 SEARCH
            PARAMS (...)`` -- input accepted, different text out.
            MilvusQL is a language we define and (via Track B's
            compiler) mostly generate ourselves, so there is no legacy
            corpus to be lenient toward, and silently reordering a
            user's SQL is worse than refusing it.

            ``positions`` is whatever :func:`_recording_modifier`
            collected for *this* query, so "the LIMIT clause" here means
            the clause and never a column, alias or placeholder that
            happens to be spelled ``limit``. ``LIMIT n`` and ``FETCH
            FIRST n ROWS ONLY`` fill the same ``limit`` slot; ``OFFSET``
            is recorded separately, so the canonical ``LIMIT ... OFFSET
            ...`` pair is treated as one family with two possible
            positions -- HYBRID SEARCH must precede the earlier of the
            two, SEARCH PARAMS and CONSISTENCY LEVEL must follow the
            later.

            Canonical order: ``HYBRID SEARCH ... LIMIT ... OFFSET ...
            SEARCH PARAMS ... CONSISTENCY LEVEL``.
            """
            first: dict[str, tuple[int, t.Any]] = {}
            for key, index, token in positions:
                first.setdefault(key, (index, token))

            hybrid = first.get(HYBRID_ARG)
            params = first.get(SEARCH_PARAMS_ARG)
            consistency = first.get(CONSISTENCY_ARG)
            limit_family = [
                p for p in (first.get("limit"), first.get("offset")) if p
            ]
            earliest_limit = min(
                limit_family, default=None, key=lambda p: p[0]
            )
            latest_limit = max(limit_family, default=None, key=lambda p: p[0])

            if hybrid and earliest_limit and hybrid[0] > earliest_limit[0]:
                self.raise_error("HYBRID SEARCH must precede LIMIT", hybrid[1])
            if params and latest_limit and params[0] < latest_limit[0]:
                self.raise_error("SEARCH PARAMS must follow LIMIT", params[1])
            if (
                consistency
                and latest_limit
                and consistency[0] < latest_limit[0]
            ):
                self.raise_error(
                    "CONSISTENCY LEVEL must follow LIMIT", consistency[1]
                )
            if hybrid and params and hybrid[0] > params[0]:
                self.raise_error(
                    "HYBRID SEARCH must precede SEARCH PARAMS", hybrid[1]
                )
            if hybrid and consistency and hybrid[0] > consistency[0]:
                self.raise_error(
                    "HYBRID SEARCH must precede CONSISTENCY LEVEL",
                    hybrid[1],
                )
            if params and consistency and params[0] > consistency[0]:
                self.raise_error(
                    "SEARCH PARAMS must precede CONSISTENCY LEVEL",
                    params[1],
                )

    class Generator(generator.Generator):
        TRANSFORMS: t.ClassVar = {
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
            # Both clause bodies are guarded on the rendered text: an
            # empty `expressions` list satisfies arg_types at
            # construction time but renders as "SEARCH PARAMS ()", which
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
            ConsistencyLevel: lambda self, e: (
                f"{self.seg('CONSISTENCY LEVEL')} {self.sql(e, 'this')}"
            ),
        }

        def offset_limit_modifiers(
            self,
            expression: exp.Expr,
            fetch: bool,
            limit: exp.Fetch | exp.Limit | None,
        ) -> list[str]:
            # HYBRID SEARCH belongs immediately before LIMIT (D5).
            # Appending it to query_modifiers' *sqls instead -- the
            # obvious spelling -- splices it in ahead of JOIN/WHERE/GROUP
            # BY/HAVING, i.e. directly after FROM, which is not a valid
            # clause position in any SQL surface and does not re-parse to
            # the same AST.
            return [
                self.sql(expression, HYBRID_ARG),
                *super().offset_limit_modifiers(expression, fetch, limit),
            ]

        def after_limit_modifiers(self, expression: exp.Expr) -> list[str]:
            return [
                *super().after_limit_modifiers(expression),
                self.sql(expression, SEARCH_PARAMS_ARG),
                self.sql(expression, CONSISTENCY_ARG),
            ]

        def ordered_sql(self, expression: exp.Ordered) -> str:
            # MilvusQL has no NULLS FIRST/LAST grammar, but sqlglot's own
            # builders and the MySQL front end set nulls_first=True for a
            # plain `ORDER BY id`. Emitting it would invent a clause the
            # user never wrote and this parser would have to read back.
            if expression.args.get("nulls_first"):
                self.unsupported(
                    "MilvusQL has no NULLS FIRST/LAST; null ordering is fixed"
                )
                expression = expression.copy()
                expression.set("nulls_first", False)
            return super().ordered_sql(expression)

        def not_sql(self, expression: exp.Not) -> str:
            """Render ``IS NOT`` / ``NOT IN`` / ``NOT BETWEEN`` infix, not
            as a ``NOT (...)`` prefix.

            The base generator has no infix path for these:
            ``exp.Not(exp.Is/In/Between)`` comes out prefixed in every
            shipped dialect, including postgres and mysql, which
            MilvusQL otherwise borrows its operator spellings from.
            ``exp.Is`` already carries a ``negate`` flag for exactly this
            case; ``In`` and ``Between`` do not, so those two are
            rendered by hand, mirroring the base ``in_sql`` /
            ``between_sql`` bodies with ``NOT`` spliced in. A ``BETWEEN
            SYMMETRIC`` falls back to the inherited prefix form: negating
            its OR-expanded rendering (``SUPPORTS_BETWEEN_FLAGS``
            defaults to ``False``) is not a matter of inserting one
            word, and MilvusQL has no SYMMETRIC syntax to round-trip
            regardless.
            """
            this = expression.this
            if isinstance(this, exp.Is):
                return self.is_sql(
                    exp.Is(
                        this=this.this,
                        expression=this.expression,
                        negate=True,
                    )
                )
            if isinstance(this, exp.In):
                query = this.args.get("query")
                unnest = this.args.get("unnest")
                field = this.args.get("field")
                is_global = " GLOBAL" if this.args.get("is_global") else ""
                if query:
                    in_sql = self.sql(query)
                elif unnest:
                    in_sql = self.in_unnest_op(unnest)
                elif field:
                    in_sql = self.sql(field)
                else:
                    in_sql = "({})".format(
                        self.expressions(
                            this,
                            dynamic=True,
                            new_line=True,
                            skip_first=True,
                            skip_last=True,
                        )
                    )
                return f"{self.sql(this, 'this')}{is_global} NOT IN {in_sql}"
            if isinstance(this, exp.Between) and not this.args.get(
                "symmetric"
            ):
                return (
                    f"{self.sql(this, 'this')} NOT BETWEEN "
                    f"{self.sql(this, 'low')} AND {self.sql(this, 'high')}"
                )
            return super().not_sql(expression)

        def datatype_sql(self, expression: exp.DataType) -> str:
            # ARRAY<T>(n): the capacity renders in parens, matching the
            # spec's own spelling. The base rendering of a nested type's
            # `values` arg uses brackets for ARRAY -- correct for the
            # unrelated `INT[3]` fixed-size-array spelling, wrong for
            # ours.
            if expression.is_type(
                exp.DataType.Type.ARRAY
            ) and expression.args.get("values"):
                capacity = self.expressions(
                    expression, key="values", flat=True
                )
                without_capacity = expression.copy()
                without_capacity.set("values", None)
                base = super().datatype_sql(without_capacity)
                return f"{base}({capacity})"
            return super().datatype_sql(expression)

        def matchagainst_sql(self, expression: exp.MatchAgainst) -> str:
            # Identical to the base rendering except for the space
            # before "(", which the MilvusQL spec requires:
            # MATCH(text) AGAINST (:q).
            modifier = expression.args.get("modifier")
            modifier = f" {modifier}" if modifier else ""
            func = self.func("MATCH", *expression.expressions)
            this = self.sql(expression, "this")
            return f"{func} AGAINST ({this}{modifier})"

        def indexparameters_sql(self, expression: exp.IndexParameters) -> str:
            # This override renders only what MilvusQL has syntax for, so
            # anything else present on the node is *dropped*. Silently
            # widening a partial index to the whole table is the same
            # class of failure the <=> guard exists to prevent, so say
            # so.
            for key in _UNSUPPORTED_INDEX_ARGS:
                if expression.args.get(key):
                    self.unsupported(
                        f"MilvusQL indexes do not support {key.upper()}"
                    )

            columns = self.expressions(expression, key="columns", flat=True)
            columns = f" ({columns})" if columns else ""
            using = self.sql(expression, "using")
            using = f" USING {using.upper()}" if using else ""
            with_storage = self.expressions(
                expression, key="with_storage", flat=True
            )
            with_storage = f" WITH ({with_storage})" if with_storage else ""
            return f"{columns}{using}{with_storage}"
