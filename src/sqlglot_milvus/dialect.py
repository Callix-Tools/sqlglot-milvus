"""The ``milvus`` sqlglot dialect: tokenizer, parser and generator for MilvusQL."""

from __future__ import annotations

import typing as t

from sqlglot import exp, generator, tokens
from sqlglot.dialects.dialect import Dialect
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


def _null_safe_eq_unsupported(self: generator.Generator, expression: exp.NullSafeEQ) -> str:
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


class Milvus(Dialect):
    """MilvusQL -- a SQL surface over Milvus collections, indexes and vector search.

    Deliberately derived from the bare :class:`Dialect` rather than from ``Postgres`` or ``MySQL``.
    ``MySQL`` is the only built-in dialect that *emits* ``<=>``, so inheriting from it would open a
    silent round-trip corruption channel for the very operator we rebind; ``Postgres`` would drag in
    ``%(name)s`` placeholder output, which conflicts with the ``named`` paramstyle Track B relies on.
    """

    SUPPORTS_USER_DEFINED_TYPES = True

    # Milvus has no NULLS FIRST/LAST concept; declaring the default suppresses a spurious
    # " NULLS LAST" in generated index column lists.
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
        EQUALITY = {k: v for k, v in BaseParser.EQUALITY.items() if k is not TokenType.NULLSAFE_EQ}

        FUNCTIONS = {**BaseParser.FUNCTIONS, **BM25Score.default_parser_mappings()}

        # RELEASE owns a TokenType so it can open a statement, which would otherwise make it a
        # reserved word. Add it back everywhere identifiers are accepted: STATEMENT_PARSERS is
        # consulted before any expression parsing, so "RELEASE TABLE x" still wins at statement
        # position while "SELECT release FROM t" keeps working.
        ID_VAR_TOKENS = BaseParser.ID_VAR_TOKENS | {TT_RELEASE}
        TABLE_ALIAS_TOKENS = BaseParser.TABLE_ALIAS_TOKENS | {TT_RELEASE}
        COLON_PLACEHOLDER_TOKENS = ID_VAR_TOKENS

        STATEMENT_PARSERS = {
            **BaseParser.STATEMENT_PARSERS,
            TokenType.LOAD: lambda self: self._parse_milvus_load(),
            TT_RELEASE: lambda self: self._parse_milvus_release(),
        }

        QUERY_MODIFIER_PARSERS = {
            **BaseParser.QUERY_MODIFIER_PARSERS,
            TT_HYBRID_SEARCH: lambda self: (HYBRID_ARG, self._parse_hybrid_search()),
            TT_SEARCH_PARAMS: lambda self: (SEARCH_PARAMS_ARG, self._parse_search_params()),
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
            return self.expression(LoadTable(this=this, properties=self._parse_properties()))

        def _parse_milvus_release(self):
            """``RELEASE TABLE x``, falling through to RELEASE-as-identifier."""
            index = self._index
            if not self._match(TokenType.TABLE):
                # Not our statement. Rewind *past* the RELEASE token itself and parse from there
                # as an ordinary expression -- retreating to the token and re-dispatching would
                # land back in this method and recurse forever.
                self._retreat(index - 1)
                return self._parse_expression()
            return self.expression(ReleaseTable(this=self._parse_table_parts()))

        def _parse_milvus_alter_add(self):
            if self._match_text_seq("FIELD"):
                return ensure_list(self.expression(AddField(this=self._parse_field_def())))
            return self._parse_alter_table_add()

        # ---------------------------------------------------------------------------------
        # Search clauses
        # ---------------------------------------------------------------------------------

        def _parse_search_params(self):
            self._advance()
            return self.expression(SearchParams(expressions=self._parse_wrapped_properties()))

        def _parse_hybrid_search(self):
            self._advance()
            arms = self._parse_wrapped_csv(self._parse_search_arm)
            rerank = None
            if self._match_text_seq("RERANK"):
                kind = self._parse_var(any_token=True)
                params = (
                    self._parse_wrapped_properties()
                    if self._match(TokenType.L_PAREN, advance=False)
                    else None
                )
                rerank = self.expression(Rerank(this=kind, expressions=params))
            return self.expression(HybridSearch(expressions=arms, rerank=rerank))

        def _parse_search_arm(self):
            # _parse_assignment, never _parse_expression: the latter consumes a trailing bare word
            # as an implicit alias, so "embedding <=> :dv WEIGHT 0.7" would parse as an expression
            # aliased "WEIGHT" and then choke on the number.
            this = self._parse_assignment()
            weight = self._parse_number() if self._match_text_seq("WEIGHT") else None
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
            using = self._parse_var(any_token=True) if self._match(TokenType.USING) else None
            with_storage = self._match(TokenType.WITH) and self._parse_wrapped_properties()
            where = self._parse_where()
            return self.expression(
                exp.IndexParameters(
                    using=using, columns=columns, with_storage=with_storage, where=where
                )
            )

        # ---------------------------------------------------------------------------------
        # Clause-order enforcement
        # ---------------------------------------------------------------------------------

        def _parse_query_modifiers(self, this):
            start = self._index
            this = super()._parse_query_modifiers(this)
            if isinstance(this, exp.Select):
                self._validate_clause_order(this, start, self._index)
            return this

        def _validate_clause_order(self, select: exp.Select, start: int, end: int) -> None:
            """Reject non-canonical clause order.

            sqlglot's modifier loop is a bare ``while True`` with no positional state, so it
            happily accepts ``SEARCH PARAMS (...) LIMIT 10`` and then regenerates it as
            ``LIMIT 10 SEARCH PARAMS (...)`` -- input accepted, different text out. MilvusQL is a
            language we define and (via Track B's compiler) mostly generate ourselves, so there is
            no legacy corpus to be lenient toward, and silently reordering a user's SQL is worse
            than refusing it.
            """
            if not (select.args.get(HYBRID_ARG) or select.args.get(SEARCH_PARAMS_ARG)):
                return

            positions: t.Dict[TokenType, t.Tuple[int, t.Any]] = {}
            depth = 0
            for index in range(start, min(end, len(self._tokens))):
                token = self._tokens[index]
                token_type = token.token_type
                if token_type is TokenType.L_PAREN:
                    depth += 1
                elif token_type is TokenType.R_PAREN:
                    depth -= 1
                elif depth == 0 and token_type in _ORDERED_CLAUSES and token_type not in positions:
                    positions[token_type] = (index, token)

            hybrid = positions.get(TT_HYBRID_SEARCH)
            params = positions.get(TT_SEARCH_PARAMS)
            limit = positions.get(TokenType.LIMIT)

            if hybrid and limit and hybrid[0] > limit[0]:
                self.raise_error("HYBRID SEARCH must precede LIMIT", hybrid[1])
            if params and limit and params[0] < limit[0]:
                self.raise_error("SEARCH PARAMS must follow LIMIT", params[1])
            if hybrid and params and hybrid[0] > params[0]:
                self.raise_error("HYBRID SEARCH must precede SEARCH PARAMS", hybrid[1])

    class Generator(generator.Generator):
        TRANSFORMS = {
            **generator.Generator.TRANSFORMS,
            InnerProduct: lambda self, e: self.binary(e, "<#>"),
            L1Distance: lambda self, e: self.binary(e, "<+>"),
            CosineDistance: lambda self, e: self.binary(e, "<=>"),
            exp.NullSafeEQ: _null_safe_eq_unsupported,
            LoadTable: lambda self, e: (
                f"LOAD TABLE {self.sql(e, 'this')}"
                + (f" {self.sql(e, 'properties')}" if e.args.get("properties") else "")
            ),
            ReleaseTable: lambda self, e: f"RELEASE TABLE {self.sql(e, 'this')}",
            AddField: lambda self, e: f"ADD FIELD {self.sql(e, 'this')}",
            SearchParams: lambda self, e: (
                f"{self.seg('SEARCH PARAMS')} ({self.expressions(e, flat=True)})"
            ),
            SearchArm: lambda self, e: (
                self.sql(e, "this")
                + (f" WEIGHT {self.sql(e, 'weight')}" if e.args.get("weight") else "")
            ),
            Rerank: lambda self, e: (
                self.sql(e, "this")
                + (f"({self.expressions(e, flat=True)})" if e.args.get("expressions") else "")
            ),
            HybridSearch: lambda self, e: (
                f"{self.seg('HYBRID SEARCH')} ({self.expressions(e, flat=True)})"
                + (f" RERANK {self.sql(e, 'rerank')}" if e.args.get("rerank") else "")
            ),
        }

        def query_modifiers(self, expression, *sqls):
            # Thin delegation rather than a wholesale override, so that upstream reordering of the
            # built-in modifiers keeps working.
            return super().query_modifiers(expression, *sqls, self.sql(expression, HYBRID_ARG))

        def after_limit_modifiers(self, expression):
            return super().after_limit_modifiers(expression) + [
                self.sql(expression, SEARCH_PARAMS_ARG)
            ]

        def matchagainst_sql(self, e: exp.MatchAgainst) -> str:
            # Identical to the base rendering except for the space before "(", which the MilvusQL
            # spec requires: MATCH(text) AGAINST (:q).
            modifier = e.args.get("modifier")
            modifier = f" {modifier}" if modifier else ""
            return f"{self.func('MATCH', *e.expressions)} AGAINST ({self.sql(e, 'this')}{modifier})"

        def indexparameters_sql(self, e: exp.IndexParameters) -> str:
            columns = self.expressions(e, key="columns", flat=True)
            columns = f" ({columns})" if columns else ""
            using = self.sql(e, "using")
            using = f" USING {using.upper()}" if using else ""
            with_storage = self.expressions(e, key="with_storage", flat=True)
            with_storage = f" WITH ({with_storage})" if with_storage else ""
            return f"{columns}{using}{with_storage}"


_ORDERED_CLAUSES = frozenset({TT_HYBRID_SEARCH, TT_SEARCH_PARAMS, TokenType.LIMIT})
