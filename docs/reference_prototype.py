"""V8: FINAL reference prototype -- all residual issues fixed. This is the shape
recommended by SQLGLOT_API_GROUND_TRUTH.md."""
from __future__ import annotations

import sqlglot
from sqlglot import exp, generator, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.helper import ensure_list
from sqlglot.tokens import TokenType

TT_IP      = TokenType.SPACE        # <#>
TT_L1      = TokenType.BREAK        # <+>
TT_HYBRID  = TokenType.LANGUAGE     # "HYBRID SEARCH"
TT_SPARAMS = TokenType.PROPERTIES   # "SEARCH PARAMS"
TT_RELEASE = TokenType.SOUNDS_LIKE  # "RELEASE"

class LoadTable(exp.Expression):    arg_types = {"this": True, "properties": False}
class ReleaseTable(exp.Expression): arg_types = {"this": True}
class SearchParams(exp.Expression): arg_types = {"expressions": True}
class SearchArm(exp.Expression):    arg_types = {"this": True, "weight": False}
class HybridSearch(exp.Expression): arg_types = {"expressions": True, "rerank": False}
class Rerank(exp.Expression):       arg_types = {"this": True, "expressions": False}
class AddField(exp.Expression):     arg_types = {"this": True}
class InnerProduct(exp.Expression, exp.Binary): pass
class L1Distance(exp.Expression, exp.Binary): pass
class CosineDistance(exp.Expression, exp.Binary): pass
class BM25Score(exp.Expression, exp.Func):
    _sql_names = ["BM25_SCORE"]
    arg_types = {"this": True, "expression": True}

exp.Select.arg_types["search_params"] = False
exp.Select.arg_types["hybrid"] = False


class Milvus(Dialect):
    SUPPORTS_USER_DEFINED_TYPES = True
    NULL_ORDERING = "nulls_are_last"

    class Tokenizer(tokens.Tokenizer):
        KEYWORDS = {**tokens.Tokenizer.KEYWORDS,
                    "<#>": TT_IP, "<+>": TT_L1,
                    "HYBRID SEARCH": TT_HYBRID,
                    "SEARCH PARAMS": TT_SPARAMS,
                    "RELEASE": TT_RELEASE}

    class Parser(parser.Parser):
        FACTOR = {**parser.Parser.FACTOR, TT_IP: InnerProduct, TT_L1: L1Distance,
                  TokenType.NULLSAFE_EQ: CosineDistance}
        EQUALITY = {k: v for k, v in parser.Parser.EQUALITY.items()
                    if k is not TokenType.NULLSAFE_EQ}
        FUNCTIONS = {**parser.Parser.FUNCTIONS, **BM25Score.default_parser_mappings()}

        # keep RELEASE usable as an identifier despite owning a dedicated TokenType
        ID_VAR_TOKENS = parser.Parser.ID_VAR_TOKENS | {TT_RELEASE}
        TABLE_ALIAS_TOKENS = parser.Parser.TABLE_ALIAS_TOKENS | {TT_RELEASE}
        COLON_PLACEHOLDER_TOKENS = ID_VAR_TOKENS

        STATEMENT_PARSERS = {**parser.Parser.STATEMENT_PARSERS,
                             TokenType.LOAD: lambda self: self._parse_milvus_load(),
                             TT_RELEASE: lambda self: self._parse_milvus_release()}
        QUERY_MODIFIER_PARSERS = {**parser.Parser.QUERY_MODIFIER_PARSERS,
                                  TT_HYBRID: lambda self: ("hybrid", self._parse_hybrid_search()),
                                  TT_SPARAMS: lambda self: ("search_params", self._parse_search_params())}
        QUERY_MODIFIER_TOKENS = set(QUERY_MODIFIER_PARSERS)
        ALTER_PARSERS = {**parser.Parser.ALTER_PARSERS,
                         "ADD": lambda self: self._parse_milvus_alter_add()}

        def _parse_milvus_load(self):
            index = self._index
            if not self._match(TokenType.TABLE):
                self._retreat(index)
                return super()._parse_load()
            this = self._parse_table_parts()
            return self.expression(LoadTable(this=this, properties=self._parse_properties()))

        def _parse_milvus_release(self):
            index = self._index
            if not self._match(TokenType.TABLE):
                self._retreat(index - 1)          # not our statement -> let it be an identifier
                return super()._parse_statement()
            return self.expression(ReleaseTable(this=self._parse_table_parts()))

        def _parse_search_params(self):
            self._advance()
            return self.expression(SearchParams(expressions=self._parse_wrapped_properties()))

        def _parse_hybrid_search(self):
            self._advance()
            arms = self._parse_wrapped_csv(self._parse_search_arm)
            rerank = None
            if self._match_text_seq("RERANK"):
                kind = self._parse_var(any_token=True)
                params = (self._parse_wrapped_properties()
                          if self._match(TokenType.L_PAREN, advance=False) else None)
                rerank = self.expression(Rerank(this=kind, expressions=params))
            return self.expression(HybridSearch(expressions=arms, rerank=rerank))

        def _parse_search_arm(self):
            this = self._parse_assignment()       # NOT _parse_expression (alias trap)
            weight = self._parse_number() if self._match_text_seq("WEIGHT") else None
            return self.expression(SearchArm(this=this, weight=weight))

        def _parse_milvus_alter_add(self):
            if self._match_text_seq("FIELD"):
                return ensure_list(self.expression(AddField(this=self._parse_field_def())))
            return self._parse_alter_table_add()

        def _parse_index_params(self):
            """MilvusQL order:  (cols) USING <method> WITH (...)
            Base order is:      USING <method> (cols) WITH (...)"""
            columns = (self._parse_wrapped_csv(self._parse_with_operator)
                       if self._match(TokenType.L_PAREN, advance=False) else None)
            using = self._parse_var(any_token=True) if self._match(TokenType.USING) else None
            with_storage = self._match(TokenType.WITH) and self._parse_wrapped_properties()
            where = self._parse_where()
            return self.expression(exp.IndexParameters(
                using=using, columns=columns, with_storage=with_storage, where=where))

    class Generator(generator.Generator):
        TRANSFORMS = {**generator.Generator.TRANSFORMS,
            InnerProduct:   lambda s, e: s.binary(e, "<#>"),
            L1Distance:     lambda s, e: s.binary(e, "<+>"),
            CosineDistance: lambda s, e: s.binary(e, "<=>"),
            LoadTable: lambda s, e: (f"LOAD TABLE {s.sql(e, 'this')}"
                                     + (f" {s.sql(e, 'properties')}" if e.args.get("properties") else "")),
            ReleaseTable: lambda s, e: f"RELEASE TABLE {s.sql(e, 'this')}",
            SearchParams: lambda s, e: f"{s.seg('SEARCH PARAMS')} ({s.expressions(e, flat=True)})",
            SearchArm: lambda s, e: (s.sql(e, "this")
                                     + (f" WEIGHT {s.sql(e, 'weight')}" if e.args.get("weight") else "")),
            Rerank: lambda s, e: (f"{s.sql(e, 'this')}"
                                  + (f"({s.expressions(e, flat=True)})" if e.args.get("expressions") else "")),
            HybridSearch: lambda s, e: (f"{s.seg('HYBRID SEARCH')} ({s.expressions(e, flat=True)})"
                                        + (f" RERANK {s.sql(e, 'rerank')}" if e.args.get("rerank") else "")),
            AddField: lambda s, e: f"ADD FIELD {s.sql(e, 'this')}"}

        def query_modifiers(self, expression, *sqls):
            return super().query_modifiers(expression, *sqls, self.sql(expression, "hybrid"))

        def after_limit_modifiers(self, expression):
            return super().after_limit_modifiers(expression) + [self.sql(expression, "search_params")]

        def matchagainst_sql(self, e):
            modifier = e.args.get("modifier")
            modifier = f" {modifier}" if modifier else ""
            return f"{self.func('MATCH', *e.expressions)} AGAINST ({self.sql(e, 'this')}{modifier})"

        def indexparameters_sql(self, e):
            columns = self.expressions(e, key="columns", flat=True)
            columns = f" ({columns})" if columns else ""
            using = self.sql(e, "using")
            using = f" USING {using.upper()}" if using else ""
            ws = self.expressions(e, key="with_storage", flat=True)
            ws = f" WITH ({ws})" if ws else ""
            return f"{columns}{using}{ws}"


SPEC = [
 "CREATE TABLE items (id BIGINT PRIMARY KEY AUTO_INCREMENT, embedding VECTOR(768), category VARCHAR(64)) WITH (shards=2, consistency_level='Bounded', partition_key=category)",
 "CREATE INDEX idx_emb ON items (embedding) USING HNSW WITH (metric_type='COSINE', M=16, ef_construction=200)",
 "LOAD TABLE items WITH (replicas=2)",
 "RELEASE TABLE items",
 "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
 "DELETE FROM items WHERE category = :cat",
 "SELECT id, category FROM items WHERE category = :cat ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)",
 "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7, sparse_emb <#> :sv WEIGHT 0.3) RERANK RRF(k=60) LIMIT 10",
 "SELECT id FROM items WHERE MATCH(text) AGAINST (:q) ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10",
 "ALTER TABLE items ADD FIELD tag VARCHAR(32)",
 "SELECT a <#> b, c <+> d, e <=> f, g <-> h FROM t",
 "DROP TABLE items",
 "SELECT release, search, hybrid, params, weight, field, rerank, text FROM t",
 "SELECT category, COUNT(*) FROM items GROUP BY category HAVING COUNT(*) > 1",
 "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 10",
 "SELECT id FROM items WHERE category = :cat AND price > 10 ORDER BY embedding <#> :q LIMIT 5 SEARCH PARAMS (nprobe=16, ef=64)",
]
print("=" * 96)
print("FINAL PROTOTYPE -- text-identical round-trip over the full MilvusQL spec")
print("=" * 96)
ok = bad = 0
for s in SPEC:
    try:
        a = sqlglot.parse_one(s, read=Milvus)
        out = a.sql(dialect=Milvus)
        a2 = sqlglot.parse_one(out, read=Milvus)
        good = out == s and repr(a) == repr(a2) and not isinstance(a, exp.Command)
        ok += good; bad += not good
        print(f"{'OK  ' if good else 'DIFF'} {type(a).__name__:14s} {s[:66]}")
        if not good:
            print(f"       out: {out}")
    except Exception as e:
        bad += 1
        print(f"FAIL {'-':14s} {s[:66]}\n       {type(e).__name__}: {str(e).splitlines()[0][:90]}")
print(f"\n==> {ok} identical / {bad} not identical")

print("\n" + "=" * 96)
print("REGRESSIONS THAT MUST KEEP WORKING")
print("=" * 96)
for label, s in [
    ("LOAD DATA (hive) delegation",  "LOAD DATA INPATH 'x' INTO TABLE t"),
    ("release as table alias",       "SELECT a FROM t release"),
    (":release placeholder",         "SELECT :release"),
    ("release as column",            "SELECT release FROM t"),
    ("GROUP BY",                     "SELECT c, COUNT(*) FROM t GROUP BY c"),
    ("subquery",                     "SELECT * FROM (SELECT id FROM t) AS x"),
    ("CTE",                          "WITH c AS (SELECT 1 AS a) SELECT a FROM c"),
    ("JSON arrow gone (by design)",  "SELECT a -> 'b' FROM t"),
    ("dup HYBRID SEARCH",            "SELECT id FROM t HYBRID SEARCH (a) HYBRID SEARCH (b)"),
    ("garbage tail",                 "SELECT a FROM t GARBAGE TOKENS HERE"),
]:
    try:
        print(f"  {label:30s} -> {sqlglot.parse_one(s, read=Milvus).sql(dialect=Milvus)}")
    except Exception as e:
        print(f"  {label:30s} -> {type(e).__name__}: {str(e).splitlines()[0][:60]}")

print("\n" + "=" * 96)
print("PRETTY MODE")
print("=" * 96)
print(sqlglot.parse_one(SPEC[7], read=Milvus).sql(dialect=Milvus, pretty=True))
print()
print(sqlglot.parse_one(SPEC[6], read=Milvus).sql(dialect=Milvus, pretty=True))
