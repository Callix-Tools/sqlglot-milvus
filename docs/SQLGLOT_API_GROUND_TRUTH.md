# sqlglot 30.16.0 — API Ground Truth for `sqlglot-milvus`

> **Status: authoritative.** This document is the contract for the implementation phase.
> Every claim below was executed against `/home/neko/startup/milvus/sqlglot-milvus/.venv/bin/python`
> (sqlglot 30.16.0, pure Python). Where the six recon reports contradicted each other, the
> experiment was re-run and the verified answer is stated, with the losing claim named explicitly
> (see §9, *Reconciliation log*).
>
> Source citations use paths relative to
> `/home/neko/startup/milvus/sqlglot-milvus/.venv/lib/python3.12/site-packages/sqlglot/`.
>
> A **complete, working reference implementation** of the entire MilvusQL grammar lives at
> `/home/neko/startup/milvus/sqlglot-milvus/docs/reference_prototype.py`. It round-trips
> **16 / 16** spec statements text-identically. Read it alongside this document; where prose and
> prototype disagree, the prototype is right (it is executable).

---

## 1. Version and environment

### 1.1 Installed versions

| Item | Value |
|---|---|
| `sqlglot.__version__` | `30.16.0` |
| Install path | `/home/neko/startup/milvus/sqlglot-milvus/.venv/lib/python3.12/site-packages/sqlglot/` |
| Python | 3.12 |
| `sqlglot.tokens.SQLGLOTC_INSTALLED` | `False` |
| `sqlglot.tokenizer_core.__file__` | `.../tokenizer_core.py` (pure Python, **not** `.so`) |
| `sqlglotrs` | not installed |
| `sqlglotc` | not installed |

```
$ .venv/bin/python -c "import sqlglot, sqlglot.tokens as t, sqlglot.tokenizer_core as tc; \
    print(sqlglot.__version__, t.SQLGLOTC_INSTALLED, tc.__file__)"
30.16.0 False .../site-packages/sqlglot/tokenizer_core.py
```

### 1.2 Package layout changed in 30.x — this invalidates most tutorials

sqlglot 30.x split the monolith. **`sqlglot/dialects/<name>.py` no longer contains the Parser or
Generator.**

```
sqlglot/dialects/<name>.py   -> Dialect subclass + nested Tokenizer only
sqlglot/parsers/<name>.py    -> <Name>Parser
sqlglot/generators/<name>.py -> <Name>Generator
sqlglot/expressions/         -> a PACKAGE (core.py, ddl.py, dml.py, query.py, math.py, string.py,
                                datatypes.py, ...), no longer a single expressions.py
sqlglot/tokens.py            -> 592 lines, a thin CONFIG shell
sqlglot/tokenizer_core.py    -> 1217 lines, all the actual scanning (class TokenizerCore)
```

Line counts of the files you will fight with:

```
   592 tokens.py            1217 tokenizer_core.py     10420 parser.py
  6341 generator.py         2657 dialects/dialect.py    3116 expressions/core.py
    51 expressions/__init__.py    17 parsers/base.py
```

When `sqlglot.parsers` / `sqlglot.generators` appeared (verified in throwaway venvs by the
dialect-plugin recon):

| version | `PLUGIN_GROUP_NAME` | `sqlglot.parsers` | `sqlglot.generators` | `exp.DType` |
|---|---|---|---|---|
| 28.5.0 | **absent** | no | no | no |
| 28.6.0 | `sqlglot.dialects` | no | no | no |
| 29.0.0 | `sqlglot.dialects` | no | no | no |
| 30.0.0 | `sqlglot.dialects` | yes | **no** | yes |
| 30.5.0 / 30.16.0 | `sqlglot.dialects` | yes | yes | yes |

### 1.3 🔴 PROJECT BLOCKER: `sqlglot[c]` makes third-party dialects impossible

`sqlglot[c]` overlays mypyc-compiled `.so` files onto sqlglot's modules. mypyc **disables runtime
subclassing** of the compiled classes. Shipped dialects survive only because
`sqlglot/parsers/*.py` and `sqlglot/generators/*.py` are themselves compiled.

Upstream states this in the shipped wheel — `sqlglot-30.16.0.dist-info/METADATA:504`:

> Note: when `sqlglot[c]` is installed, subclassing may not work properly, because runtime
> subclassing of the mypyc-compiled classes has been disabled for performance reasons. Use the
> pure Python version if you need custom dialects.

and `METADATA:145`:

> Keep in mind that subclassing may not work properly with `sqlglot[c]` installed, so custom
> dialects may require the pure Python version.

**Verified side-by-side** (`final_verify/v7_compiled.py`, run in both the project venv and an
isolated `sqlglot[c]` venv at `.../scratchpad/cvenv`):

| operation | pure Python | `sqlglot[c]` |
|---|---|---|
| `class T(tokens.Tokenizer)` — class statement | OK | OK |
| `class P(parser.Parser)` — class statement | OK | OK |
| `class G(generator.Generator)` — class statement | OK | OK |
| `class E(exp.Expression)` — class statement | OK | OK |
| `T().tokenize("SELECT 1")` | OK | **OK** |
| `P()` | OK | **`TypeError: interpreted classes cannot inherit from compiled`** |
| `G()` | OK | **`TypeError: interpreted classes cannot inherit from compiled`** |
| `E(this=...)` | OK | **`TypeError: interpreted classes cannot inherit from compiled`** |
| tokenizer-only 3rd-party Dialect, full `parse_one` | OK | **OK** |
| full 3rd-party Dialect (Parser+Generator) `parse_one` | OK | **`TypeError`** |
| shipped dialects (postgres → duckdb) | OK | OK |

Critically: **the `class` statement succeeds; the failure is at instantiation.** So
`sqlglot_milvus` will *import cleanly* and then explode on the first `parse_one` for any user
who has `sqlglot[c]` or `sqlglot[rs]` installed.

**Mandatory actions:**

1. `pyproject.toml`: `dependencies = ["sqlglot>=30.16.0,<31"]` — never the `c` / `rs` extras.
2. Import-time guard in `sqlglot_milvus/__init__.py`:
   ```python
   from sqlglot.tokens import SQLGLOTC_INSTALLED
   if SQLGLOTC_INSTALLED:
       raise ImportError(
           "sqlglot-milvus is incompatible with sqlglot[c]/sqlglot[rs]: mypyc-compiled sqlglot "
           "forbids runtime subclassing of Parser/Generator/Expression. "
           "Reinstall pure-Python sqlglot: pip uninstall sqlglotc && pip install --force-reinstall sqlglot==30.16.0"
       )
   ```
   Note `SQLGLOTC_INSTALLED` is computed at `tokens.py:13` as
   `not getattr(tokenizer_core, "__file__", "py").endswith(".py")`.
3. CI job that installs `sqlglot[c]` and asserts the guard fires with a readable message.
4. README install note.

`sqlglot[rs]` is deprecated in this version — `tokens.py:15-26` imports `sqlglotrs` only to emit a
`DeprecationWarning` pointing at `sqlglot[c]`. There is **no `USE_RS_TOKENIZER` flag** in 30.16.

### 1.4 Version pin

**`sqlglot>=30.16.0,<31`.** Rationale: the APIs we import (`sqlglot.parsers.*`,
`sqlglot.generators.*`, `exp.DType`, the instance-based `Parser.expression()`, the
`exp.Expression` vs `exp.Expr` split) do not exist below 30.x, and `sqlglot.generators` only
appeared between 30.0.0 and 30.5.0. sqlglot makes breaking parser/generator changes at minor
versions. `sqlglot>=28.6` (the plugin-mechanism floor cited by the architecture plan) will
`ImportError`. `requires-python = ">=3.9"` is fine (sqlglot 30.16.0 declares the same).

---

## 2. Tokenizer

### 2.1 Architecture

`Tokenizer` in `tokens.py` is a **config shell**; it holds `__slots__ = ("dialect", "_core")` and
delegates to `TokenizerCore` (`tokenizer_core.py`). All scanning behaviour is driven by class
attributes that are baked into derived structures **at class-creation time** by
`_TokenizerBase.__init_subclass__` (`tokens.py:78-118`).

The single most important line — `tokens.py:109-118`:

```python
cls._KEYWORD_TRIE = new_trie(
    key.upper()
    for key in (
        *cls.KEYWORDS,
        *cls._COMMENTS,
        *cls._QUOTES,
        *cls._FORMAT_STRINGS,
    )
    if " " in key or any(single in key for single in cls.SINGLE_TOKENS)
)
```

Consequences:

* A `KEYWORDS` entry enters the trie **only** if it contains a space **or** at least one
  `SINGLE_TOKENS` character. Pure-alphanumeric keywords (`SELECT`, `RELEASE`) are *not* in the
  trie; they are resolved by `_scan_var` via `self.keywords.get(text.upper(), TokenType.VAR)`
  (`tokenizer_core.py:1116-1120`).
* Multi-character **operators** (`<#>`, `<+>`) and **multi-word** keywords (`SEARCH PARAMS`) both
  go into `KEYWORDS` and are matched by the trie in `_scan_keywords` (`tokenizer_core.py:802-860`).
* The trie is a plain Python dict passed to `TokenizerCore.__init__` as `keyword_trie=`
  (`tokens.py:568`). **There is no precompiled/frozen trie.** This is true under `sqlglot[c]` too —
  `tokens.py` stays pure Python.

### 2.2 Adding multi-character operators — `KEYWORDS`, not `SINGLE_TOKENS`

`SINGLE_TOKENS` is strictly one character (a `dict[str, TokenType]` indexed by the current char).
Multi-char operators go in `KEYWORDS`. This is exactly how the base tokenizer ships `<->`, `#>>`,
`-|-` (`tokens.py:206-232`) and how SingleStore ships `:>`, `!:>`, `::$`
(`dialects/singlestore.py:54-58`).

Base tokenizer behaviour for the four pgvector operators (verified):

| operator | base spelling in `tokens.py` | base tokens |
|---|---|---|
| `<->` | `tokens.py:224 "<->": TokenType.LR_ARROW` | one token `LR_ARROW` ✅ |
| `<=>` | `tokens.py:218 "<=>": TokenType.NULLSAFE_EQ` | one token `NULLSAFE_EQ` ✅ (wrong meaning, see §7.9) |
| `<#>` | — | `LT('<')` + `HASH_ARROW('#>')` ❌ |
| `<+>` | — | `LT('<')` + `PLUS('+')` + `GT('>')` ❌ |

`<->` and `<<->>` come from the **base** tokenizer, not from postgres. Postgres in 30.16 spells
**none** of the pgvector operators; it only adds `~ @@ @? @> <@ ?& ?| #- |/ ||/`
(`dialects/postgres.py:82-93`).

Working registration (`final_verify/v2_contradictions.py`):

```python
class Tokenizer(tokens.Tokenizer):
    KEYWORDS = {**tokens.Tokenizer.KEYWORDS, "<#>": TT_IP, "<+>": TT_L1}
```
```
TokA (class body)   'a <#> b': [('VAR','a'), ('SPACE','<#>'), ('VAR','b')]
```
No regressions on `<`, `<=`, `<>`, `<->`, `<=>`. Longest-match wins in the trie.

### 2.3 Three ways `KEYWORDS` registration fails

All three verified in `final_verify/v2_contradictions.py`:

**(a) Post-hoc mutation does NOT rebuild the trie.**
```python
class TokB(tokens.Tokenizer): pass
TokB.KEYWORDS = {**tokens.Tokenizer.KEYWORDS, "<#>": TokenType.SPACE}   # too late
```
```
TokB (post-hoc)     'a <#> b': [('VAR','a'), ('LT','<'), ('HASH_ARROW','#>'), ('VAR','b')]
```
`__init_subclass__` already ran. Postgres avoids this by doing `KEYWORDS.pop("/*+")` **inside the
class body** (`dialects/postgres.py:130-131`).

**(b) In-place mutation of the inherited dict corrupts sqlglot globally.**
```python
class TokC(tokens.Tokenizer): pass
TokC.KEYWORDS["<@@>"] = TokenType.SPACE
```
```
TokC in-place: did it leak into base Tokenizer.KEYWORDS? True
TokC 'a <@@> b': [('VAR','a'), ('LT','<'), ('PARAMETER','@'), ('PARAMETER','@'), ('GT','>'), ('VAR','b')]
```
It pollutes `sqlglot.tokens.Tokenizer.KEYWORDS` for the whole process **and** still doesn't work.
**Always `{**tokens.Tokenizer.KEYWORDS, ...}`.**

**(c) Lowercase `KEYWORDS` keys crash at tokenize time.**
The trie is built from `key.upper()` but the lookup at `tokenizer_core.py:852-853` is a **direct
index** `self.keywords[word]`, not `.get()`.
```
class TokL(Tokenizer): KEYWORDS = {**Tokenizer.KEYWORDS, "search params": TokenType.SPACE}
TokL().tokenize("SEARCH PARAMS (x=1)")
-> TokenError: Error tokenizing 'SEARCH PARAMS (x=1'   (__cause__: KeyError('SEARCH PARAMS'))
```
**All `KEYWORDS` keys MUST be UPPERCASE.**

### 2.4 `TokenType` cannot be extended — and the failure is silent

`TokenType` is an `IntEnum` with **443 members** (`tokenizer_core.py:13`).

```
class T2(TokenType): X = auto()      -> TypeError: <enum 'T2'> cannot extend <enum 'TokenType'>
TokenType.MILVUSTEST1 = "HYBRID"     -> "OK"; type is str; is member? False   <- SILENT TRAP
TokenType.MILVUSTEST2 = 9001         -> "OK"; type is int; is member? False   <- SILENT TRAP
TokenType._member_map_["X"] = 9002   -> "OK"; yields a plain int, NOT a member <- SILENT TRAP
```

Both recon reports were right about their own case: assigning a **str** yields a `str`, assigning
an **int** yields an `int`. Neither becomes an enum member. The assignment appears to succeed,
pollutes the global enum process-wide, and then `_match_set` (dicts keyed by real `TokenType`
members) never matches it — you get a mystery parse failure, not a `TypeError`.
**Never assign to `TokenType`.** Recycle existing members.

### 2.5 Which `TokenType` members are actually free — DEFINITIVE

Two independent checks were run over the whole installed package
(`final_verify/v1_tokentypes.py`): (i) `grep -rn 'TokenType.<NAME>\b'` across every `.py`, and
(ii) reflection over every `dict`/`set`/`list`/`tuple` class attribute of `Parser` and `Generator`.

**Reflection alone is NOT sufficient** — the base parser also does inline
`self._match(TokenType.X)`. Example: `TokenType.SUMMARIZE` shows as "FREE" by reflection but is
matched inline at `parser.py:4138`. Only the grep count is conclusive.

**Tier A — provably free (ZERO `TokenType.X` references anywhere in the package):**

| TokenType | in `ID_VAR_TOKENS` | in `TABLE_ALIAS_TOKENS` | base spelling |
|---|---|---|---|
| `SPACE` | no | no | none |
| `BREAK` | no | no | none |
| `LANGUAGE` | no | no | none |
| `ORDERED` | no | no | none |
| `PROPERTIES` | no | no | none |
| `SOUNDS_LIKE` | no | no | none |
| `COLUMN_DEF` | no | no | none |

These are the **only** members with zero package-wide references. Because they are absent from
`ID_VAR_TOKENS` / `TABLE_ALIAS_TOKENS` / `FUNC_TOKENS` / `COLON_PLACEHOLDER_TOKENS`, hosting a
keyword on one of them **automatically reserves that word** — it stops being usable as an
identifier or table alias. That is a feature for `HYBRID SEARCH` / `SEARCH PARAMS`, and a bug for
`RELEASE`; §3.3 shows how to un-reserve selectively.

**Tier B — free from base-`Parser` dispatch dicts but spelled by other shipped dialects and
present in `ID_VAR_TOKENS`/`TABLE_ALIAS_TOKENS`:** `POOL, POLICY, PACKAGE, ROLE, RULE, VARIADIC,
VOLUME, INSTALL, INTEGRATION, MEMBER_OF, ONLY, QUOTE, SEPARATOR, UNDROP, GLOBAL, OPTION,
SEMANTIC_VIEW, STREAMLIT, SINK, SOURCE, NAMESPACE, EXPORT, DETACH, ATTACH, POINT, RING, TAG,
PUT, GET`. Usable, but each carries an inherited `ID_VAR_TOKENS` membership (so it *will* be
swallowed as a table alias, §7.5) and `SUMMARIZE` / `OPERATOR` / `UNCACHE` additionally have live
base-parser behaviour. **Prefer Tier A.**

**Never reuse these — they have live base-`Parser` semantics that will silently rewrite your AST:**

| TokenType | base spelling | base binding |
|---|---|---|
| `HASH_ARROW` | `#>` | `COLUMN_OPERATORS` → `exp.JSONBExtract` |
| `DHASH_ARROW` | `#>>` | `COLUMN_OPERATORS` → `exp.JSONBExtractScalar` |
| `DARROW` | `->>` | `COLUMN_OPERATORS` → `exp.JSONExtractScalar` |
| `ARROW` | `->` | `COLUMN_OPERATORS` → `exp.JSONExtract` |
| `AMP_LT` / `AMP_GT` | `&<` / `&>` | `RANGE_PARSERS` → ExtendsLeft/Right |
| `LR_ARROW` | `<->` | `FACTOR` → `exp.Distance` (**we want this one**) |
| `LLRR_ARROW` | `<<->>` | `FACTOR` → `exp.DistanceNd` |
| `NULLSAFE_EQ` | `<=>` | `EQUALITY` → `exp.NullSafeEQ` (**see §7.9**) |
| `PLACEHOLDER` | `?` | `COLUMN_OPERATORS` → `exp.JSONBContains` |

Proof that reuse breaks semantics (`final_verify/v3_more.py`, C5):
```
host=HASH_ARROW/DARROW, COLUMN_OPERATORS untouched:
  SELECT a <#> b, c <+> d FROM t -> [JSONBExtract, JSONExtractScalar]
                                 -> SELECT JSONB_EXTRACT(a, b), JSON_EXTRACT_SCALAR(c, d) FROM t
host=HASH_ARROW/DARROW, removed from COLUMN_OPERATORS:
                                 -> [IP, L1] -> SELECT a <#> b, c <+> d FROM t
host=SPACE/BREAK (Tier A):       -> [IP, L1] -> SELECT a <#> b, c <+> d FROM t   (no stripping needed)
```

**Decision (see §10 D1): host `<#>` on `TokenType.SPACE`, `<+>` on `TokenType.BREAK`.** Zero
collisions, no `COLUMN_OPERATORS` surgery, no risk of a future sqlglot release adding behaviour to
a token we depend on being inert.

### 2.6 Multi-word keywords — the highest-leverage tool we have

Space-containing keys are trie-eligible by construction (`tokens.py:117`). `_scan_keywords`
normalizes runs of whitespace to a single space while walking (`tokenizer_core.py:831-840`).

```python
class Tokenizer(tokens.Tokenizer):
    KEYWORDS = {**tokens.Tokenizer.KEYWORDS,
                "SEARCH PARAMS": TokenType.PROPERTIES,
                "HYBRID SEARCH": TokenType.LANGUAGE}
```

Verified (`final_verify/v9_last.py`, E8):

```
'SEARCH PARAMS (x=1)'          -> [('PROPERTIES','SEARCH PARAMS'), ('L_PAREN','('), ...]
'SEARCH   PARAMS (x=1)'        -> [('PROPERTIES','SEARCH PARAMS'), ...]   # runs collapsed
'SEARCH\nPARAMS (x=1)'         -> [('PROPERTIES','SEARCH PARAMS'), ...]   # newline ok
'search params (x=1)'          -> [('PROPERTIES','SEARCH PARAMS'), ...]   # case-insensitive
'SEARCH x'                     -> [('VAR','SEARCH'), ('VAR','x')]         # partial -> VAR
'SEARCH PARAMSX'               -> [('VAR','SEARCH'), ('VAR','PARAMSX')]   # boundary enforced
'SELECT hybrid, search FROM t' -> [..., ('VAR','search'), ('FROM','FROM'), ('VAR','t')]
```

**This is the key insight for the grammar:** registering `HYBRID SEARCH` and `SEARCH PARAMS` as
*pairs* makes each clause a single `_match_set` target **while leaving the bare words `HYBRID`,
`SEARCH`, `PARAMS` fully usable as column and table names.** No `ID_VAR_TOKENS` surgery is needed
for them. The token `text` is the **normalized uppercase form** (`SEARCH PARAMS`), not the source
spelling.

Two hazards:

1. **The pair swallows legitimate identifier pairs.**
   ```
   'SELECT x FROM search params'   -> [..., FROM, ('PROPERTIES','SEARCH PARAMS')]
   'SELECT x FROM "search" params' -> [..., FROM, ('IDENTIFIER','search'), ('VAR','params')]
   ```
   A table named `search` aliased `params` is destroyed. Quoting rescues it. Acceptable; add a
   negative test.
2. **Punctuation-containing word keywords have a sticky boundary.** The guard at
   `tokenizer_core.py:850` is `if prev_space or single_token or not char`, and `single_token` is
   set once *any* char seen during the walk is in `SINGLE_TOKENS`. So a registered `FOO-BAR`
   matches inside `FOO-BARX`:
   ```
   'FOO-BARX' -> [('SPACE','FOO-BAR'), ('VAR','X')]
   ```
   whereas `SEARCH PARAMSX` correctly does not match. **Never register a word keyword containing
   punctuation.**

### 2.7 `COMMANDS` / rest-of-statement swallow — do NOT use it

`tokens.py:515-523`:
```
COMMANDS              = {COMMAND, EXECUTE, FETCH, RENAME, SHOW}
COMMAND_PREFIX_TOKENS = {SEMICOLON, BEGIN}
KEYWORDS -> TokenType.COMMAND:  CALL, EXPLAIN, OPTIMIZE, PREPARE, VACUUM
```

Two independent mechanisms:

**(a) Tokenizer-level swallow** — `TokenizerCore._add`, `tokenizer_core.py:789-800`. Everything
after the command token up to `;` becomes one raw `STRING` token. Verified:
```
'VACUUM ANALYZE foo' -> [('COMMAND','VACUUM'), ('STRING','ANALYZE foo')]
'SELECT VACUUM x'    -> [('SELECT',..), ('COMMAND','VACUUM'), ('VAR','x')]   # not first -> no swallow
'; VACUUM x'         -> [SEMICOLON, COMMAND, ('STRING','x')]
'VACUUM;'            -> [COMMAND, SEMICOLON]                                 # peek==';' -> no swallow
```

**(b) Parser-level fallback** — `_parse_statement`, `parser.py:2379-2381`, reads
`self.dialect.tokenizer_class.COMMANDS`, so overriding `COMMANDS` in your nested `Tokenizer`
changes parser behaviour too.

**For MilvusQL this is the wrong tool.** Mapping `RELEASE → TokenType.COMMAND` yields
`[('COMMAND','RELEASE'), ('STRING','TABLE items')]` — an opaque blob with no structured AST.
Host `RELEASE` on a Tier-A TokenType and register a real `STATEMENT_PARSERS` entry (§3.3).
**Do not add our tokens to `COMMANDS`.**

### 2.8 Dialect ↔ Tokenizer wiring

`DialectMeta.__new__`, `dialects/dialect.py:287-299`:

```python
klass.tokenizer_class = klass.__dict__.get("Tokenizer", type("Tokenizer", base_tokenizer, {}))
klass.jsonpath_tokenizer_class = klass.__dict__.get("JSONPathTokenizer", type(...))
klass.parser_class    = klass.__dict__.get("Parser",    klass.__dict__.get("parser_class",    base_parser[0]))
klass.generator_class = klass.__dict__.get("Generator", klass.__dict__.get("generator_class", base_generator[0]))
```

**🔴 ASYMMETRY — `tokenizer_class = X` is SILENTLY IGNORED.** `Parser`/`Generator` accept both a
nested class *and* a `parser_class = X` / `generator_class = X` assignment. `Tokenizer` accepts
**only** the nested `class Tokenizer`. Verified (`final_verify/v3_more.py`, C7):

```
class WrongTok(Dialect):
    tokenizer_class = TkA
-> WrongTok.tokenizer_class is TkA : False     (silently replaced by a synthesized subclass)
```

**Always declare nested `class Tokenizer` / `class Parser` / `class Generator`.**

Other metaclass facts:
* If you omit `Tokenizer`, the metaclass synthesizes `type("Tokenizer", (BaseTokenizer,), {})`,
  which still triggers `__init_subclass__` (safe to inherit and mutate).
* Tokenizer instances are **not cached**: `Dialect.tokenizer()` → `self.tokenizer_class(dialect=self)`
  on every call (`dialect.py:1190`), and `Tokenizer.__init__` rebuilds a `TokenizerCore` each time
  (`tokens.py:544-573`). The class-level trie is reused, so cost is bounded — but never put
  expensive work in a custom `__init__`.
* `dialect.py:307-345` derives `QUOTE_START/END`, `IDENTIFIER_START/END`,
  `BIT/HEX/BYTE/UNICODE_START/END`, `STRINGS_SUPPORT_ESCAPED_SEQUENCES`, and
  `SUPPORTS_COLUMN_JOIN_MARKS = "(+)" in klass.tokenizer_class.KEYWORDS` **from** the tokenizer.

---

## 3. Parser

### 3.1 `STATEMENT_PARSERS`

**Definition:** `parser.py:1154-1185`. Keys are `TokenType`; values `Callable[[Parser], exp.Expr]`.

**Dispatch — `parser.py:2369-2377` (verbatim):**
```python
def _parse_statement(self) -> exp.Expr | None:
    if not self._curr:
        return None

    if self._match_set(self.STATEMENT_PARSERS):
        comments = self._prev_comments
        stmt = self.STATEMENT_PARSERS[self._prev.token_type](self)
        stmt.add_comments(comments, prepend=True)
        return stmt

    if self._match_set(self.dialect.tokenizer_class.COMMANDS):
        return self._parse_command()
    ...
```

Contract:
* **The key TokenType is already consumed** by `_match_set` before your callable runs.
  `self._prev` is that token; `self._curr` is the next one.
* **The callable MUST return a non-`None` `exp.Expr`** — `stmt.add_comments(...)` is unconditional.
  Verified: returning `None` → `AttributeError: 'NoneType' object has no attribute 'add_comments'`.
* `self._curr` is **never `None`**: `parser.py:332 SENTINEL_NONE = Token(TokenType.SENTINEL, "SENTINEL")`,
  assigned past EOF by `_advance` (`parser.py:1982-1996`). It is falsy
  (`bool(SENTINEL_NONE) is False`) but `.token_type` / `.text` are always safe. So
  `self._curr.text.upper()` never `AttributeError`s and `if not self._curr` still works as an EOF
  test.

### 3.2 `LOAD TABLE x WITH (...)` — override, don't replace

`TokenType.LOAD` already exists and is claimed by `_parse_load` (`parser.py:3797`, Hive
`LOAD DATA INPATH`). Override and delegate:

```python
STATEMENT_PARSERS = {**parser.Parser.STATEMENT_PARSERS,
                     TokenType.LOAD: lambda self: self._parse_milvus_load()}

def _parse_milvus_load(self):
    index = self._index
    if not self._match(TokenType.TABLE):
        self._retreat(index)
        return super()._parse_load()        # keeps Hive LOAD DATA working
    this = self._parse_table_parts()
    return self.expression(LoadTable(this=this, properties=self._parse_properties()))
```
Verified:
```
LOAD TABLE items WITH (replicas=2)  -> LoadTable(this=Table(items),
                                         properties=Properties([Property(Var(replicas), Literal(2))]))
                                    -> LOAD TABLE items WITH (replicas=2)      text-identical
LOAD DATA INPATH 'x' INTO TABLE t   -> LoadData -> LOAD DATA INPATH 'x' INTO TABLE t
```

### 3.3 `RELEASE TABLE x` — dedicated TokenType + selective un-reservation

There is no `TokenType.RELEASE`, and `RELEASE` is not in base `KEYWORDS`. Base:
`RELEASE TABLE items` → `ParseError` (sqlglot has no `RELEASE` support at all, not even
`RELEASE SAVEPOINT`).

Recommended shape:

```python
TT_RELEASE = TokenType.SOUNDS_LIKE          # Tier A

class Tokenizer(tokens.Tokenizer):
    KEYWORDS = {**tokens.Tokenizer.KEYWORDS, "RELEASE": TT_RELEASE}

class Parser(parser.Parser):
    STATEMENT_PARSERS = {**parser.Parser.STATEMENT_PARSERS,
                         TT_RELEASE: lambda self: self._parse_milvus_release()}
    # un-reserve: keep `release` usable as a column / alias / :placeholder
    ID_VAR_TOKENS            = parser.Parser.ID_VAR_TOKENS | {TT_RELEASE}
    TABLE_ALIAS_TOKENS       = parser.Parser.TABLE_ALIAS_TOKENS | {TT_RELEASE}
    COLON_PLACEHOLDER_TOKENS = ID_VAR_TOKENS
```

Effect verified (`final_verify/v6_fixes.py`, F1):

| statement | without un-reservation | with un-reservation |
|---|---|---|
| `RELEASE TABLE items` | `RELEASE TABLE items` | `RELEASE TABLE items` |
| `SELECT release FROM t` | **ParseError** | `SELECT release FROM t` |
| `SELECT a FROM t release` | **ParseError** | `SELECT a FROM t AS release` |
| `SELECT :release` | **ParseError** | `SELECT :release` |
| `CREATE TABLE t (release INT)` | works | works |

Un-reservation is safe because `_parse_statement` consults `STATEMENT_PARSERS` **first**, before
any expression parsing.

**🔴 `_retreat` + `super()._parse_statement()` INFINITE-RECURSES.** Discovered while building the
prototype (`final_verify/v10_recursion.py`):

```python
def _parse_milvus_release(self):                     # WRONG
    index = self._index
    if not self._match(TokenType.TABLE):
        self._retreat(index - 1)
        return super()._parse_statement()            # re-matches TT_RELEASE -> recursion
```
```
'RELEASE items' -> RecursionError
'RELEASE'       -> RecursionError
```
Two correct forms:
```python
def _parse_milvus_release(self):                     # RIGHT (strict) -- recommended
    if not self._match(TokenType.TABLE):
        self.raise_error("Expected TABLE after RELEASE")
    return self.expression(ReleaseTable(this=self._parse_table_parts()))
# 'RELEASE items' -> ParseError: Expected TABLE after RELEASE. Line 1, Col: 13.
```
```python
def _parse_milvus_release(self):                     # RIGHT (permissive)
    index = self._index
    if not self._match(TokenType.TABLE):
        self._retreat(index - 1)
        return self._parse_expression() or self._parse_as_command(self._curr)
    return self.expression(ReleaseTable(this=self._parse_table_parts()))
# 'RELEASE items' -> Alias -> RELEASE AS items
```
**Never call `super()._parse_statement()` from inside a `STATEMENT_PARSERS` callable.**

### 3.4 `QUERY_MODIFIER_PARSERS`

**Definition:** `parser.py:1595-1621`. Keys `TokenType`; value signature
`Callable[[Parser], tuple[str, exp.Expr | None]]`.

**Consumer — `parser.py:4338-4384` (verbatim, abridged):**
```python
def _parse_query_modifiers(self, this):
    if isinstance(this, self.MODIFIABLES):
        for join in self._parse_joins():
            this.append("joins", join)
        for lateral in iter(self._parse_lateral, None):
            this.append("laterals", lateral)

        while True:
            if self._match_set(self.QUERY_MODIFIER_PARSERS, advance=False):
                modifier_token = self._curr
                parser = self.QUERY_MODIFIER_PARSERS[modifier_token.token_type]
                key, expression = parser(self)

                if expression:
                    if this.args.get(key):
                        self.raise_error(
                            f"Found multiple '{modifier_token.text.upper()}' clauses",
                            token=modifier_token,
                        )
                    this.set(key, expression)
                    if key == "limit":
                        ...   # limit/offset re-splicing
                    continue

            if self._curr.text.upper() == "START":
                ...
            break
```

Key facts:
* **`advance=False` — the trigger token is NOT consumed.** Your callable must consume it itself
  (`self._advance()` first). cf. clickhouse `parsers/clickhouse.py:446`.
* Must return a **2-tuple**. Returning `None` → `TypeError` on unpack.
* Result lands via `this.set(key, expression)` → `exp.Select.args[key]`. Arbitrary keys are
  accepted at runtime (but see §6.5 for the pytest trap).
* Falsy `expression` → the loop falls through and **breaks** — a "probe and decline" modifier is
  safe *provided you consumed nothing*.
* Duplicate-key guard message uses `modifier_token.text.upper()`. With a multi-word keyword the
  message is nicely readable: `ParseError: Found multiple 'SEARCH PARAMS' clauses`.
* **The loop is completely UNORDERED.** There is no positional constraint at all. `SEARCH PARAMS`
  after `LIMIT`, before `LIMIT`, before `WHERE` — all parse. Order can only be enforced
  post-parse (§10 D5).
* `MODIFIABLES = (exp.Query, exp.Table, exp.TableFromRows, exp.Values)` (`parser.py:1834`).

Recommended shape (dedicated multi-word TokenTypes, no `VAR` key):

```python
QUERY_MODIFIER_PARSERS = {**parser.Parser.QUERY_MODIFIER_PARSERS,
                          TT_HYBRID:  lambda self: ("hybrid", self._parse_hybrid_search()),
                          TT_SPARAMS: lambda self: ("search_params", self._parse_search_params())}
QUERY_MODIFIER_TOKENS = set(QUERY_MODIFIER_PARSERS)

def _parse_search_params(self):
    self._advance()                       # advance=False -> consume the trigger ourselves
    return self.expression(SearchParams(expressions=self._parse_wrapped_properties()))
```

### 3.5 🔴 `QUERY_MODIFIER_TOKENS` — the trap is real but CONDITIONAL

`parser.py:1622`: `QUERY_MODIFIER_TOKENS: t.ClassVar = set(QUERY_MODIFIER_PARSERS)` — a class-body
snapshot. It is consumed in exactly **one** place, `_parse_group` (`parser.py:5544`), as an
"empty `GROUP BY`" lookahead:

```python
if self._match_set(self.QUERY_MODIFIER_TOKENS, advance=False):
    return self.expression(exp.Group(**elements), comments=comments)
```

The parser recon and the pitfalls recon contradicted each other here. **Both were right about
different configurations.** Full 3×2 matrix run in `final_verify/v2_contradictions.py` (C3):

| `QUERY_MODIFIER_TOKENS` | modifier key | `SEARCH PARAMS` | `GROUP BY category` |
|---|---|---|---|
| inherited (stale) | `TokenType.VAR` | OK | OK |
| inherited (stale) | dedicated TT | OK | OK |
| **`set(QUERY_MODIFIER_PARSERS)` (local)** | **`TokenType.VAR`** | OK | **🔴 `ParseError`** |
| `set(QUERY_MODIFIER_PARSERS)` (local) | dedicated TT | OK | OK |
| `set(parser.Parser.QUERY_MODIFIER_PARSERS)` | `TokenType.VAR` | OK | OK |
| `set(parser.Parser.QUERY_MODIFIER_PARSERS)` | dedicated TT | OK | OK |

```
-- mode=recompute-local  key=VAR     key in QUERY_MODIFIER_TOKENS: True
   OK   SELECT id FROM items LIMIT 10 SEARCH PARAMS (ef_search=64)
   FAIL SELECT category, COUNT(*) FROM items GROUP BY category -> ParseError: Line 1, Col: 54.
   FAIL SELECT category FROM items GROUP BY category LIMIT 5   -> ParseError: Line 1, Col: 44.
```

**Verified rule:** the modifier itself works in every configuration, because
`_parse_query_modifiers` matches against `QUERY_MODIFIER_PARSERS` (the dict), **not**
`QUERY_MODIFIER_TOKENS`. The only thing `QUERY_MODIFIER_TOKENS` controls is empty-`GROUP BY`
detection. Putting `TokenType.VAR` in it makes *every* `GROUP BY <column>` look empty.

**Our configuration is safe:** we key on **dedicated** TokenTypes, so
`QUERY_MODIFIER_TOKENS = set(QUERY_MODIFIER_PARSERS)` is both correct and desirable (it lets
`GROUP BY` correctly detect `... GROUP BY` immediately followed by `SEARCH PARAMS`). **Redefine it
alongside `QUERY_MODIFIER_PARSERS`, and never key a modifier on `TokenType.VAR`.**

### 3.6 Binary operator wiring — `FACTOR`

Precedence dicts, `parser.py:955-987`:
```python
EQUALITY   = {EQ: exp.EQ, NEQ: exp.NEQ, NULLSAFE_EQ: exp.NullSafeEQ}         # :955
COMPARISON = {GT: exp.GT, GTE: exp.GTE, LT: exp.LT, LTE: exp.LTE}
BITWISE    = {AMP: exp.BitwiseAnd, CARET: exp.BitwiseXor, PIPE: exp.BitwiseOr}
TERM       = {DASH: exp.Sub, PLUS: exp.Add, MOD: exp.Mod, COLLATE: exp.Collate}   # :974
FACTOR     = {DIV: exp.IntDiv, LR_ARROW: exp.Distance, LLRR_ARROW: exp.DistanceNd,
              SLASH: exp.Div, STAR: exp.Mul}                                      # :981
```

Driver — `_parse_factor`, `parser.py:6340-6360`:
```python
while self._match_set(self.FACTOR):
    klass = self.FACTOR[self._prev.token_type]
    comments = self._prev_comments
    expression = parse_method()
    ...
    this = self.expression(klass(this=this, expression=expression), comments=comments)
```
The class is called as `klass(this=..., expression=...)`, so any `(exp.Expression, exp.Binary)`
node works.

Verified recipe:
```python
class Parser(parser.Parser):
    FACTOR = {**parser.Parser.FACTOR,
              TT_IP: InnerProduct, TT_L1: L1Distance,
              TokenType.NULLSAFE_EQ: CosineDistance}
    EQUALITY = {k: v for k, v in parser.Parser.EQUALITY.items()
                if k is not TokenType.NULLSAFE_EQ}         # MANDATORY, see §7.9
```
```
SELECT a <#> b, c <+> d, e <=> f, g <-> h FROM t
-> [InnerProduct, L1Distance, CosineDistance, Distance]
-> SELECT a <#> b, c <+> d, e <=> f, g <-> h FROM t        text-identical
```

**Precedence consistency matters.** `<->` and `<#>` are in `FACTOR` (bind like `*`); `<=>` ships in
`EQUALITY` (binds looser than `+`). Verified (`final_verify/v3_more.py`, C5):

| expression | `<=>` left in `EQUALITY` | `<=>` moved to `FACTOR` |
|---|---|---|
| `a <-> b + c` | `Add(Distance(a,b), c)` | `Add(Distance(a,b), c)` |
| `a <#> b + c` | `Add(IP(a,b), c)` | `Add(IP(a,b), c)` |
| **`a <=> b + c`** | **`NullSafeEQ(a, Add(b,c))`** ← inconsistent | **`Add(Cos(a,b), c)`** ✅ |

Moving `NULLSAFE_EQ` from `EQUALITY` to `FACTOR` is required to give the four sibling distance
operators identical precedence.

**If instead you host on `HASH_ARROW`/`DARROW`, you MUST also strip `COLUMN_OPERATORS`**
(`parser.py:1077-1107`), because `_parse_column_ops` runs *inside* the factor operand and wins.
Tier-A hosts make this unnecessary.

### 3.7 `_parse_properties` and `WITH (...)` — CORRECTED

The parser recon and my first re-run disagreed. **The parser recon was right**; my first driver
failed to set `_tokens_size`, so `_curr` was `SENTINEL_NONE` from the start. Correct driver
protocol for any manual sub-parser invocation:

```python
p = Parser(); toks = Tokenizer().tokenize(sql)
p.reset(); p._tokens = toks; p._tokens_size = len(toks); p._index = -1; p._advance()
```

Verified results (`final_verify/v4_props.py`):

| input | `_parse_properties()` | `_parse_wrapped_properties()` |
|---|---|---|
| `WITH (replicas=2)` | `Properties([Property(Var(replicas), Literal(2))])`, **consumes everything incl. `WITH`** | `ParseError: Expecting (. Line 1, Col: 4.` |
| `WITH (shards=2, consistency_level='Bounded', partition_key=category)` | full `Properties`, consumes all | `ParseError` |
| `WITH (metric_type='COSINE', M=16, ef_construction=200)` | full `Properties`, consumes all | `ParseError` |
| `(ef_search=64)` | **`None`**, consumes nothing | `[Property(Var(ef_search), Literal(64))]` |

**Rules:**
* `_parse_properties()` **consumes the `WITH` keyword itself** via `PROPERTY_PARSERS["WITH"]`
  (`parser.py:1413`, a **string**-keyed dict). Do **not** pre-consume `WITH`.
* For a bare parenthesised bag with no `WITH` (i.e. `SEARCH PARAMS (...)`, `RERANK RRF(...)`), use
  `_parse_wrapped_properties()` (`parser.py:2883` = `self._parse_wrapped_csv(self._parse_property)`),
  which returns a **`list`**, not a `Properties` node.

Value shapes produced by `_parse_key_value_property` (`parser.py:2915`): the key becomes
`exp.Var`, and a bare-identifier value is **demoted from `Column` to `Var`**:
```
partition_key=category -> Property(this=Var(partition_key), value=Var(category))    # NOT a Column
consistency_level='Bounded' -> Property(this=Var(...), value=Literal('Bounded', is_string=True))
M=16 -> Property(this=Var(M), value=Literal(16))     # case preserved: `M`, not `m`
```

### 3.8 `ALTER TABLE items ADD FIELD tag VARCHAR(32)`

`ALTER_PARSERS` — `parser.py:1507-1519`, keys are **uppercase STRINGS**, not TokenTypes.

Dispatch — `_parse_alter`, `parser.py:9194-9244`. Three things to know:

1. The action keyword is consumed by a **blind `self._advance()`** and matched via
   `self._prev.text.upper()`. **Any word works as an `ALTER_PARSERS` key** — no TokenType needed.
2. **`if not self._curr and actions:`** — the `Alter` node is produced *only* if the statement is
   fully consumed. Leave one token unconsumed and it silently degrades to `exp.Command`. This is
   the #1 cause of "my ALTER extension does nothing".
3. `ALTERABLES = {INDEX, SESSION, TABLE, VIEW}` (`parser.py:719`).

```python
ALTER_PARSERS = {**parser.Parser.ALTER_PARSERS,
                 "ADD": lambda self: self._parse_milvus_alter_add()}

def _parse_milvus_alter_add(self):
    if self._match_text_seq("FIELD"):
        return ensure_list(self.expression(AddField(this=self._parse_field_def())))
    return self._parse_alter_table_add()          # keep ADD COLUMN / ADD CONSTRAINT
```
```
ALTER TABLE items ADD FIELD tag VARCHAR(32)
-> Alter(this=Table(items), kind=TABLE, exists=False,
         actions=[AddField(this=ColumnDef(this=Identifier(tag),
                     kind=DataType(this=DType.VARCHAR, expressions=[DataTypeParam(Literal(32))])))],
         only=False, not_valid=False, check=False, cascade=False, iceberg=False)
-> ALTER TABLE items ADD FIELD tag VARCHAR(32)     text-identical
```

### 3.9 `CREATE INDEX ... (cols) USING <m> WITH (...)` — needs a parser override

**Negative finding: the MilvusQL word order does NOT parse in any shipped dialect.**
```
CREATE INDEX i ON t (e) USING HNSW WITH (m=16)   -> exp.Command (opaque)
CREATE INDEX i ON t USING HNSW (e) WITH (m=16)   -> exp.Create/exp.Index (structured)
```
Base `_parse_index_params` (`parser.py:4747-4778`) reads `USING` **before** the column list:
```python
def _parse_index_params(self) -> exp.IndexParameters:
    using = self._parse_var(any_token=True) if self._match(TokenType.USING) else None
    if self._match(TokenType.L_PAREN, advance=False):
        columns = self._parse_wrapped_csv(self._parse_with_operator)
    else:
        columns = None
    ...
    with_storage = self._match(TokenType.WITH) and self._parse_wrapped_properties()
    ...
```
Working override (verified text-identical round-trip):
```python
def _parse_index_params(self):
    columns = (self._parse_wrapped_csv(self._parse_with_operator)
               if self._match(TokenType.L_PAREN, advance=False) else None)
    using = self._parse_var(any_token=True) if self._match(TokenType.USING) else None
    with_storage = self._match(TokenType.WITH) and self._parse_wrapped_properties()
    where = self._parse_where()
    return self.expression(exp.IndexParameters(
        using=using, columns=columns, with_storage=with_storage, where=where))
```
plus the matching `indexparameters_sql` in the Generator (§4.7).

`exp.IndexParameters.arg_types = {'using','include','columns','with_storage','partition_by',
'tablespace','where','on'}` (all optional). `using` is an `exp.Var` preserving the source case;
`columns` is a list of `exp.Ordered` (possibly wrapping `exp.Opclass`); `with_storage` is the list
of `exp.Property` from `WITH (...)`.

### 3.10 Helper method reference (all signatures verified via `inspect.signature`)

| Method | Line | Signature / behaviour |
|---|---|---|
| `_match_text_seq` | 2050 | `(*texts: str, advance: bool = True) -> bool`. **Self-retreats on partial failure** — safe for speculative probing. Skips `TEXT_MATCH_EXCLUDED_TOKENS` (`parser.py:666`, all string/identifier token types), so a quoted `"SEARCH"` never text-matches. |
| `_match_texts` | 2040 | `(texts, advance=True) -> bool`. Single token only. |
| `_parse_table_parts` | 4909 | `(schema=False, is_db_reference=False, wildcard=False, fast=False)`. Also parses trailing `_parse_changes` / `_parse_historical_data` / `_parse_pivots`. `raise_error`s when no table found. |
| `_parse_properties` | 2977 | `(before: bool | None = None) -> exp.Properties | None`. Consumes `WITH`. §3.7. |
| `_parse_wrapped_properties` | 2883 | `() -> list[exp.Expr | list[exp.Expr]]`. Returns a **list**. Requires `(`. |
| `_parse_wrapped_csv` | 8878 | `(parse_method, sep=TokenType.COMMA, optional=False) -> list[T]` |
| `_parse_csv` | 8860 | `(parse_method, sep=TokenType.COMMA) -> list[T]`. `parse_method` is required — no unsafe default. |
| `_parse_placeholder` | 8843 | Backtracks with `self._advance(-1)` if the matched `PLACEHOLDER_PARSER` returns `None`. |
| `_parse_number` | 8784 | ⚠️ **falls back to `_parse_placeholder()`** — `_parse_number()` on `:x` returns `Placeholder`, not `None`. Same for `_parse_string`, `_parse_var`, `_parse_identifier`, `_parse_null`, `_parse_boolean`, `_parse_star`. **Cannot be used as a numeric type check.** |
| `_parse_expression` | 5978 | `return self._parse_alias(self._parse_assignment())` — **eats a trailing bare word as an alias**. §7.7. |
| `_parse_assignment` | 5981 | The alias-free variant. Use this inside `HYBRID SEARCH (...)`. |
| `_parse_field_def` | 7479 | Produces `ColumnDef` incl. constraints — exactly what `ADD FIELD` needs. |
| `_try_parse` | 2107 | `(parse_method, retreat=False) -> T | None`. Temporarily sets `ErrorLevel.IMMEDIATE`, catches `ParseError`, retreats. |
| `expression` | 2187 | `(instance: E, token: Token | None = None, comments: list[str] | None = None) -> E`. **See §8, plan correction C7.** |
| `validate_expression` | 2097 | Bumps `_node_count` vs `max_nodes`, then `expression.error_messages(args)`. |
| `_retreat` / `_advance` | 2005 / 1982 | Idiom: `index = self._index; ...; self._retreat(index)`. |

---

## 4. Generator

### 4.1 🔴 `<snake_case>_sql` METHODS DO NOT WORK FOR THIRD-PARTY NODES

30.16 replaced the old runtime `getattr(self, f"{e.key}_sql")` lookup with a **precomputed dispatch
table**. `generator.py:79-93`:

```python
def _build_dispatch(cls) -> dict[type[exp.Expr], t.Callable[..., str]]:
    dispatch: dict[type[exp.Expr], t.Callable[..., str]] = dict(cls.TRANSFORMS)
    for attr_name in dir(cls):
        if not attr_name.endswith("_sql") or attr_name.startswith("_"):
            continue
        expr_key = attr_name[:-4]
        expr_cls = exp.EXPR_CLASSES.get(expr_key)          # <-- THE TRAP
        if expr_cls and expr_cls not in dispatch:
            dispatch[expr_cls] = getattr(cls, attr_name)
    return dispatch
```

`exp.EXPR_CLASSES` is built **once**, at `sqlglot.expressions` import time, by scanning **only that
package's** members — `expressions/__init__.py:51`:
```python
EXPR_CLASSES: dict[str, type[Expr]] = {cls.key: cls for cls in subclasses(__name__, Expr)}
```
`subclasses()` (`helper.py:150-156`) does `inspect.getmembers(sys.modules[module_name], ...)`.
**A third-party class is never in it.**

Dispatch site — `generator.py:1121-1130`:
```python
handler = self._dispatch.get(expression.__class__)
if handler:
    sql = handler(self, expression)
elif isinstance(expression, exp.Func):
    sql = self.function_fallback_sql(expression)
elif isinstance(expression, exp.Property):
    sql = self.property_sql(expression)
else:
    raise ValueError(f"Unsupported expression type {expression.__class__.__name__}")
```

Verified (`final_verify/v3_more.py`, C10):

| definition | result |
|---|---|
| `class MyNode(exp.Expression)` + `def mynode_sql` | **`ValueError: Unsupported expression type MyNode`** — method silently ignored |
| `TRANSFORMS = {**Generator.TRANSFORMS, MyNode: lambda s, e: ...}` | ✅ `FROM_TRANSFORMS` |
| `TRANSFORMS` entry **and** `foo_sql` method on the same built-in class | **`TRANSFORMS` wins** (`T_WINS`) |
| custom `(exp.Expression, exp.Func)` node, no handler | **silently** `MY_FN(a)` via `function_fallback_sql` |
| custom `exp.Property` subclass, generated standalone | **silently** `'None=a'` + `unsupported("Unsupported property myprop")` |
| genuinely unknown node | `ValueError` regardless of `unsupported_level` |

Note the **exact-type** lookup: there is **no MRO fallback**. A subclass of `exp.Binary` with no
own entry raises.

**MANDATE for `sqlglot-milvus`: register EVERY custom node in `TRANSFORMS`.** It is
order-independent, needs no global mutation, and is the only mechanism that cannot silently no-op.

### 4.2 `_DISPATCH_CACHE` is permanent and class-keyed — no late mutation

`generator.py:76` + `generator.py:934-939` (in `__init__`):
```python
cls = type(self)
dispatch = _DISPATCH_CACHE.get(cls)
if dispatch is None:
    dispatch = _build_dispatch(cls)
    _DISPATCH_CACHE[cls] = dispatch
self._dispatch = dispatch
```

Verified: registering into `exp.EXPR_CLASSES` *after* the first instance of that Generator class
leaves the cache stale **forever**, for existing *and future* instances:
```
'mynode' in exp.EXPR_CLASSES: False
after late EXPR_CLASSES registration -> ValueError Unsupported expression type MyNode
after clearing _DISPATCH_CACHE       -> FROM_METHOD
```

**Rule: define `TRANSFORMS` entirely at class-definition/import time. Never patch afterwards.**

`exp.EXPR_CLASSES[MyNode.key] = MyNode` at module import top-level *does* work and enables
method-based dispatch — but it is a **global process-wide dict**, and naming a class `Index` /
`Load` / `Insert` would clobber `exp.Index` etc. for every dialect in the process. **Do not use it.**

Also note `dialects/dialect.py:300-305`: the metaclass does `gen_cls.TRANSFORMS.pop(part, None)`
for JSON-path parts not in `SUPPORTED_JSON_PATH_PARTS`. If your nested `class Generator` does not
define its own `TRANSFORMS`, this pops from the **inherited** dict. Harmless with defaults
(`SUPPORTED_JSON_PATH_PARTS = ALL_JSON_PATH_PARTS.copy()`, `generator.py:517`; base `TRANSFORMS`
size 142 → 142), but do not shrink `SUPPORTED_JSON_PATH_PARTS` without defining your own
`TRANSFORMS`.

### 4.3 `TRANSFORMS` signature

`generator.py:136`: `TRANSFORMS: t.ClassVar[dict[type[exp.Expr], t.Callable[..., str]]]`.
Keys are **expression classes**, not strings. Values are called as `f(generator, expression)` —
exactly **two positional args, no kwargs** (verified by capturing `*a, **kw`:
`{'nargs': 2, 'types': ['G4', 'MyThing'], 'kw': {}}`).

**Always write `TRANSFORMS = {**Generator.TRANSFORMS, ...}`.** A bare `TRANSFORMS = {...}` drops
all 142 base entries. Corollary: ~142 base behaviours ship as `TRANSFORMS` lambdas, so adding a
`foo_sql` method in your subclass to override one of *those* **does nothing** — you must override
the `TRANSFORMS` entry.

### 4.4 `query_modifiers` — the injection mechanism

`generator.py:3270-3300`, quoted in full:

```python
    def query_modifiers(self, expression: exp.Expr, *sqls: str) -> str:
        limit = expression.args.get("limit")

        if self.LIMIT_FETCH == "LIMIT" and isinstance(limit, exp.Fetch):
            count = limit.args.get("count")
            limit = exp.Limit(
                expression=exp.maybe_copy(count) if count is not None else exp.Literal.number(1)
            )
        elif self.LIMIT_FETCH == "FETCH" and isinstance(limit, exp.Limit):
            limit = exp.Fetch(direction="FIRST", count=exp.maybe_copy(limit.expression))

        return csv(
            *sqls,
            *[self.sql(join) for join in expression.args.get("joins") or []],
            self.sql(expression, "match"),
            *[self.sql(lateral) for lateral in expression.args.get("laterals") or []],
            self.sql(expression, "prewhere"),
            self.sql(expression, "where"),
            self.sql(expression, "connect"),
            self.sql(expression, "group"),
            self.sql(expression, "having"),
            *[gen(self, expression) for gen in self.AFTER_HAVING_MODIFIER_TRANSFORMS.values()],
            self.sql(expression, "order"),
            *self.offset_limit_modifiers(expression, isinstance(limit, exp.Fetch), limit),
            *self.after_limit_modifiers(expression),
            self.options_modifier(expression),
            self.sql(expression, "for_"),
            sep="",
        )
```

`csv` is `helper.py:120-131` → `sep.join(arg for arg in args if arg)`. With `sep=""`,
**every fragment must carry its own leading separator** (`self.seg(...)`, `generator.py:1005`).
Failure mode if you forget: `... LIMIT 10SEARCH PARAMS (ef_search=64)`.

Callers: `select_sql` (`generator.py:3334`, passes
`sqls = (f"SELECT{...}{expressions}", into_sql, from_sql)`), `subquery_sql` (`:3464`), and the
VALUES path (`:2718`). No shipped dialect in 30.16 overrides `query_modifiers`.

**Injection points, ranked:**

| wanted position | hook | line |
|---|---|---|
| right after `FROM`, **before** `WHERE` | `super().query_modifiers(expression, *sqls, <frag>)` — extra positional sqls are splatted first | 3270 |
| between `HAVING` and `ORDER BY` | add a key to `AFTER_HAVING_MODIFIER_TRANSFORMS` (string-keyed `t.ClassVar`, values `(self, e) -> str`) | 670 |
| immediately after `LIMIT`/`OFFSET` | override `after_limit_modifiers(expression) -> list[str]` | 3329 |
| very end | `options_modifier` | 3302 |

**This is how MilvusQL gets text-identical round-trip for both clauses**
(`final_verify/v4_props.py`, verified):

```python
def query_modifiers(self, expression, *sqls):
    # extra positional sqls land right after FROM, before joins/WHERE
    return super().query_modifiers(expression, *sqls, self.sql(expression, "hybrid"))

def after_limit_modifiers(self, expression):
    return super().after_limit_modifiers(expression) + [self.sql(expression, "search_params")]
```
```
SELECT id FROM items HYBRID SEARCH (a, b) WHERE c = :x ORDER BY e <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)
```
Pretty mode works (`self.seg` → newline):
```
SELECT
  id
FROM items
HYBRID SEARCH (a, b)
WHERE
  c = :x
ORDER BY
  e <-> :q
LIMIT 10
SEARCH PARAMS (ef_search=64)
```
Both overrides are safe on the Subquery / VALUES call paths (missing arg → `""` → dropped by
`csv`); verified unchanged for `SELECT * FROM (SELECT id FROM t) AS x` and
`SELECT * FROM (VALUES (1), (2)) AS t(a)`.

**Do NOT reimplement `query_modifiers` wholesale.** The thin `super()` delegation above survives
future upstream changes to the tail order.

A **base** generator silently drops the custom args:
```
BASE generator -> SELECT id FROM items WHERE c = :x ORDER BY e <-> :q LIMIT 10
```

### 4.5 `self.expressions(...)` — `flat=True` is the only one-line guarantee

`generator.py:4720-4735`:
```python
    def expressions(
        self, expression=None, key=None, sqls=None, flat=False, indent=True,
        skip_first=False, skip_last=False, sep=", ", prefix="", dynamic=False, new_line=False,
    ) -> str:
        expressions = expression.args.get(key or "expressions") if expression else sqls
        if not expressions:
            return ""
        if flat:
            return sep.join(sql for sql in (self.sql(e) for e in expressions) if sql)
        ...
```

Verified matrix:

| call | `pretty=False` | `pretty=True` |
|---|---|---|
| `expressions(node, flat=True)` | `"a=1, b='x'"` | **`"a=1, b='x'"`** ✅ |
| `expressions(node, indent=False)` | `"a=1, b='x'"` | `"a=1,\nb='x'"` ❌ |
| `expressions(node)` | `"a=1, b='x'"` | `"  a=1,\n  b='x'"` ❌ |

`indent=False` is **not** sufficient (`result_sql = "\n".join(...)` at `:4761`).
**Use `self.expressions(e, flat=True)`** for `SEARCH PARAMS (...)`, `HYBRID SEARCH (...)`,
`RERANK RRF(...)`, `WITH (...)`, index column lists. `flat=True` also ignores `prefix`,
`skip_first/last`, `new_line`, `dynamic` and drops comments. Safe on missing key / `None` → `""`.

### 4.6 `self.sql(e, "key")` edge matrix

`generator.py:1103-1121`. Verified:

| input | result |
|---|---|
| key absent from `args` | `''` |
| arg is `None` | `''` |
| `expression is None` | `''` |
| expression is a `str` | returned verbatim (raw-SQL escape hatch) |
| arg is `exp.Literal.number(0)` | `'0'` (Expression defines no `__bool__`, always truthy) |
| arg is python `False` | `''` — ⚠️ **boolean flags must be read via `expression.args.get(...)`** |
| arg is `[]` | `''` |
| arg is a **non-empty list** | 💥 `ValueError: Unsupported expression type list` |

**Rule: never `self.sql(e, key)` on a list-valued arg; use `self.expressions(e, key=key, flat=True)`.**

### 4.7 Properties and `WITH (...)` rendering

Chain: `create_sql` (`:1314`) → `locate_properties` (`:2089`, buckets by
`self.PROPERTIES_LOCATION[p.__class__]`) → `properties_sql` (`:2041`) → `root_properties` (`:2066`)
/ `with_properties` (`:2086`) → `properties` (`:2071`).

```python
    def with_properties(self, properties: exp.Properties) -> str:
        return self.properties(properties, prefix=self.seg(self.WITH_PROPERTIES_PREFIX, sep=""))
```
`WITH_PROPERTIES_PREFIX = "WITH"` at `generator.py:558` — the single knob that turns `WITH (...)`
into `OPTIONS (...)` / `TBLPROPERTIES (...)`. Leave it alone.

Verified location table for a plain `exp.Property`:
```
POST_CREATE      -> CREATE shards=2 TABLE t (id INT)
POST_NAME        -> CREATE TABLE t (id INT)              <- SILENTLY DROPPED
POST_SCHEMA      -> CREATE TABLE t (id INT) shards=2
POST_WITH        -> CREATE TABLE t (id INT) WITH (shards=2)   <- DEFAULT, what we want
POST_ALIAS       -> CREATE TABLE t (id INT)              <- SILENTLY DROPPED
POST_EXPRESSION  -> CREATE TABLE t (id INT) shards=2
POST_INDEX       -> CREATE TABLE t (id INT) shards=2
UNSUPPORTED      -> CREATE TABLE t (id INT)              <- dropped + unsupported()
```
Base default is already `exp.Property: exp.Properties.Location.POST_WITH` (`generator.py:757`).

**🔴 Custom `exp.Property` SUBCLASSES CRASH.** `properties_sql:2046` and `locate_properties:2092`
use **direct indexing** `self.PROPERTIES_LOCATION[p.__class__]`. Verified:
```
Create with custom Property subclass -> RAISED KeyError <class '__main__.MilvusProperty'>
```
**Mitigation: emit plain `exp.Property(this=exp.var(k), value=<literal>)` and add nothing.**
(`exp.Property` MRO is `['Property', 'Expression', 'Expr', 'object']` — it *is* concrete, but
subclassing it buys nothing and costs a `PROPERTIES_LOCATION` entry plus a `TRANSFORMS` entry, or
you get literal `None=x`.)

Manual `properties()` shapes:
```
properties(props, prefix="WITH")   -> 'WITH (replicas=2)'
properties(props, prefix=" WITH")  -> ' WITH (replicas=2)'
properties(props, wrapped=False)   -> 'replicas=2'
with_properties(props)             -> 'WITH (replicas=2)'
self.sql(props)                    -> 'WITH (replicas=2)'
self.sql(exp.Properties(expressions=[]))  -> ''
```
So `LoadTable` needs only:
```python
LoadTable: lambda s, e: (f"LOAD TABLE {s.sql(e, 'this')}"
                         + (f" {s.sql(e, 'properties')}" if e.args.get("properties") else ""))
```
`self.sql(e, "properties")` dispatches `properties_sql` → `"WITH (replicas=2)"`.

Pretty-mode caveat: `with_properties` uses `self.seg(...)`, so pretty output becomes multi-line
`WITH (\n  replicas=2\n)`. Acceptable for us.

**`indexparameters_sql`** — base at `generator.py:1948-1965` puts `using` FIRST. MilvusQL override
(verified, produces the exact target string):
```python
def indexparameters_sql(self, e):
    columns = self.expressions(e, key="columns", flat=True)
    columns = f" ({columns})" if columns else ""
    using = self.sql(e, "using")
    using = f" USING {using.upper()}" if using else ""
    ws = self.expressions(e, key="with_storage", flat=True)
    ws = f" WITH ({ws})" if ws else ""
    return f"{columns}{using}{ws}"
# -> CREATE INDEX idx_emb ON items (embedding) USING HNSW WITH (metric_type='COSINE', M=16, ef_construction=200)
```
`indexparameters_sql` **is** dispatched by method name (the class is a built-in and *is* in
`EXPR_CLASSES`) — the §4.1 trap does not apply to overrides of base handlers.

⚠️ **`Ordered(nulls_first=False)` artifact:** the *default* dialect emits a spurious ` NULLS LAST`
inside index column lists (`ordered_sql`, `generator.py:3149-3172`, driven by
`Dialect.NULL_ORDERING`). Set `NULL_ORDERING = "nulls_are_last"` on the Milvus dialect (verified:
removes the artifact).

### 4.8 Custom binary operators

`self.binary(expression, op)` — `generator.py:4637-4657` — is iterative (avoids recursion depth on
long chains), always emits `left <space> op <space> right`, adds no parentheses, and flattens
same-type chains. The `node.args.get("operator")` branch is postgres `OPERATOR(schema.op)` support;
leave it unset.

`<->` **already exists**: `expressions/core.py:2160 class Distance(Expression, Binary)` and
`generator.py:4476-4477 def distance_sql(...): return self.binary(expression, "<->")`
(`DistanceNd` → `<<->>` at `:4479`). **Reuse `exp.Distance` for L2.** Do not host anything else on
`LR_ARROW` — `distance_sql` hardcodes `"<->"`.

For the other three:
```python
class InnerProduct(exp.Expression, exp.Binary): pass
class L1Distance(exp.Expression, exp.Binary): pass
class CosineDistance(exp.Expression, exp.Binary): pass

TRANSFORMS = {**Generator.TRANSFORMS,
              InnerProduct:   lambda s, e: s.binary(e, "<#>"),
              L1Distance:     lambda s, e: s.binary(e, "<+>"),
              CosineDistance: lambda s, e: s.binary(e, "<=>")}
```

Pre-existing *function-form* distance nodes exist but are **not** operators
(`expressions/math.py:98-124`, all `(Expression, Func)`):
```
exp.CosineDistance    -> base: COSINE_DISTANCE(a, b)     postgres: COSINE_DISTANCE(a, b)
exp.DotProduct        -> base: DOT_PRODUCT(a, b)         postgres: DOT_PRODUCT(a, b)
exp.ManhattanDistance -> base: MANHATTAN_DISTANCE(a, b)  postgres: MANHATTAN_DISTANCE(a, b)
exp.EuclideanDistance -> ...
```
⚠️ **Name collision warning:** if you name your operator node `CosineDistance`, it shadows nothing
at runtime (different module), but it *will* confuse readers and would collide catastrophically if
anyone ever does `exp.EXPR_CLASSES["cosinedistance"] = ...`. Consider `CosineDist` /
`InnerProductOp` / `L1Dist` to keep the namespaces visually distinct.

### 4.9 `MATCH ... AGAINST` rendering — one-character divergence

`generator.py:3771-3786` (verbatim):
```python
    def matchagainst_sql(self, expression: exp.MatchAgainst) -> str:
        if self.MATCH_AGAINST_TABLE_PREFIX:
            expressions = []
            for expr in expression.expressions:
                if isinstance(expr, exp.Table):
                    expressions.append(f"TABLE {self.sql(expr)}")
                else:
                    expressions.append(expr)
        else:
            expressions = expression.expressions

        modifier = expression.args.get("modifier")
        modifier = f" {modifier}" if modifier else ""
        return (
            f"{self.func('MATCH', *expressions)} AGAINST({self.sql(expression, 'this')}{modifier})"
        )
```
Note **`AGAINST(`** — no space. The MilvusQL spec writes `AGAINST (:q)`. So
`MATCH(text) AGAINST (:q)` parses fine but regenerates as `AGAINST(:q)` — not text-identical.
Verified fix:
```python
def matchagainst_sql(self, e):
    modifier = e.args.get("modifier")
    modifier = f" {modifier}" if modifier else ""
    return f"{self.func('MATCH', *e.expressions)} AGAINST ({self.sql(e, 'this')}{modifier})"
# base      : SELECT id FROM items WHERE MATCH(text) AGAINST(:q)
# overridden: SELECT id FROM items WHERE MATCH(text) AGAINST (:q)
```
(This is a base-class handler override, so the method-name route is legal here.)

### 4.10 Making unsupported nodes raise

`errors.py:15` — `ErrorLevel = {IGNORE, WARN, RAISE, IMMEDIATE}`.
`generator.py:997-999`:
```python
    def unsupported(self, message: str) -> None:
        if self.unsupported_level == ErrorLevel.IMMEDIATE:
            raise UnsupportedError(message)
        self.unsupported_messages.append(message)
```
`generator.py:960-971` (end of `generate`): IGNORE → return; WARN → `logger.warning` per message;
RAISE → one `UnsupportedError(concat_messages(...))`; IMMEDIATE already raised.

Verified for a handler calling `self.unsupported(...)`:
```
IGNORE     -> ''
WARN       -> ''    (+ logged "Weird is not supported in MilvusQL")
RAISE      -> RAISED UnsupportedError: Weird is not supported in MilvusQL
IMMEDIATE  -> RAISED UnsupportedError: Weird is not supported in MilvusQL
```

**Genuinely unknown nodes already hard-raise regardless of `unsupported_level`:**
```
unknown node @ ErrorLevel.IGNORE -> ValueError | is SqlglotError: False | is UnsupportedError: False
unknown node @ ErrorLevel.RAISE  -> ValueError | is SqlglotError: False | is UnsupportedError: False
```
⚠️ **`ValueError` is NOT a `SqlglotError`.** `except SqlglotError` will not catch a missing
`TRANSFORMS` entry, and `unsupported_level=IGNORE` will not suppress it.

`Dialect.generator(**opts)` (`dialect.py:1200-1202`) passes kwargs straight through, so the strict
entry point is:
```python
Milvus().generator(unsupported_level=ErrorLevel.RAISE).generate(ast)
# or:  ast.sql(dialect="milvus", unsupported_level=ErrorLevel.IMMEDIATE)
```
`sqlglot.transpile(...)` defaults to `ErrorLevel.WARN` — **the test suite must pin `RAISE`.**

### 4.11 Useful generator helpers

`self.sep(sep=" ")` (`:1002`), `self.seg(sql, sep=" ")` (`:1005`), `self.wrap(expr_or_str)`
(`:1054`, pretty-aware parens), `self.func(name, *args, prefix="(", suffix=")", normalize=True)`
(`:4684`), `self.format_args(*args, sep=", ")` (`:4695`), `self.indent(...)` (`:1080`),
`self.function_fallback_sql` (`:4665`), the `unsupported_args("arg", ("arg2","msg"))` decorator
(`:32-62`).

---

## 5. Dialect and entry points

### 5.1 The `_Dialect` metaclass

`dialects/dialect.py:164-182, 256-259`:
```python
164: class _Dialect(type):
165:     _classes: dict[str, Type[Dialect]] = {}
167:     def __eq__(cls, other): ...        # cls == "milvus" -> True via cls.get(other)
177:     def __hash__(cls): return hash(cls.__name__.lower())
256:     def __new__(cls, clsname, bases, attrs):
257:         klass = super().__new__(cls, clsname, bases, attrs)
258:         enum = Dialects.__members__.get(clsname.upper())
259:         cls._classes[enum.value if enum is not None else clsname.lower()] = klass
```

**Every `Dialect` subclass is auto-registered globally at class-definition time under
`clsname.lower()`.** For `class Milvus(Dialect)`, the key is `"milvus"` — no entry point needed
*if the module has already been imported*. Verified:
```
auto-registered key: ['milvus']
parse_one("SELECT 1", read="milvus").sql(dialect="milvus") -> 'SELECT 1'
```

### 5.2 Nested-class handling and the sharing hazard

See §2.8 for the source. Two behaviours to internalise:

* **`Tokenizer` / `JSONPathTokenizer`**: omitting the nested class makes the metaclass synthesize a
  **fresh anonymous subclass** of the parent's. Safe to mutate.
* **`Parser` / `Generator`**: omitting them sets `parser_class` / `generator_class` to the
  **parent's class object itself — no subclass is created.** Verified:
  ```
  Child.parser_class    is Bare.parser_class    -> True   (SHARED, dangerous)
  Child.tokenizer_class is Bare.tokenizer_class -> False  (fresh subclass)
  ```
  **🔴 If you write `class Milvus(Postgres): pass` and then mutate
  `Milvus.parser_class.STATEMENT_PARSERS`, you corrupt `sqlglot.parsers.postgres.PostgresParser`
  globally for the whole process.** Always declare explicit nested classes.

Also `dialect.py:300-305` mutates `generator_class.TRANSFORMS` in place — another reason to always
subclass your Generator.

### 5.3 Default classes for a bare `Dialect` subclass

```
Bare.parser_class    = <class 'sqlglot.parsers.base.BaseParser'>
Bare.generator_class = <class 'sqlglot.generator.Generator'>
Bare.tokenizer_class = <class 'sqlglot.dialects.dialect.Tokenizer'>  (auto-synthesized subclass)
```
`sqlglot/parsers/base.py` (17 lines, full):
```python
class BaseParser(parser.Parser):
    NO_PAREN_FUNCTIONS = {**parser.Parser.NO_PAREN_FUNCTIONS,
                          TokenType.LOCALTIME: exp.Localtime,
                          TokenType.LOCALTIMESTAMP: exp.Localtimestamp,
                          TokenType.CURRENT_CATALOG: exp.CurrentCatalog,
                          TokenType.SESSION_USER: exp.SessionUser}
    ID_VAR_TOKENS      = parser.Parser.ID_VAR_TOKENS      - {TokenType.STRAIGHT_JOIN}
    TABLE_ALIAS_TOKENS = parser.Parser.TABLE_ALIAS_TOKENS - {TokenType.STRAIGHT_JOIN}
```
**Subclass `BaseParser`, not `parser.Parser`, for parity with built-ins** — unless you specifically
want `STRAIGHT_JOIN` back as an identifier. (The reference prototype subclasses `parser.Parser`
and is fine; `BaseParser` is the marginally more correct choice.)

### 5.4 Metaclass-derived attributes you must NOT set

`Dialect` defines 103 upper-case class constants; **25 are metaclass-derived**:

`BIT_END, BIT_START, BYTE_END, BYTE_START, BYTE_STRINGS_SUPPORT_ESCAPED_SEQUENCES,
ESCAPED_SEQUENCES, FORMAT_TRIE, HEX_END, HEX_START, IDENTIFIER_END, IDENTIFIER_START,
INITCAP_SUPPORTS_CUSTOM_DELIMITERS, INVERSE_CREATABLE_KIND_MAPPING, INVERSE_FORMAT_MAPPING,
INVERSE_FORMAT_TRIE, INVERSE_TIME_MAPPING, INVERSE_TIME_TRIE, QUOTE_END, QUOTE_START,
STRINGS_SUPPORT_ESCAPED_SEQUENCES, SUPPORTS_COLUMN_JOIN_MARKS, TIME_TRIE, UNICODE_END,
UNICODE_START, VALID_INTERVAL_UNITS`

Two that bite specifically:
* `dialect.py:345` `klass.SUPPORTS_COLUMN_JOIN_MARKS = "(+)" in klass.tokenizer_class.KEYWORDS` —
  **overwrites** whatever you set.
* `dialect.py:347-348`:
  ```python
  if enum not in ("", "bigquery", "snowflake"):
      klass.INITCAP_SUPPORTS_CUSTOM_DELIMITERS = False
  ```
  For a third-party dialect `enum is None`, so this **always** fires. Verified: an explicit
  `INITCAP_SUPPORTS_CUSTOM_DELIMITERS = True` is silently clobbered to `False`.

Settings we care about: `SUPPORTS_USER_DEFINED_TYPES = True` (keep, for `FLOAT_VECTOR` etc.),
`NULL_ORDERING = "nulls_are_last"` (§4.7), `NORMALIZATION_STRATEGY`,
`ALTER_TABLE_ADD_REQUIRED_FOR_EACH_COLUMN` (default `True`),
`INVERSE_VECTOR_TYPE_ALIASES` (exists specifically for vector dialects — see SingleStore).

`Dialect.__init__` (`:1027`) accepts only `SUPPORTED_SETTINGS = {"normalization_strategy",
"version"}`; extend that set if you want e.g. `"milvus, milvus_version = 2.4"`.

### 5.5 Entry-point plugin mechanism

```python
# dialect.py:35
from importlib.metadata import entry_points
# dialect.py:78
PLUGIN_GROUP_NAME = "sqlglot.dialects"
```

Loader — `_Dialect._try_load`, `dialect.py:188-233`:
```python
189:     def _try_load(cls, key):
194:         if key in DIALECT_MODULE_NAMES:                  # 1. built-ins WIN
195:             module = importlib.import_module(f"sqlglot.dialects.{key}")
206:             return
209:         try:                                             # 2. entry points
210:             all_eps = entry_points()
212:             if hasattr(all_eps, "select"):
213:                 eps = all_eps.select(group=PLUGIN_GROUP_NAME, name=key)
215:             else:
216:                 group_eps = all_eps.get(PLUGIN_GROUP_NAME, [])
217:                 eps = [ep for ep in group_eps if ep.name == key]
218:             for entry_point in eps:
219:                 dialect_class = entry_point.load()
221:                 if isinstance(dialect_class, type) and issubclass(dialect_class, Dialect):
223:                     if key not in cls._classes:
224:                         cls._classes[key] = dialect_class
225:                     return
226:         except ImportError:
227:             pass
230:         try:                                             # 3. legacy direct import
231:             importlib.import_module(f"sqlglot.dialects.{key}")
232:         except ImportError:
233:             pass
```

| question | verified answer |
|---|---|
| group | **`sqlglot.dialects`** — the TOML key must be quoted: `[project.entry-points."sqlglot.dialects"]` |
| value | `module.path:ClassName`, must resolve to a `Dialect` **subclass** |
| when loaded | **lazily**, on the first `Dialect.get(name)` / `Dialect[name]` / `get_or_raise(name)` miss |
| name source | the **entry-point NAME**, used verbatim (the class name is *also* registered independently by the metaclass as a side effect of importing the module) |
| `milvus` free? | ✅ `'milvus' in DIALECT_MODULE_NAMES` → `False` |

**Verified end-to-end** by the dialect-plugin recon, which built, `uv pip install -e`'d, exercised
and then uninstalled a real package:
```
modules before: []                                    <- lazy, not imported yet
1) parse_one SELECT 1 read=milvus -> 'SELECT 1'
2) transpile postgres->milvus     -> ['SELECT CAST(a AS INT) FROM t']
4) custom stmt AST: ReleaseTable(this=Table(this=Identifier(this=items, quoted=False)))
5) custom stmt roundtrip: LOAD TABLE items
6) modules after: ['milvusprobe2', 'milvusprobe2.dialect']
7) Dialect['milvus'] -> <class 'milvusprobe2.dialect.Milvus'>
CLI: python -m sqlglot "SELECT a::int FROM t" --read postgres --write milvus
     -> SELECT CAST("a" AS INT) FROM "t"
```
Post-uninstall the env was verified clean (`entry_points().select(group='sqlglot.dialects') -> []`,
`Dialect.get_or_raise('milvus') -> ValueError`).

Working `pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sqlglot-milvus"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["sqlglot>=30.16.0,<31"]

[project.entry-points."sqlglot.dialects"]
milvus = "sqlglot_milvus.dialect:Milvus"

[tool.hatch.build.targets.wheel]
packages = ["src/sqlglot_milvus"]
```

### 5.6 Entry-point gotchas (all reproduced)

1. **Entry-point names are NOT lowercased.** An entry point named `UPPERNAME` resolves only as
   `"UPPERNAME"`. → **use an all-lowercase name.**
2. **`AttributeError` from a bad entry point escapes** — only `ImportError` is caught
   (`dialect.py:226`). A dangling `:DoesNotExist` gives a raw `AttributeError`, not a friendly
   `ValueError`.
3. **Non-`Dialect` / module-only targets fail silently** as `Unknown dialect '...'`.
4. **Plugins CANNOT shadow built-in dialect names** — step 1 short-circuits on
   `key in DIALECT_MODULE_NAMES`. Entry points named `postgres` / `druid` still resolve to the
   built-ins.
5. **`Dialect.classes` does NOT discover plugins** (`dialect.py:180-186` iterates only
   `DIALECT_MODULE_NAMES`). Consequence: `get_or_raise`'s "Did you mean…?" list
   (`dialect.py:1004`, `set(DIALECT_MODULE_NAMES) | set(cls._classes.keys())`) will not suggest
   your plugin until it has already been loaded.
6. **No `Dialects` enum member is created for plugins.** Anything iterating
   `sqlglot.dialects.Dialects` will not see us.
7. **Duplicate class-name registration silently clobbers.** Defining `class Milvus(Dialect)` twice
   (stale module reload, or entry-point package plus a local import) replaces the first with no
   warning. → single definition site; add a test asserting
   `type(Dialect.get_or_raise("milvus")) is Milvus`.

### 5.7 `Dialect.get_or_raise` — STRICTLY case-sensitive

`dialect.py:955-1011`; the lookup at `:1002` is `cls.get(dialect_name.strip())` — **there is no
`.lower()` anywhere in the path.** Verified:
```
get_or_raise('milvus'    ) -> Milvus
get_or_raise(' milvus '  ) -> Milvus                      (strip() only)
get_or_raise('Milvus'    ) -> ValueError: Unknown dialect 'Milvus'. Did you mean milvus?
get_or_raise('MILVUS'    ) -> ValueError: Unknown dialect 'MILVUS'.
get_or_raise('milvuz'    ) -> ValueError: Unknown dialect 'milvuz'. Did you mean milvus?
```
Error is a plain `ValueError` from `helper.py:53-64 suggest_closest_match_and_fail`. It surfaces
unchanged out of `parse_one(..., read=...)` and `transpile(..., write=...)`.

Settings suffix works: `"milvus, normalization_strategy = case_sensitive"` →
`NormalizationStrategy.CASE_SENSITIVE`. Unknown key → `ValueError: Unknown setting 'foo'.`
A bare key defaults to `True`.

**Equality/hash quirk:** `_Dialect.__eq__` makes `Milvus == "milvus"` → `True`; `__hash__` is
`hash(cls.__name__.lower())`. **Keep entry-point name == lowercased class name** (`milvus` /
`Milvus`) or `hash(Milvus) != hash(entry_point_name)` while `==` is still `True`.

### 5.8 Templates worth reading

| file | lines | why |
|---|---|---|
| `dialects/druid.py` | 9 | absolute minimum: `Parser = DruidParser`, `Generator = DruidGenerator` |
| `dialects/risingwave.py` | 24 | **closest analogue**: derives from Postgres, adds tokenizer keywords → TokenTypes, `PROPERTY_PARSERS`, `CONSTRAINT_PARSERS`, generator `PROPERTIES_LOCATION` |
| `parsers/risingwave.py` | 66 | shows `self.expression(Instance(...))` calling convention, and `def _parse_table_hints(self): return None` to neutralize a base rule |
| `dialects/singlestore.py` | 64 | **multi-char operators in `KEYWORDS`** (`:>`, `!:>`, `::$`) + `VECTOR_TYPE_ALIASES` / `INVERSE_VECTOR_TYPE_ALIASES` |
| `dialects/materialize.py` | 13 | leanest "derive from an existing dialect" form |
| `dialects/solr.py` | 20 | `class Solr(Dialect)` with nested `class Tokenizer(tokens.Tokenizer)` |

---

## 6. Placeholders and custom expressions

### 6.1 `:name` works out of the box — ZERO dialect opt-in

Evidence chain:

1. **Tokenizer**: `SINGLE_TOKENS[":"] == TokenType.COLON`. There is no `PLACEHOLDER` handling for
   `:` in the tokenizer. Only `"?"` is `TokenType.PLACEHOLDER` and `"@"` is `TokenType.PARAMETER`;
   `"$"` is not a single token at all.
   ```
   Tokenizer().tokenize(":emb") -> [('COLON', ':'), ('VAR', 'emb')]
   ```
2. **Parser** — `parser.py:1249-1257`:
   ```python
   PLACEHOLDER_PARSERS: t.ClassVar = {
       TokenType.PLACEHOLDER: lambda self: self.expression(exp.Placeholder()),
       TokenType.PARAMETER:   lambda self: self._parse_parameter(),
       TokenType.COLON: lambda self: (
           self.expression(exp.Placeholder(this=self._prev.text))
           if self._match_set(self.COLON_PLACEHOLDER_TOKENS)
           else None
       ),
   }
   ```
   with `parser.py:850: COLON_PLACEHOLDER_TOKENS: t.ClassVar = ID_VAR_TOKENS`.
3. **Entry point** — `_parse_placeholder`, `parser.py:8843-8849`, called as the fallback tail of
   `_parse_primary`, `_parse_string`, `_parse_number`, `_parse_identifier`, `_parse_var`,
   `_parse_null`, `_parse_boolean`, `_parse_star`.
4. **Generator** — `generator.py:682` `NAMED_PLACEHOLDER_TOKEN = ":"` and `:3450-3451`:
   ```python
   def placeholder_sql(self, expression: exp.Placeholder) -> str:
       return f"{self.NAMED_PLACEHOLDER_TOKEN}{expression.name}" if expression.this else "?"
   ```

A bare `class Milvus(Dialect)` inherits all of it. **Do not override anything for `:name`.**

Node shape (`expressions/core.py:1847-1848`):
```python
class Placeholder(Expression, Condition):
    arg_types = {"this": False, "kind": False, "widget": False, "jdbc": False}
```
`this` is stored as a **plain `str`** by the base parser (`this=self._prev.text`), not an
Identifier. `.name` → `'emb'`. `widget` is Spark-only; `jdbc` is postgres-only. Ignore both.

### 6.2 Bind-parameter cross-dialect matrix (verified)

`SELECT <input>`; `RT` = `parse_one(s, read=D).sql(dialect=D)`.

| input | dialect | node | args | RT |
|---|---|---|---|---|
| `:emb` | **base / milvus** | `Placeholder` | `{'this': 'emb'}` (str) | `:emb` ✅ |
| `:emb` | oracle / snowflake / tsql / mysql | `Placeholder` | `{'this': 'emb'}` | `:emb` ✅ |
| `:emb` | postgres | `Placeholder` | `{'this': 'emb'}` | `%(emb)s` |
| `:emb` | bigquery | `Placeholder` | `{'this': 'emb'}` | `@emb` |
| `:emb` | clickhouse | `Placeholder` | `{'this': 'emb'}` | `{emb: }` ⚠️ **not re-parseable** |
| `:emb` | **duckdb** | **ParseError in a projection list**¹ | — | — |
| `%(emb)s` | postgres | `Placeholder` | `{'this': Identifier(emb)}` ← **Identifier, not str** | `%(emb)s` |
| `%(emb)s` | mysql | `Alias(Anonymous('%',[emb]) AS s)` 😱 **silently wrong** | — | `%(emb) AS s` |
| `?` | base | `Placeholder` | `{}` | `?` |
| `@emb` | base | **`Parameter`** | `{'this': Var(emb)}` | `@emb` |
| `@emb` | **duckdb** | **`Abs`** 😱 (`@` is prefix ABS) | `{'this': Column(emb)}` | `ABS(emb)` |
| `$emb` | base | `Column(Identifier('$emb'))` 😱 | — | `$emb` |
| `$emb` | duckdb | `Placeholder` | `{'this': 'emb'}` | `$emb` |
| `$emb` | clickhouse | **TokenError** | — | — |
| `:1` | **snowflake only** | `Placeholder` | `{'this': '1'}` | `:1` |
| `:1` | all others | **ParseError** | — | — |

¹ `parse_one("SELECT :emb", read="duckdb")` → `ParseError: Required keyword: 'this' missing for
Alias` — duckdb uses `:` for `expr : alias` sugar. Works in `WHERE`, in `VALUES`, and parenthesized.

Second-pass stability (`parse→sql→parse→sql`) holds for all of the above **except clickhouse**,
which throws `ParseError` on its own output.

### 6.3 `:name` reserved-word limits

`COLON_PLACEHOLDER_TOKENS = ID_VAR_TOKENS` (258 members) excludes `SELECT`, `FROM`, `VALUES`.
Verified across our whole vocabulary:
```
:emb :cat :q :k :limit :table :order :index :key :search :hybrid :release :weight  -> OK
:select :from :values :1                                                          -> ParseError
```
**If you host a new keyword on a Tier-A TokenType, `:thatword` stops working** unless you add the
TokenType back to `ID_VAR_TOKENS` (§3.3). Our multi-word `HYBRID SEARCH` / `SEARCH PARAMS` approach
avoids the problem entirely for those words; only `RELEASE` needs the un-reservation.

Whitespace between `:` and the name is accepted (`x = : emb`, `x = :\nemb` → `Placeholder('emb')`).
Arguably too tolerant; not our problem.

### 6.4 🔴 `exp.Expr`, `exp.Func`, `exp.Binary`, `exp.Condition` are ABSTRACT TRAITS

`expressions/core.py:51-52`:
```python
@trait
class Expr:
```
with `core.py:291-292`:
```python
    def _set_parent(self, arg_key: str, value: object, index: int | None = None) -> None:
        raise NotImplementedError
```
The concrete implementation with `__slots__` is `core.py:824 class Expression(Expr)`.

Verified MROs and construction:
```
Expr         MRO=['Expr', 'object']
Expression   MRO=['Expression', 'Expr', 'object']
Func         MRO=['Func', 'Condition', 'Expr', 'object']       <- does NOT inherit Expression
Binary       MRO=['Binary', 'Condition', 'Expr', 'object']
Condition    MRO=['Condition', 'Expr', 'object']
Property     MRO=['Property', 'Expression', 'Expr', 'object']  <- concrete

subclass Expr       -> FAIL NotImplementedError: (no message)
subclass Func       -> FAIL NotImplementedError: (no message)
subclass Binary     -> FAIL NotImplementedError: (no message)
subclass Condition  -> FAIL NotImplementedError: (no message)
subclass (Expression, Func) -> OK, key = ok, sql_names = ['OK']
```

The failure is at **construction time**, with a ~30-frame traceback and **no error message** — the
single most confusing failure mode in the whole library.

**MANDATE: `exp.Expression` FIRST, traits second.**
```python
class LoadTable(exp.Expression):                    arg_types = {"this": True, "properties": False}
class InnerProduct(exp.Expression, exp.Binary):     pass
class BM25Score(exp.Expression, exp.Func):          _sql_names = ["BM25_SCORE"]; arg_types = {...}
```
This is exactly how sqlglot itself declares nodes: `core.py:2160 class Distance(Expression, Binary)`,
`core.py:2144 class EQ(Expression, Binary, Predicate)`, `string.py:89 class MatchAgainst(Expression, Func)`.

Other `@trait` classes to watch: `Predicate` (`:1579`), `AggFunc`, `ColumnConstraintKind`,
`Connector` (`:1605`, `:1610`, `:1623`, `:1636`, `:1696`, `:2027`, `:2072`).

### 6.5 `arg_types` semantics — `True` = REQUIRED

`expressions/core.py:103-111`:
```python
    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.key = cls.__name__.lower()                                   # LoadTable -> "loadtable"
        cls.required_args = {k for k, v in cls.arg_types.items() if v}   # True == REQUIRED
```

⚠️ **The `Expr` docstring at `core.py:61` is WRONG.** It says arg_types "maps arg keys to booleans
that indicate whether the corresponding args are **optional**". The example two lines below
(`core.py:75-78`) and `required_args` both prove `True` = required.

Verified:
```
class N(exp.Expression): arg_types = {"this": True, "opt": False}
N.required_args = {'this'}
N()                  -> constructs FINE
N().error_messages() -> ["Required keyword: 'this' missing for <class '__main__.N'>"]
```

**Validation happens at PARSE time only, never at construction.** It is invoked only from
`Parser.expression` → `Parser.validate_expression` (`parser.py:2097`, `:2187`) and
`expressions/builders.py:990`. If you build nodes by hand (transforms, tests, the
`ast_to_pymilvus` translator in Track B), **nothing validates them.**

An empty list counts as missing (`core.py:1315-1316`).

### 6.6 🔴 The `UNITTEST` flag — your tests are STRICTER than production

`expressions/core.py:48`:
```python
UNITTEST: bool = "unittest" in sys.modules or "pytest" in sys.modules
```
`core.py:1306-1310`:
```python
    def error_messages(self, args=None) -> list[str]:
        if UNITTEST:
            for k in self.args:
                if k not in self.arg_types:
                    raise TypeError(f"Unexpected keyword: '{k}' for {self.__class__}")
```

Verified in-process (`core.UNITTEST = False`) vs under pytest (`final_verify/test_unittest_flag.py`,
4 passed):
```
core.UNITTEST = True
RAISED TypeError: Unexpected keyword: 'bogus' for <class '...N'>
validate UNPATCHED RAISED: TypeError Unexpected keyword: 'search_params' for <class 'sqlglot.expressions.query.Select'>
validate PATCHED: ok
sql() with unknown arg key -> SELECT id FROM t LIMIT 10          # generation never validates
```

So `SELECT ... SEARCH PARAMS (...)` parses fine at runtime (because `_parse_query_modifiers` uses
`this.set(...)`, which does not validate) and then blows up as a **`TypeError`, not a `ParseError`**
the moment any code path calls `validate_expression` under pytest.

**MANDATORY at dialect import time:**
```python
exp.Select.arg_types["search_params"] = False
exp.Select.arg_types["hybrid"] = False
```
`required_args` is computed from truthy values in `__init_subclass__`, so an optional key needs
nothing further. Verified: after the patch, `validate_expression ok`.

### 6.7 Custom `exp.Func` — registries and a working example

| registry | file:line | populated with | mutable after import? |
|---|---|---|---|
| `exp.ALL_FUNCTIONS` | `expressions/__init__.py:49` | 562 entries | snapshot, no effect |
| `exp.FUNCTION_BY_NAME` | `expressions/__init__.py:50` | 628 entries | **no effect on `Parser.FUNCTIONS`** |
| `exp.EXPR_CLASSES` | `expressions/__init__.py:51` | 1043 entries | affects `_build_dispatch`, but see §4.2 — **do not use** |
| `Parser.FUNCTIONS` | `parser.py:373` | `{name: cls.from_arg_list}`, **str keys** | you merge into your subclass |
| `Parser.FUNCTION_PARSERS` | `parser.py:1558` | **str keys**, for custom syntax (`"MATCH"`) | you merge |
| `Parser.NO_PAREN_FUNCTION_PARSERS` | `parser.py:1540` | **str keys** | — |
| `Parser.NO_PAREN_FUNCTIONS` | `parser.py:477` | **TokenType keys** | — |
| `Generator.TRANSFORMS` | `generator.py:136` | `type[Expr] -> callable` | ✅ the reliable path |

`Parser.FUNCTIONS` is a class-body snapshot of `exp.FUNCTION_BY_NAME`
(`parser.py:373-374`), so mutating `exp.FUNCTION_BY_NAME` later does nothing. Verified:
`exp.FUNCTION_BY_NAME["LATE_FN"] = Late` → `"LATE_FN" in parser.Parser.FUNCTIONS` is `False`.
The `Dialect` metaclass does **no** auto-merging — all merging is manual `{**Base.X, ...}`.

`Func` machinery (`core.py:1658-1694`):
* `from_arg_list` maps positional args onto `arg_types` **keys in declaration order**. So
  `arg_types = {"this": True, "expression": True}` ⇒ `BM25_SCORE(a, b)` → `this=a, expression=b`.
* `sql_names()` derives from the class name if `_sql_names` is unset:
  `class HybridRerank` → `['HYBRID_RERANK']`, but `class BM25Score` → `['BM25SCORE']`.
  **Always set `_sql_names` explicitly.**
* `default_parser_mappings()` returns `{name: cls.from_arg_list for name in sql_names()}`.

Working, verified:
```python
class BM25Score(exp.Expression, exp.Func):        # Expression FIRST
    _sql_names = ["BM25_SCORE"]
    arg_types = {"this": True, "expression": True}

class Parser(parser.Parser):
    FUNCTIONS = {**parser.Parser.FUNCTIONS, **BM25Score.default_parser_mappings()}
```
```
parse_one("SELECT id FROM items ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10", read=Milvus)
  node: BM25Score(this=Column(Identifier(text)), expression=Placeholder(this=q))
  regen: SELECT id FROM items ORDER BY BM25_SCORE(text, :q) DESC LIMIT 10
  round-trip AST equal? True
```
Arity is enforced at parse time:
```
1 arg  -> ParseError: Required keyword: 'expression' missing for <class 'BM25Score'>
2 args -> BM25Score
3 args -> ParseError: The number of provided arguments (3) is greater than the maximum (2)
```
**Generation needs NO registration** — `generator.py:1123-1124`'s
`elif isinstance(expression, exp.Func): function_fallback_sql(...)` handles it. Only *custom
formatting* needs `TRANSFORMS`. ⚠️ The flip side: a `bm25score_sql` method would be **silently
ignored** and fall through to the fallback with no error (§4.1).

⚠️ Without registering in `Parser.FUNCTIONS`, `BM25_SCORE(text, :q)` becomes `exp.Anonymous` and
**still round-trips textually** — a missing registration is a *silent semantic loss*, not a parse
error. Test on node type, not on text.

### 6.8 `MATCH ... AGAINST` — reuse `exp.MatchAgainst`, it is in the BASE parser

`expressions/string.py:89-90`:
```python
class MatchAgainst(Expression, Func):
    arg_types = {"this": True, "expressions": True, "modifier": False}
```
`parser.py:1580` (inside the base `FUNCTION_PARSERS`, a **string**-keyed dict):
```python
        "MATCH": lambda self: self._parse_match_against(),
```
Implementation `parser.py:8434-8463`; the whole thing is text-driven —
`self._match_text_seq(")", "AGAINST", "(")`.

**`TokenType.MATCH` exists but the base tokenizer never emits it.**
`Tokenizer.KEYWORDS.get("MATCH") is None`; `TokenType.MATCH in Parser.FUNC_TOKENS` is `False`.
Only sqlite maps it (`dialects/sqlite.py:31`). Verified tokenization:
```
tokenize("MATCH(a) AGAINST (:q)") ->
 [('VAR','MATCH'),('L_PAREN','('),('VAR','a'),('R_PAREN',')'),('VAR','AGAINST'),
  ('L_PAREN','('),('COLON',':'),('VAR','q'),('R_PAREN',')')]
```
Both `MATCH` and `AGAINST` are plain `VAR`; dispatch is purely string-based. **This is the exact
pattern MilvusQL should copy** (`FUNCTION_PARSERS` key + `_match_text_seq`) wherever a multi-word
construct does not need a modifier slot.

**⚠️ The args are INVERTED from intuition:**
```
parse_one("SELECT * FROM t WHERE MATCH(a) AGAINST ('x')", read="mysql")
-> MatchAgainst(this=Literal('x', is_string=True),        # <- the QUERY goes in `this`
                expressions=[Column(Identifier(a))],       # <- the COLUMNS go in `expressions`
                modifier=None)
```
Anyone writing `MatchAgainst(this=col, expressions=[q])` by hand produces backwards SQL.
`modifier` is a raw Python `str`, not a node (`'IN BOOLEAN MODE'`).

`MATCH(text) AGAINST (:q)` works **only by luck**: `parser.py:8447 this = self._parse_string()` and
`_parse_string` falls through to `_parse_placeholder`. Verified:
```
MATCH(text) AGAINST (:q) -> MatchAgainst(this=Placeholder(this=q), expressions=[Column(Identifier(text))])
```
`AGAINST (some_column)` would **not** work — `_parse_string` accepts only STRING tokens or
placeholders. Since MilvusQL's spec uses `:q`, we are fine; document it.

Cross-dialect regeneration: base / mysql / tsql / snowflake → `MATCH(a) AGAINST('x')`;
postgres → `a @@ 'x'` (and `(a @@ 'x' OR b @@ 'x')` for two columns).

### 6.9 VECTOR type — `DType.VECTOR` already exists

* `tokenizer_core.py:234` `VECTOR = auto()` (TokenType)
* `tokens.py:498` `"VECTOR": TokenType.VECTOR` (base tokenizer keyword)
* `expressions/datatypes.py:163` `VECTOR = auto()` (in `class DType(AutoName)`)
* `expressions/datatypes.py:192` `Type: t.ClassVar[Type[DType]] = DType` — **`exp.DataType.Type is
  exp.DType` → `True`**. Reprs print `DType.VECTOR`, not `Type.VECTOR`.
* `TokenType.VECTOR in Parser.TYPE_TOKENS` → `True`

Verified `CREATE TABLE` behaviour (base dialect, **zero custom code**):

| declared | `kind` | base regen |
|---|---|---|
| `VECTOR(768)` | `DataType(this=DType.VECTOR, expressions=[DataTypeParam(Literal(768))])` | `VECTOR(768)` ✅ |
| `VECTOR` | `DataType(this=DType.VECTOR)` | `VECTOR` ✅ |
| `FLOAT_VECTOR(768)` | `DType.USERDEFINED, kind=FLOAT_VECTOR` | `FLOAT_VECTOR(768)` ✅ |
| `BINARY_VECTOR(128)` | `DType.USERDEFINED, kind=BINARY_VECTOR` | `BINARY_VECTOR(128)` ✅ |
| `SPARSE_FLOAT_VECTOR` | `DType.USERDEFINED, kind=SPARSE_FLOAT_VECTOR` | `SPARSE_FLOAT_VECTOR` ✅ |
| `VECTOR(FLOAT, 768)` | `this=VECTOR, expressions=[DataType(FLOAT), DataTypeParam(768)]` | ⚠️ postgres → `VECTOR(DOUBLE PRECISION, 768)`, duckdb → `VECTOR(REAL, 768)` |
| `ARRAY(INT)` | `DataType(ARRAY, ..., nested=True)` | ⚠️ base → `ARRAY<INT>`, postgres/duckdb → `INT[]` |

Unknown type names do **not** error; they become `DType.USERDEFINED` with `kind=<name>` and
round-trip verbatim including the parameter list. Subtle inconsistency: in base/duckdb `kind` is a
bare `Var`-like node, in **postgres** it is `Identifier(quoted=False)`. **Do not pattern-match on
`kind`'s node type across dialects.**

`exp.DataType.build("VECTOR(768)")` works; `exp.DataType.build("SPARSE_FLOAT_VECTOR(768)")`
→ `ParseError`. Pass `udt=True` for user-defined types.

**Do not invent a custom Expression node for column types.**

### 6.10 `AUTO_INCREMENT` and `CREATE TABLE ... WITH (...)` are free

`tokens.py:240-241` maps both `AUTOINCREMENT` and `AUTO_INCREMENT` → `TokenType.AUTO_INCREMENT`.
Verified in the **base** dialect with no custom code, text-identical round-trip:
```
CREATE TABLE items (id BIGINT PRIMARY KEY AUTO_INCREMENT, embedding VECTOR(768), category VARCHAR(64))
  WITH (shards=2, consistency_level='Bounded', partition_key=category)
```
→ `Create(this=Schema(...), properties=Properties([Property(Var(shards), Literal(2)), ...]))`.
Order-independent (`AUTO_INCREMENT PRIMARY KEY` also works).

`INSERT INTO items (embedding, category) VALUES (:emb, :cat)` and
`DELETE FROM items WHERE category = :cat` are likewise free and text-identical.

### 6.11 Custom nodes and the rest of the sqlglot ecosystem

Verified for a `Select` carrying `search_params` / `hybrid` in `args`:

| operation | result |
|---|---|
| `.set()` unknown key | accepted, no validation |
| `.sql()` on a **base** generator | **silently drops it**, no warning |
| `.copy()` | preserved, deep-copied |
| `walk()` / `find_all()` | traverses in — the node and its children are reachable |
| `repr()` | shows the key |
| `parent` / `arg_key` | correctly set |
| `qualify()` / `optimize()` | **preserved** (verified with a schema) |
| `.select()/.where()/.limit()/.subquery()` | preserved |
| `sqlglot.serde.dump/load` | round-trips, key preserved; records `'module.ClassName'` |
| `__eq__` | correctly distinguishes |

Custom `exp.Expression` subclasses also get working `copy()`, `__eq__`, `__hash__`.

`ORDER BY embedding <-> :q LIMIT 10` survives `qualify()` and `optimize()` intact:
`ORDER BY "items"."embedding" <-> :q LIMIT 10`.

---

## 7. CONFIRMED PITFALLS

Ordered by severity. Each is reproduced, with a verified mitigation.

### P1 🔴 BLOCKER — `sqlglot[c]` / `sqlglot[rs]` kill third-party dialects
See §1.3. Instantiating any interpreted subclass of `Parser` / `Generator` / `Expression` raises
`TypeError: interpreted classes cannot inherit from compiled`. The class statement succeeds, so the
package imports cleanly and explodes on first use.
**Mitigation:** pin `sqlglot>=30.16.0,<31` without extras; import-time guard on
`sqlglot.tokens.SQLGLOTC_INSTALLED`; CI job that installs `sqlglot[c]` and asserts the guard fires.

### P2 🔴 BLOCKER — `class Foo(exp.Func)` raises a bare `NotImplementedError`
The pattern in every sqlglot 25.x/26.x tutorial, and in the architecture plan (§2.3,
`class BM25Score(exp.Func)`). §6.4.
**Mitigation:** `class BM25Score(exp.Expression, exp.Func)` — `Expression` FIRST. Same for
`Binary`, `Condition`, `Predicate`, `Expr`.

### P3 🔴 BLOCKER — `<key>_sql` generator methods are silently ignored for third-party nodes
§4.1. `ValueError: Unsupported expression type LoadTable` for non-Func nodes; for
`(Expression, Func)` nodes it is **worse** — silent fallthrough to `function_fallback_sql` with
plausible-but-wrong output and no error.
**Mitigation:** every custom node goes in `TRANSFORMS = {**Generator.TRANSFORMS, Node: fn}`.
Never mutate `exp.EXPR_CLASSES`; never patch `TRANSFORMS` after class definition (`_DISPATCH_CACHE`
is permanent, §4.2).

### P4 🔴 HIGH — `TokenType` assignment silently produces a `str`/`int`, not a member
§2.4. `TokenType.HYBRID = "HYBRID"` "works", pollutes the global enum, and then never matches in
`_match_set`. The symptom is a mystery parse failure, not a `TypeError`.
**Mitigation:** never assign to `TokenType`. Recycle Tier-A members (§2.5).

### P5 🔴 HIGH — a new keyword after the table name is swallowed as a table alias
```
base: parse_one("SELECT id FROM items HYBRID") -> SELECT id FROM items AS HYBRID
```
`parser.py:836 TABLE_ALIAS_TOKENS = ID_VAR_TOKENS - {...}` — 249 of 258 tokens are legal aliases.
The resulting `ParseError` points at the *next* word, not at the keyword, which is deeply
misleading.
**Mitigation (verified, better than the recon's suggestion):** use a **multi-word keyword on a
Tier-A TokenType**. `HYBRID SEARCH` becomes one token that is absent from `TABLE_ALIAS_TOKENS`, so
it is never eaten as an alias — **and the bare words `HYBRID` / `SEARCH` remain usable as
identifiers.** No `ID_VAR_TOKENS` subtraction needed. Verified:
```
SELECT id FROM items HYBRID SEARCH (embedding <=> :dv WEIGHT 0.7, sparse_emb <#> :sv WEIGHT 0.3)
    RERANK RRF(k=60) LIMIT 10                          -> parses, text-identical round-trip
SELECT hybrid FROM t          -> SELECT hybrid FROM t
SELECT id FROM items hybrid   -> SELECT id FROM items AS hybrid
```

### P6 🔴 HIGH — `<=>` is ALREADY BOUND to `exp.NullSafeEQ`
`tokens.py:218 "<=>": TokenType.NULLSAFE_EQ`; `parser.py:955 EQUALITY = {..., NULLSAFE_EQ: exp.NullSafeEQ}`.
Today, with no dialect: `parse_one("SELECT a <=> b")` → `NullSafeEQ`, rendered as
`SELECT a IS NOT DISTINCT FROM b`. **A naive Milvus dialect silently produces a boolean predicate
where the user wrote a cosine distance.**

Precedence differs too (§3.6): `<->` is `FACTOR`, `<=>` is `EQUALITY`, so
`a <-> b + c` → `Add(Distance(a,b), c)` but `a <=> b + c` → `NullSafeEQ(a, Add(b,c))`.

The killer scenario — a Milvus dialect **inheriting from MySQL** (a tempting choice, since MySQL is
the only dialect that *emits* `<=>`): MySQL `NullSafeEQ` renders as `a <=> b`, which our parser
re-reads as a cosine distance. Identical text, silently different meaning, no error at any step.

**Mitigation (all four, verified):**
1. Remove `NULLSAFE_EQ` from `EQUALITY` **and** add it to `FACTOR` — gives all four operators
   identical precedence.
2. **Do NOT inherit from MySQL.** Inherit from bare `Dialect`, so `exp.NullSafeEQ` renders as
   `IS NOT DISTINCT FROM`.
3. Add a `TRANSFORMS` entry for `exp.NullSafeEQ` in the Milvus generator that calls
   `self.unsupported(...)`, so inbound MySQL can never round-trip into a distance op.
4. Test `SELECT a <=> b` asserts `CosineDistance`, never `NullSafeEQ`.

### P7 🔴 HIGH — `WEIGHT 0.7` is parsed as an implicit column alias
`parser.py:5978`: `def _parse_expression(self): return self._parse_alias(self._parse_assignment())`.
Verified by driving the parser directly on `a <-> b WEIGHT 0.7`:
```
_parse_expression   -> Alias(this=Distance(a,b), alias=WEIGHT)   remaining=['0.7']
_parse_assignment   -> Distance(a,b)                              remaining=['WEIGHT','0.7']
```
Base-dialect confirmation: `SELECT embedding <-> q WEIGHT` → `SELECT embedding <-> q AS WEIGHT`.
**Mitigation:** inside `HYBRID SEARCH (...)` use `self._parse_csv(one)` where `one()` calls
**`self._parse_assignment()`**, never `_parse_expression`. `_parse_csv` (`parser.py:8860`) takes
the method as a required arg, so there is no unsafe default. Verified end-to-end.

### P8 🔴 HIGH — `CREATE INDEX ... USING HNSW`, `LOAD TABLE`, `ALTER TABLE ... ADD FIELD` silently degrade to `exp.Command` AND ROUND-TRIP PERFECTLY
Base dialect:
```
Command  'CREATE INDEX i ON t (e) USING HNSW'          -> identical text
Command  'LOAD TABLE items WITH (replicas=2)'          -> identical text
Command  'ALTER TABLE items ADD FIELD tag VARCHAR(32)' -> identical text
Create   'CREATE INDEX i ON t USING HNSW (e)'          -> structured (USING before the column list)
```
The node is completely opaque:
```
type: Command | args: ['this', 'expression']
find_all(Column): []      find_all(Identifier): []
sql() == input: True
repr: Command(this=CREATE, expression=INDEX i ON t (e) USING HNSW WITH (m=16))
```
The warning at `parser.py:2266-2268` is a `logger.warning`, invisible under any normal logging
config.
**Mitigation: a round-trip test is WORTHLESS as a correctness signal here.** Every test must assert
`not isinstance(ast, exp.Command)` **and** assert on node types. Our parser overrides (§3.2, §3.8,
§3.9) eliminate all three.

### P9 🟠 MEDIUM — `error_level=RAISE` does NOT catch a single trailing garbage word
`parser.py:2225-2226`: `if self._index < self._tokens_size: self.raise_error(...)`. But one
trailing word is absorbed as an alias *before* that check. And WARN/IGNORE silently truncate.
Verified:
```
RAISE    SELECT a FROM t GARBAGE TOKENS HERE  -> ParseError               ✅
RAISE    SELECT a FROM t GARBAGE              -> 'SELECT a FROM t AS GARBAGE'   ⚠️
WARN     SELECT a FROM t GARBAGE TOKENS HERE  -> 'SELECT a FROM t AS GARBAGE'   ⚠️ HERE vanished
IGNORE   SELECT a FROM t GARBAGE TOKENS HERE  -> 'SELECT a FROM t AS GARBAGE'   ⚠️
```
**Mitigation:** never let the test suite pass `IGNORE`/`WARN`; assert on the exact AST, never merely
that parsing "succeeded". **A successful parse is not evidence your grammar matched.**

### P10 🟠 MEDIUM — a custom `Select` arg vanishes from `.sql()` with ZERO diagnostics
`generator.py:3270-3300 query_modifiers` reads a **fixed** key list. An unwired custom arg is
dropped silently, and the error type for a missing `TRANSFORMS` entry is a plain `ValueError` that
is **not** a `SqlglotError` and **not** suppressible by `unsupported_level`.
**Mitigation:** wire `query_modifiers` / `after_limit_modifiers` (§4.4) and add a regression test
asserting the clause survives `.sql()`.

### P11 🟠 MEDIUM — `QUERY_MODIFIER_TOKENS` breaks `GROUP BY` — but only if you key on `TokenType.VAR`
§3.5, full matrix verified. Keying on `VAR` **and** recomputing `QUERY_MODIFIER_TOKENS` locally
breaks every `GROUP BY <column>`.
**Mitigation:** key modifiers on **dedicated** TokenTypes and redefine
`QUERY_MODIFIER_TOKENS = set(QUERY_MODIFIER_PARSERS)`.

### P12 🟠 MEDIUM — `STATEMENT_PARSERS` + `_retreat` + `super()._parse_statement()` = infinite recursion
§3.3. `'RELEASE items'` → `RecursionError`.
**Mitigation:** raise a clear `ParseError`, or fall back to `self._parse_expression()` /
`self._parse_as_command(...)`. Never re-enter `_parse_statement`.

### P13 🟠 MEDIUM — `_parse_number()` on `:x` returns a `Placeholder`, not `None`
`parser.py:8784` falls back to `_parse_placeholder()`. Same for `_parse_string`, `_parse_var`,
`_parse_identifier`, `_parse_null`, `_parse_boolean`, `_parse_star`. So
`WEIGHT :w` / `ef_search=:n` are silently accepted. Probably desirable — but **`_parse_number`
cannot be used as a numeric type check.**

### P14 🟠 MEDIUM — `RRF(k=60)` produces a boolean comparison, not a named argument
```
SELECT RRF(k = 60)  -> Anonymous(RRF, [EQ(this=Column(k), expression=Literal(60))])
SELECT RRF(k => 60) -> Anonymous(RRF, [Kwarg(this=Var(k), expression=Literal(60))])
SELECT RRF(k := 60) -> Anonymous(RRF, [PropertyEQ(this=Identifier(k), expression=Literal(60))])
```
`EQ` creates a **phantom column reference `k`** that `qualify` / lineage will try to resolve against
the schema and fail on.
**Mitigation (verified, keeps the spec's `k=60` spelling):** do **not** let the generic function
parser see it. Parse the parenthesised bag yourself with `_parse_wrapped_properties()` into
`exp.Property(Var(k), Literal(60))` and render with `self.expressions(e, flat=True)`:
```
exp.Property via expressions(flat=True) -> RRF(k=60)     ✅ exact spec spelling, no phantom column
```
(`exp.PropertyEQ` → `k := 60`, `exp.Kwarg` → `k => 60`, `exp.EQ` → `k = 60` — **none** gives bare
`k=60`; only `exp.Property` does, via `property_sql`, `generator.py:2105-2108`.)

### P15 🟡 LOW — `WITH (partition_key=category)` value is `exp.Var`, invisible to column-aware passes
`_parse_key_value_property` (`parser.py:2915`) demotes a bare-identifier value from `Column` to
`Var`. So `find_all(exp.Column)` on that `CREATE` returns `[]`. `qualify`, lineage, column-renaming
and schema-validation passes **silently skip it**: rename the column and the property still points
at the old name, with no error.
**Mitigation:** either post-process `Property` values whose key is `partition_key` into
`exp.Column`, or accept and document the opacity. (`M=16` preserves case even under `identify=True`.)

### P16 🟡 LOW — custom `exp.Property` subclasses raise `KeyError`
`properties_sql:2046` / `locate_properties:2092` index `PROPERTIES_LOCATION[p.__class__]` directly.
**Mitigation:** emit plain `exp.Property`. §4.7.

### P17 🟡 LOW — dialect registry silently clobbers on duplicate class name
§5.6 item 7.

### P18 🟢 NEGATIVE FINDINGS — worth banking, no action needed

* **Clause ordering is a non-issue for the parser.** `_parse_query_modifiers` is an unordered
  `while True:` loop. `SEARCH PARAMS` after `LIMIT` parses fine; so does `SEARCH PARAMS` before
  `LIMIT` (see D5 for why that is a *problem*, not a relief).
* **Keyword-collision identifiers are broadly a non-issue.** 31 candidate identifiers
  (`text, category, embedding, tag, id, k, weight, shards, replicas, metric_type, search, params,
  hybrid, rerank, match, against, load, release, field, vector, sparse_emb, consistency_level,
  partition_key, value, index, key, order, limit, M, ef_construction`) all parse as `exp.Column`
  in base/mysql/postgres and all work as `CREATE TABLE` column names. **`text` is NOT a problem** —
  it tokenizes as `TokenType.TEXT` but `TEXT ∈ ID_VAR_TOKENS`, and `MATCH(text)`, `MATCH(t.text)`,
  `BM25_SCORE(text, q)` all produce correct `exp.Column` nodes. Only `values` and `key` fail, and
  only in the **mysql** dialect — another reason not to inherit from MySQL.
* **`<->` is free** — `parser.py:983 LR_ARROW: exp.Distance`, `generator.py:4476 distance_sql`.
  Do not redefine it.
* **`self._curr` is never `None`** (`SENTINEL_NONE`, `parser.py:332`), so
  `_parse_query_modifiers`' unguarded `self._curr.text.upper() == "START"` is safe, and so is
  yours.
* **`Tokenizer.COMMANDS` does not contain `LOAD`**, so the tokenizer will not swallow the rest of a
  `LOAD TABLE` statement as raw text.
* **`:name` placeholders work everywhere** in the grammar — `VALUES (:emb, :cat)`,
  `WHERE category = :cat`, `ORDER BY embedding <-> :q`, `AGAINST (:q)`, `WEIGHT :w`.

---

## 8. Corrections to the architecture plan

Every place where `/home/neko/startup/milvus/milvusdialectarchitectureplan.md` states something
factually **wrong** for sqlglot 30.16. Plan claims are quoted verbatim (Russian original),
with the verified correct fact. Line numbers refer to the plan file.

---

### C1 — `class BM25Score(exp.Func)` does not work · plan §2.3, line 148

> ```python
> class BM25Score(exp.Func):
>     arg_types = {"this": True, "expression": True}
> ```

**WRONG.** `exp.Func` is a `@trait` (`expressions/core.py:1641-1642`) whose MRO is
`['Func', 'Condition', 'Expr', 'object']` — it does **not** inherit `Expression`. Constructing an
instance raises a bare, message-less `NotImplementedError` from `Expr._set_parent`
(`core.py:291-292`).

**CORRECT:**
```python
class BM25Score(exp.Expression, exp.Func):     # Expression FIRST
    _sql_names = ["BM25_SCORE"]
    arg_types = {"this": True, "expression": True}
```
The other six node declarations in plan §2.3 (`LoadTable`, `ReleaseTable`, `SearchArm`,
`HybridSearch`, `Rerank`, `MatchAgainst`) use `exp.Expression` and **are correct**.
See §6.4.

---

### C2 — `STATEMENT_PARSERS` is keyed by `TokenType`, never by text · plan §2.5, lines 188, 197-201

> «Новый top-level statement — регистрируется в словаре `STATEMENT_PARSERS`
> (ключ — **токен/текст** ключевого слова, значение — метод парсера)»
> ```python
> STATEMENT_PARSERS = {
>     **_Parser.STATEMENT_PARSERS,
>     "LOAD": lambda self: self._parse_load_table(),
>     "RELEASE": lambda self: self._parse_release_table(),
> }
> ```

**WRONG.** `STATEMENT_PARSERS` (`parser.py:1154-1185`) is keyed **exclusively by `TokenType`**.
Dispatch is `self._match_set(self.STATEMENT_PARSERS)` then
`self.STATEMENT_PARSERS[self._prev.token_type](self)` (`parser.py:2374-2376`). A string key is
simply never looked up — it is dead weight, with no error.

**CORRECT:**
```python
STATEMENT_PARSERS = {**parser.Parser.STATEMENT_PARSERS,
                     TokenType.LOAD: lambda self: self._parse_milvus_load(),
                     TT_RELEASE:     lambda self: self._parse_milvus_release()}
```
(String-keyed dispatch dicts do exist in sqlglot — `PROPERTY_PARSERS` (`:1301`), `ALTER_PARSERS`
(`:1507`), `FUNCTION_PARSERS` (`:1558`), `SET_PARSERS` — but `STATEMENT_PARSERS` is not one of
them.) See §3.1.

---

### C3 — `QUERY_MODIFIER_PARSERS` is keyed by `TokenType`, and the value must be a *2-tuple-returning callable*, not a lambda returning a tuple of your own key · plan §2.5, lines 218-222

> ```python
> QUERY_MODIFIER_PARSERS = {
>     **_Parser.QUERY_MODIFIER_PARSERS,
>     "hybrid_search": lambda self: ("hybrid_search", self._parse_hybrid_search()),
>     "search_params": lambda self: ("search_params", self._parse_search_params()),
> }
> ```

**WRONG on the key.** `QUERY_MODIFIER_PARSERS` (`parser.py:1595-1621`) is keyed by `TokenType`;
the match is `self._match_set(self.QUERY_MODIFIER_PARSERS, advance=False)` (`parser.py:4346`).
String keys are never matched.

**Also missing three contract details** the plan does not mention:
1. `advance=False` — **the trigger token is NOT consumed**; your callable must `self._advance()`
   first.
2. The callable **must** return a 2-tuple; returning `None` → `TypeError` on unpack.
3. `QUERY_MODIFIER_TOKENS = set(QUERY_MODIFIER_PARSERS)` (`parser.py:1622`) is a separate class-body
   snapshot that must be redefined alongside (§3.5).

**CORRECT:**
```python
QUERY_MODIFIER_PARSERS = {**parser.Parser.QUERY_MODIFIER_PARSERS,
                          TT_HYBRID:  lambda self: ("hybrid", self._parse_hybrid_search()),
                          TT_SPARAMS: lambda self: ("search_params", self._parse_search_params())}
QUERY_MODIFIER_TOKENS = set(QUERY_MODIFIER_PARSERS)

def _parse_search_params(self):
    self._advance()                 # consume the trigger ourselves
    return self.expression(SearchParams(expressions=self._parse_wrapped_properties()))
```
See §3.4.

---

### C4 — `HYBRID`/`RERANK` mapped to `TokenType.VAR` does not work · plan §2.4, lines 166-181

> «С нуля размечать как `TokenType.VAR` (обычный идентификатор) нужно только по-настоящему новые
> слова … `HYBRID`, `RERANK`, `SEARCH`, `PARAMS`, `MATCH`, `AGAINST`.»
> ```python
> KEYWORDS = {**_Tokenizer.KEYWORDS, "HYBRID": TokenType.VAR, "RERANK": TokenType.VAR}
> ```

**FACTUALLY VACUOUS AND ACTIVELY HARMFUL.**

* **Vacuous:** these words *already* tokenize as `TokenType.VAR` — that is the default in
  `_scan_var` (`tokenizer_core.py:1116-1120`: `self.keywords.get(text.upper(), TokenType.VAR)`).
  Registering them explicitly changes nothing:
  ```
  tokenize("SELECT hybrid, search FROM t") -> [..., ('VAR','hybrid'), ..., ('VAR','search'), ...]
  ```
  Verified: none of `MATCH, AGAINST, ADD, RELEASE, SEARCH, HYBRID, RERANK, WEIGHT, FIELD, HNSW`
  are in base `KEYWORDS`.
* **Harmful:** `TokenType.VAR ∈ TABLE_ALIAS_TOKENS`, so `HYBRID` after a table name is eaten as an
  alias (P5): `SELECT id FROM items HYBRID` → `SELECT id FROM items AS HYBRID`, and the subsequent
  `ParseError` points at `SEARCH`, not at `HYBRID`.
* **Harmful:** keying a `QUERY_MODIFIER_PARSER` on `TokenType.VAR` and recomputing
  `QUERY_MODIFIER_TOKENS` breaks **every** `GROUP BY <column>` (§3.5).

**CORRECT:** register `HYBRID SEARCH` and `SEARCH PARAMS` as **multi-word keywords** on Tier-A
TokenTypes. This makes each clause a single `_match_set` target that is absent from
`TABLE_ALIAS_TOKENS`, **while leaving the bare words `HYBRID`, `SEARCH`, `PARAMS`, `RERANK`,
`WEIGHT` fully usable as identifiers.** See §2.6, §2.5, P5.

---

### C5 — `LOAD`/`RELEASE` are NOT matched by text at the `STATEMENT_PARSERS` level · plan §2.4, lines 168-169

> «Сами ключевые слова `LOAD`/`RELEASE` как начала стейтментов тоже не нужно нигде объявлять
> отдельно — они матчатся по тексту прямо на уровне `STATEMENT_PARSERS`.»

**WRONG for both, for different reasons.**

* `LOAD` **is** already a `TokenType` and **is** already in `STATEMENT_PARSERS`
  (`parser.py:1172`, → `_parse_load`, the Hive `LOAD DATA INPATH` parser, `parser.py:3797`). You
  must **override** that entry and delegate to `super()._parse_load()` for the `DATA` branch, or
  you break `LOAD DATA` inheritance.
* `RELEASE` has **no** `TokenType` and is **not** in base `KEYWORDS`. sqlglot has no `RELEASE`
  support at all (not even `RELEASE SAVEPOINT`). Verified:
  `RELEASE TABLE items` → `ParseError: Invalid expression / Unexpected token. Line 1, Col: 19.`
  Since `STATEMENT_PARSERS` is TokenType-keyed (C2), text matching is not available there.

**CORRECT:** map `"RELEASE"` to a Tier-A TokenType in `Tokenizer.KEYWORDS`, add a
`STATEMENT_PARSERS` entry, and add that TokenType back to `ID_VAR_TOKENS` /
`TABLE_ALIAS_TOKENS` / `COLON_PLACEHOLDER_TOKENS` so `release` stays usable as an identifier.
See §3.2, §3.3.

---

### C6 — `MATCH`/`AGAINST` need no work at all; `exp.MatchAgainst` already exists · plan §2.3 line 145, §2.4 line 167

> ```python
> class MatchAgainst(exp.Expression):
>     arg_types = {"this": True, "expression": True}        # MATCH(text) AGAINST (:q)
> ```
> «нужно … `MATCH`, `AGAINST`»

**WRONG — this is wasted work, and the arg names are backwards.**

`exp.MatchAgainst` ships in the **base** parser (`expressions/string.py:89-90`;
`parser.py:1580 FUNCTION_PARSERS["MATCH"]`; impl `parser.py:8434-8463`). Verified with zero custom
code:
```
SELECT id FROM items WHERE MATCH(text) AGAINST (:q)
-> MatchAgainst(this=Placeholder(this=q), expressions=[Column(Identifier(text))])
```
Note the shape: `arg_types = {"this": True, "expressions": True, "modifier": False}` — **`this` is
the search TERM and `expressions` is the COLUMN LIST**, the opposite of the plan's
`{this, expression}`. Defining a second `MatchAgainst` with the plan's signature would shadow
nothing at runtime but would silently generate backwards SQL.

Neither `MATCH` nor `AGAINST` needs a tokenizer entry — both are plain `VAR` and dispatch is
string-based.

**Only correction needed:** the base generator emits `AGAINST(` with **no space**
(`generator.py:3784`), so `AGAINST (:q)` is not text-identical. Override `matchagainst_sql` (§4.9).

---

### C7 — `self.expression(Cls, **kwargs)` no longer exists · plan §2.5, lines 207, 229, 238

> ```python
> return self.expression(LoadTable, this=name, properties=props)
> return self.expression(HybridSearch, expressions=arms, rerank=rerank)
> return self.expression(SearchArm, this=column, op=op, param=param, weight=weight)
> ```

**WRONG.** `parser.py:2187`:
```python
def expression(self, instance: E, token: Token | None = None, comments: list[str] | None = None) -> E:
```
Verified:
```
self.expression(Cls, **kwargs) -> TypeError: Parser.expression() got an unexpected keyword argument 'this'
self.expression(Cls(**kwargs)) -> RT(this=Var(this=x))
```
**CORRECT:** `self.expression(LoadTable(this=name, properties=props))`.
In-tree confirmation: `parsers/risingwave.py:21` `self.expression(exp.WatermarkColumnConstraint(...))`.

---

### C8 — `_parse_properties()` consumes `WITH` itself; the plan's `_parse_load_table` double-handles it · plan §2.5, lines 203-207

> ```python
> def _parse_load_table(self):
>     self._match_text_seq("TABLE")
>     name = self._parse_table_parts()
>     props = self._parse_properties()  # переиспользуем общий парсер WITH (...)
> ```

**Half right.** `_parse_properties()` **does** handle `WITH (...)` — including consuming the `WITH`
keyword, via `PROPERTY_PARSERS["WITH"]` (`parser.py:1413`). Verified:
```
'WITH (replicas=2)'  _parse_properties() -> Properties([Property(Var(replicas), Literal(2))])  remaining=[]
```
But two details are wrong/missing:
1. `self._match_text_seq("TABLE")` is the wrong matcher — `TABLE` is a real `TokenType`, so
   `self._match(TokenType.TABLE)` is correct and cheaper. More importantly, the plan **ignores the
   return value**, so `LOAD DATA INPATH ...` would fall through into `_parse_table_parts` and
   break Hive inheritance (C5).
2. For a **bare** `(...)` bag with no `WITH` — which is what `SEARCH PARAMS (...)` and
   `RERANK RRF(...)` are — `_parse_properties()` returns `None` and consumes nothing. Those need
   `_parse_wrapped_properties()`, which returns a **`list`**, not a `Properties` node.

See §3.7.

---

### C9 — `SearchArm.op` as a raw string is the wrong model · plan §2.3 lines 135-137, §2.6 lines 253-256

> ```python
> class SearchArm(exp.Expression):
>     arg_types = {"this": True, "op": True, "param": True, "weight": False}
> ```
> ```python
> SearchArm: lambda self, e: (f"{self.sql(e, 'this')} {e.args['op']} {self.sql(e, 'param')}" ...)
> ```

**Not "wrong" but architecturally inconsistent and needlessly costly.** The distance operators must
already exist as first-class `FACTOR` entries producing real binary nodes
(`InnerProduct`, `L1Distance`, `CosineDistance`, `exp.Distance`) for the *`ORDER BY`* form of the
grammar. Storing the operator as a bare Python `str` inside `SearchArm` means:
* two independent code paths for the same operator, which will drift;
* the `op` string is invisible to `find_all` / `walk` / `qualify` / lineage;
* the Track-B `ast_to_pymilvus` translator needs a separate string→`metric_type` map instead of one
  node-class→`metric_type` map.

**CORRECT (verified, text-identical round-trip):**
```python
class SearchArm(exp.Expression):
    arg_types = {"this": True, "weight": False}     # `this` IS the binary distance node

def _parse_search_arm(self):
    this = self._parse_assignment()                 # NOT _parse_expression -- P7
    weight = self._parse_number() if self._match_text_seq("WEIGHT") else None
    return self.expression(SearchArm(this=this, weight=weight))
```
```
SearchArm(this=CosineDistance(this=Column(embedding), expression=Placeholder(dv)), weight=Literal(0.7))
```
One operator model, everywhere.

---

### C10 — `SearchParams` is NOT "just an alias for `exp.Properties`" · plan §2.3, line 152

> «`HybridSearch` и `SearchParams` (последний — просто алиас на `exp.Properties`)»

**Won't work.** If you store a bare `exp.Properties` under `Select.args["search_params"]`, the
generator dispatches `properties_sql` (`generator.py:2041`), which renders
`WITH (ef_search=64)` — you would get `... LIMIT 10 WITH (ef_search=64)`, not
`SEARCH PARAMS (ef_search=64)`. There is no per-instance way to change the prefix;
`WITH_PROPERTIES_PREFIX` (`generator.py:558`) is a class-level constant shared with `CREATE TABLE`.

**CORRECT:** a distinct node, so it gets its own `TRANSFORMS` entry:
```python
class SearchParams(exp.Expression):
    arg_types = {"expressions": True}       # expressions: list[exp.Property]

SearchParams: lambda s, e: f"{s.seg('SEARCH PARAMS')} ({s.expressions(e, flat=True)})"
```
(The *contents* are plain `exp.Property` nodes — that part of the plan's intent is right, and it is
why `_parse_wrapped_properties()` is the right parser.)

---

### C11 — the generator hook position is wrong, and there is no "`select_sql` method" to add · plan §2.6, lines 264-265

> «Плюс метод, добавляющий вывод `hybrid_search`/`search_params` в нужном месте общего `select_sql`
> (сразу после `WHERE`/`ORDER BY`, **перед `LIMIT`**) — аналогично тому, как в базовом генераторе
> уже вставлены `QUALIFY`/`WINDOW`.»

**Three errors.**

1. **Wrong method.** `select_sql` (`generator.py:3334`) does not emit modifiers; it delegates to
   `query_modifiers` (`generator.py:3270-3300`), which emits from a **hard-coded ordered list**.
   No shipped dialect overrides `query_modifiers`.
2. **Wrong position, and it contradicts the plan's own grammar.** Plan §2.1 line 76 puts
   `SEARCH PARAMS (...)` **after** `LIMIT 10`, and line 77 puts `HYBRID SEARCH (...) RERANK ...`
   **before** `LIMIT 10` — i.e. the two clauses go in *different* places, not both "before LIMIT".
3. **`QUALIFY`/`WINDOW` are not a usable analogy** — they are hard-coded entries in
   `AFTER_HAVING_MODIFIER_TRANSFORMS` / the `csv(...)` list, not an extension mechanism you can
   append to for arbitrary positions.

**CORRECT (verified, both clauses land exactly where §2.1 specifies, text-identical):**
```python
def query_modifiers(self, expression, *sqls):
    # extra positional sqls are splatted FIRST -> right after FROM, before joins/WHERE
    return super().query_modifiers(expression, *sqls, self.sql(expression, "hybrid"))

def after_limit_modifiers(self, expression):
    return super().after_limit_modifiers(expression) + [self.sql(expression, "search_params")]
```
Each fragment must carry its own leading `self.seg(...)`, because `csv(..., sep="")`. See §4.4.

---

### C12 — `TRANSFORMS` values, not `*_sql` methods (the plan gets this right, but for the wrong reason) · plan §2.6, lines 247-261

The plan's `TRANSFORMS` dict is **correct**, and is in fact the *only* mechanism that works. But
the plan never says why, and elsewhere (§2.6 line 264) reaches for a method. For the record:
`*_sql` methods on a Generator subclass resolve through
`exp.EXPR_CLASSES.get(attr_name[:-4])` (`generator.py:89`), a registry built once from
`sqlglot.expressions`' own members (`expressions/__init__.py:51`). **Third-party classes are never
in it.** See §4.1, P3.

One bug in the plan's `LoadTable` transform (line 250-251):
```python
LoadTable: lambda self, e: f"LOAD TABLE {self.sql(e, 'this')}{self.sql(e, 'properties')}"
```
`self.sql(e, 'properties')` returns `'WITH (replicas=2)'` with **no leading space** →
`LOAD TABLE itemsWITH (replicas=2)`. Correct:
```python
LoadTable: lambda s, e: (f"LOAD TABLE {s.sql(e, 'this')}"
                         + (f" {s.sql(e, 'properties')}" if e.args.get("properties") else ""))
```

---

### C13 — `CREATE INDEX ... USING HNSW` in the plan's word order does NOT parse for free · plan §2.3, lines 119-122

> «А `CREATE INDEX ... USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64)` для
> pgvector в диалекте `postgres` в sqlglot и так парсится … HNSW/IVFFLAT-параметры достаются нам
> почти бесплатно.»

**Half right, and the half that is wrong is the half we need.** The *postgres* word order
(`USING` before the column list) does parse for free, in the base dialect too. But **the MilvusQL
word order specified in the plan's own §2.1 line 72** —
`CREATE INDEX idx_emb ON items (embedding) USING HNSW WITH (...)` — silently degrades to
`exp.Command`:
```
CREATE INDEX i ON t (e) USING HNSW WITH (m=16)   -> exp.Command   (opaque, but round-trips!)
CREATE INDEX i ON t USING HNSW (e) WITH (m=16)   -> exp.Create/exp.Index   (structured)
```
Base `_parse_index_params` (`parser.py:4747-4778`) reads `USING` first.

**CORRECT:** either (a) override `_parse_index_params` + `indexparameters_sql` to swap the order
(§3.9, §4.7 — verified text-identical), or (b) change the MilvusQL spec to pgvector's word order
and get it truly free. See D6.

Related, also unstated in the plan: the **default** dialect injects a spurious ` NULLS LAST` into
index column lists via `ordered_sql` (`generator.py:3149`); set `NULL_ORDERING = "nulls_are_last"`.

---

### C14 — plan §2.7's uncertainty about the placeholder class is resolvable, and the answer is "nothing to do" · plan §2.7, lines 269-272

> «в терминологии sqlglot обычно это `exp.Placeholder`/`exp.Parameter` — точное имя класса стоит
> свериться с версией библиотеки»

**Resolved:** `:name` → **`exp.Placeholder(this='<name>')`**, where `this` is a plain Python `str`.
`exp.Parameter` is what `@name` produces. **No dialect opt-in is required at all** — a bare
`class Milvus(Dialect)` parses and regenerates `:emb` correctly, because
`NAMED_PLACEHOLDER_TOKEN = ":"` is the base default (`generator.py:682`). See §6.1, §6.2.

The plan's design intent ("вектор никогда не превращается в строку") is fully supported.

---

### C15 — plan §2.9's claim about upstreaming is wrong · plan §2.9, lines 288-291

> «Нельзя добавлять новые члены в общий `TokenType` без форка sqlglot … Если проект дозреет до
> апстрима в основной репозиторий sqlglot (как community dialect, по образцу RisingWave/Materialize),
> это ограничение снимается — тогда можно расширять `TokenType` напрямую в общем файле.»

**The first sentence is right; the second is a red herring, and the framing understates a real
constraint.**

* Upstreaming is **not** a plausible path: `METADATA:674` explicitly classifies third-party dialects
  as *Plugin Dialects* maintained in external repositories, and states «The SQLGlot team does not
  provide support or maintenance for plugin dialects». RisingWave/Materialize are in-tree for
  historical reasons, not because there is an open door.
* The **real** constraint the plan omits is much sharper: the 443-member `TokenType` enum is not
  merely un-extendable, it is un-extendable **in a way that fails silently** (P4) — and there are
  only **seven** provably-inert members to recycle (§2.5 Tier A).
* The plan also omits the actual blocker for a plugin dialect: `sqlglot[c]` (§1.3, P1). That belongs
  in §2.9's "honest limitations" list far more than the upstreaming remark does.

---

### C16 — plan §2.10's version floor is too low · plan §2.10, line 299

> «механизм plugin-диалектов появился в sqlglot с v28.6.0»

**True but irrelevant, and following it will break the build.** `PLUGIN_GROUP_NAME` does first
appear in 28.6.0 (verified in throwaway venvs; absent in 28.5.0). But the APIs `sqlglot-milvus`
must import — `sqlglot.parsers.*`, `sqlglot.generators.*`, `exp.DType`, the instance-based
`Parser.expression()`, the `exp.Expression`/`exp.Expr` split — do not exist below 30.x, and
`sqlglot.generators` only appeared between 30.0.0 and 30.5.0.

**CORRECT:** `dependencies = ["sqlglot>=30.16.0,<31"]`. See §1.4.

---

### C17 — plan §2.8's round-trip test strategy is insufficient · plan §2.8, lines 278-284

> «**Round-trip**: `sql → parse_one(read="milvus") → .sql(dialect="milvus") == sql` … на ~50–100
> golden-примерах»
> «**Негативные тесты**: … должен либо кидать `ParseError` …, либо … падать в `exp.Command`,
> а не тихо портить дерево.»

**The round-trip assertion is a false safety signal, and the `exp.Command` fallback is named as an
acceptable outcome when it is precisely the failure mode to guard against.**

`exp.Command` round-trips **byte-identically** by construction — it stores the raw source text
(`Command(this='CREATE', expression='INDEX i ON t (e) USING HNSW WITH (m=16)')`) and is completely
opaque (`find_all(exp.Column) == []`). So all three of `CREATE INDEX ... USING HNSW`,
`LOAD TABLE ...`, `ALTER TABLE ... ADD FIELD ...` **pass a round-trip test today, in the base
dialect, with zero lines of our code written.** See P8.

**CORRECT test contract — every golden case must assert all four:**
```python
ast = sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)
assert not isinstance(ast, exp.Command)          # 1. structured, not a raw blob
assert isinstance(ast, EXPECTED_NODE_TYPE)       # 2. the right node
assert ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE) == sql   # 3. text-identical
assert repr(sqlglot.parse_one(ast.sql(dialect="milvus"), read="milvus")) == repr(ast)  # 4. AST-stable
```
Plus: never pass `ErrorLevel.IGNORE`/`WARN` (P9), and remember a *single* trailing garbage word is
absorbed as an alias even at `RAISE`.

---

### C18 — minor: the plan's file layout omits the `sqlglotc` guard and the entry-point TOML quoting

Plan §2.2 (lines 91-107) and §2.10 (lines 301-305) are otherwise fine. Two additions:
* `src/sqlglot_milvus/__init__.py` must contain the `SQLGLOTC_INSTALLED` guard (§1.3).
* The TOML key **must be quoted**: `[project.entry-points."sqlglot.dialects"]` — the dot is
  otherwise a table separator. The plan's snippet already quotes it correctly.
* Entry-point name must be lowercase and equal to `ClassName.lower()` (§5.6, §5.7).

---

### C19 — what the plan gets RIGHT (do not "fix" these)

For balance, these plan claims were checked and are **correct**:

| plan claim | verdict |
|---|---|
| §2.3: `CREATE TABLE` is a stock `exp.Create`, no custom parsing needed | ✅ verified, text-identical round-trip incl. `VECTOR(768)`, `AUTO_INCREMENT`, `WITH (...)` |
| §2.3: `WITH (...)` reuses `exp.Properties`/`exp.Property` | ✅ verified |
| §2.3: hybrid/search-params live as **keys in `Select.args`**, not as wrappers | ✅ correct and necessary (`_parse_query_modifiers` does `this.set(key, ...)`) — but you MUST also declare `exp.Select.arg_types[...] = False` (§6.6) |
| §2.3: `LoadTable`/`ReleaseTable`/`HybridSearch`/`Rerank` as `exp.Expression` | ✅ correct base class |
| §2.1: calling the object `TABLE` rather than `COLLECTION` saves parsing work | ✅ confirmed — `TABLE` is a base TokenType and `_parse_table_parts` works unmodified |
| §2.1: distance operators inherited from pgvector 1:1 | ✅ `<->` is free; the other three are cheap (§2.5) |
| §2.6: `TRANSFORMS` as the generator mechanism | ✅ the only mechanism that works (C12) |
| §2.10: entry point group `sqlglot.dialects`, value `module:Class` | ✅ verified by a real install/uninstall cycle (§5.5) |
| §2.7: vectors ride as bind params, only structure passes through sqlglot | ✅ fully supported (§6.1) |

---

## 9. Reconciliation log — where the recon reports contradicted each other

Each was re-run on `/home/neko/startup/milvus/sqlglot-milvus/.venv/bin/python`.

| # | Disagreement | Verified answer |
|---|---|---|
| R1 | **`_parse_properties()` and `WITH`.** Parser recon: "consumes `WITH` itself". My first re-run: "returns `None`, consumes nothing". | **Parser recon is RIGHT.** My first driver failed to set `_tokens_size`, so `_curr` was `SENTINEL_NONE` from token 0 and every sub-parser saw EOF. Correct driver protocol in §3.7. `_parse_properties()` on `WITH (replicas=2)` returns a full `Properties` and consumes everything. |
| R2 | **`QUERY_MODIFIER_TOKENS`.** Parser recon: "`GROUP BY category` breaks — active trap". Pitfalls recon: "could not construct a failing case — latent". | **Both right, different configs.** Full 3×2 matrix in §3.5. It breaks *only* when the modifier is keyed on `TokenType.VAR` **and** `QUERY_MODIFIER_TOKENS` is recomputed locally. With a dedicated TokenType (our design) recomputing is safe and correct. |
| R3 | **`TokenType.X = ...` result type.** Pitfalls: "silently becomes a `str`". Parser recon: "yields a plain `int`". | **Both right.** The assigned value keeps its own type; neither becomes an enum member. `TokenType.A = "s"` → `str`; `TokenType.B = 9001` → `int`; `TokenType.A in list(TokenType)` → `False` for both. §2.4. |
| R4 | **Host TokenTypes for `<#>`/`<+>`.** Tokenizer recon: `SPACE`/`BREAK` (zero package refs). Parser recon: `HASH_ARROW`/`DARROW` + strip `COLUMN_OPERATORS`. | **Both work; `SPACE`/`BREAK` is strictly safer.** Verified: with `HASH_ARROW`/`DARROW` and `COLUMN_OPERATORS` untouched you silently get `JSONBExtract`/`JSONExtractScalar`. Tier-A hosts need no surgery. §2.5, D1. |
| R5 | **The "free TokenType" list.** Pitfalls listed `SEMANTIC_VIEW, STREAMLIT, SINK, SOURCE, NAMESPACE, OPERATOR, EXPORT, DETACH, ATTACH, POINT, RING, TAG, SUMMARIZE, PUT, GET` as "verified-available spares". Tokenizer recon listed 46 members "free" by reflection over Parser dicts. | **Both lists are over-broad.** Reflection over dict/set class attributes is *necessary but not sufficient* — the base parser also does inline `self._match(TokenType.X)` (e.g. `SUMMARIZE` at `parser.py:4138`, `OPERATOR` at `:10417`, `ONLY` at `:5050`, `SEPARATOR` at `:10388`). Only a package-wide grep is conclusive. **Exactly seven members have zero references: `SPACE, BREAK, LANGUAGE, ORDERED, PROPERTIES, SOUNDS_LIKE, COLUMN_DEF`.** All of pitfalls' "spares" are additionally in `ID_VAR_TOKENS`/`TABLE_ALIAS_TOKENS`. §2.5. |
| R6 | **`HYBRID SEARCH` clause position in output.** Pitfalls: "the generator always emits it *after* `LIMIT`; round-trip is not text-idempotent; the only clean hook is `after_limit_modifiers`". Generator recon: "extra positional `*sqls` land right after `FROM`". | **Generator recon is RIGHT.** `super().query_modifiers(expression, *sqls, <frag>)` splats extras first, i.e. immediately after `from_` and before joins/`WHERE` — exactly the spec position. Verified text-identical round-trip for the full hybrid statement. §4.4, C11. |
| R7 | **Reserving `HYBRID`/`SEARCH`.** Pitfalls: "you must do `ID_VAR_TOKENS - {TT_HYBRID, TT_SEARCH, TT_RERANK}`; those words become unusable as bare table aliases — an accepted trade". | **Unnecessary with multi-word keywords.** Registering the *pair* `"HYBRID SEARCH"` on a Tier-A TokenType solves the alias-swallow problem while leaving the bare words fully usable. Verified: `SELECT hybrid, search, rerank, weight, params FROM t` and `SELECT id FROM items hybrid` both work in the final prototype. §2.6, P5. |
| R8 | **Custom `exp.Property` subclass failure mode.** Generator recon: `KeyError`. My isolated probe: `'None=a'`. | **Both, depending on path.** Generated *standalone*, it hits the `isinstance(e, exp.Property)` fallback → `property_sql` → `'None=a'` + an `unsupported` warning. Generated *inside a `Properties` bag on a `Create`*, `locate_properties` indexes `PROPERTIES_LOCATION[p.__class__]` → `KeyError`. §4.7, P16. |
| R9 | **`RRF(k=60)` representation.** Pitfalls: "use `=>` (`exp.Kwarg`) — it is the only form yielding a named-argument node". Generator recon: "none of `PropertyEQ`/`Kwarg`/`EQ` gives bare `k=60`; use `exp.Property`". | **Generator recon is RIGHT and preserves the spec spelling.** `exp.Property` renders `k=60` exactly, has no phantom column, and is produced for free by `_parse_wrapped_properties()`. No need to change the spec to `k => 60`. §7 P14. |
| R10 | **`text` as an identifier.** Tokenizer recon: "`text` is a reserved datatype token — `SELECT text, file FROM t` gives `TEXT`/`FILE`, not `VAR`; your `MATCH(text)` example will hit this." Pitfalls recon: "`text` is NOT a problem; all 31 identifiers parse as `exp.Column`". | **Pitfalls recon is RIGHT.** `text` *does* tokenize as `TokenType.TEXT`, but `TEXT ∈ ID_VAR_TOKENS`, so it parses as a normal `exp.Column` everywhere. Verified: `MATCH(text) AGAINST (:q)`, `BM25_SCORE(text, :q)`, `CREATE TABLE t (text VARCHAR(9))` all correct. The tokenizer recon confused token type with usability. |
| R11 | **`sqlglot[c]` impact.** Tokenizer recon called it "a project blocker"; the dialect-plugin recon only cited the README. | **Confirmed as a hard blocker by direct execution** in an isolated `sqlglot[c]==30.16.0` venv. Full matrix in §1.3. The class statement succeeds; instantiation raises `TypeError: interpreted classes cannot inherit from compiled`. |
| R12 | **Parser recon's `_parse_milvus_release` shape** (retreat + `super()._parse_statement()`). | **That pattern infinite-recurses** for `RELEASE items` / bare `RELEASE`. Neither recon tested the non-`TABLE` branch. Two correct forms in §3.3, P12. |

---

## 10. DECISIONS REQUIRED

Design choices the pitfalls **force** on us. Each has a concrete recommendation; all
recommendations are implemented together in `docs/reference_prototype.py`, which round-trips
16/16 spec statements text-identically.

---

### D1 — Which `TokenType` members host `<#>` and `<+>`?

**Forced by:** P4 (`TokenType` un-extendable), §2.5 (reuse breaks semantics).

| option | cost |
|---|---|
| (a) `HASH_ARROW` / `DARROW` | must also strip both from `COLUMN_OPERATORS`; loses JSON `->>`/`#>` (fine for us); token names actively lie in debug output; a future sqlglot release could add behaviour |
| (b) **Tier-A: `SPACE` / `BREAK`** | zero package-wide references, zero surgery, zero collision risk; token names are meaningless but at least neutral |

> **RECOMMENDATION: (b).** `TT_IP = TokenType.SPACE`, `TT_L1 = TokenType.BREAK`.
> Define module-level aliases (`TT_IP`, `TT_L1`) so the recycling is documented in one place, and
> add a unit test asserting `len(list(TokenType)) == 443` so a sqlglot upgrade that renumbers or
> repurposes them fails loudly.

---

### D2 — How are `HYBRID SEARCH`, `SEARCH PARAMS`, `RELEASE` lexed?

**Forced by:** P5 (alias swallow), P11 (`GROUP BY` breakage on `VAR` keys), §6.3 (`:name`
reservation).

> **RECOMMENDATION:**
> * `"HYBRID SEARCH"` → `TokenType.LANGUAGE`, `"SEARCH PARAMS"` → `TokenType.PROPERTIES` — as
>   **multi-word** keywords. The bare words `HYBRID`, `SEARCH`, `PARAMS` stay `VAR` and remain
>   usable as identifiers, aliases and `:placeholders`. **No `ID_VAR_TOKENS` surgery.**
> * `"RELEASE"` → `TokenType.SOUNDS_LIKE` (single word, so it *is* reserved) **plus**
>   `ID_VAR_TOKENS |= {TT_RELEASE}`, `TABLE_ALIAS_TOKENS |= {TT_RELEASE}`,
>   `COLON_PLACEHOLDER_TOKENS = ID_VAR_TOKENS` to un-reserve it. `STATEMENT_PARSERS` is consulted
>   before any expression parsing, so `RELEASE TABLE x` still wins at statement position.
> * `RERANK` and `WEIGHT` stay plain `VAR`, matched with `_match_text_seq` inside the
>   `HYBRID SEARCH` parser.
>
> **Accepted losses (add negative tests):** a table literally named `search` aliased `params`
> (`FROM search params`) is destroyed; quoting (`FROM "search" params`) rescues it. Same for
> `FROM hybrid search`.

---

### D3 — Do we rebind `<=>` to cosine distance at all?

**Forced by:** P6. `<=>` already means MySQL null-safe equality in sqlglot, in the *base* tokenizer.
Two SQL dialects will disagree about the same six characters, silently, with no error at any step.

| option | consequence |
|---|---|
| (a) rebind `<=>` → cosine (pgvector parity, plan §2.1 line 87) | text-level collision with MySQL `<=>` is **unfixable**; inbound MySQL SQL silently becomes a distance query |
| (b) drop `<=>`, spell cosine differently (e.g. `<~>`) | breaks pgvector parity, which is the plan's stated reason for the operator set; hurts the `postgres → milvus` migration story |
| (c) rebind, and defend | pgvector parity kept; collision contained by explicit guards |

> **RECOMMENDATION: (c) — rebind, with all four guards mandatory:**
> 1. `FACTOR[TokenType.NULLSAFE_EQ] = CosineDist` **and**
>    `EQUALITY = {k: v for k, v in Parser.EQUALITY.items() if k is not TokenType.NULLSAFE_EQ}`.
>    Without the removal, `EQUALITY` wins and you silently get `NullSafeEQ`. This also makes all
>    four operators share `FACTOR` precedence (verified: `a <=> b + c` → `Add(Cos(a,b), c)`).
> 2. **`class Milvus(Dialect)`, NOT `class Milvus(MySQL)`.** MySQL is the only dialect that *emits*
>    `<=>`; inheriting from it creates a silent round-trip corruption channel. It also drags in
>    MySQL's `values`/`key` identifier failures (P18).
> 3. `TRANSFORMS[exp.NullSafeEQ]` in the Milvus generator calls
>    `self.unsupported("NULL-safe equality (<=>) has no MilvusQL equivalent")` — so
>    `mysql → milvus` transpilation fails loudly instead of changing meaning.
> 4. A test asserting `parse_one("SELECT a <=> b", read="milvus")` yields `CosineDist`, and a test
>    asserting `transpile("SELECT * FROM t WHERE a <=> b", read="mysql", write="milvus",
>    unsupported_level=RAISE)` raises.
>
> Document the collision prominently in the README migration section.

---

### D4 — Canonical clause order in MilvusQL, and where the generator emits it

**Forced by:** §3.4 (the parser's modifier loop is completely unordered), C11.

The spec (plan §2.1) places `HYBRID SEARCH ... RERANK ...` immediately after `FROM`, and
`SEARCH PARAMS (...)` after `LIMIT`. **Both positions are achievable and text-identical**
(§4.4) — the pitfalls recon's claim that `HYBRID SEARCH` must move after `LIMIT` is wrong (R6).

> **RECOMMENDATION: keep the spec order exactly as written.**
> ```python
> def query_modifiers(self, expression, *sqls):
>     return super().query_modifiers(expression, *sqls, self.sql(expression, "hybrid"))
> def after_limit_modifiers(self, expression):
>     return super().after_limit_modifiers(expression) + [self.sql(expression, "search_params")]
> ```
> Each fragment must emit its own leading `self.seg(...)` (because `csv(..., sep="")`).
> Do **not** override `query_modifiers` wholesale — the thin `super()` delegation survives upstream
> reordering.

---

### D5 — Is the grammar strict or permissive about clause position?

**Forced by:** §3.4. The parser physically cannot enforce order — `_parse_query_modifiers` is a
bare `while True:` with no positional state. So today
`SELECT id FROM items SEARCH PARAMS (ef=64) LIMIT 10` **parses**, and regenerates as
`... LIMIT 10 SEARCH PARAMS (ef=64)` — accepted input, different output text.

| option | cost |
|---|---|
| (a) permissive: accept any order, normalize on output | round-trip is not text-idempotent for non-canonical input; users get silently-reordered SQL; hides typos |
| (b) strict: reject non-canonical order | needs a post-parse validation pass (a `Parser.parse` override or a `_parse_select` wrapper comparing token indices) |

> **RECOMMENDATION: (b) strict, for MVP.** MilvusQL is a language *we* define and *we* generate
> (Track B's `MilvusSQLCompiler` is the primary producer); there is no legacy corpus to be lenient
> toward. A permissive grammar makes the round-trip test suite (plan §2.8) meaningless for these
> two clauses.
>
> Cheapest implementation: record the token index at which each modifier fired, and validate the
> sequence in an overridden `Parser.parse`/`_parse_statement` wrapper. Raise `ParseError` with the
> offending token. Explicitly test both directions:
> ```
> SELECT ... LIMIT 10 SEARCH PARAMS (a=1)   -> OK
> SELECT ... SEARCH PARAMS (a=1) LIMIT 10   -> ParseError: SEARCH PARAMS must follow LIMIT
> SELECT ... FROM t HYBRID SEARCH (...) LIMIT 10  -> OK
> SELECT ... FROM t LIMIT 10 HYBRID SEARCH (...)  -> ParseError
> ```
> If the effort is judged too high for MVP, fall back to (a) **and** delete the round-trip claim for
> non-canonical input from the test plan — do not leave it ambiguous.

---

### D6 — `CREATE INDEX` word order: MilvusQL's or pgvector's?

**Forced by:** P8 / C13. The spec's `... ON items (embedding) USING HNSW WITH (...)` degrades to
`exp.Command` in every shipped dialect; pgvector's `... ON items USING hnsw (embedding) WITH (...)`
parses structurally for free.

| option | cost |
|---|---|
| (a) keep MilvusQL order | ~15 lines: override `_parse_index_params` + `indexparameters_sql` (both verified working, text-identical) |
| (b) adopt pgvector order | zero code; better `postgres → milvus` transpile fidelity; but the spec and all user-facing docs change, and `(cols) USING <m>` reads more naturally to SQL users |

> **RECOMMENDATION: (a), keep the spec order.** The cost is genuinely ~15 lines and it is already
> proven. But **accept pgvector's order too** — the base `_parse_index_params` logic can be kept as
> a fallback when the token after the table name is `USING` rather than `(`. That gives free
> `postgres → milvus` index transpilation, which plan §2.8 lists as a headline deliverable.
> Also set `NULL_ORDERING = "nulls_are_last"` on the dialect to suppress the spurious ` NULLS LAST`
> in index column lists.

---

### D7 — `RERANK RRF(k=60)`: which node represents the named argument?

**Forced by:** P14. The generic function parser turns `k=60` into `EQ(Column(k), Literal(60))` — a
phantom column reference that `qualify`/lineage will try to resolve against the schema.

> **RECOMMENDATION: a dedicated `Rerank` node whose params are plain `exp.Property`.**
> ```python
> class Rerank(exp.Expression):
>     arg_types = {"this": True, "expressions": False}   # this=Var(RRF), expressions=[Property]
>
> if self._match_text_seq("RERANK"):
>     kind = self._parse_var(any_token=True)
>     params = (self._parse_wrapped_properties()
>               if self._match(TokenType.L_PAREN, advance=False) else None)
>     rerank = self.expression(Rerank(this=kind, expressions=params))
> ```
> Renders as `RERANK RRF(k=60)` — **the spec spelling exactly**, via
> `self.expressions(e, flat=True)`. No phantom column, no spec change to `k => 60`.
> Verified text-identical.

---

### D8 — `SearchArm`: does it own the operator, or wrap a binary node?

**Forced by:** C9. The plan stores the operator as a raw string in `SearchArm.args["op"]`, but the
same operators must already exist as `FACTOR` binary nodes for the `ORDER BY` form.

> **RECOMMENDATION: wrap the binary node.** `arg_types = {"this": True, "weight": False}` where
> `this` is `CosineDist` / `InnerProduct` / `L1Distance` / `exp.Distance`. One operator model
> everywhere; the Track-B translator needs a single `node class → metric_type` map; `find_all` /
> `qualify` / lineage see the columns and placeholders.
> Parse the arm with **`_parse_assignment()`**, never `_parse_expression()` (P7).

---

### D9 — `SearchParams`: dedicated node or `exp.Properties`?

**Forced by:** C10. A bare `exp.Properties` renders `WITH (...)`, not `SEARCH PARAMS (...)`, and
the prefix is a class-level constant shared with `CREATE TABLE`.

> **RECOMMENDATION: dedicated `class SearchParams(exp.Expression)` with
> `arg_types = {"expressions": True}`,** holding plain `exp.Property` children produced by
> `_parse_wrapped_properties()`. Same for `HybridSearch`. Both get `TRANSFORMS` entries.
> **And declare the `Select` args at import time** — `exp.Select.arg_types["search_params"] = False`
> and `["hybrid"] = False` — or `validate_expression` raises `TypeError` under pytest (P-§6.6).

---

### D10 — Dialect base class

**Forced by:** D3 guard 2, P18 (`values`/`key` fail in mysql), §5.2 (shared `parser_class` hazard).

| option | consequence |
|---|---|
| `class Milvus(Postgres)` | inherits `$`-heredocs, postgres `@@`/`<@`/`@>` operators, `%(name)s` placeholder output — all wrong for us; and `%(name)s` output breaks Track B's `paramstyle = "named"` |
| `class Milvus(MySQL)` | 🔴 `<=>` corruption channel (D3), `values`/`key` identifier failures |
| **`class Milvus(Dialect)`** | clean slate; `:name` placeholders for free; `NullSafeEQ` renders as `IS NOT DISTINCT FROM` |

> **RECOMMENDATION: `class Milvus(Dialect)`** with nested `class Tokenizer(tokens.Tokenizer)`,
> `class Parser(BaseParser)`, `class Generator(generator.Generator)`.
> Note `Dialect.parser_class` defaults to **`sqlglot.parsers.base.BaseParser`**, not
> `sqlglot.parser.Parser` — subclass `BaseParser` for parity with built-ins.
> **Never** use `tokenizer_class = X` / `parser_class = X` assignment for the Tokenizer (§2.8).

---

### D11 — Packaging, version pin, and the `sqlglot[c]` guard

**Forced by:** P1, C16.

> **RECOMMENDATION:**
> ```toml
> [project]
> requires-python = ">=3.9"
> dependencies = ["sqlglot>=30.16.0,<31"]      # NEVER sqlglot[c] / sqlglot[rs]
>
> [project.entry-points."sqlglot.dialects"]
> milvus = "sqlglot_milvus.dialect:Milvus"     # lowercase, == ClassName.lower()
> ```
> * Import-time guard on `sqlglot.tokens.SQLGLOTC_INSTALLED` in
>   `src/sqlglot_milvus/__init__.py` (§1.3), raising `ImportError` with remediation instructions.
> * CI matrix job: `pip install sqlglot[c]` → assert the guard raises with a readable message.
> * CI job: `pip install 'sqlglot==30.16.0'` → full test suite.
> * A test asserting `type(Dialect.get_or_raise("milvus")) is Milvus` (guards P17, duplicate
>   registration).
> * A test asserting `len(list(TokenType)) == 443` (guards D1's recycled hosts against an upstream
>   renumbering).

---

### D12 — Test contract

**Forced by:** P8 (`exp.Command` round-trips perfectly), P9 (garbage absorption), P10 (silent arg
drop), §6.6 (`UNITTEST` strictness), §4.10 (`ValueError` is not a `SqlglotError`).

> **RECOMMENDATION: every golden case asserts all four properties**, and the suite pins
> `ErrorLevel.RAISE` / `unsupported_level=ErrorLevel.RAISE` everywhere:
> ```python
> def assert_milvus(sql, node_type):
>     ast = sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)
>     assert not isinstance(ast, exp.Command), "degraded to opaque Command"
>     assert isinstance(ast, node_type)
>     out = ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
>     assert out == sql, f"text drift: {out!r}"
>     assert repr(sqlglot.parse_one(out, read="milvus")) == repr(ast), "AST not stable"
> ```
> Plus dedicated suites for:
> * **tokenizer-level** tests for all four operators and both multi-word keywords (a wrong split
>   produces an error naming a totally unrelated node, e.g. `JSONBExtract`);
> * **identifier survival**: `SELECT release, search, hybrid, params, weight, field, rerank, text
>   FROM t` and `SELECT id FROM items hybrid`;
> * **negative order** tests (D5) and **duplicate-clause** tests
>   (`ParseError: Found multiple 'SEARCH PARAMS' clauses`);
> * **`<=>` semantics**: asserts `CosineDist`, and `mysql → milvus` raises (D3);
> * **`exp.Command` guard** on every DDL statement;
> * **`sqlglot[c]` guard** test.
>
> Do **not** rely on `except SqlglotError` — a missing `TRANSFORMS` entry raises a plain
> `ValueError` (§4.10).

---

### Summary of forced decisions

| # | Decision | Recommendation |
|---|---|---|
| D1 | operator token hosts | `SPACE` (`<#>`) / `BREAK` (`<+>`) — Tier A |
| D2 | keyword lexing | multi-word `HYBRID SEARCH` / `SEARCH PARAMS`; `RELEASE` single-word + un-reserved |
| D3 | `<=>` rebinding | rebind, with 4 mandatory guards; never inherit from MySQL |
| D4 | clause emission position | keep spec order via `query_modifiers(*sqls, hybrid)` + `after_limit_modifiers` |
| D5 | strict vs permissive order | **strict**, enforced post-parse |
| D6 | `CREATE INDEX` word order | keep MilvusQL order, also accept pgvector's |
| D7 | `RERANK RRF(k=60)` | dedicated `Rerank` node + `exp.Property` params |
| D8 | `SearchArm` shape | wraps the binary distance node; parse with `_parse_assignment` |
| D9 | `SearchParams` shape | dedicated node; declare `Select.arg_types` at import |
| D10 | dialect base class | bare `Dialect`; `Parser(BaseParser)` |
| D11 | packaging | `sqlglot>=30.16.0,<31`, no extras, import-time `sqlglotc` guard |
| D12 | test contract | 4-way assertion, `ErrorLevel.RAISE` pinned everywhere |

---

## Appendix — reproduction assets

| file | contents |
|---|---|
| `/home/neko/startup/milvus/sqlglot-milvus/docs/reference_prototype.py` | **complete working Milvus dialect**, 16/16 text-identical round-trip |
| `.../scratchpad/final_verify/v1_tokentypes.py` | package-wide free-TokenType scan (§2.5) |
| `.../scratchpad/final_verify/v2_contradictions.py` | R2, R3, trie staleness (§2.3, §3.5) |
| `.../scratchpad/final_verify/v3_more.py` | alias swallow, `COLUMN_OPERATORS`, precedence, Dialect defaults, traits, dispatch (§2.5, §3.6, §4.1, §5.3, §6.4) |
| `.../scratchpad/final_verify/v4_props.py` | R1 `_parse_properties`, `PROPERTIES_LOCATION`, modifier positions, `expressions(flat=)`, free wins (§3.7, §4.4-4.7, §6.9) |
| `.../scratchpad/final_verify/test_unittest_flag.py` | `UNITTEST` semantics under pytest (§6.6) |
| `.../scratchpad/final_verify/v5_proto.py` | first prototype (12/15) — shows the three residual failures |
| `.../scratchpad/final_verify/v6_fixes.py` | `RELEASE` un-reservation, `AGAINST` spacing, `_parse_index`, distance nodes, entry-point resolution, `Parser.expression` signature |
| `.../scratchpad/final_verify/v7_compiled.py` | 🔴 `sqlglot[c]` blocker, run in both venvs (§1.3) |
| `.../scratchpad/final_verify/v8_final.py` | final prototype (16/16) |
| `.../scratchpad/final_verify/v9_last.py` | `Command` opacity, error levels, `COMMANDS` swallow, `WEIGHT` trap, `RRF` forms, multi-word hazards, `:name` limits, `unsupported_level` |
| `.../scratchpad/final_verify/v10_recursion.py` | P12 recursion + correct fixes, `ErrorLevel` matrix |
| `.../scratchpad/cvenv/bin/python` | isolated `sqlglot[c]==30.16.0` venv (project venv untouched) |

Scratchpad root:
`/tmp/claude-1000/-home-neko-startup-milvus/c80804a5-f327-40c3-9a2b-7fdedb8aec26/scratchpad/`
