"""Packaging and wiring guarantees.

Nothing here parses a whole statement. These tests cover the seams where this package meets
sqlglot and the installed distribution metadata -- and every one of those seams has a failure
mode that is *silent* where it happens and only surfaces much later as something baffling:

* two `Milvus` classes registered under one name  -> "my transform never fires"
* a renumbered `TokenType` enum                   -> `<#>` silently lexes as something else
* the `exp.Select.arg_types` patch regressing     -> `TypeError` from deep inside sqlglot
* a renamed entry point                           -> `ValueError: Unknown dialect 'milvus'`
* `sqlglot[c]` installed                          -> `TypeError: interpreted classes cannot
                                                     inherit from compiled` on the first parse
"""

from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys
from importlib.metadata import entry_points, version

import pytest
import sqlglot
import sqlglot.tokens
from sqlglot import exp, generator
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.parsers.base import BaseParser
from sqlglot.tokens import TokenType

import sqlglot_milvus
from sqlglot_milvus import _tokens, expressions
from sqlglot_milvus.dialect import Milvus
from sqlglot_milvus.expressions import HYBRID_ARG, SEARCH_PARAMS_ARG

#: The dialect name, spelled the way `read=` / `write=` must spell it. `get_or_raise` does not
#: lowercase, and neither does the entry-point loader.
DIALECT = "milvus"

#: The entry-point group sqlglot scans (`dialects/dialect.py:78`).
PLUGIN_GROUP = "sqlglot.dialects"


# ------------------------------------------------------------------------------------------
# Cold-interpreter helper
# ------------------------------------------------------------------------------------------


def _run_cold(script: str, *args: str) -> dict:
    """Run `script` in a fresh interpreter and return the JSON dict it prints on its last line.

    Two things this file has to prove are statements about a *cold* process: that the entry
    point resolves with no prior ``import sqlglot_milvus``, and that the ``sqlglot[c]`` guard
    fires during import. Neither can be shown in-process. `importlib.reload` would come close,
    but re-executing ``dialect.py`` re-runs ``class Milvus(Dialect)``, and the `_Dialect`
    metaclass then silently rebinds the "milvus" key to the *new* class object -- corrupting
    registration for every later test in the session (ground truth §5.6, gotcha 7).
    """
    proc = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"child interpreter exited {proc.returncode}; the test below can only be trusted if the "
        f"child itself ran cleanly.\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------------
# 1. Entry-point registration
# ------------------------------------------------------------------------------------------


def test_dialect_resolves_to_exactly_one_class_object() -> None:
    # `_Dialect.__eq__` makes `Milvus == "milvus"` true by *name*, so equality proves nothing
    # here: a duplicate registration (stale reload, or the entry point loading a second copy of
    # the module) yields two distinct classes that still compare equal, while transforms
    # registered on one of them never fire. Identity is the only assertion that catches it.
    assert Dialect.get(DIALECT) is Milvus
    assert sqlglot_milvus.Milvus is Milvus, (
        "__init__ must re-export the class, not a copy"
    )

    # get_or_raise returns an *instance*, not the class (ground truth §5.7).
    assert type(Dialect.get_or_raise(DIALECT)) is Milvus


def test_nested_parser_and_generator_are_not_shared_with_the_base_dialect() -> (
    None
):
    # Omitting the nested `class Parser` / `class Generator` sets parser_class/generator_class to
    # the *parent's own class object* -- no subclass is synthesized -- and every mutation we make
    # would then land on sqlglot's shared BaseParser/Generator process-wide (§5.2).
    assert Milvus.parser_class is not BaseParser
    assert issubclass(Milvus.parser_class, BaseParser)
    assert Milvus.generator_class is not generator.Generator
    assert issubclass(Milvus.generator_class, generator.Generator)


_COLD_ENTRY_POINT_SCRIPT = """
import json, sys

import sqlglot
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ErrorLevel

result = {"preimported": "sqlglot_milvus" in sys.modules}

sql = (
    "SELECT id, category FROM items WHERE category = :cat "
    "ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)"
)
ast = sqlglot.parse_one(sql, read="milvus", error_level=ErrorLevel.RAISE)
result["node_type"] = type(ast).__name__
result["is_command"] = isinstance(ast, exp.Command)
result["columns"] = len(list(ast.find_all(exp.Column)))
result["out"] = ast.sql(dialect="milvus", unsupported_level=ErrorLevel.RAISE)
result["loaded_lazily"] = "sqlglot_milvus" in sys.modules

import sqlglot_milvus

result["same_class"] = Dialect.get("milvus") is sqlglot_milvus.Milvus
result["registered_module"] = Dialect.get("milvus").__module__
print(json.dumps(result))
"""


def test_entry_point_resolves_without_importing_the_package_first() -> None:
    # The interesting case is the one this session can never reproduce: conftest.py imports
    # sqlglot_milvus, so in-process the metaclass has already registered "milvus" and the
    # entry-point loader is never even consulted.
    result = _run_cold(_COLD_ENTRY_POINT_SCRIPT)

    assert result["preimported"] is False
    assert result["loaded_lazily"] is True, (
        "read='milvus' did not pull the package in"
    )
    assert result["same_class"] is True, (
        "entry point loaded a different class object"
    )
    assert result["registered_module"] == "sqlglot_milvus.dialect"

    # A dialect that failed to load does not error: sqlglot degrades the statement to exp.Command,
    # which stores the raw text and therefore round-trips byte-identically while being opaque.
    assert result["is_command"] is False
    assert result["node_type"] == "Select"
    assert result["columns"] > 0
    assert result["out"] == (
        "SELECT id, category FROM items WHERE category = :cat "
        "ORDER BY embedding <-> :q LIMIT 10 SEARCH PARAMS (ef_search=64)"
    )


# ------------------------------------------------------------------------------------------
# 2. Recycled TokenType members
# ------------------------------------------------------------------------------------------

#: alias name -> the name of the member it is pinned to. Deliberately spelled as *strings* and
#: compared against what `_tokens` actually holds, so that repointing an alias has to be done
#: twice, on purpose -- and so the tests below can be driven by the live values instead of by
#: this table (a test that only reads its own literal is a test of nothing).
PINNED_HOSTS = {
    "TT_INNER_PRODUCT": "SPACE",
    "TT_L1": "BREAK",
    "TT_HYBRID_SEARCH": "LANGUAGE",
    "TT_SEARCH_PARAMS": "PROPERTIES",
    "TT_RELEASE": "SOUNDS_LIKE",
}

#: What `_tokens` currently binds. Every assertion below reads this, not `PINNED_HOSTS`.
LIVE_HOSTS = {alias: getattr(_tokens, alias) for alias in PINNED_HOSTS}


def test_token_type_member_count_is_pinned() -> None:
    actual = len(list(TokenType))
    assert actual == _tokens.EXPECTED_TOKEN_TYPE_COUNT, (
        f"sqlglot {sqlglot.__version__} exposes {actual} TokenType members; this dialect was "
        f"written against {_tokens.EXPECTED_TOKEN_TYPE_COUNT}. TokenType is a plain Enum and "
        "cannot be extended, so MilvusQL recycles members that sqlglot itself never references "
        "(_tokens.py). A different member count means the enum was edited upstream: before "
        "bumping EXPECTED_TOKEN_TYPE_COUNT, re-verify by grep that every recycled host "
        f"({', '.join(sorted(PINNED_HOSTS.values()))}) is still unused anywhere in the sqlglot "
        "package -- otherwise <#>, <+>, HYBRID SEARCH, SEARCH PARAMS or RELEASE will silently "
        "lex as something else."
    )


def test_recycled_aliases_match_the_pinned_hosts() -> None:
    assert {
        alias: member.name for alias, member in LIVE_HOSTS.items()
    } == PINNED_HOSTS


def test_recycled_aliases_are_pairwise_distinct() -> None:
    # A copy-paste making two aliases the same member would not raise anywhere: the later entry
    # would just win in FACTOR/KEYWORDS and one operator would quietly parse as the other.
    assert len(set(LIVE_HOSTS.values())) == len(LIVE_HOSTS)


@pytest.mark.parametrize(
    "alias", sorted(PINNED_HOSTS), ids=sorted(PINNED_HOSTS)
)
def test_recycled_host_is_still_unclaimed_by_sqlglot(alias) -> None:
    """The count pin catches renumbering; this catches *repurposing*.

    We derive from the bare `Dialect`, so only the base tokenizer and `BaseParser` can collide
    with us. If upstream ever maps a real keyword to one of these members -- or if someone
    repoints an alias at a member sqlglot already uses -- that keyword starts lexing as a
    MilvusQL clause opener.
    """
    member = LIVE_HOSTS[alias]
    claimed_by = [
        kw
        for kw, tt in sqlglot.tokens.Tokenizer.KEYWORDS.items()
        if tt is member
    ]
    assert claimed_by == [], (
        f"base tokenizer now maps {claimed_by} to {member}"
    )

    for registry in (
        "STATEMENT_PARSERS",
        "QUERY_MODIFIER_PARSERS",
        "FACTOR",
        "EQUALITY",
    ):
        assert member not in getattr(BaseParser, registry), (
            f"BaseParser.{registry} now attaches meaning to {member}"
        )


# ------------------------------------------------------------------------------------------
# 3. The exp.Select.arg_types patch
# ------------------------------------------------------------------------------------------


def test_select_arg_types_carry_the_milvus_modifier_keys() -> None:
    # `Expression.error_messages` raises TypeError -- not ParseError -- for an arg key missing
    # from arg_types, but only when sqlglot's UNITTEST flag is set, i.e. exactly under pytest.
    # So a regression of the import-time patch in expressions.py cannot be reproduced in
    # production and shows up only here, as a confusing TypeError from a parse.
    assert exp.Select.arg_types.get(HYBRID_ARG) is False
    assert exp.Select.arg_types.get(SEARCH_PARAMS_ARG) is False

    # `required_args` is frozen in __init_subclass__ and never recomputed, so an accidental
    # `= True` would declare a required key that nothing can ever satisfy.
    assert HYBRID_ARG not in exp.Select.required_args
    assert SEARCH_PARAMS_ARG not in exp.Select.required_args


@pytest.mark.parametrize(
    "node", [exp.Subquery, exp.SetOperation], ids=["subquery", "set-operation"]
)
@pytest.mark.parametrize(
    "key", [HYBRID_ARG, SEARCH_PARAMS_ARG], ids=["hybrid", "search-params"]
)
def test_subquery_and_set_operation_carry_the_keys_too(node, key) -> None:
    # BaseParser.MODIFIABLES is (Query, Table, TableFromRows, Values), so a query modifier attaches
    # to a Subquery or a SetOperation exactly as it does to a Select. `Expression.set` writes an
    # undeclared key anyway, so patching only Select did not keep the clause off those nodes -- it
    # only made `key in node.arg_types` lie about it, and `unnest().sql()` drop it.
    assert node.arg_types.get(key) is False
    assert key not in node.required_args


def test_select_arg_type_keys_are_the_documented_strings() -> None:
    # The generator addresses these args through the constants, but the AST is a public contract
    # (Track B reads `select.args["search_params"]`), so the *strings* are what must not drift.
    assert (HYBRID_ARG, SEARCH_PARAMS_ARG) == ("hybrid", "search_params")


def test_select_validates_with_the_milvus_modifiers_attached() -> None:
    # The direct reproduction of the failure mode above: unpatched, this raises
    # TypeError: Unexpected keyword: 'search_params' for <class '...Select'>
    arm = expressions.SearchArm(this=exp.column("embedding"))
    select = exp.select("id").from_("items")
    select.set(HYBRID_ARG, expressions.HybridSearch(expressions=[arm]))
    select.set(
        SEARCH_PARAMS_ARG,
        expressions.SearchParams(expressions=[exp.var("ef_search")]),
    )
    assert select.error_messages() == []


# ------------------------------------------------------------------------------------------
# 4. Custom node arg_types
# ------------------------------------------------------------------------------------------

#: node class -> the minimal kwargs that satisfy its required args.
#: Factories, not literal dicts: constructing a node re-parents its children in place, so sharing
#: one `exp.Column` across parametrized cases would silently hand later tests a node whose
#: `parent` points at an unrelated tree.
CUSTOM_NODES = {
    expressions.LoadTable: lambda: {"this": exp.to_table("items")},
    expressions.ReleaseTable: lambda: {"this": exp.to_table("items")},
    expressions.AddField: lambda: {"this": exp.column("tag")},
    expressions.InnerProduct: lambda: {
        "this": exp.column("embedding"),
        "expression": exp.var("q"),
    },
    expressions.L1Distance: lambda: {
        "this": exp.column("embedding"),
        "expression": exp.var("q"),
    },
    expressions.CosineDistance: lambda: {
        "this": exp.column("embedding"),
        "expression": exp.var("q"),
    },
    expressions.SearchParams: lambda: {"expressions": [exp.var("ef_search")]},
    expressions.SearchArm: lambda: {"this": exp.column("embedding")},
    expressions.Rerank: lambda: {"this": exp.var("RRF")},
    expressions.HybridSearch: lambda: {
        "expressions": [expressions.SearchArm(this=exp.column("embedding"))]
    },
    expressions.BM25Score: lambda: {
        "this": exp.column("text"),
        "expression": exp.var("q"),
    },
    expressions.ConsistencyLevel: lambda: {"this": exp.var("Bounded")},
}

_NODE_IDS = [cls.__name__ for cls in CUSTOM_NODES]


def test_every_custom_node_is_covered_by_this_file() -> None:
    # Without this, adding a node and forgetting a CUSTOM_NODES entry is a silent gap.
    declared = {
        obj
        for obj in vars(expressions).values()
        if isinstance(obj, type)
        and issubclass(obj, exp.Expression)
        and obj.__module__ == expressions.__name__
    }
    assert declared == set(CUSTOM_NODES)


@pytest.mark.parametrize(
    ("cls", "make_kwargs"), CUSTOM_NODES.items(), ids=_NODE_IDS
)
def test_custom_node_constructs_with_its_required_args(
    cls, make_kwargs
) -> None:
    node = cls(**make_kwargs())
    assert node.error_messages() == []

    # `required_args` is what actually gates parsing, and the `Expr` docstring upstream claims
    # arg_types values mean "optional" -- they mean *required*. Pin the real semantics.
    assert cls.required_args == {
        k for k, required in cls.arg_types.items() if required
    }


@pytest.mark.parametrize(
    ("cls", "make_kwargs"), CUSTOM_NODES.items(), ids=_NODE_IDS
)
def test_custom_node_rejects_missing_required_args(cls, make_kwargs) -> None:
    empty = cls()

    # Constructing never validates -- only Parser.expression() does -- so check both halves:
    # the node knows what it is missing, and the parser path actually refuses it.
    reported = " ".join(empty.error_messages())
    assert reported, f"{cls.__name__} declares no required args at all"
    for key in cls.required_args:
        assert f"'{key}'" in reported

    parser = Dialect.get_or_raise(DIALECT).parser(
        error_level=ErrorLevel.IMMEDIATE
    )
    with pytest.raises(ParseError, match="Required keyword"):
        parser.expression(empty)

    # Same parser, satisfied node: proves the raise above came from the missing arg and not from
    # the node type being unacceptable to validate_expression for some other reason.
    assert parser.expression(cls(**make_kwargs())) is not None


# ------------------------------------------------------------------------------------------
# 5. The sqlglot[c] import guard
# ------------------------------------------------------------------------------------------

_SQLGLOTC_GUARD_SCRIPT = """
import json, sys

import sqlglot.tokens

# sqlglot[c] ships mypyc-compiled modules and sets this flag. We cannot install that wheel into
# the project venv (it would break every other test), so forge the flag on the very module object
# sqlglot_milvus/__init__.py reads it from.
if sys.argv[1] == "pretend-compiled":
    sqlglot.tokens.SQLGLOTC_INSTALLED = True

# Deliberately unguarded: if upstream ever renames the flag, this raises AttributeError, the
# child exits non-zero, and _run_cold fails loudly instead of the test passing vacuously.
result = {"flag": sqlglot.tokens.SQLGLOTC_INSTALLED}

try:
    import sqlglot_milvus
except BaseException as exc:  # noqa: BLE001 - the exact type is the thing under test
    result["raised"] = type(exc).__name__
    result["message"] = str(exc)
else:
    result["raised"] = None
    result["message"] = ""
print(json.dumps(result))
"""


def test_sqlglotc_flag_still_exists_upstream() -> None:
    # The guard reads the flag with a getattr default, so a rename upstream turns it into dead
    # code that can never fire -- and the failure it exists to prevent (TypeError: interpreted
    # classes cannot inherit from compiled, thrown from deep inside sqlglot on the first parse)
    # comes back. Nothing else in the suite would notice.
    assert hasattr(sqlglot.tokens, "SQLGLOTC_INSTALLED")
    assert sqlglot.tokens.SQLGLOTC_INSTALLED is False, (
        "pure-Python sqlglot is required"
    )


def test_import_succeeds_when_sqlglotc_is_absent() -> None:
    # Control for the test below: same script, same interpreter, flag untouched. Without it, an
    # ImportError caused by a broken sys.path would read as the guard working.
    result = _run_cold(_SQLGLOTC_GUARD_SCRIPT, "pretend-pure-python")
    assert result["flag"] is False
    assert result["raised"] is None, result["message"]


def test_import_fails_loudly_when_sqlglotc_is_installed() -> None:
    result = _run_cold(_SQLGLOTC_GUARD_SCRIPT, "pretend-compiled")

    assert result["flag"] is True
    # Exactly ImportError: ModuleNotFoundError is a *subclass*, so `pytest.raises(ImportError)`
    # would happily pass on a merely uninstalled package.
    assert result["raised"] == "ImportError", result["message"]

    message = result["message"]
    assert "sqlglot[c]" in message
    # Remediation, not just a diagnosis -- the whole point of failing at import time.
    assert "pip uninstall" in message
    assert "pip install" in message


# ------------------------------------------------------------------------------------------
# 6. Exports and version
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", sqlglot_milvus.__all__, ids=list(sqlglot_milvus.__all__)
)
def test_all_export_resolves(name) -> None:
    assert hasattr(sqlglot_milvus, name), (
        f"__all__ advertises {name!r} but nothing binds it"
    )


def test_all_has_no_duplicates() -> None:
    assert len(sqlglot_milvus.__all__) == len(set(sqlglot_milvus.__all__))


def test_version_matches_pyproject(repo_root) -> None:
    pyproject = _load_pyproject(repo_root / "pyproject.toml")
    assert sqlglot_milvus.__version__ == pyproject["project"]["version"]


def test_version_matches_installed_distribution_metadata() -> None:
    # Diverges when pyproject.toml is bumped without reinstalling; the wheel then ships one
    # version while `sqlglot_milvus.__version__` reports another.
    assert version("sqlglot-milvus") == sqlglot_milvus.__version__


# ------------------------------------------------------------------------------------------
# 7. pyproject entry-point declaration
# ------------------------------------------------------------------------------------------


def _load_pyproject(path: pathlib.Path) -> dict:
    try:
        import tomllib  # noqa: PLC0415
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip(
            "tomli", reason="no TOML reader available"
        )
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_pyproject_declares_a_lowercase_milvus_entry_point(repo_root) -> None:
    pyproject = _load_pyproject(repo_root / "pyproject.toml")
    group = pyproject["project"]["entry-points"][PLUGIN_GROUP]

    # Entry-point names are used *verbatim*: sqlglot never lowercases them, and get_or_raise
    # never lowercases the requested name either. An entry point named "Milvus" would only ever
    # resolve as read="Milvus".
    assert list(group) == [DIALECT]
    assert group[DIALECT] == "sqlglot_milvus.dialect:Milvus"


def test_pyproject_entry_point_target_resolves_to_our_class(repo_root) -> None:
    pyproject = _load_pyproject(repo_root / "pyproject.toml")
    module_path, _, attr = pyproject["project"]["entry-points"][PLUGIN_GROUP][
        DIALECT
    ].partition(":")
    # A dangling `:DoesNotExist` escapes sqlglot's loader as a raw AttributeError (only
    # ImportError is caught), and a module-only target fails silently as "Unknown dialect".
    assert getattr(importlib.import_module(module_path), attr) is Milvus


def test_entry_point_name_matches_the_lowercased_class_name() -> None:
    # `_Dialect.__hash__` is hash(cls.__name__.lower()) while `__eq__` compares by name. If the
    # entry-point name and the class name ever diverge, `Milvus == "<ep name>"` stays True while
    # the hashes differ, and anything using a dialect as a dict key breaks in a very quiet way.
    assert Milvus.__name__.lower() == DIALECT
    assert Milvus == DIALECT
    assert hash(Milvus) == hash(DIALECT)


def test_installed_entry_point_is_lowercase_milvus() -> None:
    # The installed metadata, not pyproject.toml -- these diverge when the source is edited
    # without reinstalling, and it is the metadata that sqlglot's loader actually reads. This
    # also keeps item 7's guarantee alive on interpreters where the TOML tests above skip.
    all_eps = entry_points()
    # importlib.metadata grew `.select()` in 3.10; mirror sqlglot's own compatibility shim.
    if hasattr(all_eps, "select"):
        found = list(all_eps.select(group=PLUGIN_GROUP))
    else:  # pragma: no cover - Python 3.9
        found = list(all_eps.get(PLUGIN_GROUP, []))

    # Filter to ours: the venv may legitimately host other dialect plugins.
    ours = [ep for ep in found if ep.value.startswith("sqlglot_milvus")]
    assert [ep.name for ep in ours] == [DIALECT], (
        "installed metadata disagrees with pyproject.toml -- reinstall the package "
        f"(found {[ep.name for ep in found]})"
    )
    assert ours[0].load() is Milvus
