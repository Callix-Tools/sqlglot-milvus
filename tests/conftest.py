"""Session-wide setup for the MilvusQL suite.

Deliberately thin. Every test module owns its own assertion helpers; what lives here is only
what has to be true before *any* module runs, plus the two or three facts about the environment
that turn a baffling failure into an obvious one.
"""

from __future__ import annotations

import pathlib

import pytest
import sqlglot
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType

# Importing the package registers `Milvus` under "milvus" via the `_Dialect` metaclass, which
# runs at class-definition time. Without this, registration would depend on whichever test module
# happened to import first -- or on the entry point, which is exactly what tests/test_packaging.py
# exercises in a *cold* interpreter and must therefore not be pre-empted here.
import sqlglot_milvus

#: The dialect name every test passes to `read=` / `write=`. `get_or_raise` does not lowercase,
#: so the string is case-sensitive (see docs/SQLGLOT_API_GROUND_TRUTH.md §5.7).
DIALECT = "milvus"

TESTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def pytest_report_header() -> list[str]:
    # Nearly every way this suite can break is "sqlglot moved underneath us" or "an old copy of
    # the package is on sys.path". Both are one glance away if the header says so.
    return [
        f"sqlglot {sqlglot.__version__} ({len(list(TokenType))} TokenType members)",
        f"sqlglot-milvus {sqlglot_milvus.__version__} from {sqlglot_milvus.__file__}",
        f"dialect {DIALECT!r} -> {Dialect.get(DIALECT)}",
    ]


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    """The distribution root -- the only path any test should need to hard-code."""
    return REPO_ROOT
