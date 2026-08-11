"""MilvusQL -- a SQL dialect for Milvus, as a sqlglot plugin.

Installing this package registers the dialect under the name ``milvus``::

    import sqlglot
    sqlglot.parse_one("SELECT id FROM items ORDER BY embedding <-> :q LIMIT 5", read="milvus")
    sqlglot.transpile(pgvector_sql, read="postgres", write="milvus")
"""

from __future__ import annotations

from sqlglot import tokens as _tokens

# sqlglot[c] ships mypyc-compiled modules, and mypyc disables runtime subclassing of the compiled
# classes. The failure is unusually nasty: the `class Parser(BaseParser)` statement below succeeds,
# so this package imports perfectly cleanly, and then the first parse_one() raises
# "TypeError: interpreted classes cannot inherit from compiled" from deep inside sqlglot. Fail here
# instead, with instructions. Upstream documents the limitation in the wheel's own METADATA.
if getattr(
    _tokens, "SQLGLOTC_INSTALLED", False
):  # pragma: no cover - env-dependent
    msg = (
        "sqlglot-milvus is incompatible with sqlglot[c] / sqlglot[rs]: the mypyc-compiled build "
        "forbids runtime subclassing of Parser/Generator/Expression, which every third-party "
        "dialect requires. Install the pure-Python build instead:\n"
        "    pip uninstall -y sqlglotc\n"
        "    pip install --force-reinstall 'sqlglot>=30.16.0,<31'"
    )
    raise ImportError(msg)

from .dialect import Milvus
from .expressions import (
    METRIC_TYPES,
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

__version__ = "0.1.0"

__all__ = [
    "METRIC_TYPES",
    "AddField",
    "BM25Score",
    "CosineDistance",
    "HybridSearch",
    "InnerProduct",
    "L1Distance",
    "LoadTable",
    "Milvus",
    "ReleaseTable",
    "Rerank",
    "SearchArm",
    "SearchParams",
    "__version__",
]
