"""Recycled ``TokenType`` members.

``TokenType`` is a plain ``Enum`` defined once for all of sqlglot. Python
enums cannot be extended by subclassing once the base has members, and a
plain class-body assignment such as ``TokenType.HYBRID = "HYBRID"`` silently
creates an ordinary attribute rather than a member -- the parser
dictionaries keyed by ``TokenType`` then never match. A third-party dialect
therefore has exactly one option: **recycle existing members that sqlglot
itself never references.**

Each alias below was chosen from the set of members with zero references
anywhere in the sqlglot package (verified by a package-wide grep), so
recycling them cannot perturb any built-in dialect. The names are
meaningless in our context -- that is the price of the mechanism -- which
is why every use site refers to the alias, never to the underlying member.

``tests/test_packaging.py`` pins ``len(list(TokenType))`` so that an
upstream release which renumbers or gives behaviour to one of these members
fails loudly instead of silently changing how MilvusQL is lexed.
"""

from __future__ import annotations

from sqlglot.tokens import TokenType

#: ``<#>`` -- inner-product distance (pgvector spelling).
TT_INNER_PRODUCT = TokenType.SPACE

#: ``<+>`` -- L1 / Manhattan distance (pgvector spelling).
TT_L1 = TokenType.BREAK

#: ``HYBRID SEARCH`` -- registered as a *multi-word* keyword so that the
#: bare words ``HYBRID`` and ``SEARCH`` remain ordinary identifiers.
TT_HYBRID_SEARCH = TokenType.LANGUAGE

#: ``SEARCH PARAMS`` -- likewise multi-word.
TT_SEARCH_PARAMS = TokenType.PROPERTIES

#: ``RELEASE`` -- necessarily a single word, hence genuinely reserved by the
#: tokenizer. The parser adds it back to ``ID_VAR_TOKENS`` /
#: ``TABLE_ALIAS_TOKENS`` so it stays usable as an identifier.
TT_RELEASE = TokenType.SOUNDS_LIKE

#: ``CONSISTENCY LEVEL`` -- multi-word, so ``consistency`` and ``level``
#: stay identifiers.
TT_CONSISTENCY_LEVEL = TokenType.ORDERED

#: Number of ``TokenType`` members in the sqlglot release this dialect was
#: written against.
EXPECTED_TOKEN_TYPE_COUNT = 443

__all__ = [
    "EXPECTED_TOKEN_TYPE_COUNT",
    "TT_CONSISTENCY_LEVEL",
    "TT_HYBRID_SEARCH",
    "TT_INNER_PRODUCT",
    "TT_L1",
    "TT_RELEASE",
    "TT_SEARCH_PARAMS",
]
