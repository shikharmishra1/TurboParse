from enum import Enum

# ── Module-level O(1) lookup caches (avoids Enum metaclass conflicts) ──
_token_type_cache = None


def _ensure_token_type_cache():
    global _token_type_cache
    if _token_type_cache is None:
        members = tuple(TokenType)
        _token_type_cache = {
            "members_tuple": members,
            "name_to_member": {m.name: m for m in members},
            "value_to_index": {m: i for i, m in enumerate(members)},
        }


class TokenType(Enum):
    FORMULA = "Formula"
    FOOTNOTE = "Footnote"
    LIST_ITEM = "List item"
    TABLE = "Table"
    PICTURE = "Picture"
    TITLE = "Title"
    TEXT = "Text"
    PAGE_HEADER = "Page header"
    SECTION_HEADER = "Section header"
    CAPTION = "Caption"
    PAGE_FOOTER = "Page footer"

    @staticmethod
    def from_text(text: str):
        _ensure_token_type_cache()
        try:
            return _token_type_cache["name_to_member"][text.upper()]
        except KeyError:
            return TokenType.TEXT

    @staticmethod
    def from_index(index: int):
        _ensure_token_type_cache()
        try:
            return _token_type_cache["members_tuple"][index]
        except IndexError:
            return TokenType.TEXT

    def get_index(self) -> int:
        _ensure_token_type_cache()
        return _token_type_cache["value_to_index"][self]
