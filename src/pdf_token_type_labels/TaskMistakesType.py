from enum import Enum


class TaskMistakesType(Enum):
    CORRECT = "CORRECT"
    WRONG = "WRONG"
    MISSING = "MISSING"

    # ── Precomputed O(1) lookups ────────────────────────────────────
    _members_tuple = None
    _value_to_index = None

    @classmethod
    def _ensure_lookups(cls):
        if cls._members_tuple is None:
            cls._members_tuple = tuple(cls)
            cls._value_to_index = {m: i for i, m in enumerate(cls._members_tuple)}

    @staticmethod
    def contains(key: str):
        return key.upper() in [e.value for e in TaskMistakesType]

    @staticmethod
    def from_text(text: str):
        try:
            return TaskMistakesType[text.upper()]
        except KeyError:
            return TaskMistakesType.WRONG

    def get_index(self) -> int:
        TaskMistakesType._ensure_lookups()
        return TaskMistakesType._value_to_index[self]
