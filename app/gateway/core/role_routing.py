from __future__ import annotations

import re

from app.contracts import RoleId

from .models import RoleSelection


ROLE_ORDER = (
    RoleId.DEVELOPMENT,
    RoleId.REVIEW,
    RoleId.IMPROVEMENT,
)


class RoleResolver:
    """사용자 호명과 마지막 대화 상대를 결정적으로 해석한다."""

    GROUP_MARKERS = ("얘들아", "애들아", "셋 다", "셋다", "모두")
    PLURAL_RESPONSE_MARKERS = ("둘 다", "둘다", "각자", "각각", "두 역할")
    DIRECT_ADDRESS_SUFFIXES = ("아", "야", "님", "은", "는", "이", "가")
    CONDITIONAL_MARKERS = re.compile(r"모르면|필요하면")
    DELEGATION_END = re.compile(r"물어\s*봐|불러도\s*돼|말해도\s*돼|부탁해")
    ROLE_LIST_CONNECTOR = re.compile(r"\s*(?:,|/|·|나|이랑|와|과|랑|하고|및)\s*")

    def __init__(self, role_names: dict[str, str] | None = None):
        configured = role_names or {}
        self.names = {
            role: configured.get(role.value, "").strip()
            for role in ROLE_ORDER
        }

    def resolve(self, text: str, last_active: str | RoleId) -> RoleSelection:
        normalized = text.strip().lower()
        single_role = self._single_role_request(normalized)
        if single_role is not None:
            return RoleSelection((single_role,), explicit=True)
        if self._positive_group_request(normalized):
            return RoleSelection(ROLE_ORDER, explicit=True, group_call=True)

        conditional_ranges = self._conditional_ranges(normalized)
        direct_role = self._leading_direct_role(normalized, conditional_ranges)
        if direct_role is not None:
            return RoleSelection((direct_role,), explicit=True)

        positioned = self._mentioned_roles(normalized, conditional_ranges)
        if (
            len(positioned) > 1
            and any(marker in normalized for marker in self.PLURAL_RESPONSE_MARKERS)
        ):
            return RoleSelection(
                tuple(role for _, role in positioned),
                explicit=True,
            )

        try:
            fallback = last_active if isinstance(last_active, RoleId) else RoleId(last_active)
        except ValueError:
            fallback = RoleId.DEVELOPMENT
        return RoleSelection((fallback,))

    def _single_role_request(self, text: str) -> RoleId | None:
        for role, name in self.names.items():
            if name and re.search(
                rf"{re.escape(name.lower())}(?:만|에게만|한테만)\s*(?:답|말|설명|검토|봐)",
                text,
            ):
                return role
        return None

    def _positive_group_request(self, text: str) -> bool:
        if not any(marker in text for marker in self.GROUP_MARKERS):
            return False
        if re.search(r"(?:부르지|호출하지|답하지|말하지|모이지|하지)\s*(?:마|말아|않)|(?:왜|무엇|뭐|어떻게).*?(?:모두|셋\s*다)", text):
            return False
        if re.search(r"(?:모두|셋\s*다)(?:라는|란|이라고|라는\s*단어)", text):
            return False
        return bool(re.search(r"(?:답|의견|말|모여|봐|검토|설명|소개|호출|불러|참여|논의)", text))

    def _mentioned_roles(
        self, text: str, conditional_ranges: tuple[tuple[int, int], ...]
    ) -> list[tuple[int, RoleId]]:
        positioned: list[tuple[int, RoleId]] = []
        for role in ROLE_ORDER:
            name = self.names[role].lower()
            if not name:
                continue
            start = 0
            while (position := text.find(name, start)) >= 0:
                if not self._in_conditional_range(position, conditional_ranges):
                    positioned.append((position, role))
                    break
                start = position + len(name)
        positioned.sort(key=lambda item: item[0])
        return positioned

    def _leading_direct_role(
        self, text: str, conditional_ranges: tuple[tuple[int, int], ...]
    ) -> RoleId | None:
        if self._in_conditional_range(0, conditional_ranges):
            return None
        for role, name in sorted(
            self.names.items(), key=lambda item: len(item[1]), reverse=True
        ):
            normalized_name = name.lower()
            if not normalized_name or not text.startswith(normalized_name):
                continue
            suffix = text[len(normalized_name) :]
            if not suffix or suffix[0].isspace() or suffix[0] in ".!?…:;":
                return role
            if suffix.startswith(self.DIRECT_ADDRESS_SUFFIXES):
                return role
            if suffix[0] == "," and not self._starts_with_role_name(
                suffix[1:].lstrip()
            ):
                return role
        return None

    def _starts_with_role_name(self, text: str) -> bool:
        return any(
            name and text.startswith(name.lower()) for name in self.names.values()
        )

    def _conditional_ranges(self, text: str) -> tuple[tuple[int, int], ...]:
        ranges = [
            self._range_from_position(text, match.start())
            for match in self.CONDITIONAL_MARKERS.finditer(text)
        ]
        ranges.extend(
            self._delegation_range(text, match.start())
            for match in self.DELEGATION_END.finditer(text)
        )
        return tuple(ranges)

    @staticmethod
    def _in_conditional_range(
        position: int, conditional_ranges: tuple[tuple[int, int], ...]
    ) -> bool:
        return any(
            range_start <= position < range_end
            for range_start, range_end in conditional_ranges
        )

    @staticmethod
    def _range_from_position(text: str, start: int) -> tuple[int, int]:
        end = start
        while end < len(text) and text[end] not in ".!?\n":
            end += 1
        return start, end

    def _delegation_range(self, text: str, position: int) -> tuple[int, int]:
        start = position
        while start > 0 and text[start - 1] not in ".!?\n,":
            start -= 1
        end = self._range_from_position(text, start)[1]
        mentioned = self._role_positions(text, start, position)
        if not mentioned:
            return start, end

        target_index = len(mentioned) - 1
        target_start = mentioned[target_index][0]
        while target_index > 0:
            previous_start, previous_end = mentioned[target_index - 1]
            if not self.ROLE_LIST_CONNECTOR.fullmatch(
                text[previous_end:target_start]
            ):
                break
            target_index -= 1
            target_start = previous_start
        return target_start, end

    def _role_positions(
        self, text: str, start: int, end: int
    ) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        for name in self.names.values():
            normalized_name = name.lower()
            if not normalized_name:
                continue
            position = text.find(normalized_name, start, end)
            while position >= 0:
                positions.append((position, position + len(normalized_name)))
                position = text.find(normalized_name, position + len(normalized_name), end)
        return sorted(positions)
