from __future__ import annotations

import re

from app.contracts import RoleId
from app.services.message_intent import (
    UserIntent, explicit_task_intent, forbids_repository_reads, forbids_writes,
)

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
    DELEGATION_END = re.compile(r"물어\s*봐|불러도\s*돼|말해도\s*돼|부탁해|(?:검토|의견|확인).{0,20}요청해|의견(?:을|도)?\s*(?:받아|들어)\s*(?:줘|주세요)|이야기\s*해서|이야기\s*해|논의\s*해|상의\s*해")
    ROLE_LIST_CONNECTOR = re.compile(r"\s*(?:,|/|·|나|이랑|와|과|랑|하고|및)\s*")
    GENERIC_CONSULTATION = re.compile(
        r"다른\s*(?:에이전트|역할|애(?:들)?|팀원).{0,40}(?:불러|물어|호출|검토|의견|확인)|"
        r"(?:불러|물어|호출|검토|의견|확인).{0,40}다른\s*(?:에이전트|역할|애(?:들)?|팀원)"
    )
    NEGATED_REQUEST = re.compile(r"(?:하지|보지|부르지)\s*(?:마|말|않)|안\s*돼|금지")

    def __init__(self, role_names: dict[str, str] | None = None):
        configured = role_names or {}
        defaults = dict(zip(ROLE_ORDER, ("빌더", "센티널", "피니셔"), strict=True))
        self.names = {
            role: configured.get(role.value, defaults[role]).strip()
            for role in ROLE_ORDER
        }

    def resolve(self, text: str, last_active: str | RoleId) -> RoleSelection:
        normalized = text.strip().lower()
        handoff = self.handoff_target(text)
        if handoff is not None:
            return RoleSelection((handoff,), explicit=True)
        single_role = self._single_role_request(normalized)
        if single_role is not None:
            return RoleSelection((single_role,), explicit=True)
        conditional_ranges = self._conditional_ranges(normalized)
        group_text = ''.join(
            char if not self._in_conditional_range(index, conditional_ranges) else ' '
            for index, char in enumerate(normalized)
        )
        if self._positive_group_request(group_text):
            return RoleSelection(ROLE_ORDER, explicit=True, group_call=True)

        direct_role = self._leading_direct_role(normalized, conditional_ranges)
        if direct_role is not None:
            return RoleSelection((direct_role,), explicit=True)

        for role, name in self.names.items():
            matches = re.finditer(
                rf"{re.escape(name.lower())}(?:이|가|은|는|에게|한테)\s*(?:봐|검토|확인|답|설명)",
                normalized,
            ) if name else ()
            for match in matches:
                end = self._range_from_position(normalized, match.start())[1]
                following = self._role_positions(normalized, match.end(), end)
                if following:
                    end = following[0][0]
                if (not self._in_conditional_range(match.start(), conditional_ranges)
                    and not self.NEGATED_REQUEST.search(normalized[match.start():end])):
                    return RoleSelection((role,), explicit=True)

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

    def handoff_target(self, text: str) -> RoleId | None:
        """원문의 긍정적인 단일 담당 인계만 인정한다."""
        normalized = text.strip().casefold()
        for role, name in self.names.items():
            if name and re.fullmatch(
                rf"(?:앞으로|이제부터)\s*{re.escape(name.casefold())}(?:가|이)\s*"
                r"(?:맡아|담당해)(?:줘|주세요)?[.!]?", normalized,
            ):
                return role
        return None

    def interpret(self, text: str, last_active: str | RoleId) -> UserIntent:
        selection = self.resolve(text, last_active)
        normalized = re.sub(r'"[^"\n]*"|\x27[^\x27\n]*\x27|“[^”\n]*”|‘[^’\n]*’|`[^`\n]*`',
                            lambda match: " " * len(match.group()), text.casefold())
        delegates: set[RoleId] = set()
        # 상담 대상은 현재 원문의 허용 구절에서만 가져온다.
        for match in self.DELEGATION_END.finditer(normalized):
            start, _ = self._delegation_range(normalized, match.start())
            clause = normalized[start:match.end()]
            if (self.NEGATED_REQUEST.search(clause)
                or re.match(r"\s*(?:라고|라는|란|인지|는지)", normalized[match.end():])):
                continue
            delegates.update(
                role for role, name in self.names.items()
                if name and name.casefold() in clause
            )
            if self.GENERIC_CONSULTATION.search(clause):
                delegates.update(ROLE_ORDER)
        # 같은 원문의 대상별 금지는 앞의 일반 상담 허용보다 우선한다.
        for role, name in self.names.items():
            if name and re.search(
                rf"{re.escape(name.casefold())}(?:에게|한테)?(?:은|는)?\s*"
                r"(?:부르지|물어\s*보지|(?:호출|검토|이야기|논의|상의|말|부탁|요청)\s*하지|"
                r"의견(?:을|은|는|도)?\s*(?:받지|듣지|(?:받아|들어)\s*주지))"
                r"\s*(?:마|말|않)", normalized,
            ):
                delegates.discard(role)
        return UserIntent(
            selection.roles, tuple(role for role in ROLE_ORDER if role in delegates),
            explicit_task_intent(text), forbids_writes(text),
            selection.explicit, selection.group_call,
            text.strip(),
            read_forbidden=forbids_repository_reads(text),
        )

    def _single_role_request(self, text: str) -> RoleId | None:
        for role, name in self.names.items():
            if name and re.search(
                rf"{re.escape(name.lower())}(?:만|에게만|한테만)\s*(?:답|말|설명|검토|봐)",
                text,
            ):
                return role
        return None

    def _positive_group_request(self, text: str) -> bool:
        unnamed_plural = (
            re.match(r"^(?:각자|각각)(?:\s|,)", text)
            and not any(name and name.lower() in text for name in self.names.values())
        )
        if not unnamed_plural and not any(marker in text for marker in self.GROUP_MARKERS):
            return False
        if (self.NEGATED_REQUEST.search(text)
            or re.search(r"(?:알려\s*주지|모이지)\s*(?:마|말|않)|(?:왜|무엇|뭐|어떻게).*?(?:모두|셋\s*다)", text)):
            return False
        if re.search(r"(?:모두|셋\s*다|각자|각각)\s*(?:라는|란|이라고|라는\s*단어)", text):
            return False
        return bool(re.search(r"(?:답|의견|말|모여|봐|검토|설명|소개|호출|불러|참여|논의|알려)", text))

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
            if re.match(r"(?:가|이|은|는)\s*(?:만든|작성한|구현한|수정한)", suffix):
                continue
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
        while start > 0 and text[start - 1] not in ".!?\n":
            start -= 1
        end = self._range_from_position(text, start)[1]
        comma = text.find(',', position, end)
        if comma >= 0:
            end = comma
        mentioned = [
            item for item in self._role_positions(text, start, position)
            if not re.match(r"(?:아|야|님)|(?:가|이|은|는)\s*(?:만든|작성한|구현한|수정한)", text[item[1]:])
        ]
        if not mentioned:
            condition = list(self.CONDITIONAL_MARKERS.finditer(text, start, position))
            if condition:
                return condition[-1].start(), end
            generic = self.GENERIC_CONSULTATION.search(text, start, end)
            if generic:
                return generic.start(), end
            return position, end

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
