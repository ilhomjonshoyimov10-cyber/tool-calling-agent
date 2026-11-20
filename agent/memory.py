"""Short-term conversation memory per session (in process memory)."""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any


class SessionStore:
    """Keeps the last `max_messages` user/assistant messages for up to `max_sessions` sessions (LRU).

    Only final answers are stored, not the intermediate tool messages: the next turn gets the
    conversational context without paying tokens for old tool output.
    """

    def __init__(self, max_messages: int = 10, max_sessions: int = 1000) -> None:
        self._max_messages = max_messages
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[dict[str, Any]]] = OrderedDict()

    def history(self, session_id: str) -> list[dict[str, Any]]:
        if session_id not in self._sessions:
            return []
        self._sessions.move_to_end(session_id)
        return list(self._sessions[session_id])

    def append(self, session_id: str, user_message: str, answer: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self._max_messages)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        self._sessions.move_to_end(session_id)
        self._sessions[session_id].extend(
            [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]
        )

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
