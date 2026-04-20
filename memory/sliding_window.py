import litellm

from memory.base import BaseMemory
from model.schema import Message


class SlidingWindowMemory(BaseMemory):
    """Sliding Window Memory: truncates by message count + token count"""

    def __init__(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        model: str = "gpt-4o",
    ):
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._model = model
        self._messages: list[Message] = []
        self._token_counts: list[int] = []

    def add(self, message: Message) -> None:
        tokens = self._count_tokens(message)
        self._messages.append(message)
        self._token_counts.append(tokens)
        self._truncate()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._token_counts.clear()

    @property
    def token_count(self) -> int:
        return sum(self._token_counts)

    def _count_tokens(self, message: Message) -> int:
        """Count tokens for a single message using litellm"""
        msg_dict = message.model_dump(exclude_none=True)
        return litellm.token_counter(model=self._model, messages=[msg_dict])

    def _truncate(self) -> None:
        """Execute truncation: prioritize keeping recent messages"""
        # Truncate by message count
        if self._max_messages and len(self._messages) > self._max_messages:
            excess = len(self._messages) - self._max_messages
            self._messages = self._messages[excess:]
            self._token_counts = self._token_counts[excess:]

        # Truncate by token count
        if self._max_tokens:
            while self._messages and self.token_count > self._max_tokens:
                self._messages.pop(0)
                self._token_counts.pop(0)

