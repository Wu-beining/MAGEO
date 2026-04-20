from abc import ABC, abstractmethod
from typing import Any

from config.base import is_debug
from debug.log import Logger
from memory.base import BaseMemory
from memory.sliding_window import SlidingWindowMemory
from model.base import BaseLLM
from model.schema import Message


class BaseAgent(ABC):
    """Base Agent: holds model, system_prompt, maintains context history"""

    def __init__(
        self,
        model: BaseLLM,
        system_prompt: str = "",
        memory: BaseMemory | None = None,
    ):
        self._model = model
        self._system_prompt = system_prompt
        self._memory = memory or SlidingWindowMemory()
        self._log = Logger(self.__class__.__name__)

    @property
    def _debug(self) -> bool:
        return is_debug()

    @property
    def history(self) -> list[Message]:
        return self._memory.get_messages()

    def add_message(self, message: Message) -> None:
        self._memory.add(message)

    def clear_history(self) -> None:
        self._memory.clear()

    def _build_messages(self) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        for m in self._memory.get_messages():
            msgs.append(m.model_dump(exclude_none=True))
        return msgs

    @abstractmethod
    async def run(self, user_input: str) -> str:
        pass
