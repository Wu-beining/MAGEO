from abc import ABC, abstractmethod
from typing import Any

from model.schema import LLMResponse


class BaseLLM(ABC):
    @abstractmethod
    async def call(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Single call: user_prompt + optional system_prompt -> response"""
        pass

    @abstractmethod
    async def call_with_history(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> LLMResponse:
        """Call with message history for multi-turn conversations"""
        pass
