import json
from typing import Any

import litellm

from config.base import is_debug
from debug.log import Color, Logger
from model.base import BaseLLM
from model.schema import LLMResponse, ToolCall

_log = Logger("LiteLLM")


class LiteLLMModel(BaseLLM):
    def __init__(self, model: str, **kwargs):
        self._model = model
        self._kwargs = kwargs

    async def call(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return await self._do_call(messages, **kwargs)

    async def call_with_history(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> LLMResponse:
        return await self._do_call(messages, **kwargs)

    async def _do_call(self, messages: list[dict[str, Any]], **kwargs) -> LLMResponse:
        merged_kwargs = {**self._kwargs, **kwargs}

        if is_debug():
            _log.debug(f"Request: model={self._model}, messages={len(messages)}")

        resp = await litellm.acompletion(model=self._model, messages=messages, **merged_kwargs)
        choice = resp.choices[0].message

        tool_calls = None
        if choice.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    name=tc.function.name,
                    arguments=_parse_args(tc.function.arguments),
                )
                for tc in choice.tool_calls
            ]

        usage = resp.usage.model_dump() if resp.usage else {}
        cost = litellm.completion_cost(completion_response=resp)

        if is_debug():
            tokens = f"in={usage.get('prompt_tokens', 0)}, out={usage.get('completion_tokens', 0)}"
            _log.info(f"Response: {tokens}, cost=${cost:.6f}", color=Color.MAGENTA)

        return LLMResponse(
            content=choice.content or "",
            reasoning_content=getattr(choice, "reasoning_content", None),
            tool_calls=tool_calls,
            usage={**usage, "cost": cost},
        )


def _parse_args(arguments: str) -> dict[str, Any]:
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {"raw": arguments}

