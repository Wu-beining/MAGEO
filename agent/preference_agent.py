from __future__ import annotations

import json
import re
from typing import Any

from agent.base import BaseAgent
from debug.log import Color
from memory.base import BaseMemory
from model.base import BaseLLM
from model.schema import Message
from prompt import preference_system_prompt, preference_user_prompt


class PreferenceAgent(BaseAgent):
    """Construct a reusable engine preference profile from raw rules."""

    def __init__(
        self,
        model: BaseLLM,
        system_prompt: str | None = None,
        ensure_json: bool = True,
        memory: BaseMemory | None = None,
    ):
        super().__init__(model, system_prompt or preference_system_prompt(), memory)
        self._ensure_json = ensure_json

    async def run(self, engine_id: str, engine_rules: str) -> str:
        user_prompt = preference_user_prompt(engine_id=engine_id, engine_rules=engine_rules)

        resp = await self._model.call(
            user_prompt=user_prompt,
            system_prompt=self._system_prompt,
        )
        content = (resp.content or "").strip()

        if self._debug:
            self._log.info(f"Raw preference output: {content[:200]}...", color=Color.GRAY)

        if self._ensure_json:
            content = self._normalize_to_json(content, engine_id, engine_rules)

        self.add_message(Message.user(engine_id))
        self.add_message(Message.assistant(content))
        return content

    def _normalize_to_json(self, text: str, engine_id: str, engine_rules: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned, count=1).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        data: Any | None = None
        try:
            data = json.loads(cleaned)
        except Exception:
            data = None

        if not isinstance(data, dict):
            lines = [
                line.strip("-• \t")
                for line in engine_rules.splitlines()
                if line.strip()
            ]
            data = {
                "engine_id": engine_id,
                "preference_profile": {
                    "format_preferences": lines[:2],
                    "content_preferences": lines[2:4],
                    "risk_constraints": lines[4:6],
                    "style_preferences": lines[6:8],
                },
                "summary": engine_rules.strip(),
            }

        data.setdefault("engine_id", engine_id)
        data.setdefault("preference_profile", {})
        data.setdefault("summary", "")
        return json.dumps(data, ensure_ascii=False, indent=2)
