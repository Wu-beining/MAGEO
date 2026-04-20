"""
PlannerAgent for MAGEO optimization system.

Generates structured revision plans based on:
- User query
- Current document version
- Engine preference rules
- Retrieved memory examples
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.base import BaseAgent
from debug.log import Color
from memory.base import BaseMemory
from model.base import BaseLLM
from model.schema import Message
from prompt import planner_system_prompt, planner_user_prompt


class PlannerAgent(BaseAgent):
    """
    PlannerAgent - Generates revision plans for content optimization.

    Purpose:
      - Analyze current content against engine preferences
      - Generate structured revision plan steps
      - Leverage memory examples for pattern-based planning

    Output schema (JSON string):
      {
        "plan_steps": [
          {
            "step_id": str,
            "target_span": str,
            "edit_type": str,  # Structure/Evidence/Safety/Style
            "target_metrics": [str],
            "risk_constraints": [str],
            "rationale": str,
            "suggested_operations": [str],
            "inspired_by_examples": [str]  # optional
          },
          ...
        ]
      }
    """

    def __init__(
        self,
        model: BaseLLM,
        system_prompt: str | None = None,
        ensure_json: bool = True,
        memory: BaseMemory | None = None,
    ):
        """
        Args:
          model: LLM model implementation
          system_prompt: Optional override; defaults to planner_system_prompt()
          ensure_json: If True, normalizes output to strict JSON schema
          memory: Optional memory; not required for single-turn planning
        """
        super().__init__(model, system_prompt or planner_system_prompt(), memory)
        self._ensure_json = ensure_json

    async def run(
        self,
        query: str,
        document: str,
        engine_rules: str,
        memory_examples: str,
    ) -> str:
        """
        Execute a single planning turn.

        Args:
          query: User query to optimize for
          document: Current document content with span annotations
          engine_rules: Target engine preference rules
          memory_examples: Retrieved memory examples (JSON string)

        Returns:
          JSON string with plan_steps array.
        """
        # Build prompts
        user_prompt = planner_user_prompt(
            query=query,
            document_with_spans=document,
            engin_rules=engine_rules,
            retrieved_memory_example=memory_examples,
        )

        if self._debug:
            self._log.debug(f"User (planner): query={query[:50]}...")

        # Single call
        resp = await self._model.call(
            user_prompt=user_prompt,
            system_prompt=self._system_prompt,
        )

        content = (resp.content or "").strip()

        if self._debug:
            self._log.info(f"Raw planner output: {content[:200]}...", color=Color.GRAY)

        # Normalize to strict JSON
        if self._ensure_json:
            content = self._normalize_to_json(content)

        # Save into history
        self.add_message(Message.user(f"Query: {query}"))
        self.add_message(Message.assistant(content))

        if self._debug:
            self._log.info(f"Final plan JSON: {content[:200]}...", color=Color.CYAN)

        return content

    # ----------------------------
    # JSON normalization utilities
    # ----------------------------

    def _normalize_to_json(self, text: str) -> str:
        """
        Coerce model output into the required JSON schema:
          { "plan_steps": [...] }
        """
        cleaned = self._strip_code_fences(text)
        payload = self._extract_outermost_braces(cleaned)

        # Try parse
        data: Any | None = None
        if payload:
            data = self._try_json_parse(payload)

        # Fallback: try to find the plan_steps array
        if data is None:
            data = self._try_extract_plan_steps(cleaned)

        # Build canonical structure
        result = self._coerce_to_schema(data if data is not None else text)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _strip_code_fences(self, s: str) -> str:
        """Remove Markdown code fences."""
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s, count=1).strip()
            if s.endswith("```"):
                s = s[:-3].strip()
        return s

    def _extract_outermost_braces(self, s: str) -> str | None:
        """Extract from first '{' to last '}'."""
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1].strip()
        return None

    def _try_json_parse(self, s: str) -> Any | None:
        try:
            return json.loads(s)
        except Exception:
            # Try relaxed: single quotes to double
            if ("{" in s and "}" in s) or ("[" in s and "]" in s):
                relaxed = s.replace("'", '"')
                try:
                    return json.loads(relaxed)
                except Exception:
                    return None
            return None

    def _try_extract_plan_steps(self, s: str) -> dict[str, Any]:
        """Try to extract plan_steps array and wrap it."""
        # Try to find "plan_steps" key
        match = re.search(r'"plan_steps"\s*:\s*\[', s)
        if match:
            start = match.start()
            # Find matching bracket
            depth = 0
            in_string = False
            escape = False
            for i in range(match.start() + len(match.group()), len(s)):
                c = s[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "[" or c == "{":
                    depth += 1
                elif c == "]" or c == "}":
                    depth -= 1
                    if depth == 0:
                        # Found the end
                        json_str = s[start : i + 1]
                        parsed = self._try_json_parse(json_str)
                        if parsed and isinstance(parsed, dict):
                            return parsed
                        break
        return {"plan_steps": []}

    def _coerce_to_schema(self, data: Any) -> dict[str, Any]:
        """Coerce arbitrary data into {plan_steps: [...]} schema."""
        plan_steps: list[dict[str, Any]] = []

        if isinstance(data, dict):
            raw_steps = data.get("plan_steps", [])
            if isinstance(raw_steps, list):
                for item in raw_steps:
                    if isinstance(item, dict):
                        plan_steps.append(self._coerce_plan_step(item))

        elif isinstance(data, list):
            # Treat list as plan_steps
            for item in data:
                if isinstance(item, dict):
                    plan_steps.append(self._coerce_plan_step(item))

        return {"plan_steps": plan_steps}

    def _coerce_plan_step(self, item: dict[str, Any]) -> dict[str, Any]:
        """Coerce a single plan step to required schema."""
        result: dict[str, Any] = {}

        # step_id
        result["step_id"] = self._first_non_empty_str(
            item, keys=["step_id", "id", "step", "index"]
        ) or "step_1"

        # target_span
        result["target_span"] = self._first_non_empty_str(
            item, keys=["target_span", "span", "section", "paragraph"]
        ) or "unknown"

        # edit_type
        valid_types = ["Structure", "Evidence", "Safety", "Style", "Formatting"]
        edit_type = self._first_non_empty_str(
            item, keys=["edit_type", "type", "edit", "category"]
        )
        if edit_type in valid_types:
            result["edit_type"] = edit_type
        else:
            result["edit_type"] = "Structure"  # default

        # target_metrics (list)
        result["target_metrics"] = self._coerce_to_str_list(
            item.get("target_metrics") or item.get("metrics") or []
        )

        # risk_constraints (list)
        result["risk_constraints"] = self._coerce_to_str_list(
            item.get("risk_constraints") or item.get("constraints") or []
        )

        # rationale
        result["rationale"] = self._first_non_empty_str(
            item, keys=["rationale", "reason", "why", "explanation", "description"]
        ) or ""

        # suggested_operations (list)
        result["suggested_operations"] = self._coerce_to_str_list(
            item.get("suggested_operations") or item.get("operations") or item.get("ops") or []
        )

        # inspired_by_examples (optional)
        inspired = item.get("inspired_by_examples") or item.get("examples") or []
        if inspired:
            result["inspired_by_examples"] = self._coerce_to_str_list(inspired)

        return result

    def _first_non_empty_str(self, obj: dict[str, Any], keys: list[str]) -> str:
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _coerce_to_str_list(self, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [
                str(x).strip()
                for x in v
                if isinstance(x, (str, int, float)) and str(x).strip()
            ]
        if isinstance(v, str):
            # Split by common delimiters
            parts = re.split(r"[,\n;，；]+", v)
            return [p.strip() for p in parts if p.strip()]
        return []
