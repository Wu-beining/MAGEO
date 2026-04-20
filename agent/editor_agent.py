"""
EditorAgent for MAGEO optimization system.

Generates K candidate document versions based on:
- Current document version
- Planner's revision plan
- Engine preference rules

Implements four expert roles:
1. Structure Editor - Structure and flow
2. Evidence Editor - Evidence and citations
3. Risk/Safety Editor - Factuality and safety
4. Style Editor - Tone and style
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
from prompt import editor_system_prompt, editor_user_prompt


class EditorAgent(BaseAgent):
    """
    EditorAgent - Generates candidate document versions.

    Purpose:
      - Generate K different candidate versions (e.g., K=2 or 3)
      - Each candidate has a different optimization focus
      - Track applied edit operations for each candidate

    Output schema (JSON string):
      {
        "candidates": [
          {
            "candidate_id": str,  # "V1", "V2", etc.
            "description": str,
            "applied_edit_ops": [
              {
                "edit_type": str,
                "target_span": str,
                "op_pattern": str
              },
              ...
            ],
            "revised_content": str
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
          system_prompt: Optional override; defaults to editor_system_prompt()
          ensure_json: If True, normalizes output to strict JSON schema
          memory: Optional memory; not required for single-turn editing
        """
        super().__init__(model, system_prompt or editor_system_prompt(), memory)
        self._ensure_json = ensure_json

    async def run(
        self,
        document: str,
        revision_plan: str,
        engine_rules: str,
        k: int = 2,
    ) -> str:
        """
        Execute a single editing turn.

        Args:
          document: Current document content with span annotations
          revision_plan: Planner's revision plan (JSON string)
          engine_rules: Target engine preference rules
          k: Number of candidate versions to generate

        Returns:
          JSON string with candidates array.
        """
        # Build prompts
        user_prompt = editor_user_prompt(
            document_with_spans=document,
            revision_plan=revision_plan,
            engine_rules=engine_rules,
            k=k,
        )

        if self._debug:
            self._log.debug(f"User (editor): k={k}, doc length={len(document)}")

        # Single call
        resp = await self._model.call(
            user_prompt=user_prompt,
            system_prompt=self._system_prompt,
        )

        content = (resp.content or "").strip()

        if self._debug:
            self._log.info(f"Raw editor output: {content[:200]}...", color=Color.GRAY)

        # Normalize to strict JSON
        if self._ensure_json:
            content = self._normalize_to_json(content)

        # Save into history
        self.add_message(Message.user(f"Generate {k} candidates"))
        self.add_message(Message.assistant(content))

        if self._debug:
            self._log.info(f"Final editor JSON: {content[:200]}...", color=Color.CYAN)

        return content

    # ----------------------------
    # JSON normalization utilities
    # ----------------------------

    def _normalize_to_json(self, text: str) -> str:
        """
        Coerce model output into the required JSON schema:
          { "candidates": [...] }
        """
        cleaned = self._strip_code_fences(text)
        payload = self._extract_outermost_braces(cleaned)

        # Try parse
        data: Any | None = None
        if payload:
            data = self._try_json_parse(payload)

        # Fallback: try to find the candidates array
        if data is None:
            data = self._try_extract_candidates(cleaned)

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

    def _try_extract_candidates(self, s: str) -> dict[str, Any]:
        """Try to extract candidates array and wrap it."""
        # Try to find "candidates" key
        match = re.search(r'"candidates"\s*:\s*\[', s)
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
        return {"candidates": []}

    def _coerce_to_schema(self, data: Any) -> dict[str, Any]:
        """Coerce arbitrary data into {candidates: [...]} schema."""
        candidates: list[dict[str, Any]] = []

        if isinstance(data, dict):
            raw_candidates = data.get("candidates", [])
            if isinstance(raw_candidates, list):
                for item in raw_candidates:
                    if isinstance(item, dict):
                        candidates.append(self._coerce_candidate(item))

        elif isinstance(data, list):
            # Treat list as candidates
            for item in data:
                if isinstance(item, dict):
                    candidates.append(self._coerce_candidate(item))

        return {"candidates": candidates}

    def _coerce_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        """Coerce a single candidate to required schema."""
        result: dict[str, Any] = {}

        # candidate_id
        result["candidate_id"] = self._first_non_empty_str(
            item, keys=["candidate_id", "id", "candidate", "version", "name"]
        ) or "V1"

        # description
        result["description"] = self._first_non_empty_str(
            item, keys=["description", "desc", "summary", "focus", "strategy"]
        ) or ""

        # revised_content
        result["revised_content"] = self._first_non_empty_str(
            item,
            keys=[
                "revised_content",
                "content",
                "text",
                "document",
                "article",
                "revised_document",
            ],
        )

        # applied_edit_ops
        raw_ops = item.get("applied_edit_ops") or item.get("edit_ops") or item.get("ops") or []
        if isinstance(raw_ops, list):
            result["applied_edit_ops"] = [
                self._coerce_edit_op(op) if isinstance(op, dict) else {}
                for op in raw_ops
            ]
        else:
            result["applied_edit_ops"] = []

        return result

    def _coerce_edit_op(self, item: dict[str, Any]) -> dict[str, Any]:
        """Coerce an edit operation to required schema."""
        result: dict[str, Any] = {}

        # edit_type
        valid_types = ["Structure", "Evidence", "Safety", "Style", "Formatting"]
        edit_type = self._first_non_empty_str(
            item, keys=["edit_type", "type", "edit", "category"]
        )
        if edit_type in valid_types:
            result["edit_type"] = edit_type
        else:
            result["edit_type"] = "Structure"  # default

        # target_span
        result["target_span"] = self._first_non_empty_str(
            item, keys=["target_span", "span", "section", "paragraph", "location"]
        ) or "unknown"

        # op_pattern
        result["op_pattern"] = self._first_non_empty_str(
            item, keys=["op_pattern", "pattern", "operation", "op"]
        ) or ""

        return result

    def _first_non_empty_str(self, obj: dict[str, Any], keys: list[str]) -> str:
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
