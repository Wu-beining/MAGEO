from __future__ import annotations

import json
import re
from typing import Any, Iterable

from agent.base import BaseAgent
from debug.log import Color
from memory.base import BaseMemory
from model.base import BaseLLM
from model.schema import Message
from prompt import query_rewriter_system_prompt, query_rewriter_user_prompt


class QueryRewriteAgent(BaseAgent):
    """
    QueryRewriteAgent

    Purpose:
      - Rewrite a user natural-language query into search-friendly JSON using prompt templates.
      - Leverages a single model call with a system prompt and a templated user prompt.
      - Optionally coerces/normalizes the model output to a strict JSON structure.

    Output schema (stringified JSON):
      {
        "main_query": "<string>",
        "alternative_queries": ["<string>", ...]
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
          system_prompt: Optional override for the system prompt; defaults to query_rewriter_system_prompt()
          ensure_json: If True, attempts to coerce/normalize model output into the required JSON schema
          memory: Optional memory implementation; not required for single-turn rewrite
        """
        super().__init__(model, system_prompt or query_rewriter_system_prompt(), memory)
        self._ensure_json = ensure_json

    async def run(self, user_input: str) -> str:
        """
        Execute a single rewrite turn.

        Args:
          user_input: Raw user query

        Returns:
          JSON string matching the required schema.
        """
        # Build prompts
        user_prompt = query_rewriter_user_prompt(user_input)

        if self._debug:
            self._log.debug(f"User (rewrite): {user_input}")

        # Single call; no tool use, no multi-turn
        resp = await self._model.call(
            user_prompt=user_prompt,
            system_prompt=self._system_prompt,
        )

        content = (resp.content or "").strip()

        if self._debug:
            self._log.info(f"Raw rewrite output: {content}", color=Color.GRAY)

        # Optionally normalize to strict JSON
        if self._ensure_json:
            content = self._normalize_to_required_json(content)

        # Save into history as a single turn (optional, but consistent with BaseAgent usage)
        self.add_message(Message.user(user_input))
        self.add_message(Message.assistant(content))

        if self._debug:
            self._log.info(f"Final rewrite JSON: {content}", color=Color.CYAN)

        return content

    # ----------------------------
    # JSON normalization utilities
    # ----------------------------

    def _normalize_to_required_json(self, text: str) -> str:
        """
        Attempt to coerce model output into the required JSON schema:
          - Strip code fences
          - Extract the first top-level JSON object if present
          - Parse and map likely keys to the canonical schema
          - Fall back to using the raw text as main_query

        Returns:
          A valid JSON string with keys: main_query (str), alternative_queries (list[str])
        """
        cleaned = self._strip_code_fences(text)
        payload = self._extract_json_object(cleaned)

        # Try direct parse
        data: Any | None = None
        if payload:
            data = self._try_json_parse(payload)

        # If parse failed, try more lenient approaches
        if data is None:
            # If the model returned a JSON-like list, treat it as [main, *alts]
            if cleaned.startswith("[") and cleaned.endswith("]"):
                data = self._try_json_parse(cleaned)

        # Build canonical structure
        result = self._coerce_to_schema(data if data is not None else cleaned)
        return json.dumps(result, ensure_ascii=False)

    def _strip_code_fences(self, s: str) -> str:
        """
        Remove Markdown code fences like:
          ```json
          { ... }
          ```
        """
        s = s.strip()
        # Remove leading triple backticks blocks
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s, count=1).strip()
            # Remove trailing ```
            if s.endswith("```"):
                s = s[:-3].strip()
        return s

    def _extract_json_object(self, s: str) -> str | None:
        """
        Extract the substring between the first '{' and the last '}'.
        Returns None if not found.
        """
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1].strip()
        return None

    def _try_json_parse(self, s: str) -> Any | None:
        try:
            return json.loads(s)
        except Exception:
            # Try a relaxed attempt: replace single quotes with double if it looks like JSON-ish
            if ("{" in s and "}" in s) or ("[" in s and "]" in s):
                relaxed = s.replace("'", '"')
                try:
                    return json.loads(relaxed)
                except Exception:
                    return None
            return None

    def _coerce_to_schema(self, data: Any) -> dict[str, Any]:
        """
        Map arbitrary parsed data into:
          {
            "main_query": str,
            "alternative_queries": list[str]
          }
        """
        main_query = ""
        alternative_queries: list[str] = []

        if isinstance(data, dict):
            # Try canonical keys first
            main_query = self._first_non_empty_value(
                data,
                keys=[
                    "main_query",
                    "query",
                    "rewritten_query",
                    "main",
                    "primary",
                    "q",
                ],
            )
            alternatives = self._first_value(
                data,
                keys=[
                    "alternative_queries",
                    "alternatives",
                    "related_queries",
                    "expanded_queries",
                    "optional_queries",
                    "alts",
                ],
            )
            alternative_queries = self._coerce_to_str_list(alternatives)

        elif isinstance(data, list):
            # e.g., ["main", "alt1", "alt2"]
            str_list = self._coerce_to_str_list(data)
            if str_list:
                main_query = str_list[0]
                alternative_queries = str_list[1:]

        elif isinstance(data, str):
            # If the model returned a plain string, treat it as main_query
            main_query = data.strip()

        # Final safety
        main_query = (main_query or "").strip()
        if not main_query:
            # As a last resort, provide a minimal valid schema
            main_query = ""
        alternative_queries = [
            q for q in alternative_queries if isinstance(q, str) and q.strip()
        ]
        return {
            "main_query": main_query,
            "alternative_queries": alternative_queries,
        }

    def _first_non_empty_value(self, obj: dict[str, Any], keys: Iterable[str]) -> str:
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _first_value(self, obj: dict[str, Any], keys: Iterable[str]) -> Any:
        for k in keys:
            if k in obj:
                return obj.get(k)
        return None

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
            # Split by common delimiters if it's a single string blob
            parts = re.split(r"[,\n;，；]+", v)
            return [p.strip() for p in parts if p.strip()]
        return []
