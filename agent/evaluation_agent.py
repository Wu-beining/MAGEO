from __future__ import annotations

import json
import re
from typing import Any

from agent.base import BaseAgent
from debug.log import Color
from memory.base import BaseMemory
from model.base import BaseLLM
from model.schema import Message
from prompt import evaluation_system_prompt, evaluation_user_prompt


class EvaluationAgent(BaseAgent):
    """
    EvaluationAgent - MAGEO DSV-CF Evaluator

    Purpose:
      - Evaluate candidate content versions using the paper's DSV-CF framework.
      - Predict scores across the paper-aligned 8 dimensions on two axes: SSV and ISI.
      - Provide three-critic commentary: Metric, Safety, and Preference.

    Output schema (JSON string):
      {
        "evaluations": [
          {
            "candidate_id": str,
            "predicted_scores": { 8 dimensions, each 1-10 },
            "metric_critic_comment": str,
            "safety_critic_comment": str,
            "preference_critic_comment": str,
            "overall_comment": str
          },
          ...
        ]
      }

    The 8 score dimensions:
      - wlv, dpa, cp, si
      - aa, fa, kc, ad
    """

    # Required score fields
    SCORE_FIELDS = [
        "wlv",
        "dpa",
        "cp",
        "si",
        "aa",
        "fa",
        "kc",
        "ad",
    ]

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
          system_prompt: Optional override; defaults to evaluation_system_prompt()
          ensure_json: If True, normalizes output to strict JSON schema
          memory: Optional memory; not required for single-turn evaluation
        """
        super().__init__(model, system_prompt or evaluation_system_prompt(), memory)
        self._ensure_json = ensure_json

    async def run(
        self,
        user_query: str,
        baseline_content: str,
        candidates: str,
        engine_rules: str,
    ) -> str:
        """
        Execute a single evaluation turn.

        Args:
          user_query: The user's query
          baseline_content: Current baseline version (original or previous iteration)
          candidates: Candidate versions with candidate_id, revised_content, applied_edit_ops
          engine_rules: Target engine preference rules

        Returns:
          JSON string with evaluations array.
        """
        # Build prompts
        user_prompt = evaluation_user_prompt(
            user_query=user_query,
            baseline_content=baseline_content,
            candidates=candidates,
            engine_rules=engine_rules,
        )

        if self._debug:
            self._log.debug(f"User (eval): query={user_query[:50]}...")
            self._log.debug(f"Candidates input length: {len(candidates)} chars")

        # Single call
        resp = await self._model.call(
            user_prompt=user_prompt,
            system_prompt=self._system_prompt,
        )

        content = (resp.content or "").strip()

        if self._debug:
            self._log.info(f"Raw eval output: {content[:200]}...", color=Color.GRAY)

        # Normalize to strict JSON
        if self._ensure_json:
            content = self._normalize_to_json(content)

        # Save into history
        self.add_message(Message.user(f"Query: {user_query}"))
        self.add_message(Message.assistant(content))

        if self._debug:
            self._log.info(f"Final eval JSON: {content[:200]}...", color=Color.CYAN)

        return content

    # ----------------------------
    # JSON normalization utilities
    # ----------------------------

    def _normalize_to_json(self, text: str) -> str:
        """
        Coerce model output into the required JSON schema:
          { "evaluations": [...] }
        """
        cleaned = self._strip_code_fences(text)
        payload = self._extract_outermost_braces(cleaned)

        # Try parse
        data: Any | None = None
        if payload:
            data = self._try_json_parse(payload)

        # Fallback: try to find the evaluations array
        if data is None:
            data = self._try_extract_evaluations(cleaned)

        # Build canonical structure
        result = self._coerce_to_schema(data if data is not None else text)
        return json.dumps(result, ensure_ascii=False)

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

    def _try_extract_evaluations(self, s: str) -> dict[str, Any]:
        """Try to extract evaluations array and wrap it."""
        # Try to find "evaluations" key
        match = re.search(r'"evaluations"\s*:\s*\[', s)
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
        return {"evaluations": []}

    def _coerce_to_schema(self, data: Any) -> dict[str, Any]:
        """Coerce arbitrary data into {evaluations: [...]} schema."""
        evaluations: list[dict[str, Any]] = []

        if isinstance(data, dict):
            raw_evals = data.get("evaluations", [])
            if isinstance(raw_evals, list):
                for item in raw_evals:
                    if isinstance(item, dict):
                        evaluations.append(self._coerce_evaluation_item(item))

        elif isinstance(data, list):
            # Treat list as evaluations
            for item in data:
                if isinstance(item, dict):
                    evaluations.append(self._coerce_evaluation_item(item))

        # Ensure minimal structure
        return {"evaluations": evaluations}

    def _coerce_evaluation_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Coerce a single evaluation item to required schema."""
        result: dict[str, Any] = {}

        # candidate_id
        result["candidate_id"] = self._first_non_empty_str(
            item, keys=["candidate_id", "id", "candidate", "version"]
        ) or "unknown"

        # predicted_scores
        scores = self._extract_scores(item)
        result["predicted_scores"] = scores

        # Comments
        result["metric_critic_comment"] = self._first_non_empty_str(
            item,
            keys=[
                "metric_critic_comment",
                "metric_comment",
                "metric",
            ],
        ) or ""
        result["safety_critic_comment"] = self._first_non_empty_str(
            item,
            keys=[
                "safety_critic_comment",
                "safety_comment",
                "safety",
            ],
        ) or ""
        result["preference_critic_comment"] = self._first_non_empty_str(
            item,
            keys=[
                "preference_critic_comment",
                "preference_comment",
                "preference",
            ],
        ) or ""
        result["overall_comment"] = self._first_non_empty_str(
            item,
            keys=["overall_comment", "overall", "comment", "summary"],
        ) or ""

        return result

    def _extract_scores(self, item: dict[str, Any]) -> dict[str, Any]:
        """Extract and normalize predicted scores."""
        raw_scores = item.get("predicted_scores", {})
        if not isinstance(raw_scores, dict):
            raw_scores = item

        scores: dict[str, Any] = {}
        for field in self.SCORE_FIELDS:
            value = raw_scores.get(field)
            # Normalize to float 1-10
            if value is None:
                # Check alternate keys
                for alt_key in [field, f"{field}_score"]:
                    if alt_key in item:
                        value = item[alt_key]
                        break

            if isinstance(value, (int, float)):
                scores[field] = max(1.0, min(10.0, float(value)))
            else:
                scores[field] = 5.0  # default middle score

        return scores

    def _first_non_empty_str(self, obj: dict[str, Any], keys: list[str]) -> str:
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
