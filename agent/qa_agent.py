from __future__ import annotations

from agent.base import BaseAgent
from debug.log import Color
from memory.base import BaseMemory
from model.base import BaseLLM
from model.schema import Message
from prompt import qa_system_prompt, qa_user_prompt


class QAAgent(BaseAgent):
    """
    QAAgent - RAG-based Question Answering Agent

    Purpose:
      - Answer user questions based on provided documents using prompt templates.
      - Leverages a single model call with a system prompt and a templated user prompt.
      - Returns a text response with proper document citations.

    Output:
      - Plain text answer with citations in the format [1], [2], etc.
    """

    def __init__(
        self,
        model: BaseLLM,
        system_prompt: str | None = None,
        memory: BaseMemory | None = None,
    ):
        """
        Args:
          model: LLM model implementation
          system_prompt: Optional override for the system prompt; defaults to qa_system_prompt()
          memory: Optional memory implementation; not required for single-turn QA
        """
        super().__init__(model, system_prompt or qa_system_prompt(), memory)

    async def run(self, user_query: str, documents: str) -> str:
        """
        Execute a single QA turn.

        Args:
          user_query: User's question
          documents: Search results / context documents

        Returns:
          Text answer with document citations.
        """
        # Build prompts
        user_prompt = qa_user_prompt(user_query, documents)

        if self._debug:
            self._log.debug(f"User (QA): {user_query}")
            self._log.debug(f"Documents length: {len(documents)} chars")

        # Single call; no tool use, no multi-turn
        resp = await self._model.call(
            user_prompt=user_prompt,
            system_prompt=self._system_prompt,
        )

        content = (resp.content or "").strip()

        if self._debug:
            self._log.info(f"Raw QA output: {content}", color=Color.GRAY)

        # Save into history as a single turn
        self.add_message(Message.user(user_query))
        self.add_message(Message.assistant(content))

        if self._debug:
            self._log.info(f"Final QA answer: {content}", color=Color.CYAN)

        return content
