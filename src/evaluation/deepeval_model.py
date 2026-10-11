"""
The evaluator model DeepEval uses for its LLM-graded metrics.

It wraps the project's LangChain chat model (the judge settings: JUDGE_PROVIDER / JUDGE_MODEL,
default the assistant's provider and model), so it shares the rate limiter and keys. DeepEval
asks for JSON matching a Pydantic schema; like the router and the judge, the request is sent as
plain text with the schema appended and the reply is validated here (guided json_schema decoding
ran away to the token limit on gpt-oss). One retry with the validation error, then an error that
DeepEval records for the metric.
"""
import asyncio
import json
import logging
import re
from typing import Any, Optional, Type

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

import evaluation  # noqa: F401  (configures DeepEval before it is imported)
from deepeval.models import DeepEvalBaseLLM

logger = logging.getLogger(__name__)

JSON_INSTRUCTION = (
    "\n\nReply with ONLY one JSON object (no markdown fences, no text before or after) that matches "
    "this JSON schema:\n{schema}"
)


def extract_json(text: str) -> Any:
    """The JSON object in a reply (tolerates markdown fences and text around it)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("no JSON object in the reply")
        return json.loads(match.group(0))


class EvaluatorModel(DeepEvalBaseLLM):
    """DeepEval model backed by a LangChain chat model (default: the project's judge model)."""

    def __init__(self, chat_model=None, name: Optional[str] = None):
        self._chat_model = chat_model
        self._name = name
        super().__init__(model=name or "evaluator")

    def load_model(self):
        if self._chat_model is None:
            from agents.llm import get_judge_model, judge_model_settings

            self._chat_model = get_judge_model()
            settings = judge_model_settings()
            self._name = self._name or f"{settings['provider']}:{settings['model']}"
        return self._chat_model

    def get_model_name(self) -> str:
        return self._name or "evaluator"

    def _invoke(self, prompt: str) -> str:
        from llm.text import normalize_model_text

        reply = self.model.invoke([HumanMessage(prompt)])
        return normalize_model_text(getattr(reply, "text", None) or str(reply.content))

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        if schema is None:
            return self._invoke(prompt)
        request = prompt + JSON_INSTRUCTION.format(schema=json.dumps(schema.model_json_schema()))
        error = None
        for attempt in range(2):
            text = self._invoke(request if attempt == 0 else
                                f"{request}\n\nYour previous reply was invalid ({error}). Reply again with only the JSON object.")
            try:
                return schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as exc:
                error = str(exc).splitlines()[0][:200]
                logger.info(f"Evaluator reply not valid for {schema.__name__}: {error}")
        raise ValueError(f"The evaluator model did not return valid JSON for {schema.__name__}: {error}")

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def supports_structured_outputs(self) -> bool:
        return True  # generate() returns validated schema objects
