"""The LLM feature under test: a customer-support email classifier.

This module is deliberately the only place that knows how the feature builds
its request. Production code and the eval pipeline call the same function, so
the eval measures what actually ships.
"""

from __future__ import annotations

import asyncio
import json

from pydantic import ValidationError

from .config import CLASSIFICATION_JSON_SCHEMA, ClassificationResult, PromptConfig
from .providers import LLMResponse, Provider, build_provider


def build_messages(prompt: PromptConfig, email: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": prompt.system_prompt}]
    for ex in prompt.few_shot:
        messages.append({"role": "user", "content": ex.email})
        messages.append(
            {"role": "assistant",
             "content": json.dumps({"category": ex.category, "summary": ex.summary})}
        )
    messages.append({"role": "user", "content": email})
    return messages


def parse_output(text: str) -> ClassificationResult:
    return ClassificationResult.model_validate(json.loads(text))


async def aclassify_email(
    email: str, prompt: PromptConfig, provider: Provider
) -> tuple[ClassificationResult | None, LLMResponse, str | None]:
    """Returns (result, raw_response, error). A schema violation is a test
    failure, not a crash: it is exactly the kind of regression we want to see."""
    resp = await provider.complete(
        build_messages(prompt, email),
        model=prompt.model.name,
        temperature=prompt.model.temperature,
        max_tokens=prompt.model.max_tokens,
        json_schema=CLASSIFICATION_JSON_SCHEMA,
        schema_name="support_email_classification",
    )
    try:
        return parse_output(resp.text), resp, None
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, resp, f"invalid output: {type(exc).__name__}: {str(exc)[:200]}"


def classify_email(email: str, prompt: PromptConfig, provider: Provider | None = None) -> ClassificationResult:
    """Synchronous entry point for application code."""
    provider = provider or build_provider(prompt.model.provider, role="classifier")
    result, _, error = asyncio.run(aclassify_email(email, prompt, provider))
    if result is None:
        raise ValueError(error)
    return result
