"""LLM provider abstraction.

Two implementations:
  * OpenAIProvider  - the real thing (Chat Completions + structured outputs).
  * MockProvider    - deterministic, offline stand-in used for tests, demos and
                      CI dry runs. It exercises the whole pipeline without an
                      API key. Its scores say nothing about real model quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


class Provider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_schema: dict | None = None,
        schema_name: str = "output",
    ) -> LLMResponse: ...


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, timeout_s: float = 60.0, max_retries: int = 4):
        from openai import AsyncOpenAI  # imported lazily so mock runs need no key

        # The SDK retries 408/409/429/5xx with exponential backoff + jitter.
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout_s,
            max_retries=max_retries,
        )

    async def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_schema: dict | None = None,
        schema_name: str = "output",
    ) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
            }
        t0 = time.perf_counter()
        resp = await self.client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            model=resp.model,
        )


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("charge", "charged", "invoice", "refund", "bill", "payment", "card",
                "subscription", "price", "cobro", "factura", "rechnung", "vat", "receipt",
                "money", "$", "€", "plan", "trial"),
    "technical": ("error", "crash", "bug", "broken", "not working", "load", "sync",
                  "api", "app", "500", "slow", "freez", "upload", "export", "integration",
                  "webhook", "timeout", "doesnt work", "doesn't work", "fehler"),
    "account": ("password", "login", "log in", "locked", "2fa", "two-factor", "email address",
                "username", "delete my account", "close my account", "sso", "profile",
                "verify", "verification", "contraseña", "konto", "access", "owner"),
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _stable_rand(*parts: str) -> random.Random:
    seed = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(seed[:16], 16))


class MockProvider:
    """role='classifier' answers like the support classifier; role='judge'
    answers like the LLM judge. Set LLMREG_MOCK_ERROR_RATE (0-1) to inject
    deterministic, prompt-dependent misclassifications so different prompt
    versions produce different diffs."""

    def __init__(self, role: str, error_rate: float | None = None):
        if role not in ("classifier", "judge"):
            raise ValueError(role)
        self.role = role
        self.error_rate = (
            float(os.environ.get("LLMREG_MOCK_ERROR_RATE") or "0") if error_rate is None else error_rate
        )

    async def complete(self, messages, *, model, temperature, max_tokens, json_schema=None,
                       schema_name="output") -> LLMResponse:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        rng = _stable_rand(self.role, system, user)
        if self.role == "classifier":
            text = self._classify(system, user, rng)
        else:
            text = self._judge(user)
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text) // 4,
            latency_ms=round(rng.uniform(250, 900), 1),
            model=f"mock-{model}",
        )

    def _classify(self, system: str, email: str, rng: random.Random) -> str:
        low = email.lower()
        scores = {c: sum(low.count(k) for k in kws) for c, kws in _KEYWORDS.items()}
        best = max(scores, key=scores.get)
        category = best if scores[best] > 0 else "general"
        if rng.random() < self.error_rate:
            category = rng.choice([c for c in ("billing", "technical", "account", "general")
                                   if c != category])
        body = re.sub(r"^subject:.*\n", "", email, flags=re.I).strip()
        first = re.split(r"(?<=[.!?])\s+", body)[0] if body else "Customer sent an empty message."
        summary = ("Customer writes: " + first)[:200]
        return json.dumps({"category": category, "summary": summary})

    def _judge(self, user: str) -> str:
        ref = re.search(r"<reference_summary>(.*?)</reference_summary>", user, re.S)
        cand = re.search(r"<candidate_summary>(.*?)</candidate_summary>", user, re.S)
        r, c = _tokens(ref.group(1) if ref else ""), _tokens(cand.group(1) if cand else "")
        overlap = len(r & c) / max(len(r), 1)
        score = 1 + min(4, int(overlap * 8))
        return json.dumps({"reason": f"mock token overlap {overlap:.2f}", "score": score})


def build_provider(kind: str, role: str, timeout_s: float = 60.0, max_retries: int = 4) -> Provider:
    if kind == "mock":
        return MockProvider(role=role)
    if kind == "openai":
        return OpenAIProvider(timeout_s=timeout_s, max_retries=max_retries)
    raise ValueError(f"unknown provider: {kind}")
