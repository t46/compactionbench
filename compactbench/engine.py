"""LM engine: OpenAI-compatible chat client with token accounting and a hard call budget.

Default backend is local ollama (free). Any OpenAI-compatible endpoint works via
base_url/api_key. Every call is counted; exceeding max_calls raises BudgetExceeded
so an experiment can never silently overspend. Adapted from bet-0002/cpo/engine.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

OLLAMA_BASE = "http://localhost:11434/v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class Engine:
    model: str = "qwen2.5:3b-instruct"
    base_url: str = OLLAMA_BASE
    api_key: str | None = None
    max_calls: int = 5000
    temperature: float = 0.0
    max_tokens: int = 64
    seed: int | None = None
    # ollama: extend context window so long prompts are not silently truncated.
    num_ctx: int | None = 8192
    extra_body: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    timeout: float = 600.0

    def chat(self, messages: list[dict], **overrides) -> str:
        if self.usage.calls >= self.max_calls:
            raise BudgetExceeded(f"engine call budget exhausted ({self.max_calls})")
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            **self.extra_body,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.num_ctx is not None:
            # ollama-specific knob, passed through the OpenAI-compat shim.
            body.setdefault("options", {})["num_ctx"] = self.num_ctx
        headers = {}
        key = self.api_key or os.environ.get("CB_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        self.usage.calls += 1
        u = data.get("usage") or {}
        self.usage.prompt_tokens += u.get("prompt_tokens", 0)
        self.usage.completion_tokens += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]


def strip_think(text: str) -> str:
    """Remove a leading <think>...</think> block (reasoning-model output)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()
