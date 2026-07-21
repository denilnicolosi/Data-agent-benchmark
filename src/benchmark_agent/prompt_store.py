"""Centralized prompts: Langfuse Prompt Management with local fallback.

Opt-in via LANGFUSE_PROMPTS=1 (requires the `langfuse` package and the same
LANGFUSE_* keys Aegra already uses for tracing, see .env.example). The
authoritative fallbacks live in prompts.py: with the flag off, the package
missing, the prompt not seeded, or Langfuse unreachable: the local text is
used and the repo works fully offline.

Prompts use Langfuse's native {{mustache}} variables (fully recognized by
the Playground and UI experiments). get_prompt() returns the COMPILED text:
via the SDK's prompt.compile() when the text comes from Langfuse, via an
equivalent single-pass local substitution otherwise. A prompt's `config` may
carry {"model": "provider:model", "max_tokens": int} to centralize the LLM
choice too (anything missing falls back to the LLM_MODEL / LLM_MAX_TOKENS
env vars).
"""

import os
import re

from benchmark_agent.prompts import (
    PANDAS_PROMPT, DASK_PROMPT, NUMPY_PROMPT, POLARS_PROMPT, REPORT_COMMENTARY_PROMPT,
)

LOCAL_PROMPTS = {
    "benchmark/pandas": PANDAS_PROMPT,
    "benchmark/dask": DASK_PROMPT,
    "benchmark/numpy": NUMPY_PROMPT,
    "benchmark/polars": POLARS_PROMPT,
    "benchmark/commentary": REPORT_COMMENTARY_PROMPT,
}

_client = None

_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def compile_local(text: str, variables: dict) -> str:
    """Mustache-equivalent substitution for the fallback path.

    Single-pass regex (like the SDK's compile): inserted values are never
    re-scanned, so code containing literal braces can't corrupt the result.
    Unknown variables are left as-is, matching Langfuse behavior.
    """
    return _VAR_RE.sub(lambda m: str(variables.get(m.group(1), m.group(0))), text)


def _enabled() -> bool:
    return (
        os.getenv("LANGFUSE_PROMPTS", "0").lower() in ("1", "true", "yes")
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    )


def _langfuse():
    global _client
    if _client is None:
        from langfuse import Langfuse
        kwargs = {}
        # Aegra's tracing config names the host LANGFUSE_BASE_URL; the SDK's
        # own env var is LANGFUSE_HOST, accept both.
        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
        if host:
            kwargs["host"] = host
        _client = Langfuse(**kwargs)
    return _client


def get_prompt(name: str, variables: dict = None) -> tuple[str, dict, dict]:
    """Returns (compiled_text, config, trace_metadata) for a prompt.

    trace_metadata carries {"prompt_name", "prompt_version"} only when the
    text really came from Langfuse; the caller puts it in the LLM call's
    metadata, which is what links a trace to the prompt version that produced
    it (the OTEL-side tracing has no automatic prompt linkage).

    Blocking (SDK client cache + HTTP on cache miss): call it via
    asyncio.to_thread from async nodes.
    """
    variables = variables or {}
    local = LOCAL_PROMPTS[name]
    if not _enabled():
        return compile_local(local, variables), {}, {}
    try:
        prompt = _langfuse().get_prompt(
            name,
            label=os.getenv("LANGFUSE_PROMPT_LABEL", "production"),
            type="text",
            fallback=local,
        )
        if getattr(prompt, "is_fallback", False):
            return compile_local(local, variables), {}, {}
        return prompt.compile(**variables), dict(prompt.config or {}), {
            "prompt_name": name,
            "prompt_version": prompt.version,
        }
    except Exception as e:
        print(f"Warning: Langfuse prompt fetch failed for '{name}' ({e}); using the local fallback.")
        return compile_local(local, variables), {}, {}
