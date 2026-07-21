import asyncio
import json
import os
import re
from datetime import datetime
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from benchmark_agent.prompt_store import get_prompt
from benchmark_agent.docker_runner import run_in_docker

# The 4 benchmarked libraries. Prompts are fetched per-call through
# prompt_store.get_prompt("benchmark/<lib>"): Langfuse-managed when
# LANGFUSE_PROMPTS=1, local prompts.py otherwise.
LIBRARIES = ("pandas", "dask", "numpy", "polars")


# Max automatic fix regenerations per branch when approved code fails at
# runtime in Docker (manual "regenerate" requests from the reviewer are
# unlimited; this only caps the execution-error fix loop)
MAX_EXECUTION_FIXES = 3

# Provider-agnostic model spec: LangChain init_chat_model format "provider:model"
# (e.g. "openai:gpt-4o-mini", "anthropic:claude-sonnet-5"). The provider reads
# its own API key from the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
DEFAULT_LLM_MODEL = "openai:gpt-5-mini"

# Completion-token budget passed to every provider. Some integrations default
# to a tiny budget (ChatNVIDIA: 1024) and reasoning models (gpt-5*, o*,
# gpt-oss) burn part of it on hidden reasoning tokens, so the visible code
# gets cut mid-file with finish_reason="length". Override via LLM_MAX_TOKENS.
DEFAULT_MAX_TOKENS = 16000


class CodeGenerationError(Exception):
    """Unusable LLM output (truncated, non-compiling, missing analyze()).

    Deliberately NOT a ValueError: langgraph's default_retry_on refuses to
    retry ValueError, while a plain Exception subclass is retried, so the
    node's RetryPolicy re-samples the generation instead of failing the run.
    """


def get_llm(model_spec: str = None, max_tokens: int = None):
    """Builds the chat model via init_chat_model. Explicit args (e.g. from a
    Langfuse prompt's config) override the LLM_MODEL / LLM_MAX_TOKENS env vars."""
    # No explicit temperature: reasoning models (gpt-5*, o*) reject values != 1,
    # so we rely on each provider's default to stay model-agnostic.
    model_spec = model_spec or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    max_tokens = int(max_tokens or os.getenv("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    try:
        llm = init_chat_model(model_spec, max_tokens=max_tokens)
    except TypeError:
        # Integration without a max_tokens parameter: keep its default.
        llm = init_chat_model(model_spec)
    return model_spec, llm


class GeneratedSolution(BaseModel):
    """Structured LLM output: implementation notes and runnable script in separate fields."""
    reasoning: str = Field(description="Brief explanation (a few sentences) of the chosen approach and key optimizations")
    code: str = Field(description="Complete runnable Python script defining analyze(db_path: str), raw code without markdown fences")


def _strip_markdown_fences(content: str) -> str:
    """Safety net: some models still wrap the code field in ``` fences."""
    content = content.strip()
    if "```python" in content:
        content = content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return content


# finish/stop reasons that mean "the provider hit the token limit", across
# the metadata key spellings of the various integrations
_TRUNCATION_REASONS = {"length", "max_tokens", "MAX_TOKENS", "max_output_tokens"}


def _is_truncated(message) -> bool:
    """True if generation stopped on the token limit: whatever text (or tool
    call) the message carries is cut mid-way and must not be salvaged."""
    if message is None:
        return False
    meta = getattr(message, "response_metadata", None) or {}
    finish = meta.get("finish_reason") or meta.get("stop_reason") or meta.get("done_reason")
    return finish in _TRUNCATION_REASONS


def _strip_stray_trailing_closer(code: str, e: SyntaxError) -> str | None:
    """Some models (gpt-oss via ChatNVIDIA) leak the JSON envelope's closing
    brace into the code field, so the script ends with a bare `}`. If the
    syntax error points at a lone closer on the last line, return the code
    without it; None if that's not the failure shape."""
    lines = code.rstrip().splitlines()
    if lines and e.lineno == len(lines) and lines[-1].strip() in ("}", "]", ")"):
        return "\n".join(lines[:-1])
    return None


def _validate_code(code: str, library_name: str) -> str:
    """Gate before any generated code reaches the reviewer: it must compile
    (catches truncated output no matter which parsing path produced it) and
    define the analyze() entrypoint wrapper.py calls."""
    try:
        compile(code, f"<generated_{library_name}>", "exec")
    except SyntaxError as e:
        repaired = _strip_stray_trailing_closer(code, e)
        try:
            if repaired is None:
                raise
            compile(repaired, f"<generated_{library_name}>", "exec")
            code = repaired
        except SyntaxError:
            raise CodeGenerationError(
                f"Generated {library_name} code does not compile "
                f"(likely truncated LLM output): {e}"
            ) from e
    if not re.search(r"^\s*def\s+analyze\s*\(", code, re.M):
        raise CodeGenerationError(
            f"Generated {library_name} code lacks the required analyze(db_path) function"
        )
    return code


def _message_text(message) -> str:
    """Extracts plain text from an AIMessage (content may be str or block list)."""
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _parse_unstructured(text: str) -> tuple[str, str]:
    """Extracts (code, reasoning) from a free-text reply, for providers whose
    structured output fails. Tries, in order: a JSON object with the schema
    fields, a markdown code fence, labeled 'reasoning:'/'code:' sections."""
    text = text.strip()

    # 1) JSON body, possibly wrapped in a fence
    candidate = text
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        candidate = fence.group(1)
    if candidate.startswith("{"):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and str(data.get("code") or "").strip():
                return str(data["code"]).strip(), str(data.get("reasoning") or "").strip()
        except json.JSONDecodeError:
            pass

    # 2) markdown fence: code inside, reasoning is the text before it
    if "```" in text:
        code = _strip_markdown_fences(text)
        reasoning = re.sub(r"^reasoning\s*:\s*", "", text.split("```", 1)[0].strip(), flags=re.I)
        return code, reasoning

    # 3) labeled sections without fences ("reasoning: ... code: ...")
    match = re.search(r"\bcode\s*:\s*(.+)$", text, re.S | re.I)
    if match:
        code = match.group(1).strip()
        reasoning = re.sub(r"^reasoning\s*:\s*", "", text[:match.start()].strip(), flags=re.I)
        return code, reasoning

    # 4) last resort: the whole reply is (hopefully) code
    return text, ""


# Appended to the prompt in plain-text mode. Code embedded in a JSON string
# needs every quote/newline escaped, and several models served via NIM
# (nemotron, gpt-oss) corrupt those escapes, producing scripts that never
# compile; a markdown fence has no escaping at all, so it survives intact.
PLAIN_FORMAT_INSTRUCTIONS = """

OUTPUT FORMAT, plain text, NOT JSON, no structured output:
First the `reasoning` content as a short plain paragraph, then the complete script inside a single ```python fence. Nothing after the closing fence."""


async def _generate_plain(llm, full_prompt: str, library_name: str, cfg: dict) -> tuple[str, str]:
    """Plain-text generation: reasoning paragraph + fenced script."""
    response = await llm.ainvoke(full_prompt + PLAIN_FORMAT_INSTRUCTIONS, config=cfg)
    if _is_truncated(response):
        raise CodeGenerationError(
            f"{library_name} plain generation hit the completion-token limit "
            f"(finish_reason=length), output is truncated; consider raising LLM_MAX_TOKENS"
        )
    code, reasoning = _parse_unstructured(_message_text(response))
    if not code.strip():
        raise CodeGenerationError(f"LLM returned no usable code for {library_name}")
    return _validate_code(code, library_name), reasoning


async def generate_code_with_llm(
    library_name: str, analytical_request: str,
    rejected_code: str = None, failed_code: str = None, execution_error: str = None,
) -> tuple[str, str, str]:
    """Invokes the configured LLM. Returns (code, reasoning, model_spec).

    The prompt (and optionally the model, via the prompt's config) comes from
    prompt_store.get_prompt, Langfuse-managed with local fallback. model_spec
    is the model that actually ran, which is NOT necessarily the LLM_MODEL env
    var: a Langfuse prompt config can override it per library. It is carried
    back so the branch state records it (see _push_session_score).

    Tries structured output first; if the provider/model mangles the code
    inside the JSON field (bad escapes, leaked braces, common on NIM-served
    models), falls back to one plain-text generation with a fenced script.
    Set LLM_STRUCTURED_OUTPUT=0 to skip straight to plain mode and save the
    wasted structured call on providers known to be broken.
    """
    # get_prompt returns the text already compiled ({{mustache}} variables,
    # rendered by Langfuse's compile() or the equivalent local substitution).
    full_prompt, prompt_config, prompt_meta = await asyncio.to_thread(
        get_prompt, f"benchmark/{library_name}",
        {"analytical_request": analytical_request},
    )
    model_spec, llm = get_llm(prompt_config.get("model"), prompt_config.get("max_tokens"))
    # Span identity for observability: named run + filterable metadata
    # (library, phase, and, when the prompt came from Langfuse, the prompt
    # name/version that produced this call).
    phase = "fix" if failed_code else ("regenerate" if rejected_code else "generate")
    cfg = {
        "run_name": f"generate_{library_name}",
        "metadata": {"library": library_name, "phase": phase, "model": model_spec, **prompt_meta},
    }
    print(f"Generating {library_name} code using {model_spec}...")
    if rejected_code:
        full_prompt += (
            "\n\nIMPORTANT: a previous solution was REJECTED by the human reviewer. "
            "Write a new, meaningfully different and improved implementation, do not "
            "resubmit the same approach. The rejected code was:\n\n" + rejected_code
        )
    elif failed_code and execution_error:
        full_prompt += (
            "\n\nIMPORTANT: a previous solution was approved but FAILED at runtime "
            "with the error below. Diagnose the root cause and return a corrected "
            "implementation that avoids it.\n\nError:\n" + execution_error
            + "\n\nFailed code:\n\n" + failed_code
        )

    if os.getenv("LLM_STRUCTURED_OUTPUT", "1").lower() in ("0", "false", "no"):
        code, reasoning = await _generate_plain(llm, full_prompt, library_name, cfg)
        return code, reasoning, model_spec

    try:
        # include_raw: if the provider fails to parse the schema (e.g. the model
        # replies in free text instead of calling the tool), we still get the raw
        # message to salvage instead of a None and a wasted round-trip. Not every
        # integration implements it (langchain-nvidia raises NotImplementedError),
        # so fall back to the plain binding where needed.
        try:
            structured_llm = llm.with_structured_output(GeneratedSolution, include_raw=True)
        except NotImplementedError:
            structured_llm = llm.with_structured_output(GeneratedSolution)
        result = await structured_llm.ainvoke(full_prompt, config=cfg)
        # include_raw=True yields {"parsed", "raw", "parsing_error"}; the plain
        # binding yields the model object directly (or None on parse failure).
        raw = result.get("raw") if isinstance(result, dict) else None
        if _is_truncated(raw):
            # The reply hit the token limit: the code field is cut mid-file, and
            # _parse_unstructured would happily return the half-script as valid.
            raise CodeGenerationError(
                f"{library_name} generation hit the completion-token limit "
                f"(finish_reason=length), output is truncated; consider raising LLM_MAX_TOKENS"
            )
        solution = result.get("parsed") if isinstance(result, dict) else result
        if solution is not None and (solution.code or "").strip():
            code = _validate_code(_strip_markdown_fences(solution.code), library_name)
            return code, (solution.reasoning or "").strip(), model_spec

        # Structured parsing failed: salvage the raw reply if there is one.
        if isinstance(result, dict) and result.get("parsing_error"):
            print(f"Structured output parsing failed for {library_name}: {result['parsing_error']}")
        content = _message_text(raw)
        code, reasoning = _parse_unstructured(content) if content.strip() else ("", "")
        if not code.strip():
            raise CodeGenerationError(f"structured output unusable for {library_name}")
        return _validate_code(code, library_name), reasoning, model_spec
    except CodeGenerationError as err:
        # Anything unusable from the structured path (mangled escapes, leaked
        # braces, empty reply, truncation) gets ONE plain-text retry before
        # the node's RetryPolicy takes over.
        print(f"Structured generation failed for {library_name} ({err}); retrying as plain text...")
        code, reasoning = await _generate_plain(llm, full_prompt, library_name, cfg)
        return code, reasoning, model_spec


# ---------------------------------------------------------------------------
# Node factory functions - eliminates per-library duplication
# ---------------------------------------------------------------------------

def _make_generate_node(library_name: str):
    """Factory: creates an async generate_<lib>_node."""

    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        request = state.get("analytical_request")
        if not request or not str(request).strip():
            # ValueError on purpose: default_retry_on doesn't retry it, so a
            # run started without the required input fails fast (and with a
            # clear message) instead of burning 3 LLM retries per branch.
            raise ValueError(
                "Run input is missing 'analytical_request', it is the only "
                "required field, e.g. {\"analytical_request\": \"top 10 customers by spend\"}"
            )
        branch = state.get(library_name) or {}
        attempts = branch.get("attempts", 0) + 1
        update = {"attempts": attempts}
        # Two regeneration modes, plus the initial one:
        # - the reviewer pressed "regenerate": pass the previous code as
        #   context so the model produces a different implementation;
        # - approved code failed at runtime in Docker: pass code + error so
        #   the model fixes the problem (counted against MAX_EXECUTION_FIXES).
        rejected_code = failed_code = execution_error = None
        if branch.get("decision") == "regenerate":
            rejected_code = branch.get("code")
            print(f"Regenerating {library_name} code on reviewer request (attempt {attempts})...")
        elif branch.get("exec_error"):
            failed_code = branch.get("code")
            execution_error = branch.get("exec_error")
            fix_attempts = branch.get("exec_attempts", 0) + 1
            update["exec_attempts"] = fix_attempts
            print(f"Regenerating {library_name} code after execution error (fix {fix_attempts}/{MAX_EXECUTION_FIXES})...")
        code, reasoning, model_spec = await generate_code_with_llm(
            library_name, request, rejected_code, failed_code, execution_error
        )
        update["code"] = code
        update["reasoning"] = reasoning
        update["model"] = model_spec
        return {library_name: update}

    node.__name__ = f"generate_{library_name}_node"
    return node


def _make_approve_node(library_name: str):
    """Factory: creates an approve_<lib>_node with interrupt()."""

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        branch = state.get(library_name) or {}
        res = interrupt({
            "action": "approve_code",
            "library": library_name,
            "code": branch.get("code"),
            "reasoning": branch.get("reasoning"),
            "attempt": branch.get("attempts", 1),
            # Set when this code is a fix for a runtime failure: the reviewer
            # sees the error alongside the corrected code.
            "execution_error": branch.get("exec_error"),
            "fix_attempt": branch.get("exec_attempts", 0),
            "max_fix_attempts": MAX_EXECUTION_FIXES,
        })
        decision = "reject"
        if isinstance(res, dict):
            decision = res.get("decision") or "reject"
        if decision not in ("approve", "regenerate", "reject"):
            decision = "reject"
        return {library_name: {"decision": decision}}

    node.__name__ = f"approve_{library_name}_node"
    return node


def make_approval_router(library_name: str):
    """Factory: routes after approve_<lib>, "regenerate" loops back to
    generate_<lib> (no limit); "approve" runs in Docker; "reject" also goes to
    the docker node, which skips execution and records the rejection."""

    def route(state: Dict[str, Any]) -> str:
        if (state.get(library_name) or {}).get("decision") == "regenerate":
            return f"generate_{library_name}"
        return f"run_{library_name}_docker"

    route.__name__ = f"route_after_approve_{library_name}"
    return route


def _make_run_docker_node(library_name: str):
    """Factory: creates an async run_<lib>_docker_node."""

    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        branch = state.get(library_name) or {}
        if branch.get("decision") == "reject":
            # Reviewer discarded this script: record the rejection and let the
            # branch fall through to merge_results without executing anything.
            return {library_name: {"metrics": {
                "success": False,
                "error": "Rejected by the user."
            }}}

        metrics = await run_in_docker(library_name, branch["code"])
        return {library_name: {
            "metrics": metrics,
            # A runtime failure of the CODE feeds the fix loop (and the next
            # approval message); success clears any error from a previous
            # attempt, and infra failures (docker down, broken wrapper) never
            # enter the loop: regenerating code can't fix those.
            "exec_error": None if metrics.get("success") or metrics.get("infra") else metrics.get("error"),
        }}

    node.__name__ = f"run_{library_name}_docker_node"
    return node


def make_execution_router(library_name: str):
    """Factory: routes after run_<lib>_docker; a runtime failure of approved
    code loops back to generate_<lib> with the error (up to MAX_EXECUTION_FIXES
    times); success, rejection, or an exhausted fix budget proceed to merge."""

    def route(state: Dict[str, Any]) -> str:
        branch = state.get(library_name) or {}
        metrics = branch.get("metrics") or {}
        if metrics.get("success"):
            return "merge_results"
        if metrics.get("infra"):
            return "merge_results"  # infra failure: the LLM can't fix docker
        if branch.get("decision") != "approve":
            return "merge_results"  # rejection: no fix loop
        if branch.get("exec_attempts", 0) < MAX_EXECUTION_FIXES:
            return f"generate_{library_name}"
        return "merge_results"  # fix budget exhausted: keep the failure

    route.__name__ = f"route_after_run_{library_name}_docker"
    return route


# Dynamically create all 12 branch nodes via factory
for _lib in LIBRARIES:
    globals()[f"generate_{_lib}_node"] = _make_generate_node(_lib)
    globals()[f"approve_{_lib}_node"] = _make_approve_node(_lib)
    globals()[f"run_{_lib}_docker_node"] = _make_run_docker_node(_lib)


def merge_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("Executing Merge Results Node...")
    # Gather metrics from all 4 branches into the results structure
    results = {
        lib: (state.get(lib) or {}).get("metrics", {})
        for lib in LIBRARIES
    }

    # Hit global selection interrupt (HITL 2)
    res = interrupt({
        "action": "select_strategy",
        "results": results
    })

    selected = "pandas"
    if isinstance(res, dict):
        selected = res.get("selected_strategy", "pandas")
    elif isinstance(res, str):
        selected = res

    return {
        "results": results,
        "selected_strategy": selected
    }


def _table_cell(text: Any) -> str:
    """Error/preview text squeezed into one Markdown table cell: newlines and
    pipes would break the row."""
    return str(text).replace("\n", " ").replace("|", "/")[:100]


def _result_row(lib: str, metrics: Dict[str, Any]) -> str:
    """One Markdown table row per library. A branch whose code never ran in
    Docker (rejected) has no elapsed_seconds; every real execution, even a
    failed one, records it, so it gets a NOT EXECUTED row instead of
    misleading zeroes. Failed executions show the error instead of their
    timings: a time measured up to a crash isn't comparable with the others."""
    name = lib.capitalize()
    if "elapsed_seconds" not in metrics:
        reason = _table_cell(metrics.get("error", "no code was executed"))
        return f"| **{name}** |, |, | NOT EXECUTED | `{reason}` |"
    if not metrics.get("success"):
        reason = _table_cell(metrics.get("error", "unknown error"))
        return f"| **{name}** |, |, | FAILED | `{reason}` |"
    time_s = metrics.get("elapsed_seconds", 0)
    mem_mb = metrics.get("net_memory_used_bytes", 0) / (1024 * 1024)
    preview = _table_cell(metrics.get("preview"))
    return f"| **{name}** | {time_s:.4f}s | {mem_mb:.2f} MB | SUCCESS | `{preview}...` |"


def _analysis_section(results: Dict[str, Any], selected: str) -> str:
    """Analysis bullets derived from the MEASURED metrics, never canned prose
    that could contradict the table above it (e.g. praising a library whose
    run actually failed)."""
    executed = {lib: m for lib, m in results.items() if "elapsed_seconds" in (m or {})}
    ok = {lib: m for lib, m in executed.items() if m.get("success")}
    lines = []

    if ok:
        ranked = sorted(ok, key=lambda lib: ok[lib]["elapsed_seconds"])
        fastest = ranked[0]
        t_fast = ok[fastest]["elapsed_seconds"] or 1e-9
        parts = [f"**{lib.capitalize()}** ({ok[lib]['elapsed_seconds']:.4f}s"
                 + ("" if lib == fastest else f", {ok[lib]['elapsed_seconds'] / t_fast:.1f}× slower")
                 + ")" for lib in ranked]
        lines.append(f"- **Speed ranking:** {', '.join(parts)}.")
        lightest = min(ok, key=lambda lib: ok[lib].get("net_memory_used_bytes", 0))
        lines.append(
            f"- **Lowest memory:** {lightest.capitalize()} "
            f"({ok[lightest].get('net_memory_used_bytes', 0) / (1024 * 1024):.2f} MB net peak RSS)."
        )
        if selected in ok:
            if selected == fastest:
                lines.append(f"- **Selected strategy:** {selected.capitalize()} - also the fastest run.")
            else:
                ratio = ok[selected]["elapsed_seconds"] / t_fast
                lines.append(
                    f"- **Selected strategy:** {selected.capitalize()} - "
                    f"{ratio:.1f}× slower than {fastest.capitalize()}."
                )
        else:
            lines.append(f"- **Selected strategy:** {selected.capitalize()} - no successful run recorded.")

    for lib, m in executed.items():
        if not m.get("success"):
            reason = str(m.get("error") or "unknown error").strip().splitlines()[-1]
            lines.append(f"- **{lib.capitalize()} failed:** `{reason[:200]}`")
    for lib in LIBRARIES:
        if lib not in executed:
            reason = (results.get(lib) or {}).get("error", "no code was executed")
            lines.append(f"- **{lib.capitalize()} not executed:** {reason}")

    return "\n".join(lines) if lines else "- No branch produced measurable results."


async def _generate_commentary(
    analytical_request: str, results: Dict[str, Any], codes: Dict[str, Any], selected: str
) -> str:
    """One LLM call with the FULL session picture, every branch's metrics,
    result preview and the exact code that ran, so it can explain the
    numbers, not just restate them. Returns "" on any failure: the report
    must never die (or block) on a commentary; the metric-based
    _analysis_section is always there as the factual baseline."""
    try:
        blocks = []
        for lib in LIBRARIES:
            m = results.get(lib) or {}
            if "elapsed_seconds" not in m:
                status = f"NOT EXECUTED ({m.get('error', 'no code was run')})"
                metrics_line = ""
            else:
                status = "SUCCESS" if m.get("success") else f"FAILED: {str(m.get('error') or '').strip()[-400:]}"
                metrics_line = (
                    f"- elapsed: {m.get('elapsed_seconds', 0):.4f} s, "
                    f"net peak RSS: {m.get('net_memory_used_bytes', 0) / (1024 * 1024):.2f} MB\n"
                )
            blocks.append(
                f"### {lib.upper()}, {status}\n"
                f"{metrics_line}"
                f"- result preview: {str(m.get('preview'))[:600]}\n"
                f"```python\n{codes.get(lib) or '(no code generated)'}\n```"
            )
        prompt, prompt_config, prompt_meta = await asyncio.to_thread(
            get_prompt, "benchmark/commentary",
            {
                "analytical_request": str(analytical_request),
                "selected": selected.upper(),
                "context": "\n\n".join(blocks),
            },
        )
        model_spec, llm = get_llm(prompt_config.get("model"), prompt_config.get("max_tokens"))
        print(f"Generating report commentary using {model_spec}...")
        response = await llm.ainvoke(prompt, config={
            "run_name": "report_commentary",
            "metadata": {"phase": "commentary", "model": model_spec, **prompt_meta},
        })
        if _is_truncated(response):
            print("Warning: commentary hit the completion-token limit; section dropped.")
            return ""
        return _message_text(response).strip()
    except Exception as e:
        print(f"Warning: LLM commentary failed ({e}); the report keeps the metric-based analysis only.")
        return ""


def _session_environment(state: Dict[str, Any]) -> str:
    """The session's LLM model, slugged into a Langfuse `environment` value.

    `environment` is the only score-own dimension free enough to carry the
    model, which is what lets a dashboard widget break the session scores down
    per model (filter name=<score>, breakdown dimension environment). Every
    trace-derived dimension (sessionId, traceName, tags, observationModelName)
    is empty for session-level scores because their traceId is null, and
    `metadata` is not a valid dimension at all.

    The model is read from the branches, not from LLM_MODEL: a Langfuse prompt
    config can override it per library, so the env var is not authoritative:
    it is only the fallback for a session where no branch ever generated. Four
    branches could in principle run four different models; the session then
    belongs to none of them and is labelled "mixed" rather than mislabelled.
    """
    models = {(state.get(lib) or {}).get("model") for lib in LIBRARIES} - {None}
    if len(models) == 1:
        model = next(iter(models))
    elif models:
        model = "mixed"
    else:
        model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    # Langfuse SDK requires ^(?!langfuse)[a-z0-9_-]+$ and rejects the score otherwise.
    return re.sub(r"[^a-z0-9_-]+", "-", model.lower()).strip("-") or "unknown"


def _push_session_score(state: Dict[str, Any], thread_id) -> None:
    """Best-effort Langfuse session-level NUMERIC scores: code_errors (runtime
    errors made by the generated scripts: every failure that entered the fix
    loop plus a final non-infra failure that ended a branch; rejections and
    infra failures are not script errors), branches_failed (branches that
    produced no usable benchmark, for ANY reason: script error, user
    rejection, infra problem), and human-requested regenerations (total
    generations minus the first one minus the error-driven ones, per branch).

    All three point the same way (higher is worse), so a dashboard comparing
    two models reads without per-metric mental inversion.

    Aegra already stamps langfuse.session.id = thread_id on every span, so
    the scores land on the same Langfuse session as the traces. Requires the
    LANGFUSE_* keys (independent of LANGFUSE_PROMPTS); without them, or
    without a thread_id (no session to attach to), it's a no-op. Blocking
    (SDK ingestion client), call via asyncio.to_thread.

    Each score carries the LLM model as its Langfuse `environment`, which is
    what makes model-vs-model dashboards possible; see the comment below.
    """
    if not (thread_id and os.getenv("LANGFUSE_PUBLIC_KEY")):
        return
    results = state.get("results") or {}

    def branch_code_errors(lib: str) -> int:
        # Each exec_attempt was a runtime error the fix loop reacted to; add
        # the final failure when the last approved run also crashed (infra
        # failures and user rejections are not errors OF the script).
        branch = state.get(lib) or {}
        metrics = results.get(lib) or {}
        errors = branch.get("exec_attempts", 0)
        if (branch.get("decision") == "approve"
                and metrics and not metrics.get("success")
                and not metrics.get("infra")):
            errors += 1
        return errors

    scores = {
        "code_errors": sum(branch_code_errors(lib) for lib in LIBRARIES),
        "branches_failed": sum(
            1 for lib in LIBRARIES if not (results.get(lib) or {}).get("success")
        ),
        "human_regenerations": sum(
            max(0, (state.get(lib) or {}).get("attempts", 1) - 1
                - (state.get(lib) or {}).get("exec_attempts", 0))
            for lib in LIBRARIES
        ),
    }
    try:
        from benchmark_agent.prompt_store import _langfuse
        client = _langfuse()
        for name, value in scores.items():
            client.create_score(
                name=name,
                value=value,
                session_id=str(thread_id),
                data_type="NUMERIC",
                environment=_session_environment(state),
            )
        print(f"Langfuse session scores pushed: {scores}")
    except Exception as e:
        print(f"Warning: Langfuse session score push failed ({e}).")


async def generate_report_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    print("Executing Generate Report Node...")
    selected = state.get("selected_strategy", "pandas")
    results = state.get("results", {})
    analytical_request = state.get("analytical_request")

    # All session artifacts go into one folder: <OUTPUT_BASE_DIR>/<session id>.
    # The id is the Aegra thread id, taken from the run config rather than
    # state (it's identity, not run data); the timestamp fallback covers runs
    # executed without a checkpointer/thread.
    thread_id = (config.get("configurable") or {}).get("thread_id")
    output_dir = os.path.join(
        os.getenv("OUTPUT_BASE_DIR", "output"),
        str(thread_id) if thread_id else datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Extract codes
    codes = {lib: (state.get(lib) or {}).get("code") for lib in LIBRARIES}

    # Write the winning script
    approved_code = codes.get(selected, "")
    output_code_path = os.path.join(output_dir, "approved_analytics_solution.py")
    with open(output_code_path, "w", encoding="utf-8") as f:
        f.write(approved_code)

    # Generate charts and markdown report
    report_path = os.path.join(output_dir, "data_analysis_report.md")

    # Create matplotlib latency & memory chart. Only successful branches
    # appear: failed/not-executed ones show FAILED in the table and would
    # just be empty slots on the axis.
    ok_libs = [lib for lib in LIBRARIES if (results.get(lib) or {}).get("success")]
    if not ok_libs:
        chart_markdown = "\n*(No successful execution, chart skipped)*\n"
    else:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            lib_names = [lib.capitalize() for lib in ok_libs]
            latencies = [results[lib].get("elapsed_seconds", 0) for lib in ok_libs]
            memories = [
                results[lib].get("net_memory_used_bytes", 0) / (1024 * 1024)
                for lib in ok_libs
            ]

            fig, ax1 = plt.subplots(figsize=(8, 4))

            color = 'tab:blue'
            ax1.set_xlabel('Library')
            ax1.set_ylabel('Execution Time (seconds)', color=color)
            ax1.bar(lib_names, latencies, color=color, alpha=0.6, width=0.4, label='Latency')
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('RAM Used (MB)', color=color)
            ax2.plot(lib_names, memories, color=color, marker='o', linewidth=2, label='Memory')
            ax2.tick_params(axis='y', labelcolor=color)

            # Title must exist before tight_layout computes the margins,
            # otherwise it lands outside the figure and gets clipped.
            plt.title('Data Processing Performance Benchmark')
            fig.tight_layout()
            chart_path = os.path.join(output_dir, "performance_chart.png")
            plt.savefig(chart_path)
            plt.close()
            # Relative link: the chart lives next to the report in the same folder
            chart_markdown = "\n![Performance Chart](performance_chart.png)\n"
        except Exception as e:
            print(f"Warning: Failed to generate Matplotlib chart. Error: {e}")
            chart_markdown = "\n*(Chart generation failed or matplotlib is missing)*\n"

    # Write Markdown Report
    results_table_rows = "\n".join(_result_row(lib, results.get(lib) or {}) for lib in LIBRARIES)
    analysis_section = _analysis_section(results, selected)
    commentary = await _generate_commentary(analytical_request, results, codes, selected)
    commentary_section = f"\n## Commentary (LLM)\n\n{commentary}\n" if commentary else ""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Data Analysis & Optimization Benchmark Report

This report was auto-generated via **LangGraph** and **Aegra**, comparing different Python data processing strategies.

## Original Analytical Request
> "{analytical_request}"

## Selected Strategy Result
- **Selected Strategy:** **{selected.upper()}** (Saved in `approved_analytics_solution.py`)

## Performance Comparison Table

| Library | Execution Time (s) | Peak RAM (MB) | Docker Status | Data Preview |
| :--- | :--- | :--- | :--- | :--- |
{results_table_rows}

{chart_markdown}

## Analysis of Results
{analysis_section}
{commentary_section}
---
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

    print(f"Report written to {report_path}")
    await asyncio.to_thread(_push_session_score, state, thread_id)
    return {}
