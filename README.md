# Demo Parallelization - Benchmark Agent

A **LangGraph** workflow, served via **Aegra** (self-hosted Agent Protocol server), that takes an analytical request from a **Telegram** chat and fans it out to be processed in parallel by 4 independent pipelines: **Pandas, Dask, NumPy, Polars**. Each branch generates code with an LLM, pauses for human approval (human-in-the-loop via `interrupt()`), then runs the approved code inside an isolated Docker container while measuring time and memory. At the end the user picks the winning strategy from the chat and a Markdown report with a chart is generated.

```
                    ┌─ generate_pandas ─ approve_pandas ─ run_pandas_docker ─┐
                    ├─ generate_dask ─── approve_dask ─── run_dask_docker ───┤
START ── fan-out ───┤                                                        ├── merge_results ── generate_report ── END
                    ├─ generate_numpy ── approve_numpy ── run_numpy_docker ──┤        │
                    └─ generate_polars ─ approve_polars ─ run_polars_docker ─┘   (interrupt:
                                              │                                strategy selection)
                                        (interrupt:
                                    code approval ×4)
```

## Why this project / key strengths

- **True parallel fan-out with per-branch human-in-the-loop**: all 4 pipelines generate code concurrently, and each branch can be approved, regenerated (unlimited) or rejected **independently** while the others stay paused; resumes target a single interrupt id, not the whole run.
- **Fair, isolated benchmarking**: every script runs in its own Docker container with no network, identical RAM/CPU caps, and executions serialized so the numbers aren't skewed by contention; time and **peak RSS** are measured by a wrapper inside the container.
- **Self-healing generation**: LLM retries with backoff, structured output with automatic plain-text fallback for providers that corrupt JSON-embedded code, compile + entrypoint validation before any code reaches the reviewer, and an automatic fix loop (up to 3 attempts) when approved code fails at runtime, with infrastructure failures correctly excluded from it.
- **Provider-agnostic LLM**: one env var (`LLM_MODEL="provider:model"`) switches between OpenAI, Anthropic, Ollama, NVIDIA, ... via LangChain's `init_chat_model`.
- **Grounded reporting**: the report's analysis is computed from the measured metrics (never canned prose), plus an optional LLM commentary that sees every branch's code and results.
- **Production-style observability (optional)**: native OTEL tracing to Langfuse with zero repo code, centralized prompt management with automatic local fallback, and session-level numeric scores labelled by model so two LLMs can be compared on real sessions; the repo works fully offline without any of it.

## Components

| Component | Role |
| :--- | :--- |
| `src/benchmark_agent/` | The LangGraph graph (state, nodes, prompts, Docker execution) |
| `bot_handler.py` | External process: Telegram bot (python-telegram-bot) that starts sessions and drives HITL approvals by talking to Aegra's REST API (httpx) |
| `wrapper.py` + `Dockerfile.benchmark` | Profiler run inside the container: imports the generated script, calls `analyze(db_path)`, measures time and peak RSS |
| `data_generator.py` | Generates `data/analytics.db` (SQLite, 100k synthetic transactions) |
| `aegra.json` / `docker-compose.yml` | Configuration for the Aegra server and its services (Postgres) |

## Prerequisites

- **Python 3.12+**
- **Docker** installed and running (needed for both Aegra's Postgres and the benchmark containers)
- A **Telegram bot**: create one with [@BotFather](https://t.me/BotFather) to get the token, then retrieve your `chat_id` (e.g. by messaging [@userinfobot](https://t.me/userinfobot))
- An **API key** for the chosen LLM provider (default: OpenAI)

## Get started

### 1. Install dependencies

```bash
pip install -r requirements.txt      # or: uv sync
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Then fill in `.env`:

| Variable | Description |
| :--- | :--- |
| `AEGRA_URL` | URL of the Aegra server, e.g. `http://localhost:8000` |
| `LLM_MODEL` | Model in `provider:model` format (LangChain `init_chat_model`), e.g. `openai:gpt-5-mini`, `anthropic:claude-sonnet-5` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / ... | The API key for the provider chosen in `LLM_MODEL` |
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather (**required**) |
| `TELEGRAM_CHAT_ID` | Numeric ID of the authorized chat (**required**) |
| `LLM_MAX_TOKENS` | Completion-token budget for code generation (default `16000`) |
| `LLM_STRUCTURED_OUTPUT` | Set to `0` to skip structured output and ask for a fenced script instead (for providers that corrupt JSON-embedded code) |
| `ANALYTICS_DB_PATH` | SQLite DB used by the benchmarks, resolved server-side (default `data/analytics.db`, generated on first use) |
| `OUTPUT_BASE_DIR` | Base folder for per-session artifacts (default `output`; if changed, update `UVICORN_RELOAD_EXCLUDES` to match) |
| `BENCHMARK_MEMORY_LIMIT` | RAM cap for each benchmark container (`docker run --memory`, default `2g`) |
| `BENCHMARK_CPU_LIMIT` | CPU cap for each benchmark container (`docker run --cpus`, default `2`) |
| `OTEL_TARGETS` + `LANGFUSE_*` | Optional observability: `OTEL_TARGETS=LANGFUSE` plus `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL`. Aegra natively exports OTEL traces of every node and LLM call to Langfuse (sessions grouped by thread id), no code in this repo, and no `CallbackHandler` needed (it would double-trace). Also enables the session scores, see [Observability on Langfuse](#observability-on-langfuse-optional) |

> If you use a provider other than OpenAI, also install its integration package (e.g. `pip install langchain-anthropic`).

### 3. Start the Aegra server

```bash
aegra dev
```

Automatically starts Postgres via compose and serves the graph with hot-reload on the code. Also works on Windows **as long as this is in `.env`** (already present in `.env.example`):

```
UVICORN_RELOAD_EXCLUDES=output/*/*.py
```

This excludes the script the workflow writes into `output/<thread-id>/` at the end of a session from the hot-reload watcher; without this line the server would restart, killing the run. (The uvicorn CLI also reads options from `UVICORN_*` environment variables, and `aegra dev` loads `.env` before launching it.)

Alternative **without** hot-reload: on Linux/macOS `aegra dev --no-reload`. On Windows `--no-reload` doesn't work (uvicorn would stay on the ProactorEventLoop, incompatible with async psycopg); stick with plain `aegra dev` and the exclude pattern above.

**Run the commands from the repo root** (the DB, Docker image and `aegra.json` paths are relative to the server's cwd).

### 4. Start the Telegram bot

In a second terminal, also from the repo root:

```bash
python bot_handler.py
```

The bot sends "Ready for a new session" to the chat.

### 5. Launch a session from the chat

```
/newsession Compute the 7-day rolling average of spend for each category
            and flag users with anomalous transactions
```

(alternatively: `/newsession` alone and then the request in the next message, or directly type the request text while the bot is idle)

The session flow:

1. 🚀 The bot creates a **new thread** on Aegra and starts the run (if `data/analytics.db` doesn't exist, it's generated on the fly).
2. 🧠 The 4 branches generate code in parallel for Pandas, Dask, NumPy and Polars.
3. ✅ For each library you receive the generated code in chat with **✅ Approve / 🔄 Regenerate / ❌ Reject** buttons. **Regenerate** asks the model for a new, different implementation (no limit; the previous code is passed as context and the message shows the attempt number). **Reject** marks the branch as `FAILED` immediately, without executing anything. **Approved** branches run in Docker (120s timeout, DB mounted read-only, **no network**, RAM/CPU capped, see `BENCHMARK_*` vars); executions are **serialized** so each library is measured on identical, uncontended resources, while generation and approvals stay parallel. If approved code fails at runtime, it is automatically regenerated with the error in the prompt (up to 3 fixes) and comes back for approval with the error shown above the corrected code; infrastructure failures (e.g. Docker daemon down) fail the branch immediately instead of burning fix attempts.
4. 🏆 Once benchmarks complete, a comparison table (time/RAM) arrives: pick the strategy to promote with a button.
5. 🏁 The outputs are written, the bot posts the performance chart, the Markdown report and the winning script directly in the chat, and goes back to waiting for a new `/newsession` (state resets automatically: every session uses a fresh thread).

On first run the `benchmark-runner` Docker image is built automatically (may take a few minutes).

### Produced output

Each session writes its artifacts to a dedicated `output/<thread-id>/` folder (the id is the session's Aegra thread id, also reported in the final Telegram message):

- `approved_analytics_solution.py` - the winning strategy's script
- `data_analysis_report.md` - comparative report in Markdown: an "Analysis of Results" section computed from the measured metrics (speed ranking, lowest memory, failure reasons) plus a "Commentary (LLM)" section where the model, given metrics, result previews **and the code of every branch**, explains the outcome and flags result discrepancies (best-effort: omitted if the call fails)
- `performance_chart.png` - time/memory chart (linked from the report)

Chart, report and winning script are also sent directly to the Telegram chat when the bot runs on the same machine as the server (`aegra dev` mode); in full-container mode the files live inside the API container, so only the paths are reported.

## Full-container mode (docker compose)

As an alternative to `aegra dev`, the entire stack (Postgres + Redis + Aegra server) can run containerized:

```bash
docker compose up --build
# then, on the host: AEGRA_URL=http://localhost:2026 in .env and python bot_handler.py
```

Benchmarks also work in this mode thanks to two provisions already configured in the compose file:

- the **host's Docker socket** is mounted into the API container (`/var/run/docker.sock`), so benchmark containers start as *siblings* of the API container via the host's daemon (which is why the service runs as `user: root`);
- since the host's daemon can't see the API container's filesystem, generated scripts and the DB copy are written to a **shared named volume** (`demo_parallelization_benchmark_shared`, mounted at `/shared`), which the benchmark containers mount by name; the wrapper locates them via the `SCRIPT_PATH`/`DB_PATH` env vars. All of this is driven by the `BENCHMARK_SHARED_VOLUME` variable: if absent (`aegra dev` mode), classic bind mounts are used instead.

> ⚠️ If you already built the `benchmark-runner` image before using this mode, rebuild it (`docker build -t benchmark-runner -f Dockerfile.benchmark .`): the old wrapper version doesn't read `SCRIPT_PATH`/`DB_PATH`.

## Observability on Langfuse (optional)

Put `OTEL_TARGETS=LANGFUSE` and the three `LANGFUSE_*` keys in `.env` and the whole run becomes observable.

### Tracing and monitoring

Aegra exports the traces natively, so there is no tracing code in this repo.

- **One Langfuse session per workflow session**, keyed by the Aegra thread id, the same id that names the `output/<thread-id>/` folder, so a report on disk maps back to its trace in the UI.
- **A span per graph node and per LLM call**, with `run_id`/`thread_id`/`graph_id` as filterable metadata.
- **Identifiable generation spans**: each is named `generate_<library>` and carries `library`, `phase` (`generate` | `regenerate` | `fix` | `commentary`) and `model`, so traces can be filtered per branch, per phase or per model.
- **Prompt provenance**: when the text comes from Langfuse Prompt Management (see the next section), the span also carries `prompt_name`/`prompt_version`.

### Session scoring

At the end of every session the report node pushes 3 **session-level NUMERIC scores** (needs `pip install langfuse`; best-effort, a Langfuse problem never fails the run):

| Score | Meaning |
| :--- | :--- |
| `code_errors` | Runtime errors made by the generated scripts, fix-loop attempts included. Rejections and infrastructure failures don't count: they are not errors of the code |
| `branches_failed` | Branches that produced no usable benchmark, for any reason |
| `human_regenerations` | 🔄 Regenerate presses |

All three point the same way, **higher is worse**, so a dashboard comparing two models reads without mentally inverting one of them.

### Comparing two models

Every score also carries the model that generated the code, slugged into the score's Langfuse **`environment`** (`openai:gpt-4o-mini` → `openai-gpt-4o-mini`). Run some sessions on one model and some on another, then add one dashboard widget per metric:

- view `scores-numeric`
- filter `name = code_errors` (one widget per score)
- breakdown dimension `environment` → one bar per model

The label is read from the run, so a model set in a Langfuse prompt `config` is reported correctly too.

## Centralized prompts on Langfuse (optional)

The 5 prompts (4 code generators + report commentary) can be managed on **Langfuse Prompt Management** instead of the local [prompts.py](src/benchmark_agent/prompts.py), versioned, editable from the UI without redeploy, with the **model choice centralized too** (each prompt's `config` carries `{"model", "max_tokens"}`, overriding `LLM_MODEL`/`LLM_MAX_TOKENS`):

```bash
pip install langfuse                     # prompt client only (tracing needs no package)
python seed_langfuse_prompts.py          # one-shot: creates the 5 "benchmark/*" prompts, label production
# then in .env:  LANGFUSE_PROMPTS=1      # and restart aegra dev
```

Notes:

- prompts are fetched per generation with the SDK's ~60s client cache: an edit labeled `production` in the UI goes live within a minute, no restart;
- **automatic fallback**: flag off, package missing, prompt not found or Langfuse unreachable → the local prompts.py text is used, so the workflow never blocks on Langfuse;
- prompts use Langfuse's native **`{{variable}}`** syntax, so the Playground and UI experiments recognize the variables (fill in `analytical_request` etc. to test a prompt right from the UI); edit the text freely but keep the `{{placeholders}}`;
- each LLM span in the traces carries `prompt_name`/`prompt_version` in its metadata, so you can tell which prompt version produced which generation;
- re-running the seed script creates new versions; it never deletes anything. Use `LANGFUSE_PROMPT_LABEL` to point the server at a different label (e.g. `staging`).

## Useful commands

```bash
python data_generator.py             # (re)generates data/analytics.db with 100k synthetic rows
python seed_langfuse_prompts.py      # seeds/re-versions the prompts on Langfuse (see section above)
docker build -t benchmark-runner -f Dockerfile.benchmark .   # manual rebuild of the benchmark image
```


