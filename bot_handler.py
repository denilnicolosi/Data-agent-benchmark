"""Telegram front-end for the Aegra benchmark workflow.

Built on python-telegram-bot (async): the library owns the update loop,
handlers react to /newsession, plain text and inline-button presses. One
session at a time runs as an asyncio task; HITL answers travel from the
button handler to the workflow through asyncio Futures.
"""

import asyncio
import json
import logging
import os
import sys
from html import escape

import httpx
import structlog
from dotenv import load_dotenv
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Load env variables from .env if present
load_dotenv()

logger = structlog.getLogger(__name__)


def setup_logging():
    """Console logging in the same style as the Aegra server (structlog):
    ISO timestamp, padded level, event, logger name, key=value fields.
    Stdlib logging is routed through the same formatter so library logs
    (python-telegram-bot, httpx) come out uniform too."""
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(
                pad_level=True,
                # Rich tracebacks use Unicode box-drawing characters that the
                # Windows cp1252 console can't encode (UnicodeEncodeError).
                **({"exception_formatter": structlog.dev.plain_traceback}
                   if sys.platform == "win32" else {}),
            ),
        ],
        foreign_pre_chain=shared,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # python-telegram-bot polls getUpdates via httpx every few seconds and
    # httpx logs each request at INFO; silence it to avoid the spam.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

AEGRA_URL = os.getenv("AEGRA_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip().strip("'\"")
ASSISTANT_ID = "benchmark_agent"
LIBRARIES = ("pandas", "dask", "numpy", "polars")

READY_MESSAGE = (
    "🤖 <b>Ready for a new session!</b>\n\n"
    "Send <code>/newsession &lt;analytical request&gt;</code> "
    "or just type the request directly as a message."
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class SessionState:
    """Tracks the single active session and its pending HITL questions."""

    def __init__(self):
        self.running = False
        self.task: asyncio.Task | None = None
        # HITL futures keyed by "approve_code:<library>" or "select_strategy";
        # the callback handler resolves them when a button is pressed.
        self.pending: dict[str, asyncio.Future] = {}

    def reset(self):
        self.running = False
        self.task = None
        self.pending.clear()


session = SessionState()


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

def clip(escaped_text: str, limit: int) -> str:
    """Truncates already-escaped HTML text without cutting an entity (&lt;) in half."""
    if len(escaped_text) <= limit:
        return escaped_text
    cut = escaped_text[:limit]
    amp = cut.rfind("&")
    if amp > cut.rfind(";"):
        cut = cut[:amp]
    return cut + "..."


async def send(app: Application, text: str, reply_markup=None):
    # Callers clip long content BEFORE building the HTML (a blind cut here
    # would break tags/entities); this is only a last-resort fallback.
    try:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
    except BadRequest as e:
        if "can't parse entities" not in str(e).lower():
            raise
        # Malformed/oversized HTML: deliver the content as plain text rather
        # than killing the session.
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text[:4000],
            reply_markup=reply_markup,
        )


async def ask_with_buttons(app: Application, key: str, text: str, keyboard) -> dict:
    """Sends a message with inline buttons and waits for the matching press."""
    future = asyncio.get_running_loop().create_future()
    session.pending[key] = future
    await send(app, text, InlineKeyboardMarkup(keyboard))
    return await future


def button(text: str, data: dict) -> InlineKeyboardButton:
    # Telegram caps callback_data at 64 bytes, hence the compact separators.
    return InlineKeyboardButton(text, callback_data=json.dumps(data, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Update handlers
# ---------------------------------------------------------------------------

async def cmd_newsession(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if session.running:
        await update.message.reply_text("⏳ A session is already running. Use /cancel to abort it.")
        return
    request = " ".join(context.args).strip()
    if request:
        start_session(context.application, request)
    else:
        await update.message.reply_text("📝 Ok! Describe the analytical request for the new session:")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Plain text while idle starts a session (also the follow-up to a bare /newsession)."""
    if session.running:
        await update.message.reply_text("⏳ Session in progress, use the buttons to respond.")
        return
    start_session(context.application, update.message.text.strip())


def _button_feedback(data: dict) -> str:
    """Human-readable confirmation of the pressed button."""
    action = data.get("action")
    if action == "approve_code":
        label = {
            "approve": "✅ Approved",
            "regenerate": "🔄 Regeneration requested",
            "reject": "❌ Rejected",
        }.get(DECISIONS.get(data.get("d"), "reject"), "OK")
        library = (data.get("library") or "").upper()
        return f"{label}, {library}"
    if action == "select_strategy":
        return f"🏆 Strategy selected: {str(data.get('selected') or '').upper()}"
    return "OK"


async def _clear_buttons(query):
    """Removes the inline keyboard from the pressed message so it can't be
    pressed twice and the user sees the question as settled."""
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except BadRequest:
        # Message too old / already edited: nothing to clean up.
        pass


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        data = json.loads(query.data)
    except json.JSONDecodeError:
        await query.answer()
        return

    action = data.get("action")
    key = f"approve_code:{data.get('library')}" if action == "approve_code" else action
    future = session.pending.pop(key, None)

    if future and not future.done():
        future.set_result(data)
        # Acknowledge the press: toast on the button + keyboard removed, so
        # the question reads as settled and can't be pressed twice.
        await query.answer(_button_feedback(data))
        await _clear_buttons(query)
    else:
        # Stale press (question already answered or session gone): make the
        # leftover keyboard disappear instead of silently ignoring the tap.
        await query.answer("Already handled.")
        await _clear_buttons(query)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """First contact with the bot: confirms the chat is reachable and shows usage."""
    if session.running:
        await update.message.reply_text("⏳ Session in progress, use the buttons to respond, or /cancel to abort.")
        return
    await send(context.application, READY_MESSAGE)


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not session.running or session.task is None:
        await update.message.reply_text("No session in progress.")
        return
    # Cancelling the task unblocks any pending ask_with_buttons await;
    # run_workflow's finally block resets the state and re-sends the prompt.
    session.task.cancel()


def start_session(app: Application, analytical_request: str):
    session.running = True
    session.task = app.create_task(run_workflow(app, analytical_request))




async def notify_successes(app: Application, state_values: dict, notified: set):
    """One message per branch whose Docker benchmark succeeded, sent the first
    time the result shows up in the thread state (the bot re-reads state at
    every wake-up, so a branch would otherwise be announced repeatedly)."""
    for lib in LIBRARIES:
        if lib in notified:
            continue
        metrics = (state_values.get(lib) or {}).get("metrics") or {}
        if not metrics.get("success"):
            continue
        notified.add(lib)
        elapsed = float(metrics.get("elapsed_seconds") or 0)
        peak_mb = float(metrics.get("peak_rss_bytes") or 0) / (1024 * 1024)
        await send(
            app,
            f"✅ <b>{lib.upper()}</b> executed successfully: "
            f"{elapsed:.2f}s, peak RAM {peak_mb:.0f} MB"
        )


async def send_artifacts(app: Application, out_dir: str):
    """Delivers chart, report and winning script straight into the chat. Works
    when the bot shares the filesystem with the server (aegra dev); in
    full-container mode the files live inside the API container, so missing
    files are just skipped; the final message already lists their paths.

    Each file is sent independently with generous timeouts: one failure (e.g.
    a Telegram read timeout AFTER the upload went through) must not swallow
    the remaining artifacts, and delivery is a nicety: never let it kill the
    session epilogue."""
    async def send_one(filename: str, as_photo: bool = False):
        path = os.path.join(out_dir, filename)
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                if as_photo:
                    await app.bot.send_photo(
                        chat_id=TELEGRAM_CHAT_ID, photo=f,
                        read_timeout=30, write_timeout=30,
                    )
                else:
                    await app.bot.send_document(
                        chat_id=TELEGRAM_CHAT_ID, document=f,
                        read_timeout=30, write_timeout=30,
                    )
        except Exception as e:
            logger.warning("Could not send artifact to the chat", filename=filename, error=str(e))

    await send_one("performance_chart.png", as_photo=True)
    await send_one("data_analysis_report.md")
    await send_one("approved_analytics_solution.py")


# ---------------------------------------------------------------------------
# Aegra workflow driver
# ---------------------------------------------------------------------------

async def resume_run(aegra: httpx.AsyncClient, thread_id: str, interrupt_id: str, value):
    """Resumes an interrupted thread by targeting one specific interrupt.

    Per Aegra/LangGraph's HITL protocol, resuming is done by creating a NEW
    run on the thread with a `command.resume` mapping keyed by the interrupt's
    own `id` (not a flat value). LangGraph only routes the value to the task
    whose interrupt id matches; every other pending interrupt on the same
    thread is left untouched, which is what makes independent per-branch
    approval possible. This also means a fresh `run_id` is returned that the
    caller must switch to for subsequent polling.
    """
    return await aegra.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": ASSISTANT_ID,
            "command": {"resume": {interrupt_id: value}}
        }
    )


# Compact callback_data codes (Telegram caps callback_data at 64 bytes)
# mapped to the decision values the graph's approve node expects.
DECISIONS = {"a": "approve", "g": "regenerate", "x": "reject"}


async def handle_approve_code(app, aegra, thread_id, intr_id, payload) -> str | None:
    """Asks for code approval on Telegram, resumes the branch, returns the new run_id."""
    library = payload.get("library")
    code = payload.get("code")
    reasoning = payload.get("reasoning")
    attempt = payload.get("attempt", 1)
    execution_error = payload.get("execution_error")
    fix_attempt = payload.get("fix_attempt", 0)
    max_fix_attempts = payload.get("max_fix_attempts")
    logger.info("Interrupt: code ready for approval", library=library, attempt=attempt)

    # Budget under Telegram's 4096-char limit:
    # ~400 for the execution error, ~500 for reasoning, ~2600 for code
    title = f"Code Approval: {library.upper()}"
    if execution_error and fix_attempt:
        title += f" - fix after runtime error, attempt {fix_attempt}/{max_fix_attempts}"
    elif attempt > 1:
        title += f" - regenerated, attempt {attempt}"
    text = f"<b>{title}</b>\n\n"
    if execution_error:
        text += f"⚠️ <b>Previous execution failed:</b>\n<code>{clip(escape(execution_error), 400)}</code>\n\n"
    if reasoning:
        text += f"💡 <i>{clip(escape(reasoning), 500)}</i>\n\n"
    text += f"<pre><code class='language-python'>{clip(escape(code or ''), 2600)}</code></pre>"

    answer = await ask_with_buttons(
        app,
        f"approve_code:{library}",
        text,
        [[
            button("✅ Approve", {"action": "approve_code", "d": "a", "library": library}),
            button("🔄 Regenerate", {"action": "approve_code", "d": "g", "library": library}),
            button("❌ Reject", {"action": "approve_code", "d": "x", "library": library}),
        ]],
    )
    decision = DECISIONS.get(answer.get("d"), "reject")

    # Resume the run on Aegra, targeting only this interrupt's id so the
    # other still-pending branches are left untouched.
    logger.info("Resuming run with decision", library=library, decision=decision)
    res = await resume_run(aegra, thread_id, intr_id, {"decision": decision})
    if res.status_code != 200:
        logger.error("Error resuming the run", library=library, status_code=res.status_code, response=res.text)
        return None
    logger.info("Branch execution resumed", library=library)
    return res.json()["run_id"]


async def handle_select_strategy(app, aegra, thread_id, intr_id, payload) -> str | None:
    """Asks for the winning strategy on Telegram, resumes, returns the new run_id."""
    results = payload.get("results", {})
    logger.info("Interrupt: benchmark results ready for selection")

    table = f"{'Library':<10} | {'Time (s)':<9} | {'RAM (MB)':<8}\n" + "-" * 31 + "\n"
    for lib, metrics in results.items():
        # No elapsed_seconds at all = the code never ran in Docker (e.g.
        # rejected); a real execution, even a failed one, always has it.
        if "elapsed_seconds" not in (metrics or {}):
            table += f"{lib.upper():<10} | not executed\n"
            continue
        if not metrics.get("success"):
            # Same rule as the report table: a failed run's timings are noise
            table += f"{lib.upper():<10} | failed\n"
            continue
        time_sec = metrics.get("elapsed_seconds", 0)
        mem_mb = metrics.get("net_memory_used_bytes", 0) / (1024 * 1024)
        table += f"{lib.upper():<10} | {time_sec:<9.4f} | {mem_mb:<8.2f}\n"

    # Only branches that actually succeeded are eligible as the winner; if
    # every branch failed there is nothing meaningful to pick, so fall back
    # to all of them rather than presenting an empty keyboard.
    eligible = [lib for lib, m in results.items() if (m or {}).get("success")] or list(results)
    buttons = [button(lib.upper(), {"action": "select_strategy", "selected": lib}) for lib in eligible]
    keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]  # pairs per row

    answer = await ask_with_buttons(
        app,
        "select_strategy",
        "<b>Benchmarks Complete!</b>\n\n<pre>" + table + "</pre>\n"
        "Select the strategy to promote to production:",
        keyboard,
    )
    selected = answer.get("selected")

    logger.info("Resuming run with selected strategy", strategy=selected)
    res = await resume_run(aegra, thread_id, intr_id, {"selected_strategy": selected})
    if res.status_code != 200:
        logger.error("Error resuming the run", status_code=res.status_code, response=res.text)
        return None
    logger.info("Execution resumed, generating the report")
    return res.json()["run_id"]


async def run_workflow(app: Application, analytical_request: str):
    """Runs one full benchmark session (fresh thread) for the given request."""
    try:
        async with httpx.AsyncClient(base_url=AEGRA_URL, timeout=30) as aegra:
            # 1. Create a new thread (fresh state for every session)
            thread_id = (await aegra.post("/threads", json={})).json()["thread_id"]
            logger.info("Thread created on Aegra", thread_id=thread_id)

            # 2. Start the run
            logger.info("Starting benchmark workflow", request=analytical_request)
            await send(app, f"🚀 <b>Session started!</b>\n\nRequest: <i>{escape(analytical_request)}</i>")
            run_res = await aegra.post(
                f"/threads/{thread_id}/runs",
                json={
                    "assistant_id": ASSISTANT_ID,
                    "input": {
                        # Paths are server-side config (ANALYTICS_DB_PATH /
                        # OUTPUT_BASE_DIR env vars), not run input.
                        "analytical_request": analytical_request,
                    }
                }
            )
            if run_res.status_code != 200:
                logger.error("Error starting the run", status_code=run_res.status_code, response=run_res.text)
                await send(app, "❌ Error starting the workflow. Check the server logs.")
                return
            run_id = run_res.json()["run_id"]
            logger.info("Run started", run_id=run_id)

            # 3. Event loop. No fixed-rate polling: GET .../join blocks
            # server-side (sending a heartbeat byte every ~5s) until the run
            # reaches a terminal state (success, error or interrupted), then
            # we read the status once and react.
            handled_interrupts = set()
            notified_successes = set()
            logger.info("Waiting for the parallel branches to run")
            while True:
                try:
                    await aegra.get(
                        f"/threads/{thread_id}/runs/{run_id}/join",
                        # read timeout > heartbeat interval; heartbeats reset it,
                        # so arbitrarily long runs are fine
                        timeout=httpx.Timeout(30.0, read=60.0),
                    )
                    run_info = (await aegra.get(f"/threads/{thread_id}/runs/{run_id}")).json()
                except httpx.HTTPError as e:
                    logger.warning("Network error while waiting for the run", error=str(e))
                    await asyncio.sleep(2)
                    continue

                status = run_info.get("status")

                if status == "success":
                    logger.info("Workflow completed successfully")
                    state_values = (await aegra.get(f"/threads/{thread_id}/state")).json().get("values", {})
                    await notify_successes(app, state_values, notified_successes)
                    # Same derivation as the server's report node (the bot and
                    # the server load the same .env, so the base stays in sync)
                    out_dir = f"{os.getenv('OUTPUT_BASE_DIR', 'output')}/{thread_id}"
                    selected = state_values.get("selected_strategy", "n/a")
                    logger.info("Session output saved", out_dir=out_dir, selected_strategy=selected)
                    await send(
                        app,
                        "🏁 <b>Workflow completed successfully!</b>\n\n"
                        f"Winning strategy: <b>{str(selected).upper()}</b>\n"
                        f"Output in <code>{escape(str(out_dir))}</code>:\n"
                        "• <code>data_analysis_report.md</code>, comparative report\n"
                        "• <code>approved_analytics_solution.py</code>, winning script\n"
                        "• <code>performance_chart.png</code>, time/memory chart"
                    )
                    await send_artifacts(app, out_dir)
                    return

                if status in ("failed", "error", "cancelled", "timeout"):
                    # Aegra's Run model calls the field error_message, not error
                    reason = run_info.get("error_message") or "unknown (see server logs)"
                    logger.error("The run ended with an error", status=status, reason=reason)
                    await send(app, f"❌ <b>The workflow failed.</b>\n\nDetails: <code>{clip(escape(str(reason)), 1000)}</code>")
                    return

                if status == "interrupted":
                    # Thread state lists every currently pending interrupt
                    state_data = (await aegra.get(f"/threads/{thread_id}/state")).json()
                    # The bot wakes here right after each branch's Docker run
                    # (the resume that triggered it ends in a new interrupt or
                    # terminal state), so fresh benchmark results are announced
                    # before the next approval question goes out.
                    await notify_successes(app, state_data.get("values") or {}, notified_successes)
                    resumed = False
                    for task in state_data.get("tasks", []):
                        interrupts = task.get("interrupts", [])
                        if not interrupts:
                            continue
                        intr = interrupts[0]
                        intr_id = intr.get("id") or task.get("id")
                        if intr_id in handled_interrupts:
                            continue

                        payload = intr.get("value", {})
                        action = payload.get("action")
                        new_run_id = None
                        if action == "approve_code":
                            new_run_id = await handle_approve_code(app, aegra, thread_id, intr_id, payload)
                        elif action == "select_strategy":
                            new_run_id = await handle_select_strategy(app, aegra, thread_id, intr_id, payload)

                        if new_run_id:
                            run_id = new_run_id
                            handled_interrupts.add(intr_id)
                            resumed = True
                        break  # one interrupt at a time, then re-check status

                    if not resumed:
                        # Interrupt not yet visible in thread state (or resume
                        # failed): brief pause to avoid a tight re-join loop
                        await asyncio.sleep(1)
                    continue

                # Unexpected status right after join (e.g. still running)
                await asyncio.sleep(2)
    except asyncio.CancelledError:
        # The interrupted Aegra run is simply abandoned: the next session
        # starts on a fresh thread anyway.
        logger.info("Session cancelled by the user")
        await send(app, "🛑 <b>Session cancelled.</b>")
    except Exception as e:
        logger.exception("Unexpected error in the session")
        await send(app, f"❌ <b>Unexpected error in the session:</b> <code>{clip(escape(str(e)), 1000)}</code>")
    finally:
        session.reset()
        logger.info("Session ended, ready for a new request")
        await send(app, READY_MESSAGE)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

async def post_init(app: Application):
    me = await app.bot.get_me()
    logger.info("Telegram bot connected", username=me.username, chat_id=TELEGRAM_CHAT_ID)

    await app.bot.set_my_commands([
        BotCommand("newsession", "Start a new benchmark session with a new request"),
        BotCommand("cancel", "Cancel the current session"),
    ])

    try:
        await send(app, READY_MESSAGE)
    except BadRequest as e:
        # Typically "Chat not found": the bot can't message a chat that never
        # contacted it, or the configured id is wrong. Stay alive so a /start
        # from the right chat can unblock everything.
        logger.warning(
            "Unable to message the configured chat. Open the chat with the bot "
            "and press Start (a bot can't message first), and verify that "
            "TELEGRAM_CHAT_ID in .env is the right numeric ID (e.g. from "
            "@userinfobot). The bot stays listening: after the fix, send "
            "/start in the chat to verify the connection.",
            chat_id=TELEGRAM_CHAT_ID, username=me.username, error=e.message,
        )
    logger.info("Waiting for a new request from the Telegram chat (/newsession)")


def main():
    setup_logging()

    required = {"AEGRA_URL": AEGRA_URL, "TELEGRAM_BOT_TOKEN": TELEGRAM_TOKEN, "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID}
    missing = [name for name, value in required.items() if not value]
    if missing:
        logger.error("Missing variables in .env", missing=missing)
        return
    if not TELEGRAM_CHAT_ID.lstrip("-").isdigit():
        logger.error(
            "TELEGRAM_CHAT_ID must be the numeric chat ID (get it from @userinfobot on Telegram)",
            found=TELEGRAM_CHAT_ID,
        )
        return

    logger.info("Checking the connection to the Aegra server", url=AEGRA_URL)
    try:
        httpx.get(f"{AEGRA_URL}/threads")
    except httpx.HTTPError:
        logger.error(
            "Unable to connect to Aegra; make sure the server is running via 'aegra dev'",
            url=AEGRA_URL,
        )
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    # Only the configured chat can drive the bot
    chat_filter = filters.Chat(int(TELEGRAM_CHAT_ID))
    app.add_handler(CommandHandler("start", cmd_start, filters=chat_filter))
    app.add_handler(CommandHandler("newsession", cmd_newsession, filters=chat_filter))
    app.add_handler(CommandHandler("cancel", cmd_cancel, filters=chat_filter))
    app.add_handler(MessageHandler(chat_filter & filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(CallbackQueryHandler(on_button))

    logger.info("Telegram bot started: the initial request and HITL flow through the chat")
    # drop_pending_updates discards anything typed while the bot was offline
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
