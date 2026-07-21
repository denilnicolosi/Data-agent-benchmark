"""One-shot: seed/re-version the 5 benchmark prompts on Langfuse Prompt Management.

Creates (or adds a new version of) each "benchmark/*" prompt from the
authoritative local texts in prompts.py, labeled "production" (override with
LANGFUSE_PROMPT_LABEL). Requires `pip install langfuse` and the LANGFUSE_*
keys in .env. Re-running creates new versions; it never deletes anything.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Allow running from the repo root without installing the package
sys.path.insert(0, "src")

from benchmark_agent.prompt_store import LOCAL_PROMPTS  


def main():
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing, set them in .env first.")
    from langfuse import Langfuse

    label = os.getenv("LANGFUSE_PROMPT_LABEL", "production")
    # Aegra's tracing config names the host LANGFUSE_BASE_URL; the SDK's own
    # env var is LANGFUSE_HOST, accept both (same rule as prompt_store).
    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    client = Langfuse(**({"host": host} if host else {}))

    for name, text in LOCAL_PROMPTS.items():
        prompt = client.create_prompt(name=name, prompt=text, type="text", labels=[label])
        print(f"Seeded {name} -> version {prompt.version} (label: {label})")
    print(f"\nDone. Set LANGFUSE_PROMPTS=1 in .env and restart `aegra dev` to use them.")


if __name__ == "__main__":
    main()
