import asyncio
import os
import json
import shutil
import subprocess
import tempfile
import uuid

# Serializes DB generation so the 4 parallel branches don't race to create it
_db_lock = asyncio.Lock()

# Serializes container EXECUTION: generation and approvals stay parallel, but
# benchmarks run one at a time so the 4 containers don't compete for CPU/RAM
# and skew the timing/memory measurements.
_benchmark_lock = asyncio.Lock()


def ensure_docker_image_built():
    """Checks if the benchmark-runner docker image exists; if not, builds it."""
    print("Checking if Docker image 'benchmark-runner' exists...")
    try:
        # Run docker images to see if it is built
        result = subprocess.run(
            ["docker", "images", "-q", "benchmark-runner"],
            capture_output=True,
            text=True,
            check=True
        )
        if not result.stdout.strip():
            print("Docker image 'benchmark-runner' not found. Building it now from Dockerfile.benchmark...")
            # Run docker build
            # We assume Dockerfile.benchmark is in the workspace root
            subprocess.run(
                ["docker", "build", "-t", "benchmark-runner", "-f", "Dockerfile.benchmark", "."],
                check=True
            )
            print("Docker image built successfully.")
        else:
            print("Docker image 'benchmark-runner' is ready.")
    except Exception as e:
        print(f"Warning: Failed to check/build Docker image. Error: {e}")
        print("Please make sure Docker is installed and running.")


def _parse_json_from_stdout(stdout_content: str) -> dict:
    """Parses JSON from stdout, searching from the last line backwards.

    This is robust against user scripts that print to stdout before
    the wrapper's JSON output line.
    """
    lines = stdout_content.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            # parse_constant=str turns NaN/Infinity/-Infinity tokens (emitted
            # by the wrapper's json.dumps for non-finite floats, e.g. rolling
            # means) into their string names. Left as floats they'd reach the
            # graph state and kill the run: Postgres JSONB rejects the NaN
            # token when Aegra persists the run output.
            return json.loads(line, parse_constant=str)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No valid JSON line found in stdout", stdout_content, 0)


async def _ensure_database(db_path: str):
    """Generates the SQLite DB if missing. Runs server-side so it works both in
    host mode and inside the API container (full-container mode)."""
    async with _db_lock:
        if not os.path.exists(db_path):
            print(f"Database {db_path} not found: generating synthetic data...")
            from data_generator import generate_database
            await asyncio.to_thread(generate_database, db_path)


def _prepare_run(library_name: str, code_content: str, db_path: str):
    """Stages the script (and, in volume mode, a DB copy) and builds the docker
    command. Returns (cmd, cleanup_paths).

    Two modes:
    - Host mode (aegra dev): bind-mount the script and DB from the local
      filesystem, which the docker daemon can see directly.
    - Full-container mode (BENCHMARK_SHARED_VOLUME set): this process runs
      inside the API container and talks to the HOST docker daemon, so local
      bind mounts would resolve against the host filesystem and come up empty.
      Instead, files are staged into a named volume (mounted here at
      BENCHMARK_SHARED_DIR) that the sibling benchmark container mounts by
      name; the wrapper locates them via SCRIPT_PATH/DB_PATH env vars.
    """
    script_id = f"{library_name}_{uuid.uuid4().hex[:8]}"
    shared_volume = os.getenv("BENCHMARK_SHARED_VOLUME")

    # Applied to every benchmark container: no network (LLM-generated code
    # has no business talking to the outside, and the SQLite DB is local) and
    # capped CPU/RAM so every library is measured on identical resources.
    resource_flags = [
        "--network", "none",
        "--memory", os.getenv("BENCHMARK_MEMORY_LIMIT", "2g"),
        "--cpus", os.getenv("BENCHMARK_CPU_LIMIT", "2"),
    ]

    if shared_volume:
        shared_dir = os.getenv("BENCHMARK_SHARED_DIR", "/shared")
        scripts_dir = os.path.join(shared_dir, "scripts")
        data_dir = os.path.join(shared_dir, "data")
        os.makedirs(scripts_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        script_filepath = os.path.join(scripts_dir, f"{script_id}.py")
        with open(script_filepath, "w", encoding="utf-8") as f:
            f.write(code_content)

        # Per-run private DB copy: the unique name avoids read/write races
        # between the 4 branches staging into the same volume concurrently.
        db_copy = os.path.join(data_dir, f"{script_id}.sqlite")
        shutil.copyfile(db_path, db_copy)

        cmd = [
            "docker", "run", "--rm", *resource_flags,
            "-v", f"{shared_volume}:/shared:ro",
            "-e", f"SCRIPT_PATH=/shared/scripts/{script_id}.py",
            "-e", f"DB_PATH=/shared/data/{script_id}.sqlite",
            "benchmark-runner"
        ]
        return cmd, [script_filepath, db_copy]

    # Host mode: bind-mount script + DB read-only into the container.
    # The scripts go to the SYSTEM temp dir, NOT inside the project: `aegra dev`
    # watches the project tree for *.py changes and would hot-reload the server
    # mid-benchmark, killing the run.
    temp_dir = os.path.join(tempfile.gettempdir(), "benchmark_scripts")
    os.makedirs(temp_dir, exist_ok=True)
    script_filepath = os.path.join(temp_dir, f"{script_id}.py")
    with open(script_filepath, "w", encoding="utf-8") as f:
        f.write(code_content)

    cmd = [
        "docker", "run", "--rm", *resource_flags,
        "-v", f"{os.path.abspath(db_path)}:/data/db.sqlite:ro",
        "-v", f"{script_filepath}:/app/script.py:ro",
        "benchmark-runner"
    ]
    return cmd, [script_filepath]


def _failure(error: str, elapsed: float = 0.0, infra: bool = False) -> dict:
    """infra=True marks failures of the benchmark infrastructure itself
    (docker unavailable, staging error, broken wrapper output), not of the
    generated code. The execution router skips the LLM fix loop for these:
    regenerating the code can't fix a docker daemon that is down."""
    return {
        "success": False,
        "infra": infra,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": 0,
        "net_memory_used_bytes": 0,
        "preview": None,
        "error": error
    }


async def run_in_docker(library_name: str, code_content: str, db_path: str = None) -> dict:
    """Executes the generated code in an isolated container and returns profiling metrics.

    The DB location is deployment config, not per-run data: when db_path is not
    given it comes from the ANALYTICS_DB_PATH env var (resolved server-side).
    """
    if db_path is None:
        db_path = os.getenv("ANALYTICS_DB_PATH", "data/analytics.db")
    ensure_docker_image_built()
    await _ensure_database(db_path)

    cleanup_paths = []
    try:
        cmd, cleanup_paths = _prepare_run(library_name, code_content, db_path)

        print(f"Launching Docker container for {library_name} benchmark...")
        # Blocking subprocess in a worker thread instead of asyncio subprocess:
        # on Windows asyncio subprocesses need the Proactor loop, but the Aegra
        # server must run on the Selector loop (psycopg async requires it).
        # subprocess.run works under any event loop.
        # The lock serializes the actual measurements (see _benchmark_lock).
        try:
            async with _benchmark_lock:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    timeout=120,  # 2 minute timeout per container
                )
        except subprocess.TimeoutExpired:
            # Not infra: code that runs too slowly is exactly what the LLM
            # fix loop can address (e.g. by optimizing the query).
            return _failure("Execution timed out after 120 seconds in Docker container.", elapsed=120.0)

        stdout_content = completed.stdout.decode("utf-8", errors="replace").strip()
        stderr_content = completed.stderr.decode("utf-8", errors="replace").strip()

        if completed.returncode != 0:
            # 125 = the docker CLI/daemon failed before the container started;
            # a daemon that is down/unreachable surfaces as a connect error in
            # stderr. Both are infrastructure, not the generated code. Other
            # exit codes (user script killed, OOM 137, ...) stay fixable.
            stderr_lower = stderr_content.lower()
            infra = (
                completed.returncode == 125
                or "cannot connect to the docker daemon" in stderr_lower
                or "error during connect" in stderr_lower
            )
            return _failure(
                f"Docker run failed with exit code {completed.returncode}.\nStderr: {stderr_content}",
                infra=infra,
            )

        # Find JSON block in stdout (search from last line backwards for robustness)
        try:
            parsed_metrics = _parse_json_from_stdout(stdout_content)
        except json.JSONDecodeError:
            # Exit 0 but no JSON line: the wrapper contract itself is broken
            # (stale image, corrupted wrapper); regenerating code won't help.
            return _failure(
                f"Failed to parse JSON output from wrapper. Raw Output: {stdout_content}\nStderr: {stderr_content}",
                infra=True,
            )

        # If wrapper reported an execution error inside python
        if parsed_metrics.get("error"):
            return {
                "success": False,
                "elapsed_seconds": parsed_metrics.get("elapsed_seconds", 0),
                "peak_rss_bytes": parsed_metrics.get("peak_rss_bytes", 0),
                "net_memory_used_bytes": parsed_metrics.get("net_memory_used_bytes", 0),
                "preview": None,
                "error": parsed_metrics.get("error")
            }

        parsed_metrics["success"] = True
        return parsed_metrics

    except Exception as e:
        # Anything raised host-side (docker binary missing, staging IO error,
        # ...) happened before/around the container, infra by definition.
        return _failure(f"Failed to run Docker benchmark: {str(e)}", infra=True)
    finally:
        for path in cleanup_paths:
            if os.path.exists(path):
                os.remove(path)
