import time
import os
import json
import importlib.util
import traceback
import threading


def make_serializable(obj):
    if hasattr(obj, 'head') and hasattr(obj, 'to_dicts'):
        try:
            return make_serializable(obj.head(5).to_dicts())
        except Exception:
            pass
    if hasattr(obj, 'head') and hasattr(obj, 'to_dict'):
        try:
            return make_serializable(obj.head(5).to_dict(orient='records'))
        except Exception:
            return str(obj.head(5))
    elif hasattr(obj, 'to_dict'):
        try:
            return make_serializable(obj.to_dict())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {
            k if isinstance(k, str) else str(k): make_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif hasattr(obj, 'tolist'):
        try:
            return make_serializable(obj.tolist())
        except Exception:
            pass
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


class PeakMemoryTracker:
    """Samples RSS in a background thread to capture peak memory during execution."""

    def __init__(self, process):
        self.process = process
        self.peak = process.memory_info().rss
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        def _track():
            while not self._stop.is_set():
                try:
                    current = self.process.memory_info().rss
                    if current > self.peak:
                        self.peak = current
                except Exception:
                    pass
                self._stop.wait(0.01)  # 10ms sampling interval

        self._thread = threading.Thread(target=_track, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak


def profile():
    # Defaults match host mode (bind mounts); full-container mode passes the
    # locations inside the shared named volume via env vars.
    script_path = os.environ.get("SCRIPT_PATH", "/app/script.py")
    db_path = os.environ.get("DB_PATH", "/data/db.sqlite")

    if not os.path.exists(script_path):
        print(json.dumps({"error": f"Script not found at {script_path}"}))
        return

    import psutil
    process = psutil.Process(os.getpid())

    # Measure memory before execution
    initial_rss = process.memory_info().rss

    # Start peak memory tracker
    tracker = PeakMemoryTracker(process)
    tracker.start()

    # Load the user script as a module dynamically
    try:
        spec = importlib.util.spec_from_file_location("user_script", script_path)
        module = importlib.util.module_from_spec(spec)

        start_time = time.perf_counter()
        spec.loader.exec_module(module)

        # We expect the LLM to write an 'analyze(db_path: str)' function
        if hasattr(module, 'analyze'):
            result = module.analyze(db_path)
            preview_data = make_serializable(result)
        else:
            raise AttributeError("The script does not contain an 'analyze(db_path)' function.")

        error = None
    except Exception as e:
        preview_data = None
        error = traceback.format_exc()

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    peak_rss = tracker.stop()
    net_memory_used = max(0, peak_rss - initial_rss)

    output = {
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "net_memory_used_bytes": net_memory_used,
        "preview": preview_data,
        "error": error
    }

    # Output only the JSON result to stdout so the host can parse it cleanly.
    # default=str catches any leaf make_serializable missed; the except arm is
    # the last resort: a benchmark that ran must never die on serialization.
    try:
        print(json.dumps(output, default=str))
    except Exception:
        output["preview"] = str(output.get("preview"))[:2000]
        print(json.dumps(output, default=str))

if __name__ == "__main__":
    profile()
