"""Convenience wrapper: pilot batch via the CLI (4 parallel workers)."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "src.cli", "generate",
        "--tasks-file", "config/pilot_tasks.json",
        "--start", "2026-12-01T08:00:00+08:00",
        "--jitter-window",
        "--workers", "4",
        "--run-id", "pilot-aligned-v1",
        "--resume",
    ], check=False)
