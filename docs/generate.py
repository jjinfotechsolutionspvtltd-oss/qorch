"""Generate API documentation with pdoc.

Usage:
    pip install pdoc
    python docs/generate.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    docs_dir = Path(__file__).resolve().parent
    output_dir = docs_dir / "_build"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add src to PYTHONPATH
    src_dir = docs_dir.parent / "src"

    env = {"PYTHONPATH": str(src_dir)}
    cmd = [
        sys.executable, "-m", "pdoc",
        "--docformat", "google",
        "--output-dir", str(output_dir),
        "qorch",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env={**env, **{k: v for k, v in zip(env.keys(), env.values())}}, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Docs generated in {output_dir}")


if __name__ == "__main__":
    main()
