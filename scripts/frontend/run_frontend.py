#!/usr/bin/env python3
"""Start the Streamlit front end (the API must be running: scripts/api/run_api.py).

    python scripts/frontend/run_frontend.py                       # http://localhost:8501
    python scripts/frontend/run_frontend.py --port 8502 --api http://127.0.0.1:8080
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP = PROJECT_ROOT / "frontend" / "app.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Streamlit front end.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="localhost", help="Bind address (default: localhost only).")
    parser.add_argument("--api", default=None, help="API base URL (default: API_BASE_URL or http://localhost:8000).")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.api:
        env["API_BASE_URL"] = args.api
    print(f"Front end: http://{args.address}:{args.port}  (API: {env.get('API_BASE_URL', 'http://localhost:8000')})")
    command = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(args.port),
        "--server.address", args.address,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    # Started from the project root so Streamlit reads .streamlit/config.toml (theme, toolbar).
    sys.exit(subprocess.call(command, env=env, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
