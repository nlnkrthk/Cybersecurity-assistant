#!/usr/bin/env python3
"""Start the incident-response assistant API (FastAPI + Uvicorn).

    python scripts/api/run_api.py                 # http://127.0.0.1:8000, docs at /docs
    python scripts/api/run_api.py --port 8080 --reload
"""
import argparse
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # hide embedding progress bars
warnings.filterwarnings("ignore")

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import uvicorn  # noqa: E402

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the incident-response assistant API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, local only).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Restart on code changes (development).")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("huggingface_hub", "sentence_transformers", "transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    print(f"API docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "api.main:app",
        app_dir=str(SRC_DIR),
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(SRC_DIR)] if args.reload else None,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
