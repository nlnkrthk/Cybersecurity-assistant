"""
Evaluation of the assistant's answers with DeepEval plus deterministic checks.

    cases.py           labelled question set (cases/deepeval_cases.json) and test-case building
    checks.py          exact checks from the stored run: route, attack type, priority, guardrails, caveats
    deepeval_model.py  the evaluator model (NIM / Gemini / Claude) behind DeepEval's model interface
    runner.py          generate answers (resumable), score them, write the report

Importing this package configures DeepEval before it is imported anywhere: no telemetry, no
reading of the project's .env, and its cache folder under data/evaluation/ instead of the
project root.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT  # noqa: E402

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
os.environ.setdefault("DEEPEVAL_CACHE_FOLDER", str(EVALUATION_DIR / ".deepeval"))
