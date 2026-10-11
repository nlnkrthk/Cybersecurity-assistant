#!/usr/bin/env python3
"""
Write the small sample data set in data/sample_data/ (committed, for reviewers and demos).

    python scripts/data/make_sample_data.py

- sample_incidents.csv     up to 8 incidents per traffic type from the prepared incident library:
                           ID, label, port, the 24 model features and the indexed incident text
- triage_queue_sample.csv  a CSV for the Triage Queue page / POST /ml/prioritize/csv: incident IDs
                           and hand-entered flows (some partial)
- sample_queries.json      example questions with the facts the answer must contain and an excerpt
                           of the assistant's real answer (from the stored DeepEval runs, if present)
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from config import CLEANED_INCIDENTS_FILE  # noqa: E402
from services.feature_input import FEATURE_NAMES  # noqa: E402

OUT = ROOT / "data" / "sample_data"
PER_LABEL = 8
TRIAGE_SOURCE = ROOT / "data" / "samples" / "triage_test.csv"
CASES = ROOT / "src" / "evaluation" / "cases" / "deepeval_cases.json"
RUNS = [ROOT / "data" / "evaluation" / f"deepeval_run_20261010_{name}.jsonl" for name in ("full", "searchfix")]
SAMPLE_CASES = ["classify-incident", "profile-ddos", "risk-rank", "search-ssh", "investigate-ssh",
                "general-dos-ddos", "guard-injection", "guard-redaction"]


def incidents() -> pd.DataFrame:
    frame = pd.read_parquet(CLEANED_INCIDENTS_FILE, columns=["incident_id", "label", "dst_port", *FEATURE_NAMES,
                                                             "incident_text"])
    sample = (frame.sort_values("incident_id").groupby("label", group_keys=False).head(PER_LABEL)
              .sort_values(["label", "incident_id"]))
    sample["dst_port"] = sample["dst_port"].astype(int)
    return sample.reset_index(drop=True)


def queries() -> list:
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    answers = {}
    for run in RUNS:  # a later run (after a fix) replaces earlier answers
        if run.exists():
            for line in run.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("answer"):
                    answers[row["case_id"]] = row
    out = []
    for case_id in SAMPLE_CASES:
        case, row = cases[case_id], answers.get(case_id, {})
        expect = {k: v for k, v in case.get("expect", {}).items() if v not in (None, [], False)}
        answer = (row.get("answer") or "").strip()
        out.append({
            "id": case_id,
            "question": case["question"],
            "expected": expect,
            "route": row.get("route"),
            "agents_run": row.get("agents_run"),
            "answer_excerpt": answer[:900] + (" …" if len(answer) > 900 else ""),
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = incidents()
    sample.to_csv(OUT / "sample_incidents.csv", index=False)
    print(f"sample_incidents.csv: {len(sample)} rows, {sample['label'].nunique()} labels")
    if TRIAGE_SOURCE.exists():
        shutil.copyfile(TRIAGE_SOURCE, OUT / "triage_queue_sample.csv")
        print("triage_queue_sample.csv copied")
    rows = queries()
    (OUT / "sample_queries.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"sample_queries.json: {len(rows)} queries")


if __name__ == "__main__":
    main()
