"""
SQLite store for analyst feedback on assistant answers.

One row per rated turn (thread_id + turn_id); rating the same turn again updates it.
Each row keeps what the analyst said (rating, comment, corrected attack label) and a
snapshot of the turn taken from the conversation store (question, answer, route, model
prediction, incident / flow, input quality, judge verdict, LLM model) and who rated it, so feedback can be
analysed and used for retraining later without the conversation.
"""
import json
import sqlite3
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    thread_id             TEXT NOT NULL,
    turn_id               TEXT NOT NULL,
    rating                TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment               TEXT,
    corrected_label       TEXT,
    question              TEXT,
    answer                TEXT,
    route                 TEXT,
    agents_run            TEXT,
    predicted_label       TEXT,
    prediction_confidence REAL,
    incident_id           TEXT,
    historical_label      TEXT,
    flow_args             TEXT,
    input_quality         TEXT,
    judge_verdict         TEXT,
    judge_overall         REAL,
    llm_model             TEXT,
    username              TEXT,
    UNIQUE (thread_id, turn_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_incident ON feedback (incident_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback (created_at);
"""

JSON_COLUMNS = ("agents_run", "flow_args")
SNAPSHOT_COLUMNS = (
    "question", "answer", "route", "agents_run", "predicted_label", "prediction_confidence", "incident_id",
    "historical_label", "flow_args", "input_quality", "judge_verdict", "judge_overall", "llm_model", "username",
)
RATINGS = ("up", "down")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FeedbackStore:
    """Thread-safe SQLite feedback store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(feedback)")}
            if "username" not in columns:  # databases created before user accounts
                self._conn.execute("ALTER TABLE feedback ADD COLUMN username TEXT")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add(
        self,
        thread_id: str,
        turn_id: str,
        rating: str,
        snapshot: Dict[str, Any],
        comment: Optional[str] = None,
        corrected_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store (or update) feedback for one turn and return the stored row."""
        if rating not in RATINGS:
            raise ValueError(f"rating must be one of {RATINGS}, got {rating!r}")
        values = {column: snapshot.get(column) for column in SNAPSHOT_COLUMNS}
        for column in JSON_COLUMNS:
            if values[column] is not None:
                values[column] = json.dumps(values[column], default=str)
        now = _now()
        row = {
            "created_at": now, "updated_at": now, "thread_id": thread_id, "turn_id": turn_id, "rating": rating,
            "comment": (comment or "").strip() or None, "corrected_label": corrected_label, **values,
        }
        columns = ", ".join(row)
        placeholders = ", ".join(f":{name}" for name in row)
        updates = ", ".join(f"{name} = excluded.{name}" for name in row if name not in ("created_at", "thread_id", "turn_id"))
        with self._lock:
            self._conn.execute(
                f"INSERT INTO feedback ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT (thread_id, turn_id) DO UPDATE SET {updates}",
                row,
            )
            self._conn.commit()
            stored = self._conn.execute(
                "SELECT * FROM feedback WHERE thread_id = ? AND turn_id = ?", (thread_id, turn_id)
            ).fetchone()
        return self._row(stored)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        out = dict(row)
        for column in JSON_COLUMNS:
            if out.get(column):
                out[column] = json.loads(out[column])
        return out

    def list(
        self,
        rating: Optional[str] = None,
        route: Optional[str] = None,
        label: Optional[str] = None,
        corrected_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Newest first, optionally filtered (label matches the prediction or the correction)."""
        where, params = [], []
        if rating:
            where.append("rating = ?")
            params.append(rating)
        if route:
            where.append("route = ?")
            params.append(route)
        if label:
            where.append("(predicted_label = ? OR corrected_label = ?)")
            params += [label, label]
        if corrected_only:
            where.append("corrected_label IS NOT NULL")
        sql = "SELECT * FROM feedback"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._conn.execute(sql, [*params, limit, offset]).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    def corrections_for(self, incident_id: str) -> Dict[str, int]:
        """Analyst-corrected labels for one incident: {label: number of analysts}."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT corrected_label, COUNT(*) AS n FROM feedback "
                "WHERE incident_id = ? AND corrected_label IS NOT NULL GROUP BY corrected_label ORDER BY n DESC",
                (incident_id,),
            ).fetchall()
        return {r["corrected_label"]: r["n"] for r in rows}

    def stats(self) -> Dict[str, Any]:
        """Approval rates, judge-vs-human agreement and the most common corrections."""
        rows = self.list(limit=1_000_000)
        total = len(rows)
        up = sum(r["rating"] == "up" for r in rows)

        by_route: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "up": 0})
        for r in rows:
            entry = by_route[r["route"] or "unknown"]
            entry["n"] += 1
            entry["up"] += r["rating"] == "up"
        for entry in by_route.values():
            entry["approval_rate"] = round(entry["up"] / entry["n"], 3)

        # Judge "pass" vs analyst thumbs up; needs_review and fail count as "not pass".
        judged = [r for r in rows if r["judge_verdict"] in ("pass", "needs_review", "fail")]
        matrix = Counter(
            ("judge_pass" if r["judge_verdict"] == "pass" else "judge_not_pass", "human_up" if r["rating"] == "up" else "human_down")
            for r in judged
        )
        agree = matrix[("judge_pass", "human_up")] + matrix[("judge_not_pass", "human_down")]
        judge_agreement = {
            "n": len(judged),
            "pass_up": matrix[("judge_pass", "human_up")],
            "pass_down": matrix[("judge_pass", "human_down")],
            "not_pass_up": matrix[("judge_not_pass", "human_up")],
            "not_pass_down": matrix[("judge_not_pass", "human_down")],
            "agreement_rate": round(agree / len(judged), 3) if judged else None,
        }

        corrections = [r for r in rows if r["corrected_label"]]
        pairs = Counter((r["predicted_label"] or "unknown", r["corrected_label"]) for r in corrections)
        return {
            "total": total,
            "up": up,
            "down": total - up,
            "approval_rate": round(up / total, 3) if total else None,
            "by_route": dict(sorted(by_route.items())),
            "judge_agreement": judge_agreement,
            "corrections": {
                "total": len(corrections),
                "top_pairs": [
                    {"predicted": p, "corrected": c, "count": n} for (p, c), n in pairs.most_common(10)
                ],
            },
            "recent_comments": [
                {"rating": r["rating"], "route": r["route"], "comment": r["comment"], "created_at": r["created_at"]}
                for r in rows if r["comment"]
            ][:5],
        }
