"""SQLite run history. One file, no server, easy to cache in CI."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .metrics import RunMetrics
from .runner import CaseResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    prompt_hash      TEXT NOT NULL,
    model            TEXT NOT NULL,
    judge_model      TEXT NOT NULL,
    dataset_version  TEXT NOT NULL,
    dataset_hash     TEXT NOT NULL,
    track            TEXT NOT NULL,
    git_sha          TEXT,
    is_baseline      INTEGER NOT NULL DEFAULT 0,
    status           TEXT,
    metrics_json     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_id             INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    case_id            TEXT NOT NULL,
    data_json          TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_track ON runs(track, id);
"""


@dataclass
class RunRecord:
    id: int
    created_at: str
    prompt_version: str
    prompt_hash: str
    model: str
    judge_model: str
    dataset_version: str
    dataset_hash: str
    track: str
    git_sha: str | None
    is_baseline: bool
    status: str | None
    metrics: RunMetrics


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_run(self, *, prompt_version: str, prompt_hash: str, model: str, judge_model: str,
                 dataset_version: str, dataset_hash: str, track: str, git_sha: str | None,
                 metrics: RunMetrics, results: list[CaseResult]) -> int:
        with closing(self._conn()) as c, c:
            cur = c.execute(
                "INSERT INTO runs (created_at, prompt_version, prompt_hash, model, judge_model,"
                " dataset_version, dataset_hash, track, git_sha, metrics_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), prompt_version,
                 prompt_hash, model, judge_model, dataset_version, dataset_hash, track, git_sha,
                 metrics.model_dump_json()),
            )
            run_id = int(cur.lastrowid)
            c.executemany(
                "INSERT INTO results (run_id, case_id, data_json) VALUES (?,?,?)",
                [(run_id, r.case_id, r.model_dump_json()) for r in results],
            )
        return run_id

    def set_status(self, run_id: int, status: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))

    def mark_baseline(self, run_id: int) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE runs SET is_baseline=1 WHERE id=?", (run_id,))

    @staticmethod
    def _record(row: sqlite3.Row) -> RunRecord:
        d = dict(row)
        metrics = RunMetrics.model_validate_json(d.pop("metrics_json"))
        d["is_baseline"] = bool(d["is_baseline"])
        return RunRecord(metrics=metrics, **d)

    def get_run(self, run_id: int) -> RunRecord | None:
        with closing(self._conn()) as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._record(row) if row else None

    def latest_baseline(self, exclude_id: int | None = None) -> RunRecord | None:
        """Most recent baseline that actually ran: skips runs marked 'error' and
        runs where at least half the cases errored (e.g. a bad API key)."""
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT * FROM runs WHERE is_baseline=1 AND id != ? ORDER BY id DESC LIMIT 50",
                (exclude_id or -1,),
            ).fetchall()
        for row in rows:
            rec = self._record(row)
            broken = rec.status == "error" or (
                rec.metrics.n_cases and rec.metrics.errors * 2 >= rec.metrics.n_cases)
            if not broken:
                return rec
        return None

    def latest_run(self, track: str | None = None, exclude_id: int | None = None) -> RunRecord | None:
        q = "SELECT * FROM runs WHERE id != ?"
        args: list = [exclude_id or -1]
        if track:
            q += " AND track = ?"
            args.append(track)
        with closing(self._conn()) as c:
            row = c.execute(q + " ORDER BY id DESC LIMIT 1", args).fetchone()
        return self._record(row) if row else None

    def results(self, run_id: int) -> list[CaseResult]:
        with closing(self._conn()) as c:
            rows = c.execute("SELECT data_json FROM results WHERE run_id=? ORDER BY case_id",
                             (run_id,)).fetchall()
        return [CaseResult.model_validate_json(r["data_json"]) for r in rows]

    def history(self, track: str | None = None, limit: int = 30) -> list[RunRecord]:
        """Chronological (oldest first)."""
        q, args = "SELECT * FROM runs", []
        if track:
            q += " WHERE track = ?"
            args.append(track)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with closing(self._conn()) as c:
            rows = c.execute(q, args).fetchall()
        return [self._record(r) for r in reversed(rows)]
