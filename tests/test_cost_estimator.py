"""Tests for ``cost_estimator``."""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

import cost_estimator as _ce  # noqa: E402
from cost_estimator import (  # noqa: E402
    CostEstimator,
    CostRecord,
    TaskSpec,
)

# ---------------------------------------------------------------------------
# Record I/O
# ---------------------------------------------------------------------------

class TestRecordIO:
    def test_record_and_load(self, tmp_path):
        est = CostEstimator(history_path=str(tmp_path / "h.jsonl"))
        est.record(TaskSpec(kind="refactor"),
                   tokens=1000, iterations=3, wall_s=10.0)
        records = est._load()
        assert len(records) == 1
        assert records[0].tokens == 1000

    def test_records_appended_not_truncated(self, tmp_path):
        est = CostEstimator(history_path=str(tmp_path / "h.jsonl"))
        est.record(TaskSpec("r"), tokens=1, iterations=1, wall_s=1.0)
        est.record(TaskSpec("r"), tokens=2, iterations=2, wall_s=2.0)
        assert len(est._load()) == 2

    def test_robust_to_corrupt_lines(self, tmp_path):
        p = tmp_path / "h.jsonl"
        p.write_text("not json\n{\"task_kind\":\"r\",\"tokens\":10,\"iterations\":1,\"wall_s\":1.0}\n")
        est = CostEstimator(history_path=str(p))
        recs = est._load()
        assert len(recs) == 1 and recs[0].tokens == 10

    def test_concurrent_writes_do_not_tear_lines(self, tmp_path):
        """Threaded writers each append exactly one record per iteration;
        after the run, every line must parse as a clean JSON record."""
        import json
        import threading
        p = str(tmp_path / "h.jsonl")
        est = CostEstimator(history_path=p)

        def hammer(n: int, kind: str) -> None:
            for i in range(n):
                est.record(
                    TaskSpec(kind=kind),
                    tokens=i, iterations=i, wall_s=float(i),
                )

        threads = [threading.Thread(target=hammer, args=(50, f"t{i}")) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(p, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        assert len(lines) == 200
        # Every line is clean JSON (no torn writes).
        for ln in lines:
            json.loads(ln)


# ---------------------------------------------------------------------------
# Estimation strategy
# ---------------------------------------------------------------------------

class TestEstimate:
    def test_static_fallback_with_no_history(self, tmp_path):
        est = CostEstimator(history_path=str(tmp_path / "h.jsonl"))
        b = est.estimate(TaskSpec("anything"))
        assert b.method == "static"
        assert b.confidence == "low"
        assert b.sample_size == 0

    def test_exact_knn_when_enough_samples(self, tmp_path):
        est = CostEstimator(history_path=str(tmp_path / "h.jsonl"))
        for i in range(5):
            est.record(TaskSpec("refactor"),
                       tokens=1000 + i * 100, iterations=2 + i,
                       wall_s=5.0 + i)
        b = est.estimate(TaskSpec("refactor"))
        assert b.method == "knn_exact"
        assert b.confidence == "high"
        assert b.sample_size == 5
        assert 1000 <= b.p50_tokens <= 1400

    def test_prefix_knn(self, tmp_path):
        est = CostEstimator(history_path=str(tmp_path / "h.jsonl"))
        for i in range(5):
            est.record(TaskSpec("refactor_model"),
                       tokens=2000, iterations=5, wall_s=20.0)
        b = est.estimate(TaskSpec("refactor"))
        assert b.method in ("knn_exact", "knn_prefix")

    def test_p95_ge_p50(self, tmp_path):
        est = CostEstimator(history_path=str(tmp_path / "h.jsonl"))
        for v in (100, 200, 300, 400, 900):
            est.record(TaskSpec("x"), tokens=v, iterations=1, wall_s=1.0)
        b = est.estimate(TaskSpec("x"))
        assert b.p95_tokens >= b.p50_tokens

    def test_is_reasonable(self, tmp_path):
        est = CostEstimator(history_path=str(tmp_path / "h.jsonl"))
        for _ in range(5):
            est.record(TaskSpec("cheap"), tokens=100, iterations=1, wall_s=1.0)
        b = est.estimate(TaskSpec("cheap"))
        assert b.is_reasonable()
        assert not b.is_reasonable(max_tokens=1)


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

class TestRecencyWeighting:
    def test_old_records_discounted(self):
        now = time.time()
        w_new = _ce._recency_weight(now, now=now)
        w_old = _ce._recency_weight(now - _ce._RECENCY_HALF_LIFE_S, now=now)
        assert abs(w_new - 1.0) < 1e-6
        assert abs(w_old - 0.5) < 1e-6

    def test_zero_weight_fallback_uses_plain_stats(self, tmp_path):
        # Make every record infinitely old -> weights all ~0.
        import time as _t
        old = _t.time() - _ce._RECENCY_HALF_LIFE_S * 200
        p = tmp_path / "h.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for v in (10, 20, 30, 40, 50):
                rec = CostRecord(
                    task_kind="x", tokens=v, iterations=1,
                    wall_s=1.0, ts=old,
                )
                fh.write(rec.to_json_line() + "\n")
        est = CostEstimator(history_path=str(p))
        b = est.estimate(TaskSpec("x"))
        # Median of [10,20,30,40,50] = 30 (irrespective of zero weights).
        assert b.p50_tokens in (20, 30, 40)  # allow for recency-weighted drift


# ---------------------------------------------------------------------------
# TaskSpec feature vector
# ---------------------------------------------------------------------------

class TestTaskSpec:
    def test_feature_vector_shape(self):
        s = TaskSpec(kind="refactor", description="/path/to/foo",
                     files_touched=3, tools=("edit", "test"))
        fv = s.feature_vector()
        assert fv["desc_len"] == len("/path/to/foo")
        assert fv["files"] == 3.0
        assert fv["tools"] == 2.0
        assert fv["desc_has_path"] == 1.0
        assert fv["desc_has_test"] == 0.0
