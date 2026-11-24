"""#394: run settings live in result['config']; check_regression writes and gates on them."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from eval.check_regression import check_baseline
from eval.evaluate import EvalConfig, evaluate
from nugget_rag.paper_registry import ARXIV_MAP

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _data(tmp_path):
    chunks = [
        {
            "paper_id": ARXIV_MAP["2410.10071"],
            "text": "correct answer here",
            "nugget": "correct answer here",
        }
    ]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["correct answer here"]}]
    c_path, g_path = tmp_path / "chunks.json", tmp_path / "gold.json"
    c_path.write_text(json.dumps(chunks))
    g_path.write_text(json.dumps(gold))
    return c_path, g_path


def _eval_args():
    chunks = {1: [{"paper_id": 1, "text": "correct answer here", "nugget": "x"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["correct answer here"]}]
    return chunks, gold


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "eval.check_regression", *args],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


def _fake_result(config, n=1):
    return {
        "modes": ["full_chunk", "nugget"],
        "n_queries": n,
        "config": config,
        "full_chunk": {"recall": 1.0},
        "nugget": {"recall": 1.0},
    }


def _baseline(tmp_path, **extra):
    base = {"top_k": 5, "n": 1, "full_chunk": {"recall": 1.0}, "nugget": {"recall": 1.0}, **extra}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base))
    return str(path)


# -- result['config'] ----------------------------------------------------------------------------


def test_result_config_default():
    result = evaluate(*_eval_args())
    assert result["config"] == {
        "top_k": 5,
        "match": "exact",
        "nugget_window": 0,
        "embed_weight": 0.5,
        "nuggets_per_chunk": 3,
    }
    assert result["match"] == "exact"  # existing key unchanged


def test_result_config_reflects_eval_config():
    cfg = EvalConfig(
        top_k=3, match="normalized", nugget_window=2, nuggets_per_chunk=4, embed_weight=0.25
    )
    result = evaluate(*_eval_args(), cfg)
    assert result["config"] == {
        "top_k": 3,
        "match": "normalized",
        "nugget_window": 2,
        "embed_weight": 0.25,
        "nuggets_per_chunk": 4,
    }
    json.dumps(result["config"])  # plain data only (no estimator / callables)


# -- --write-baseline -> same args PASS ----------------------------------------------------------


def test_write_baseline_then_check_passes(tmp_path):
    c_path, g_path = _data(tmp_path)
    out = tmp_path / "written.json"
    args = ["--chunks", str(c_path), "--gold", str(g_path), "--match", "normalized"]
    written = _run([*args, "--write-baseline", str(out)])
    assert written.returncode == 0, written.stdout + written.stderr
    base = json.loads(out.read_text(encoding="utf-8"))
    assert base["top_k"] == 5 and base["match"] == "normalized" and base["nugget_window"] == 0
    assert base["n"] == 1 and base["full_chunk"]["recall"] == 1.0
    assert Path(base["gold"]).resolve() == g_path.resolve()
    checked = _run([*args, "--baseline", str(out), "--max-drop", "0.0"])
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "[PASS]" in checked.stdout


def test_write_baseline_with_window_then_check_passes(tmp_path):
    c_path, g_path = _data(tmp_path)
    out = tmp_path / "written.json"
    args = ["--chunks", str(c_path), "--gold", str(g_path), "--nugget-window", "1", "--top-k", "3"]
    assert _run([*args, "--write-baseline", str(out)]).returncode == 0
    assert json.loads(out.read_text())["nugget_window"] == 1
    assert _run([*args, "--baseline", str(out)]).returncode == 0


def test_write_baseline_in_repo_gold_is_repo_relative(tmp_path):
    out = tmp_path / "written.json"
    args = ["--chunks", "tests/fixtures/chunks_test.json"]
    args += ["--gold", "tests/fixtures/gold_test.json"]
    assert _run([*args, "--write-baseline", str(out)]).returncode == 0
    assert json.loads(out.read_text())["gold"] == "tests/fixtures/gold_test.json"
    assert _run([*args, "--baseline", str(out), "--max-drop", "0.0"]).returncode == 0


def test_write_baseline_rejects_baseline_together(tmp_path):
    c_path, g_path = _data(tmp_path)
    args = ["--chunks", str(c_path), "--gold", str(g_path)]
    r = _run([*args, "--write-baseline", str(tmp_path / "a.json"), "--baseline", "b.json"])
    assert r.returncode == 2


# -- baseline match / nugget_window are checked --------------------------------------------------


def test_match_mismatch_fails(tmp_path):
    c_path, g_path = _data(tmp_path)
    base = _baseline(tmp_path, match="exact")
    args = ["--chunks", str(c_path), "--gold", str(g_path), "--baseline", base]
    r = _run([*args, "--match", "normalized"])
    assert r.returncode == 1
    assert "match" in r.stdout


def test_nugget_window_mismatch_fails(tmp_path):
    c_path, g_path = _data(tmp_path)
    base = _baseline(tmp_path, nugget_window=0)
    args = ["--chunks", str(c_path), "--gold", str(g_path), "--baseline", base]
    r = _run([*args, "--nugget-window", "2"])
    assert r.returncode == 1
    assert "nugget_window" in r.stdout


def test_mismatch_is_warn_when_advisory(tmp_path):
    c_path, g_path = _data(tmp_path)
    base = _baseline(tmp_path, match="exact")
    args = ["--chunks", str(c_path), "--gold", str(g_path), "--baseline", base]
    r = _run([*args, "--match", "normalized", "--advisory"])
    assert r.returncode == 0
    assert "[WARN]" in r.stdout and "比較不能" in r.stdout


def test_old_baseline_without_match_and_window_is_not_checked(tmp_path):
    result = _fake_result({"top_k": 5, "match": "normalized", "nugget_window": 3})
    failures, _ = check_baseline(result, _baseline(tmp_path), 0.0, top_k=5)
    assert failures == []


def test_check_baseline_flags_match_and_window(tmp_path):
    result = _fake_result({"top_k": 5, "match": "normalized", "nugget_window": 3})
    path = _baseline(tmp_path, match="exact", nugget_window=0)
    failures, _ = check_baseline(result, path, 0.0, top_k=5)
    assert any("baseline match" in f for f in failures)
    assert any("baseline nugget_window" in f for f in failures)


def test_check_baseline_without_result_config_skips_new_checks(tmp_path):
    result = _fake_result(None)
    del result["config"]
    path = _baseline(tmp_path, match="exact", nugget_window=0)
    assert check_baseline(result, path, 0.0, top_k=5)[0] == []


def test_real_baselines_record_match_and_window():
    for name in ("baseline_hard.json", "baseline_gold.json"):
        base = json.loads((_REPO_ROOT / "eval" / name).read_text(encoding="utf-8"))
        assert base["match"] == "exact" and base["nugget_window"] == 0
