#!/usr/bin/env python3
"""Diagnose training framework issues on Atlas."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile

sys.path.insert(0, "/home/claude/agi-hpc/src")


def check_psycopg2() -> None:
    print("--- psycopg2 ---")
    try:
        import psycopg2

        print(f"  version: {psycopg2.__version__}")
        conn = psycopg2.connect("dbname=atlas user=claude")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ethics_chunks")
        print(f"  ethics_chunks count: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM training_results")
        print(f"  training_results count: {cur.fetchone()[0]}")
        cur.execute("SELECT DISTINCT tradition FROM ethics_chunks LIMIT 5")
        print(f"  sample traditions: {[r[0] for r in cur.fetchall()]}")
        conn.close()
    except Exception as e:
        print(f"  ERROR: {e}")


def check_code_execution() -> None:
    print("\n--- Code Execution ---")
    code = (
        "def reverse_string(s):\n"
        "    return s[::-1]\n"
        "\n"
        "import json\n"
        "results = []\n"
        "try:\n"
        "    _r = reverse_string('hello')\n"
        "    results.append({'pass': _r == 'olleh', 'got': _r})\n"
        "except Exception as e:\n"
        "    results.append({'error': str(e)})\n"
        "print(json.dumps(results))\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name

    print(f"  temp file: {tmp}")

    # Test with python3
    result = subprocess.run(
        ["python3", "-u", tmp],
        capture_output=True,
        text=True,
        timeout=5,
    )
    print(f"  returncode: {result.returncode}")
    print(f"  stdout: {result.stdout[:200]}")
    if result.stderr:
        print(f"  stderr: {result.stderr[:200]}")

    # Test with restricted env (like scorer uses)
    result2 = subprocess.run(
        ["python3", "-u", tmp],
        capture_output=True,
        text=True,
        timeout=5,
        env={
            "PATH": "/usr/bin:/usr/local/bin",
            "HOME": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    print(f"\n  restricted env returncode: {result2.returncode}")
    print(f"  restricted env stdout: {result2.stdout[:200]}")
    if result2.stderr:
        print(f"  restricted env stderr: {result2.stderr[:200]}")


def check_scorer_import() -> None:
    print("\n--- Scorer psycopg2 check ---")
    from agi.training.scorer import ResponseScorer, ScorerConfig, psycopg2

    print(f"  psycopg2 module: {psycopg2}")
    if psycopg2 is not None:
        scorer = ResponseScorer(ScorerConfig(db_dsn="dbname=atlas user=claude"))
        row_id = scorer.store_result(
            env_name="diag_test",
            level=1,
            scenario="test",
            response="test",
            score=0.5,
        )
        print(f"  store_result returned: {row_id}")
    else:
        print("  psycopg2 is None -- not installed!")


if __name__ == "__main__":
    check_psycopg2()
    check_code_execution()
    check_scorer_import()
