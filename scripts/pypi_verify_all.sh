#!/usr/bin/env bash
# Verify all ahb-sjsu PyPI packages install + pass tests, in parallel.
# Each package gets its own fresh venv under /tmp/pypi-verify/<repo>/.
# Results streamed to /archive/neurogolf/pypi_verify_results.jsonl.
set -u

OUT=/archive/neurogolf/pypi_verify_results.jsonl
WORKDIR=/tmp/pypi-verify
REPOS=(agi-hpc neurogolf-bundle nats-bursting atlas-portal erisml-lib atlas-burst polite-submit turboquant-pro batch-probe research-dashboard theory-radar geometric-medicine geometric-communication geometric-cognition geometric-law geometric-economics eris-ketos sqnd-probe arc-prize geometric-moderation eris-econ structural-fuzzing prd-builder compliance-os altscribe counter-apologetics prometheus fun-hypothesis non-abelian-sqnd yacht)

mkdir -p "$WORKDIR"
: > "$OUT"

# Worker: given a repo, detect Python package, install from PyPI, clone, run tests
verify_one() {
  local repo="$1"
  local dir="$WORKDIR/$repo"
  local log="$dir/verify.log"
  mkdir -p "$dir"
  exec > "$log" 2>&1

  local result='{"repo":"'$repo'"'

  # Find pyproject.toml and the package name
  local pyproj=""
  for path in pyproject.toml python/pyproject.toml; do
    local raw
    raw=$(gh api "repos/ahb-sjsu/$repo/contents/$path" --jq '.content' 2>/dev/null | base64 -d 2>/dev/null)
    if [ -n "$raw" ]; then
      pyproj="$raw"
      result="$result,\"pyproject_path\":\"$path\""
      break
    fi
  done
  if [ -z "$pyproj" ]; then
    echo "$result,\"status\":\"no_python_package\"}" >> "$OUT"
    return
  fi

  local name
  name=$(echo "$pyproj" | python3 -c "import sys,tomllib; d=tomllib.loads(sys.stdin.read()); print(d.get('project',{}).get('name',''))" 2>/dev/null)
  if [ -z "$name" ]; then
    echo "$result,\"status\":\"no_name_in_pyproject\"}" >> "$OUT"
    return
  fi
  result="$result,\"package\":\"$name\""

  # Is it on PyPI?
  local pypi_status
  pypi_status=$(curl -s -o /dev/null -w "%{http_code}" "https://pypi.org/pypi/$name/json")
  if [ "$pypi_status" != "200" ]; then
    echo "$result,\"status\":\"not_on_pypi\",\"pypi_http\":\"$pypi_status\"}" >> "$OUT"
    return
  fi

  # Fresh venv, install from PyPI
  python3 -m venv "$dir/venv" || { echo "$result,\"status\":\"venv_fail\"}" >> "$OUT"; return; }
  local PY="$dir/venv/bin/python"
  "$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1

  local install_out
  install_out=$("$PY" -m pip install "$name" 2>&1)
  local install_rc=$?
  if [ $install_rc -ne 0 ]; then
    local err
    err=$(echo "$install_out" | tail -3 | tr '\n' ' ' | sed 's/"/\\"/g')
    echo "$result,\"status\":\"install_failed\",\"err\":\"$err\"}" >> "$OUT"
    return
  fi

  # Import test
  local import_name
  import_name=$(echo "$name" | tr '-' '_')
  if ! "$PY" -c "import $import_name" 2>/dev/null; then
    # Some packages have a different module name; try lowercasing first component
    local alt
    alt=$(echo "$name" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')
    if ! "$PY" -c "import $alt" 2>/dev/null; then
      echo "$result,\"status\":\"install_ok_import_fail\"}" >> "$OUT"
      return
    fi
  fi

  # Clone repo to run tests
  if [ ! -d "$dir/src" ]; then
    git clone --depth 1 "https://github.com/ahb-sjsu/$repo.git" "$dir/src" >/dev/null 2>&1 \
      || { echo "$result,\"status\":\"install_ok_clone_fail\"}" >> "$OUT"; return; }
  fi

  # Install dev/test dependencies
  local test_cwd="$dir/src"
  if [ -f "$dir/src/python/pyproject.toml" ]; then test_cwd="$dir/src/python"; fi
  "$PY" -m pip install --quiet pytest >/dev/null 2>&1
  (cd "$test_cwd" && "$PY" -m pip install --quiet ".[dev]" 2>/dev/null || "$PY" -m pip install --quiet ".[test]" 2>/dev/null || "$PY" -m pip install --quiet -e . 2>/dev/null)

  # Find tests
  local test_dir=""
  for cand in "$dir/src/tests" "$dir/src/python/tests" "$dir/src/test"; do
    [ -d "$cand" ] && test_dir="$cand" && break
  done
  if [ -z "$test_dir" ]; then
    echo "$result,\"status\":\"install_ok_no_tests\"}" >> "$OUT"
    return
  fi

  # Run tests
  local test_out
  test_out=$(cd "$test_cwd" && timeout 600 "$PY" -m pytest "$test_dir" -x --tb=short -q 2>&1)
  local test_rc=$?
  local test_summary
  test_summary=$(echo "$test_out" | tail -1 | sed 's/"/\\"/g')

  if [ $test_rc -eq 0 ]; then
    echo "$result,\"status\":\"tests_pass\",\"summary\":\"$test_summary\"}" >> "$OUT"
  else
    echo "$result,\"status\":\"tests_fail\",\"rc\":$test_rc,\"summary\":\"$test_summary\"}" >> "$OUT"
  fi
}

export -f verify_one
export WORKDIR OUT

# Run up to 8 in parallel
printf '%s\n' "${REPOS[@]}" | xargs -I{} -P 8 bash -c 'verify_one "$@"' _ {}

echo "=== DONE ===" >> "$OUT"
echo "=== Results written to $OUT ==="
wc -l "$OUT"
