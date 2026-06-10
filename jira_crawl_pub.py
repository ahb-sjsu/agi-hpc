import os
import time
import json
import glob
import signal
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd

# =========================
# CONFIG
# =========================
JIRA_BASE_URL = "<URL>"
PROJECT_KEY = "<KEY>"
API_TOKEN = "<TOKEN."
OUTPUT_DIR = "jira_full_export"
RANGES_DIR = os.path.join(OUTPUT_DIR, "ranges")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "crawl_summary.json")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "crawl_summary.txt")

JQL = f'project = {PROJECT_KEY} ORDER BY created DESC'

DEFAULT_CHUNK_SIZE = 25
SPLIT_SEQUENCE = [25, 10, 5, 1]

CONNECT_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 0.5

FETCH_ALL_FIELDS = True
EXPAND_CHANGELOG_IN_SEARCH = False
FETCH_CHANGELOG_SECOND_PASS = False

MIN_READ_TIMEOUT = 30
MAX_READ_TIMEOUT = 120

# =========================
# INTERRUPT STATE
# =========================
STOP_REQUESTED = False


def handle_interrupt(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[WARN] Interrupt requested. Stopping after current operation and saving checkpoint...")


signal.signal(signal.SIGINT, handle_interrupt)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, handle_interrupt)


def check_stop():
    if STOP_REQUESTED:
        raise KeyboardInterrupt("Stop requested by user")


def interruptible_sleep(seconds: float, step: float = 0.25):
    end = time.time() + max(0, seconds)
    while time.time() < end:
        check_stop()
        remaining = end - time.time()
        time.sleep(min(step, remaining))


# =========================
# SESSION
# =========================
session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "Authorization": f"Bearer {API_TOKEN}",
})


# =========================
# BASIC HELPERS
# =========================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RANGES_DIR, exist_ok=True)


def range_file_path(start_at: int, size: int) -> str:
    return os.path.join(RANGES_DIR, f"range_{start_at:06d}_{size:04d}.json")


def range_key(start_at: int, size: int) -> str:
    return f"{start_at}:{size}"


def retries_for_size(size: int) -> int:
    if size >= 25:
        return 1
    if size >= 10:
        return 2
    if size >= 5:
        return 2
    return 3


def read_timeout_for_size(size: int) -> int:
    if size >= 25:
        return 45
    if size >= 10:
        return 60
    if size >= 5:
        return 90
    return 120


def next_smaller_size(size: int) -> Optional[int]:
    if size not in SPLIT_SEQUENCE:
        return None
    idx = SPLIT_SEQUENCE.index(size)
    if idx == len(SPLIT_SEQUENCE) - 1:
        return None
    return SPLIT_SEQUENCE[idx + 1]


def flatten_value(val):
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, list):
        parts = []
        for x in val:
            if isinstance(x, dict):
                parts.append(
                    x.get("name")
                    or x.get("value")
                    or x.get("displayName")
                    or x.get("key")
                    or x.get("id")
                    or json.dumps(x, ensure_ascii=False)
                )
            else:
                parts.append(str(x))
        return " | ".join(str(p) for p in parts if p is not None)
    if isinstance(val, dict):
        return (
            val.get("name")
            or val.get("value")
            or val.get("displayName")
            or val.get("emailAddress")
            or val.get("key")
            or val.get("id")
            or json.dumps(val, ensure_ascii=False)
        )
    return str(val)


# =========================
# CHECKPOINT
# =========================
def load_checkpoint() -> Dict[str, Any]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            cp = json.load(f)
    else:
        cp = {}

    cp.setdefault("project_key", PROJECT_KEY)
    cp.setdefault("jql", JQL)
    cp.setdefault("started_at", now_str())
    cp.setdefault("last_updated", None)
    cp.setdefault("interrupted", False)
    cp.setdefault("total", None)

    cp.setdefault("completed_ranges", [])
    cp.setdefault("failed_ranges", [])
    cp.setdefault("range_attempts", {})

    stats_defaults = {
        "requests_attempted": 0,
        "requests_succeeded": 0,
        "requests_failed": 0,
        "ranges_split": 0,
        "ranges_completed": 0,
        "ranges_failed": 0,
    }
    cp.setdefault("stats", {})
    for k, v in stats_defaults.items():
        cp["stats"].setdefault(k, v)

    return cp


def save_checkpoint(cp: Dict[str, Any]):
    cp["last_updated"] = now_str()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2)


def mark_interrupted(cp: Dict[str, Any], value: bool = True):
    cp["interrupted"] = value
    save_checkpoint(cp)


def completed_range_exists(cp: Dict[str, Any], start_at: int, size: int) -> bool:
    fp = range_file_path(start_at, size)
    if not os.path.exists(fp):
        return False
    for r in cp["completed_ranges"]:
        if r["start_at"] == start_at and r["size"] == size:
            return True
    return False


def add_completed_range(cp: Dict[str, Any], start_at: int, size: int, issue_count: int):
    fp = range_file_path(start_at, size)
    cp["completed_ranges"] = [
        r for r in cp["completed_ranges"]
        if not (r["start_at"] == start_at and r["size"] == size)
    ]
    cp["completed_ranges"].append({
        "start_at": start_at,
        "size": size,
        "issue_count": issue_count,
        "file": fp,
        "saved_at": now_str(),
    })
    cp["completed_ranges"].sort(key=lambda r: (r["start_at"], r["size"]))

    cp["failed_ranges"] = [
        r for r in cp["failed_ranges"]
        if not (r["start_at"] == start_at and r["size"] == size)
    ]


def add_failed_range(cp: Dict[str, Any], start_at: int, size: int, error: str):
    cp["failed_ranges"] = [
        r for r in cp["failed_ranges"]
        if not (r["start_at"] == start_at and r["size"] == size)
    ]
    cp["failed_ranges"].append({
        "start_at": start_at,
        "size": size,
        "error": (error or "")[:2000],
        "failed_at": now_str(),
    })
    cp["failed_ranges"].sort(key=lambda r: (r["start_at"], r["size"]))


# =========================
# HTTP/API
# =========================
def jira_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
    read_timeout: int = 60,
) -> Dict[str, Any]:
    last_exc = None

    for attempt in range(1, max_retries + 1):
        check_stop()

        try:
            resp = session.get(
                url,
                params=params,
                timeout=(CONNECT_TIMEOUT, max(MIN_READ_TIMEOUT, min(read_timeout, MAX_READ_TIMEOUT))),
                allow_redirects=True,
            )

            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(10, 2 ** attempt)
                print(f"[WARN] HTTP {resp.status_code} for {resp.url}")
                if attempt < max_retries:
                    print(f"[WARN] Sleeping {wait}s before retry {attempt}/{max_retries}...")
                    interruptible_sleep(wait)
                    continue

            if not resp.ok:
                print(f"[ERROR] HTTP {resp.status_code} for {resp.url}")
                print(resp.text[:2000])
                resp.raise_for_status()

            check_stop()
            return resp.json()

        except KeyboardInterrupt:
            raise

        except requests.exceptions.ReadTimeout as e:
            last_exc = e
            if attempt < max_retries:
                wait = min(10, 2 ** attempt)
                print(f"[WARN] Read timeout on attempt {attempt}/{max_retries}: {e}")
                print(f"[WARN] Sleeping {wait}s before retry...")
                interruptible_sleep(wait)

        except requests.exceptions.ConnectionError as e:
            last_exc = e
            if attempt < max_retries:
                wait = min(10, 2 ** attempt)
                print(f"[WARN] Connection error on attempt {attempt}/{max_retries}: {e}")
                print(f"[WARN] Sleeping {wait}s before retry...")
                interruptible_sleep(wait)

        except requests.exceptions.RequestException as e:
            last_exc = e
            break

    raise RuntimeError(f"Request failed after {max_retries} attempts: {url}\nLast exception: {last_exc}")


def search_issues(start_at: int, max_results: int) -> Dict[str, Any]:
    url = f"{JIRA_BASE_URL}/rest/api/2/search"
    params: Dict[str, Any] = {
        "jql": JQL,
        "startAt": start_at,
        "maxResults": max_results,
    }

    if FETCH_ALL_FIELDS:
        params["fields"] = "*all"
    else:
        params["fields"] = "summary,status,issuetype,priority,assignee,reporter,created,updated,resolution"

    if EXPAND_CHANGELOG_IN_SEARCH:
        params["expand"] = "changelog"

    return jira_get(
        url,
        params=params,
        max_retries=retries_for_size(max_results),
        read_timeout=read_timeout_for_size(max_results),
    )


def get_all_fields() -> pd.DataFrame:
    url = f"{JIRA_BASE_URL}/rest/api/2/field"
    data = jira_get(url, max_retries=3, read_timeout=60)

    rows = []
    for f in data:
        rows.append({
            "id": f.get("id"),
            "name": f.get("name"),
            "custom": f.get("custom"),
            "schema_type": (f.get("schema") or {}).get("type"),
            "schema_items": (f.get("schema") or {}).get("items"),
            "schema_custom": (f.get("schema") or {}).get("custom"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "fields_catalog.csv"), index=False, encoding="utf-8-sig")
    return df


def get_issue_changelog(issue_key: str) -> Dict[str, Any]:
    url = f"{JIRA_BASE_URL}/rest/api/2/issue/{issue_key}"
    params = {"fields": "summary", "expand": "changelog"}
    return jira_get(url, params=params, max_retries=3, read_timeout=90)


# =========================
# FETCH / SPLIT
# =========================
def fetch_range_once(start_at: int, size: int, cp: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    check_stop()

    key = range_key(start_at, size)
    cp["range_attempts"][key] = cp["range_attempts"].get(key, 0) + 1
    cp["stats"]["requests_attempted"] += 1
    save_checkpoint(cp)

    try:
        data = search_issues(start_at=start_at, max_results=size)
        cp["stats"]["requests_succeeded"] += 1
        save_checkpoint(cp)
        return True, data, None
    except KeyboardInterrupt:
        raise
    except Exception as e:
        cp["stats"]["requests_failed"] += 1
        save_checkpoint(cp)
        return False, None, str(e)


def save_range_data(start_at: int, size: int, data: Dict[str, Any], cp: Dict[str, Any]):
    check_stop()

    fp = range_file_path(start_at, size)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    issue_count = len(data.get("issues", []))
    add_completed_range(cp, start_at, size, issue_count)
    cp["stats"]["ranges_completed"] += 1
    save_checkpoint(cp)

    print(f"[INFO] Saved range startAt={start_at}, size={size}, issues={issue_count}")


def fetch_range_recursive(start_at: int, size: int, total: int, cp: Dict[str, Any]):
    check_stop()

    if start_at >= total:
        return

    size = min(size, total - start_at)

    if completed_range_exists(cp, start_at, size):
        return

    print(f"[INFO] Fetching range startAt={start_at}, size={size}")
    ok, data, err = fetch_range_once(start_at, size, cp)

    if ok and data is not None:
        save_range_data(start_at, size, data, cp)
        interruptible_sleep(SLEEP_BETWEEN_REQUESTS)
        return

    print(f"[WARN] Range failed startAt={start_at}, size={size}: {err}")

    smaller = next_smaller_size(size)
    if smaller is None:
        print(f"[ERROR] Marking unresolved range as failed startAt={start_at}, size={size}")
        add_failed_range(cp, start_at, size, err or "Unknown error")
        cp["stats"]["ranges_failed"] += 1
        save_checkpoint(cp)
        return

    cp["stats"]["ranges_split"] += 1
    save_checkpoint(cp)

    print(f"[INFO] Splitting range startAt={start_at}, size={size} into chunks of {smaller}")

    current = start_at
    end = start_at + size
    while current < end and current < total:
        check_stop()
        sub_size = min(smaller, end - current, total - current)
        fetch_range_recursive(current, sub_size, total, cp)
        current += sub_size


# =========================
# CRAWL
# =========================
def crawl_issues():
    cp = load_checkpoint()
    mark_interrupted(cp, False)

    print("[INFO] Fetching initial page to determine total...")
    first = search_issues(start_at=0, max_results=DEFAULT_CHUNK_SIZE)
    total = first.get("total", 0)
    cp["total"] = total
    save_checkpoint(cp)

    print(f"[INFO] Total visible issues: {total}")

    if total == 0:
        return

    first_size = min(DEFAULT_CHUNK_SIZE, total)
    if not completed_range_exists(cp, 0, first_size):
        save_range_data(0, first_size, first, cp)

    print("[INFO] Starting adaptive crawl...")
    current = 0
    while current < total:
        check_stop()

        size = min(DEFAULT_CHUNK_SIZE, total - current)
        if completed_range_exists(cp, current, size):
            current += size
            continue

        fetch_range_recursive(current, size, total, cp)
        current += size

    print("[INFO] Crawl pass complete.")


# =========================
# LOAD / FLATTEN
# =========================
def load_all_issues_from_ranges() -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(RANGES_DIR, "range_*.json")))
    all_issues = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
            all_issues.extend(payload.get("issues", []))

    dedup = {}
    for issue in all_issues:
        dedup[issue.get("id")] = issue

    return list(dedup.values())


def flatten_issue(issue: Dict[str, Any], field_map: Dict[str, str]) -> Dict[str, Any]:
    row = {
        "issue_id": issue.get("id"),
        "issue_key": issue.get("key"),
        "self": issue.get("self"),
    }

    fields = issue.get("fields", {})
    for field_id, value in fields.items():
        field_name = field_map.get(field_id, field_id)
        row[field_name] = flatten_value(value)

    return row


def flatten_changelog(issue: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    changelog = issue.get("changelog", {}) or {}
    histories = changelog.get("histories", []) or []

    for h in histories:
        author = h.get("author", {}) or {}
        for item in h.get("items", []) or []:
            rows.append({
                "issue_id": issue.get("id"),
                "issue_key": issue.get("key"),
                "history_id": h.get("id"),
                "history_created": h.get("created"),
                "author_displayName": author.get("displayName"),
                "author_email": author.get("emailAddress"),
                "field": item.get("field"),
                "fieldtype": item.get("fieldtype"),
                "from": item.get("from"),
                "fromString": item.get("fromString"),
                "to": item.get("to"),
                "toString": item.get("toString"),
            })
    return rows


def build_outputs():
    print("[INFO] Building consolidated outputs...")

    fields_df = get_all_fields()
    field_map = dict(zip(fields_df["id"], fields_df["name"]))

    all_issues = load_all_issues_from_ranges()
    print(f"[INFO] Loaded {len(all_issues)} unique issues from saved range files")

    raw_path = os.path.join(OUTPUT_DIR, "issues_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_issues, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote {raw_path}")

    flat_rows = [flatten_issue(issue, field_map) for issue in all_issues]
    flat_df = pd.DataFrame(flat_rows)
    flat_csv_path = os.path.join(OUTPUT_DIR, "issues_flat.csv")
    flat_df.to_csv(flat_csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Wrote {flat_csv_path}")

    if FETCH_CHANGELOG_SECOND_PASS:
        print("[INFO] Fetching changelog in second pass...")
        changelog_rows = []
        total = len(all_issues)

        for idx, issue in enumerate(all_issues, start=1):
            check_stop()
            key = issue.get("key")
            if not key:
                continue
            print(f"[INFO] Changelog {idx}/{total}: {key}")
            full_issue = get_issue_changelog(key)
            changelog_rows.extend(flatten_changelog(full_issue))
            interruptible_sleep(0.2)

        changelog_df = pd.DataFrame(changelog_rows)
        changelog_path = os.path.join(OUTPUT_DIR, "changelog_flat.csv")
        changelog_df.to_csv(changelog_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Wrote {changelog_path}")


# =========================
# SUMMARY
# =========================
def summarize_coverage(cp: Dict[str, Any]) -> Dict[str, Any]:
    total = cp.get("total") or 0

    covered_offsets = set()
    for r in cp.get("completed_ranges", []):
        for i in range(r["start_at"], min(r["start_at"] + r["size"], total)):
            covered_offsets.add(i)

    failed_offsets = set()
    for r in cp.get("failed_ranges", []):
        for i in range(r["start_at"], min(r["start_at"] + r["size"], total)):
            failed_offsets.add(i)

    missing_offsets = set(range(total)) - covered_offsets - failed_offsets

    return {
        "total_offsets_expected": total,
        "covered_offsets": len(covered_offsets),
        "failed_offsets": len(failed_offsets),
        "missing_offsets": len(missing_offsets),
        "coverage_pct": round((len(covered_offsets) / total * 100.0), 2) if total else 0.0,
        "failed_offset_examples": sorted(list(failed_offsets))[:50],
        "missing_offset_examples": sorted(list(missing_offsets))[:50],
    }


def write_summary_report():
    cp = load_checkpoint()
    coverage = summarize_coverage(cp)
    all_issues = load_all_issues_from_ranges()

    summary = {
        "run_started_at": cp.get("started_at"),
        "run_last_updated": cp.get("last_updated"),
        "interrupted": cp.get("interrupted", False),
        "project_key": cp.get("project_key"),
        "jql": cp.get("jql"),
        "total_reported_by_jira": cp.get("total"),
        "unique_issues_saved": len(all_issues),
        "completed_range_count": len(cp.get("completed_ranges", [])),
        "failed_range_count": len(cp.get("failed_ranges", [])),
        "request_stats": cp.get("stats", {}),
        "coverage": coverage,
        "failed_ranges": cp.get("failed_ranges", []),
        "range_attempts": cp.get("range_attempts", {}),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "JIRA CRAWL SUMMARY",
        "=" * 60,
        f"Run started:          {summary['run_started_at']}",
        f"Last updated:         {summary['run_last_updated']}",
        f"Interrupted:          {summary['interrupted']}",
        f"Project:              {summary['project_key']}",
        f"JQL:                  {summary['jql']}",
        f"Jira total issues:    {summary['total_reported_by_jira']}",
        f"Unique issues saved:  {summary['unique_issues_saved']}",
        "",
        "REQUEST STATS",
        "-" * 60,
        f"Requests attempted:   {summary['request_stats'].get('requests_attempted', 0)}",
        f"Requests succeeded:   {summary['request_stats'].get('requests_succeeded', 0)}",
        f"Requests failed:      {summary['request_stats'].get('requests_failed', 0)}",
        f"Ranges split:         {summary['request_stats'].get('ranges_split', 0)}",
        f"Ranges completed:     {summary['request_stats'].get('ranges_completed', 0)}",
        f"Ranges failed:        {summary['request_stats'].get('ranges_failed', 0)}",
        "",
        "COVERAGE",
        "-" * 60,
        f"Covered offsets:      {coverage['covered_offsets']}",
        f"Failed offsets:       {coverage['failed_offsets']}",
        f"Missing offsets:      {coverage['missing_offsets']}",
        f"Coverage %:           {coverage['coverage_pct']}",
        "",
        "RANGES",
        "-" * 60,
        f"Completed ranges:     {summary['completed_range_count']}",
        f"Failed ranges:        {summary['failed_range_count']}",
        "",
    ]

    if summary["failed_range_count"] > 0:
        lines.append("FAILED RANGES DETAIL")
        lines.append("-" * 60)
        for r in summary["failed_ranges"]:
            lines.append(
                f"startAt={r['start_at']}, size={r['size']}, failed_at={r['failed_at']}, error={r['error']}"
            )
        lines.append("")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Wrote {SUMMARY_JSON}")
    print(f"[INFO] Wrote {SUMMARY_TXT}")
    print("\n" + "\n".join(lines))


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    if not API_TOKEN:
        raise RuntimeError("Missing JIRA_TOKEN environment variable.")

    cp = load_checkpoint()

    try:
        print("[INFO] Starting Jira crawl...")
        crawl_issues()

        check_stop()

        print("[INFO] Building output files...")
        build_outputs()

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user.")
        cp = load_checkpoint()
        mark_interrupted(cp, True)

        try:
            print("[INFO] Building partial outputs from saved data...")
            build_outputs()
        except Exception as e:
            print(f"[WARN] Could not build partial outputs: {e}")

    finally:
        try:
            print("[INFO] Writing summary report...")
            write_summary_report()
        except Exception as e:
            print(f"[WARN] Could not write summary report: {e}")

        print("[INFO] Done.")


if __name__ == "__main__":
    main()