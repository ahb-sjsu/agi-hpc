
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

GITHUB_API = "https://api.github.com"
DEFAULT_USER_AGENT = "AtlasAI-WeeklyDigest/1.0"
DEFAULT_MAX_CHARS = 1900
DEFAULT_MAX_ITEMS = 10
DEFAULT_DAYS = 7
MAX_COMMIT_PAGES = 50


@dataclass
class RepoRef:
    owner: str
    name: str

    @property
    def full(self) -> str:
        return f"{self.owner}/{self.name}"


def _parse_repo_arg(s: str) -> RepoRef:
    parts = s.strip().split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("expected owner/name")
    return RepoRef(parts[0], parts[1])


def _parse_git_remote(url: str) -> RepoRef | None:
    url = url.strip()
    for pat in (
        r"https://github\.com/([^/]+)/([^/.]+)(?:\.git)?/?$",
        r"git@github\.com:([^/]+)/([^/.]+)(?:\.git)?$",
    ):
        m = re.match(pat, url)
        if m:
            return RepoRef(m.group(1), m.group(2))
    return None


def resolve_repo(explicit: RepoRef | None, repo_root: Path) -> RepoRef:
    if explicit:
        return explicit
    env = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if env and "/" in env:
        o, n = env.split("/", 1)
        return RepoRef(o, n)
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        parsed = _parse_git_remote(r.stdout)
        if parsed:
            return parsed
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print(
        "Could not determine repo: set GITHUB_REPOSITORY, use --repo owner/name, "
        "or run from a git clone with github.com origin.",
        file=sys.stderr,
    )
    sys.exit(2)


def _parse_github_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _is_bot_login(login: str | None) -> bool:
    return not login or login.endswith("[bot]")


def _safe_title(title: str) -> str:
    return title.replace("*", "").replace("_", "").replace("`", "").strip() or "(no title)"


class GitHubClient:
    """Minimal GitHub REST client via `gh` or urllib + token."""

    def __init__(self, prefer_gh: bool) -> None:
        self._use_gh = bool(prefer_gh and shutil.which("gh"))
        self._token: str | None = None
        if not self._use_gh:
            self._token = (
                os.environ.get("GH_TOKEN")
                or os.environ.get("GITHUB_TOKEN")
                or os.environ.get("GITHUB_PAT")
            )
            if not self._token:
                print(
                    "Authentication required: install `gh` and run `gh auth login`, "
                    "or set GH_TOKEN / GITHUB_TOKEN.",
                    file=sys.stderr,
                )
                sys.exit(1)

    def get_json(self, path: str, query: dict[str, str] | None = None) -> Any:
        if query:
            path = f"{path}?{urllib.parse.urlencode(query)}"
        if self._use_gh:
            return self._gh_api(path)
        return self._urllib_json(f"{GITHUB_API}{path}")

    def _gh_api(self, path: str) -> Any:
        cmd = ["gh", "api", "-H", "Accept: application/vnd.github+json", path]
        # Read bytes and decode ourselves so Windows codepage issues never null out stdout.
        r = subprocess.run(cmd, capture_output=True, text=False, timeout=120)
        stdout = (r.stdout or b"").decode("utf-8", errors="replace")
        stderr = (r.stderr or b"").decode("utf-8", errors="replace")
        if r.returncode != 0:
            msg = (stderr or stdout or "").strip() or f"exit {r.returncode}"
            raise RuntimeError(f"gh api failed: {msg}")
        return json.loads(stdout) if stdout.strip() else None

    def _urllib_json(self, url: str) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": DEFAULT_USER_AGENT,
            "Authorization": f"Bearer {self._token}",
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from e
        return json.loads(body) if body.strip() else None


def search_issues(
    client: GitHubClient, repo: RepoRef, query_extra: str, per_page: int = 100
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for page in range(1, 100):
        data = client.get_json(
            "/search/issues",
            {
                "q": f"repo:{repo.full} {query_extra}",
                "per_page": str(per_page),
                "page": str(page),
            },
        )
        batch = (data or {}).get("items") or []
        items.extend(batch)
        if len(batch) < per_page:
            break
    return items


def filter_by_timestamp(
    items: Iterable[dict[str, Any]],
    since_dt: datetime,
    ts_getter: Callable[[dict[str, Any]], str | None],
    predicate: Callable[[dict[str, Any]], bool] = lambda _: True,
) -> list[dict[str, Any]]:
    kept: list[tuple[datetime, dict[str, Any]]] = []
    for it in items:
        if not predicate(it):
            continue
        ts = _parse_github_dt(ts_getter(it))
        if ts and ts >= since_dt:
            kept.append((ts, it))
    kept.sort(key=lambda pair: pair[0], reverse=True)
    return [it for _, it in kept]


def count_commits_by_author(
    client: GitHubClient, repo: RepoRef, since_iso: str
) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    total = 0
    for page in range(1, MAX_COMMIT_PAGES + 1):
        data = client.get_json(
            f"/repos/{repo.full}/commits",
            {"since": since_iso, "per_page": "100", "page": str(page)},
        )
        if not data:
            break
        for c in data:
            total += 1
            author = c.get("author") if isinstance(c.get("author"), dict) else None
            login = (author or {}).get("login") or (
                ((c.get("commit") or {}).get("author") or {}).get("name") or ""
            ).strip() or None
            if _is_bot_login(login):
                continue
            counts[str(login)] += 1
        if len(data) < 100:
            break
    return counts, total


def _format_item(it: dict[str, Any]) -> str:
    user = (it.get("user") or {}).get("login", "?")
    return f"`#{it['number']}` **{user}** — {_safe_title(it.get('title', ''))}"


def _render_digest_markdown(
    repo: RepoRef,
    since_dt: datetime,
    until_dt: datetime,
    sections: list[tuple[str, list[dict[str, Any]], str]],
    top_author: tuple[str, int] | None,
    item_cap: int,
) -> str:
    since_d = since_dt.date().isoformat()
    until_d = until_dt.date().isoformat()
    lines: list[str] = [
        f"**Weekly digest** `{repo.full}` — `{since_d}` → `{until_d}` (UTC)",
        "",
    ]
    for title, rows, more_url in sections:
        lines.append(f"**{title}** ({len(rows)})")
        if not rows:
            lines.extend(["- _None_", ""])
            continue
        show = rows[:item_cap]
        for it in show:
            lines.append(f"- {_format_item(it)}")
        rest = len(rows) - len(show)
        if rest > 0:
            lines.append(f"- … and **{rest}** more: {more_url}")
        lines.append("")

    if top_author:
        lines.append(
            f"**Top contributor (commits):** `{top_author[0]}` ({top_author[1]} commits)"
        )
    else:
        lines.append("**Top contributor (commits):** _No human-authored commits in window_")
    return "\n".join(lines).strip()


def build_digest_markdown(
    repo: RepoRef,
    since_dt: datetime,
    until_dt: datetime,
    sections: list[tuple[str, list[dict[str, Any]], str]],
    top_author: tuple[str, int] | None,
    max_items: int,
    max_chars: int,
) -> str:
    cap = max(1, max_items)
    text = ""
    while True:
        text = _render_digest_markdown(
            repo, since_dt, until_dt, sections, top_author, cap
        )
        if len(text) <= max_chars or cap == 1:
            break
        cap = max(1, cap // 2)
    if len(text) > max_chars:
        cut = max_chars - 40
        text = text[:cut].rstrip() + "\n\n… *(truncated)*"
    return text


def post_to_discord(content: str, webhook_url: str, timeout: int = 30) -> None:
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"Discord webhook failed (HTTP {e.code}): {detail}") from e


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="Weekly GitHub -> Discord markdown digest."
    )
    parser.add_argument("--repo", type=_parse_repo_arg, default=None)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw counts and titles to stdout as JSON instead of markdown.",
    )
    parser.add_argument(
        "--prefer-urllib",
        action="store_true",
        help="Force urllib + token even if gh is installed.",
    )
    parser.add_argument(
        "--post-discord",
        action="store_true",
        help="Post markdown output to DISCORD_WEBHOOK_URL (or --webhook-url).",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Discord webhook URL override (otherwise uses DISCORD_WEBHOOK_URL).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    repo = resolve_repo(args.repo, repo_root)
    until_dt = datetime.now(timezone.utc)
    since_dt = until_dt - timedelta(days=args.days)
    since_iso = since_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    since_day = since_dt.date().isoformat()

    client = GitHubClient(prefer_gh=not args.prefer_urllib)

    try:
        merged = filter_by_timestamp(
            search_issues(client, repo, f"is:pr is:merged merged:>={since_day}"),
            since_dt,
            lambda it: (it.get("pull_request") or {}).get("merged_at") or it.get("closed_at"),
        )
        opened = filter_by_timestamp(
            search_issues(client, repo, f"is:issue created:>={since_day}"),
            since_dt,
            lambda it: it.get("created_at"),
            predicate=lambda it: not it.get("pull_request"),
        )
        closed = filter_by_timestamp(
            search_issues(client, repo, f"is:issue is:closed closed:>={since_day}"),
            since_dt,
            lambda it: it.get("closed_at"),
            predicate=lambda it: not it.get("pull_request") and it.get("state") == "closed",
        )
        counts, total_commits = count_commits_by_author(client, repo, since_iso)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    top: tuple[str, int] | None = None
    if counts:
        login, n = counts.most_common(1)[0]
        top = (login, n)

    if args.json:
        if args.post_discord:
            print("--post-discord cannot be used with --json.", file=sys.stderr)
            sys.exit(2)
        out = {
            "repo": repo.full,
            "since": since_iso,
            "merged_prs": [
                {"number": x["number"], "title": x.get("title"), "user": (x.get("user") or {}).get("login")}
                for x in merged
            ],
            "opened_issues": [
                {"number": x["number"], "title": x.get("title"), "user": (x.get("user") or {}).get("login")}
                for x in opened
            ],
            "closed_issues": [
                {"number": x["number"], "title": x.get("title"), "user": (x.get("user") or {}).get("login")}
                for x in closed
            ],
            "top_contributor": {"login": top[0], "commits": top[1]} if top else None,
            "total_commits_seen": total_commits,
        }
        print(json.dumps(out, indent=2))
        return

    if not merged and not opened and not closed and total_commits == 0:
        print(
            f"**Weekly digest** `{repo.full}`\n\n"
            "_Quiet week — no merged PRs, issue activity, or commits in the last "
            f"**{args.days}** days (UTC)._"
        )
        return

    sections = [
        (
            "Merged PRs",
            merged,
            f"<https://github.com/{repo.full}/pulls?q=is%3Apr+is%3Amerged+merged%3A%3E%3D{since_day}>",
        ),
        (
            "Opened issues",
            opened,
            f"<https://github.com/{repo.full}/issues?q=is%3Aissue+created%3A%3E%3D{since_day}>",
        ),
        (
            "Closed issues",
            closed,
            f"<https://github.com/{repo.full}/issues?q=is%3Aissue+is%3Aclosed+closed%3A%3E%3D{since_day}>",
        ),
    ]
    markdown = build_digest_markdown(
        repo, since_dt, until_dt, sections, top, args.max_items, args.max_chars
    )
    print(markdown)

    if args.post_discord:
        webhook_url = (args.webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")).strip()
        if not webhook_url:
            print(
                "Missing Discord webhook URL. Set DISCORD_WEBHOOK_URL or pass --webhook-url.",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            post_to_discord(markdown, webhook_url)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
