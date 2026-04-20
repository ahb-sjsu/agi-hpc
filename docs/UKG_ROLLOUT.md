# UKG rollout runbook

Operational procedure for cutting the Primer's teaching-context retrieval over from wiki-glob to the Unified Knowledge Graph. Written against the v1 spec (see `reference_ukg_v1_spec.md` in the assistant's memory for the authoritative design doc).

## Preconditions

Before any cutover step:

- Phase 1–6 code is on `main` (`ukg:` prefix commits).
- CI is green.
- The Primer's service unit (`atlas-primer.service`) is running.

## Step 1 — seed the graph from the existing wiki

The wiki holds 9 verified sensei notes today. Backfill imports each one as a `filled, verified, active` node so the Primer's graph reader has something to return.

```bash
# On Atlas
cd /home/claude/agi-hpc
git pull
python -m agi.knowledge.backfill  # or:
python scripts/ukg_backfill_wiki.py --verbose
```

Expected output:

```
imported=9 skipped_unverified=0 skipped_already_present=0 failed=0
```

Re-running is a no-op (`skipped_already_present` rises). Use `--force` only after a wiki edit if you want to refresh an existing node's metadata.

## Step 2 — verify the graph JSONL

```bash
wc -l /archive/neurogolf/knowledge_graph.jsonl
jq -r '.id + " " + .type + " verified=" + (.verified|tostring)' \
  /archive/neurogolf/knowledge_graph.jsonl | head
```

At this point the `/api/ukg/status` endpoint reports real numbers on the dashboard card (Knowledge Graph panel).

## Step 3 — let the Primer's tick-level gap import run for one cycle

With the service already running (wiki-reader default), the very next tick will pull stuck tasks out of `erebus_help_queue.json` and upsert them as `gap` nodes. Nothing to do — just wait for one poll interval (`PRIMER_POLL_S`, default 300 s) and re-check:

```bash
jq -r 'select(.type == "gap") | .id' \
  /archive/neurogolf/knowledge_graph.jsonl | sort -u
```

If gaps appear for stuck task numbers Erebus has been trying, the graph is tracking both halves of the teaching loop.

## Step 4 — flip the context reader to `graph`

Install the drop-in so the Primer reads context from the graph instead of globbing the wiki directory:

```bash
sudo install -Dm644 \
  deploy/systemd/atlas-primer.service.d/context-reader-graph.conf \
  /etc/systemd/system/atlas-primer.service.d/context-reader-graph.conf
sudo systemctl daemon-reload
sudo systemctl restart atlas-primer
```

Verify the env var is set:

```bash
systemctl show atlas-primer -p Environment | tr ' ' '\n' | grep EREBUS_CONTEXT_READER
# Expected: Environment=EREBUS_CONTEXT_READER=graph
```

## Step 5 — watch the journal

```bash
journalctl -u atlas-primer -f
```

Two signals to look for:

- `ensemble_complete` lifecycle events continuing to fire at each tick means the retrieval path is alive.
- `graph_context: zero eligible snippets for task sensei_task_NNN` warning means the graph reader found no eligible node for a task the Primer is working on. One warning per processed task is expected early in rollout (most tasks don't have a filled node yet). A persistent pattern across many tasks means either the backfill didn't run or the `body_ref` files are missing — check paths.

A `filled_excluded_missing_body` warning means the graph has a filled node pointing at a file that isn't on disk. Usually a path mismatch between `cfg.wiki_dir` and what was recorded during backfill. Re-run backfill after confirming `EREBUS_WIKI_DIR`.

## Step 6 — soak period

Leave it running for ~24 hours. Compare Primer verify rates (Primer panel on the dashboard — `verify pass/fail` column) before and after the cutover. Expected outcome: no regression. The graph reader surfaces the same notes as the wiki reader today (same 9 sensei files), just via a different index; the Primer should not notice.

## Step 7 — promote `graph` to default (optional, after soak)

Once soak confirms the graph reader is stable, flip the code default in `src/agi/knowledge/graph.py`:

```python
def context_reader_mode(default: str = "graph") -> str:  # was "wiki"
```

Then the drop-in becomes unnecessary for the happy path, but keep it around so operators can pin the value explicitly.

## Rollback

If anything looks off, the rollback is instant and local:

```bash
sudo rm /etc/systemd/system/atlas-primer.service.d/context-reader-graph.conf
sudo systemctl daemon-reload
sudo systemctl restart atlas-primer
```

The Primer reverts to wiki-glob retrieval, the graph JSONL stays intact (it is not modified by the reader), and no downstream state is lost. The Phase 3 writer hook still upserts nodes on every publish, so the graph keeps growing correctly while the reader path is off.

## What this does NOT change

- `_publish_note` still writes the markdown file to the wiki and git-commits it. The graph is an index, not a replacement for the wiki.
- Help-queue import still runs at every tick regardless of reader mode.
- `is_context_eligible()` is the single trust gate; it runs in both reader modes via the graph reader, and via `is_verified()` in the wiki reader. Same invariant, two surfaces.
