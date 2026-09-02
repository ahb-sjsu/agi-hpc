#!/bin/bash
# daily_backup.sh — Back up the AGI's life story and knowledge
#
# What gets backed up:
#   1. PostgreSQL (episodes, chunks, training_results, confidence_log)
#   2. Wiki directory (dream-consolidated articles — the life story)
#   3. Configs directory
#   4. Training logs
#
# Retention policy:
#   - Daily backups: last 30 days
#   - Weekly backups: last 12 weeks (every Sunday kept)
#
# Destination: /archive/atlas-backups/ (15TB RAID5, /dev/md5)
#
# Usage:
#   bash scripts/daily_backup.sh
#   bash scripts/daily_backup.sh --dry-run

set -euo pipefail

BACKUP_ROOT="/archive/atlas-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DAY_OF_WEEK=$(date +%u)  # 1=Monday, 7=Sunday
BACKUP_DIR="$BACKUP_ROOT/daily/$TIMESTAMP"
DRY_RUN=false
# Set true only once the dump has been written AND verified readable.
DB_OK=false

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=== Atlas Daily Backup ==="
log "Destination: $BACKUP_DIR"

if [ "$DRY_RUN" = true ]; then
    log "DRY RUN — no changes will be made"
fi

# ─── Create backup directory ─────────────────────────────────────────
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$BACKUP_DIR"
fi

# ─── 1. PostgreSQL dump ──────────────────────────────────────────────
log "Backing up PostgreSQL..."
if [ "$DRY_RUN" = false ]; then
    DUMP="$BACKUP_DIR/atlas_db.dump"

    # stderr is deliberately NOT discarded: the unit sets
    # StandardError=journal, so a failure reason lands in the journal
    # instead of vanishing into /dev/null.
    if pg_dump -U claude atlas \
            --format=custom \
            --compress=6 \
            --file="$DUMP"; then
        # A custom-format dump carries a table of contents. pg_restore
        # --list fails on a truncated or corrupt file; test -f does not.
        if pg_restore --list "$DUMP" >/dev/null 2>&1; then
            DB_OK=true
            SIZE=$(du -h "$DUMP" | cut -f1)
            log "  Database: $SIZE (verified: table of contents readable)"
        else
            log "ERROR: dump written but unreadable (pg_restore --list failed)"
            log "ERROR: removing $DUMP so it cannot pass as a good backup"
            rm -f "$DUMP"
        fi
    else
        log "ERROR: pg_dump failed; see the journal for the reason"
        rm -f "$DUMP"
    fi
fi

# ─── 2. Wiki (the AGI's life story) ─────────────────────────────────
log "Backing up wiki..."
WIKI_DIR="/home/claude/agi-hpc/wiki"
if [ -d "$WIKI_DIR" ] && [ "$DRY_RUN" = false ]; then
    tar -czf "$BACKUP_DIR/wiki.tar.gz" -C "$(dirname "$WIKI_DIR")" "$(basename "$WIKI_DIR")" 2>/dev/null
    SIZE=$(du -h "$BACKUP_DIR/wiki.tar.gz" | cut -f1)
    log "  Wiki: $SIZE"
elif [ ! -d "$WIKI_DIR" ]; then
    log "  Wiki: directory not found (skipping)"
fi

# ─── 3. Configs ──────────────────────────────────────────────────────
log "Backing up configs..."
CONFIGS_DIR="/home/claude/agi-hpc/configs"
if [ -d "$CONFIGS_DIR" ] && [ "$DRY_RUN" = false ]; then
    tar -czf "$BACKUP_DIR/configs.tar.gz" -C "$(dirname "$CONFIGS_DIR")" "$(basename "$CONFIGS_DIR")" 2>/dev/null
    log "  Configs: done"
fi

# ─── 4. Training logs ───────────────────────────────────────────────
log "Backing up training logs..."
TRAIN_LOG_DIR="/tmp/atlas-training"
if [ -d "$TRAIN_LOG_DIR" ] && [ "$DRY_RUN" = false ]; then
    tar -czf "$BACKUP_DIR/training_logs.tar.gz" -C "$(dirname "$TRAIN_LOG_DIR")" "$(basename "$TRAIN_LOG_DIR")" 2>/dev/null
    log "  Training logs: done"
fi

# ─── 5. Weekly promotion (Sunday) ────────────────────────────────────
if [ "$DAY_OF_WEEK" = "7" ] && [ "$DRY_RUN" = false ] && [ "$DB_OK" = true ]; then
    WEEKLY_DIR="$BACKUP_ROOT/weekly"
    mkdir -p "$WEEKLY_DIR"
    cp -r "$BACKUP_DIR" "$WEEKLY_DIR/$TIMESTAMP"
    log "  Weekly backup promoted (Sunday)"
fi

# ─── 6. Retention cleanup ───────────────────────────────────────────
if [ "$DRY_RUN" = false ]; then
    # Daily: keep last 30 days
    DAILY_DIR="$BACKUP_ROOT/daily"
    if [ -d "$DAILY_DIR" ]; then
        DAILY_COUNT=$(ls -1d "$DAILY_DIR"/*/ 2>/dev/null | wc -l)
        if [ "$DAILY_COUNT" -gt 30 ]; then
            REMOVE_COUNT=$((DAILY_COUNT - 30))
            ls -1d "$DAILY_DIR"/*/ 2>/dev/null | head -n "$REMOVE_COUNT" | xargs rm -rf
            log "  Cleaned $REMOVE_COUNT old daily backups"
        fi
    fi

    # Weekly: keep last 12 weeks
    WEEKLY_DIR="$BACKUP_ROOT/weekly"
    if [ -d "$WEEKLY_DIR" ]; then
        WEEKLY_COUNT=$(ls -1d "$WEEKLY_DIR"/*/ 2>/dev/null | wc -l)
        if [ "$WEEKLY_COUNT" -gt 12 ]; then
            REMOVE_COUNT=$((WEEKLY_COUNT - 12))
            ls -1d "$WEEKLY_DIR"/*/ 2>/dev/null | head -n "$REMOVE_COUNT" | xargs rm -rf
            log "  Cleaned $REMOVE_COUNT old weekly backups"
        fi
    fi
fi

# ─── Summary ─────────────────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    log ""
    log "=== Dry Run Complete ==="
elif [ "$DB_OK" = false ]; then
    log ""
    log "=== Backup INCOMPLETE: database dump failed or unverified ==="
    log "Location: $BACKUP_DIR"
    log "Other components may have been captured, but this run is NOT a"
    log "usable restore point. Exiting non-zero so systemd reports it."
    exit 1
elif [ -d "$BACKUP_DIR" ]; then
    TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
    log ""
    log "=== Backup Complete ==="
    log "Location: $BACKUP_DIR"
    log "Total size: $TOTAL_SIZE"
fi
