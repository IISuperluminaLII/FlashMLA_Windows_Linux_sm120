#!/usr/bin/env bash
# Wait (poll) for a terminal marker in a log, then print the filtered summary lines.
# Usage: bash tests/_wait_log.sh <log> <terminal-regex> <summary-regex> [max_polls(10s)]
LOG="$1"; TERM_RE="$2"; SUM_RE="$3"; N="${4:-180}"
for i in $(seq 1 "$N"); do
  if grep -qE "$TERM_RE" "$LOG" 2>/dev/null; then break; fi
  sleep 10
done
grep -E "$SUM_RE" "$LOG" | head -100
echo "WAIT_LOG_DONE"
