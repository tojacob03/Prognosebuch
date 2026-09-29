#!/usr/bin/env bash
# Commit data written by a scheduled job and push it.
#
#   scripts/commit-data.sh "<commit message>" <path> [<path> ...]
#
# - Paths that do not exist yet are skipped (e.g. scores/ before the first scored day).
# - Refuses to commit any modification, deletion or type change under forecasts/:
#   forecasts are write-once; only additions are allowed.
# - Does nothing if nothing changed.
# - Pushes with rebase and retries (set PUSH=0 to skip, used in tests).
set -euo pipefail

msg="$1"
shift

git config user.name "prognosebuch-bot"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for p in "$@"; do
  if [ -e "$p" ]; then
    git add -A -- "$p"
  fi
done

if git diff --cached --quiet; then
  echo "nothing to commit"
  exit 0
fi

if git diff --cached --no-renames --name-status --diff-filter=MDT -- forecasts | grep .; then
  echo "::error::refusing to commit a change to an existing forecast"
  exit 1
fi

git commit -q -m "$msg"

if [ "${PUSH:-1}" = "1" ]; then
  for _ in 1 2 3; do
    if git pull --rebase -q && git push -q; then
      exit 0
    fi
    sleep 10
  done
  echo "::error::push failed after 3 attempts"
  exit 1
fi
