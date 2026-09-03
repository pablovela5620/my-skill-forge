#!/usr/bin/env bash
# Stage 3 (publish): scratch upload -> browser check in app.rerun.io -> final upload -> scratch cleanup.
#
# usage: publish_hf.sh <file.rrd> <user/dataset> <target-name.rrd> --viewer-version X.Y.Z
#            [--out-dir DIR] [--skip-browser] [--dry-run]
#
# Needs: `hf` logged in as the dataset owner, `playwright-cli` with chromium (browser check), python3, curl.
# The browser check cannot use a localhost server: Chrome blocks app.rerun.io (https) from fetching localhost or
# private-network addresses. Hence the scratch upload, which is deleted after the final upload.
# The script gates only on hard failures (rrd URL not 200, console errors). The screenshots it writes are the
# real acceptance test and must be inspected by the agent: layout applied, video pixels present, time moving.
set -euo pipefail

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 3 ] || usage
FILE=$1; REPO=$2; TARGET=$3; shift 3
VIEWER_VERSION=""; OUT_DIR=""; SKIP_BROWSER=0; DRY_RUN=0
while [ $# -gt 0 ]; do
  case $1 in
    --viewer-version) VIEWER_VERSION=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    --skip-browser) SKIP_BROWSER=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option $1"; usage ;;
  esac
done
[ -f "$FILE" ] || { echo "no such file: $FILE"; exit 1; }
[ -n "$VIEWER_VERSION" ] || { echo "--viewer-version is required: the viewer must be >= the SDK that wrote the file"; exit 1; }
OUT_DIR=${OUT_DIR:-/tmp/rrd-publish/${TARGET%.rrd}}; mkdir -p "$OUT_DIR"; OUT_DIR=$(cd "$OUT_DIR" && pwd)
WHOAMI=$(hf auth whoami 2>/dev/null | grep -oE "user=[^ ]+" | head -1 || true)
echo "hf identity: ${WHOAMI:-unknown}   repo: $REPO   file: $FILE ($(stat -c %s "$FILE") bytes) -> $TARGET"

viewer_url() {  # $1 = path in repo
  python3 - "$REPO" "$1" "$VIEWER_VERSION" <<'PY'
import sys, urllib.parse
repo, path, ver = sys.argv[1:]
raw = f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(path, safe='/+')}"  # keep '+': the viewer decodes once
print(f"https://app.rerun.io/version/{ver}/index.html?url={urllib.parse.quote(raw, safe='')}")
PY
}
run() { if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] $*" >&2; else "$@"; fi; }  # stderr: callers send stdout to /dev/null

SCRATCH="scratch/$(date +%Y%m%d-%H%M%S)-$(basename "$TARGET")"
if [ "$SKIP_BROWSER" = 0 ]; then
  echo "=== 1. scratch upload -> $SCRATCH"
  run hf upload "$REPO" "$FILE" "$SCRATCH" --repo-type dataset --commit-message "scratch: browser check for $TARGET (temporary)" >/dev/null
  URL=$(viewer_url "$SCRATCH"); RAW="https://huggingface.co/datasets/$REPO/resolve/main/$SCRATCH"
  if [ "$DRY_RUN" = 0 ]; then
    CODE=$(curl -s -o /dev/null -w "%{http_code}" -I -L "$RAW"); echo "rrd url -> HTTP $CODE"; [ "$CODE" = 200 ] || { echo "scratch file not served"; exit 2; }
    echo "=== 2. browser check (headless chromium, software AV1 decode: allow 20-40 s)"
    S="rrd-publish-$$"
    ( cd "$OUT_DIR"  # playwright-cli drops a session log in the working directory; keep it out of the caller's repo
      playwright-cli -s="$S" close >/dev/null 2>&1 || true
      timeout 120 playwright-cli -s="$S" open "$URL" >/dev/null 2>&1 || true
      playwright-cli -s="$S" resize 1920 1080 >/dev/null 2>&1 || true
      sleep 20; playwright-cli -s="$S" screenshot --filename "$OUT_DIR/web-20s.png" >/dev/null 2>&1 || true
      sleep 15; playwright-cli -s="$S" screenshot --filename "$OUT_DIR/web-35s.png" >/dev/null 2>&1 || true
      playwright-cli -s="$S" console 2>/dev/null | grep -vE "^[║╔╚]" > "$OUT_DIR/console.txt" || true
      playwright-cli -s="$S" close >/dev/null 2>&1 || true )
    ERRORS=$(grep -oE "Errors: [0-9]+" "$OUT_DIR/console.txt" | grep -oE "[0-9]+" | head -1); ERRORS=${ERRORS:-?}
    echo "console errors: $ERRORS   screenshots: $OUT_DIR/web-20s.png $OUT_DIR/web-35s.png   console: $OUT_DIR/console.txt"
    if [ "$ERRORS" != 0 ]; then echo "console errors present; inspect $OUT_DIR/console.txt. Scratch file left in place: $SCRATCH"; exit 2; fi
    echo "INSPECT THE SCREENSHOTS before trusting this: blueprint layout applied, video pixels present, time cursor moving if looping."
  fi
fi

echo "=== 3. final upload -> $TARGET"
run hf upload "$REPO" "$FILE" "$TARGET" --repo-type dataset --commit-message "$TARGET: standalone rrd + blueprint (rerun $VIEWER_VERSION)" >/dev/null
if [ "$SKIP_BROWSER" = 0 ]; then
  echo "=== 4. delete scratch"
  run hf repos delete-files "$REPO" "$SCRATCH" --repo-type dataset --commit-message "remove scratch check file" >/dev/null
fi
if [ "$DRY_RUN" = 0 ]; then
  echo "=== 5. tree check"
  curl -s -m 30 "https://huggingface.co/api/datasets/$REPO/tree/main" | python3 -c "import json,sys; t=sys.argv[1]; [print(f\"{x.get('size',0)/1e6:8.1f} MB  {x['path']}\") for x in json.load(sys.stdin) if x['path']==t or x['path'].startswith('scratch/')]" "$TARGET"
fi
echo "viewer link:"; viewer_url "$TARGET"
echo "note: replaced versions stay in the dataset's git history until it is squashed."
