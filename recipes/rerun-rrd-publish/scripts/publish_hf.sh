#!/usr/bin/env bash
# Stage 3 (publish): scratch upload -> browser check in app.rerun.io -> final upload -> scratch cleanup.
#
# usage: publish_hf.sh <file.rrd> <user/dataset> <target-name.rrd> --viewer-version X.Y.Z
#            [--out-dir DIR] [--skip-browser] [--dry-run]
#
# Needs: `hf` logged in as the dataset owner, `playwright-cli` with chromium (browser check), python3, curl.
# The browser check cannot use a localhost server: Chrome blocks app.rerun.io (https) from fetching localhost or
# private-network addresses. Hence the scratch upload. That scratch copy is world-readable for as long as the
# check runs, roughly a minute, and is deleted right after the final upload.
# The script gates only on hard failures (rrd URL not 200, missing screenshots, console errors). The screenshots
# it writes are the real acceptance test and must be inspected by the agent: layout applied, video pixels
# present, time moving. They land in --out-dir, by default /tmp/rrd-publish/<target-stem>/.
set -euo pipefail

# Print the header above: from line 2 to the first line that is not a comment (never the `set -euo` line).
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1; }
[ $# -ge 3 ] || usage
FILE=$1; REPO=$2; TARGET=$3; shift 3
VIEWER_VERSION=""; OUT_DIR=""; SKIP_BROWSER=0; DRY_RUN=0
while [ $# -gt 0 ]; do
  case $1 in
    # ${2:?...} instead of $2: under `set -u` a bare $2 dies with "unbound variable" and no hint.
    --viewer-version) VIEWER_VERSION=${2:?--viewer-version needs a value, e.g. --viewer-version 0.37.0}; shift 2 ;;
    --out-dir) OUT_DIR=${2:?--out-dir needs a directory}; shift 2 ;;
    --skip-browser) SKIP_BROWSER=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option $1"; usage ;;
  esac
done
[ -f "$FILE" ] || { echo "no such file: $FILE"; exit 1; }
[ -n "$VIEWER_VERSION" ] || { echo "--viewer-version is required: the viewer must be >= the SDK that wrote the file"; exit 1; }

NEEDED="hf python3 curl"
[ "$SKIP_BROWSER" = 1 ] || NEEDED="$NEEDED playwright-cli"
MISSING=""
for tool in $NEEDED; do command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"; done
[ -z "$MISSING" ] || { echo "missing required tool(s):$MISSING (hf must be logged in as the dataset owner; playwright-cli needs chromium, and is not needed with --skip-browser)"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/rrd-publish/${TARGET%.rrd}}; mkdir -p "$OUT_DIR"; OUT_DIR=$(cd "$OUT_DIR" && pwd)
BYTES=$(wc -c <"$FILE" | tr -d ' ')  # not `stat -c %s`: that spelling is GNU-only and this also runs on macOS
WHOAMI=$(hf auth whoami 2>/dev/null | grep -oE "user=[^ ]+" | head -1 || true)
echo "hf identity: ${WHOAMI:-unknown}   repo: $REPO   file: $FILE ($BYTES bytes) -> $TARGET"
if [ "$BYTES" -gt 524288000 ]; then
  echo "WARNING: $((BYTES / 1000000)) MB is large for the web viewer, which loads the whole file into WASM memory and gives up somewhere around 1.5 GiB. Consider a clip (stage 2 --clip-seconds)."
fi

raw_url() {  # $1 = path in the repo
  python3 - "$REPO" "$1" <<'PY'
import sys, urllib.parse
repo, path = sys.argv[1:]
print(f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(path, safe='/+')}")  # keep '+': the viewer decodes once
PY
}
viewer_url() {  # $1 = path in the repo
  python3 - "$(raw_url "$1")" "$VIEWER_VERSION" <<'PY'
import sys, urllib.parse
raw, ver = sys.argv[1:]
print(f"https://app.rerun.io/version/{ver}/index.html?url={urllib.parse.quote(raw, safe='')}")
PY
}
run() { if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] $*" >&2; else "$@"; fi; }  # stderr: callers send stdout to /dev/null

SCRATCH="scratch/$(date +%Y%m%d-%H%M%S)-$(basename "$TARGET")"
SHOT_20="$OUT_DIR/web-20s.png"; SHOT_35="$OUT_DIR/web-35s.png"; CONSOLE="$OUT_DIR/console.txt"
if [ "$SKIP_BROWSER" = 0 ]; then
  echo "=== 1. scratch upload -> $SCRATCH"
  run hf upload "$REPO" "$FILE" "$SCRATCH" --repo-type dataset --commit-message "scratch: browser check for $TARGET (temporary)" >/dev/null
  URL=$(viewer_url "$SCRATCH"); RAW=$(raw_url "$SCRATCH")
  if [ "$DRY_RUN" = 0 ]; then
    CODE=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -I -L "$RAW" || true); CODE=${CODE:-000}
    echo "rrd url -> HTTP $CODE"
    [ "$CODE" != 000 ] || echo "curl never got a status back from $RAW (transport failure, not an HTTP code)"
    [ "$CODE" = 200 ] || { echo "scratch file not served; it is still in the dataset as $SCRATCH"; exit 2; }
    echo "=== 2. browser check (headless chromium, software AV1 decode: allow 20-40 s)"
    rm -f "$SHOT_20" "$SHOT_35" "$CONSOLE"  # a stale pair from an earlier run must not pass the check below
    S="rrd-publish-$$"
    ( cd "$OUT_DIR"  # playwright-cli drops a session log in the working directory; keep it out of the caller's repo
      playwright-cli -s="$S" close >/dev/null 2>&1 || true
      timeout 120 playwright-cli -s="$S" open "$URL" >/dev/null 2>&1 || true
      playwright-cli -s="$S" resize 1920 1080 >/dev/null 2>&1 || true
      sleep 20; playwright-cli -s="$S" screenshot --filename "$SHOT_20" >/dev/null 2>&1 || true
      sleep 15; playwright-cli -s="$S" screenshot --filename "$SHOT_35" >/dev/null 2>&1 || true
      playwright-cli -s="$S" console warning 2>/dev/null | grep -vE "^[║╔╚]" > "$CONSOLE" || true
      playwright-cli -s="$S" close >/dev/null 2>&1 || true )
    if [ ! -s "$SHOT_20" ] || [ ! -s "$SHOT_35" ]; then
      echo "no screenshots at $SHOT_20 and $SHOT_35: playwright-cli or chromium failed, so nothing was checked."
      echo "the scratch file is still in the dataset as $SCRATCH; delete it or re-run."
      exit 2
    fi
    # `playwright-cli console warning` prints a summary line and then the messages it returns, e.g.
    #   Total messages: 4 (Errors: 2, Warnings: 1)
    #   [ERROR] a real error line @ http://127.0.0.1:8791/probe.html:3
    # Read the summary first and count the tagged lines when it is absent. Every grep needs `|| true`: under
    # `set -e` a grep that matches nothing would end the script right here, leaving the scratch file uploaded
    # and unmentioned.
    [ -s "$CONSOLE" ] || echo "warning: the browser check captured no console output at all"
    SUMMARY=$(grep -m1 -oE "Errors: [0-9]+" "$CONSOLE" 2>/dev/null || true)
    if [ -n "$SUMMARY" ]; then
      ERRORS=${SUMMARY##* }; FORMAT="the 'Total messages: N (Errors: E, Warnings: W)' summary line"
    else
      ERRORS=$(grep -c "\[ERROR\]" "$CONSOLE" 2>/dev/null || true); ERRORS=${ERRORS:-0}
      FORMAT="a count of [ERROR] lines; no summary line was printed"
    fi
    echo "console errors: $ERRORS (from $FORMAT)"
    echo "screenshots: $SHOT_20 $SHOT_35   console: $CONSOLE"
    if [ "$ERRORS" != 0 ]; then echo "console errors present; inspect $CONSOLE. Scratch file left in place: $SCRATCH"; exit 2; fi
    echo "INSPECT THE SCREENSHOTS before trusting this: blueprint layout applied, video pixels present, time cursor moving if looping."
  fi
fi

echo "=== 3. final upload -> $TARGET"
PREVIOUS=$(curl -s -m 30 -I -L "$(raw_url "$TARGET")" 2>/dev/null | tr -d '\r' || true)
if [ "$(printf '%s\n' "$PREVIOUS" | awk '/^HTTP\//{code=$2} END{print code+0}')" = 200 ]; then
  echo "REPLACING existing $TARGET ($(printf '%s\n' "$PREVIOUS" | awk 'tolower($1)=="content-length:"{n=$2} END{printf "%.1f", n/1e6}') MB)"
fi
run hf upload "$REPO" "$FILE" "$TARGET" --repo-type dataset --commit-message "$TARGET: standalone rrd + blueprint (rerun $VIEWER_VERSION)" >/dev/null
if [ "$SKIP_BROWSER" = 0 ]; then
  echo "=== 4. delete scratch"
  run hf repos delete-files "$REPO" "$SCRATCH" --repo-type dataset --commit-message "remove scratch check file" >/dev/null
fi
if [ "$DRY_RUN" = 0 ]; then
  echo "=== 5. tree check"
  # `?recursive=true`: the flat listing returns `scratch` as one directory entry and never names the file
  # inside it, nor a target that sits in a subdirectory.
  TREE="$OUT_DIR/tree.json"
  curl -s -m 30 -o "$TREE" "https://huggingface.co/api/datasets/$REPO/tree/main?recursive=true" || true
  if [ ! -s "$TREE" ]; then
    echo "tree check skipped: could not fetch https://huggingface.co/api/datasets/$REPO/tree/main?recursive=true"
  else
    python3 - "$TREE" "$TARGET" <<'PY' || echo "tree check skipped: the tree API did not return the expected JSON ($TREE)"
import json, sys
tree, target = sys.argv[1:]
with open(tree) as f:
    entries = json.load(f)
leftover = [e for e in entries if e.get("path", "").startswith("scratch/") or (e.get("path") == "scratch" and e.get("type") == "directory")]
for e in [x for x in entries if x.get("path") == target] + leftover:
    print(f"{e.get('size', 0) / 1e6:8.1f} MB  {e['path']}  ({e.get('type', 'file')})")
if not any(e.get("path") == target for e in entries):
    print(f"WARNING: {target} is not in the dataset tree")
if leftover:
    print("WARNING: a scratch entry is still in the dataset; delete it with `hf repos delete-files`")
PY
  fi
fi
echo "viewer link:"; viewer_url "$TARGET"
echo "note: replaced versions stay in the dataset's git history until it is squashed."
