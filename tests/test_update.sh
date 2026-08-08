#!/usr/bin/env bash
# Regression tests for skill source selection.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/bin"

cat > "$WORK/recipe.yaml" <<'EOF'
context:
  upstream_path: skills/example
EOF

cat > "$WORK/bin/gh" <<'EOF'
#!/usr/bin/env bash
args="$*"
if [[ "$args" == *"commits?path=skills/example"* ]]; then
    echo path-commit
elif [[ "$args" == *"contents/skills/example?ref=path-commit"* ]]; then
    exit 0
elif [[ "$args" == *"contents/skills/example?ref=head-commit"* ]]; then
    exit 0
elif [[ "$args" == *"commits/main"* ]]; then
    echo head-commit
else
    echo "unexpected gh call: $args" >&2
    exit 1
fi
EOF
chmod +x "$WORK/bin/gh"

# shellcheck source=/dev/null
source "$REPO_ROOT/.github/scripts/update.sh"
recipe_file="$WORK/recipe.yaml"
latest_rev="$(PATH="$WORK/bin:$PATH" get_latest_git_rev example/tool)"

[[ "$latest_rev" == "path-commit" ]] \
    || { echo "FAIL: path-pinned skill followed unrelated repository HEAD: $latest_rev" >&2; exit 1; }
echo "ok: upstream_path follows its newest path-specific commit"
