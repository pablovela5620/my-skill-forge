#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT
mkdir -p "$test_dir/bin" "$test_dir/output/noarch"
touch "$test_dir/output/flat.conda" "$test_dir/output/noarch/nested.conda"
cat > "$test_dir/bin/rattler-build" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "${@: -1}" >> "$UPLOAD_LOG"
EOF
chmod +x "$test_dir/bin/rattler-build"
yq -r '.jobs.upload.steps[] | select(.name == "Upload package to Artifactory") | .run' \
  "$repo_root/.github/workflows/package.yml" > "$test_dir/upload.sh"
export UPLOAD_LOG="$test_dir/uploads.txt"
export PATH="$test_dir/bin:$PATH"
(cd "$test_dir" && bash -e upload.sh)
printf '%s\n' output/flat.conda output/noarch/nested.conda > "$test_dir/expected.txt"
sort "$UPLOAD_LOG" > "$test_dir/actual.txt"
diff -u "$test_dir/expected.txt" "$test_dir/actual.txt"
rm "$test_dir/output/flat.conda" "$test_dir/output/noarch/nested.conda"
if (cd "$test_dir" && bash -e upload.sh); then
  printf '%s\n' 'Empty artifact downloads must fail.' >&2
  exit 1
fi
printf '%s\n' 'Flat and nested package artifacts are uploaded.'
