#!/usr/bin/env bash
# Bump recipes/<skill>/recipe.yaml to its newest upstream source.
# Writes old-version=/new-version= to $GITHUB_OUTPUT (stdout outside Actions).
set -euo pipefail

usage="Usage: $0 <skill> <git-main|github-latest-release>"
skill=${1:?$usage}
strategy=${2:?$usage}
recipe="recipes/$skill/recipe.yaml"
[[ -f "$recipe" ]] || { echo "Recipe not found: $recipe" >&2; exit 1; }

path_exists_at() {
    gh api "repos/$1/contents/$2?ref=$3" --silent >/dev/null 2>&1
}

# Newest commit that touches context.upstream_path and still contains it, so a
# monorepo skill bumps only when its own files change. A commit that deletes or
# moves the path falls back to its parent. Without upstream_path: main's HEAD.
latest_rev() {
    local repo=$1 path rev parent
    path=$(yq -r '.context.upstream_path // ""' "$recipe")
    if [[ -z "$path" ]]; then
        gh api "repos/$repo/commits/main" --jq '.sha'
        return
    fi
    for rev in $(gh api "repos/$repo/commits?path=$path&per_page=10" --jq '.[].sha'); do
        if path_exists_at "$repo" "$path" "$rev"; then
            echo "$rev"
            return
        fi
        for parent in $(gh api "repos/$repo/commits/$rev" --jq '.parents[].sha'); do
            if path_exists_at "$repo" "$path" "$parent"; then
                echo "$parent"
                return
            fi
        done
    done
    echo "No recent commit in $repo contains $path" >&2
    return 1
}

case "$strategy" in
    git-main)
        repo=$(yq -r '.source.git' "$recipe" | sed -E 's|^https://github.com/||; s|\.git$||')
        old=$(yq -r '.source.rev' "$recipe")
        new=$(latest_rev "$repo")
        if [[ "$new" != "$old" ]]; then
            old_version=$(yq -r '.package.version' "$recipe")
            new_version=$(awk -F. '{print $1"."$2"."$3+1}' <<< "$old_version")
            # sed, not `yq -i`: yq drops the blank lines between sections.
            sed -E "s/^(  rev: )$old\$/\1$new/; /^package:/,/^[^ ]/ s/^(  version: \"?)$old_version(\"?)\$/\1$new_version\2/" \
                "$recipe" > "$recipe.tmp"
            mv "$recipe.tmp" "$recipe"
            [[ "$(yq -r '.source.rev + " " + .package.version' "$recipe")" == "$new $new_version" ]] \
                || { echo "Could not rewrite rev and version in $recipe" >&2; exit 1; }
        fi
        ;;
    github-latest-release)
        old=$(yq -r '.context.version' "$recipe")
        rattler-build bump-recipe --recipe "$recipe"
        new=$(yq -r '.context.version' "$recipe")
        ;;
    *)
        echo "$usage" >&2
        exit 1
        ;;
esac

printf 'old-version=%s\nnew-version=%s\n' "$old" "$new" >> "${GITHUB_OUTPUT:-/dev/stdout}"
