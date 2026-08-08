#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <skill-name> <strategy>"
    echo ""
    echo "Strategies:"
    echo "  git-main              Update to latest commit on main branch"
    echo "  github-latest-release Update to latest GitHub release"
    echo "  yolo                  Re-fetch URL and update sha256 if changed"
    echo ""
    echo "Examples:"
    echo "  $0 vercel-react-best-practices git-main"
    echo "  $0 baseline-ui github-latest-release"
    echo "  $0 rams yolo"
    exit 1
}

skill_name=""
strategy=""
recipe_file=""

# Bump patch version (e.g., 0.0.1 -> 0.0.2)
bump_patch_version() {
    local current_version
    current_version=$(yq -r '.package.version' "$recipe_file")
    local new_version
    new_version=$(echo "$current_version" | awk -F. '{print $1"."$2"."$3+1}')
    yq -i ".package.version = \"$new_version\"" "$recipe_file"
    echo "  version: $current_version -> $new_version"
}

# Extract repository URL from recipe
get_repo_from_recipe() {
    local repo_url
    # Try source.git first (for git-main strategy)
    # Format: https://github.com/owner/repo
    repo_url=$(yq -r '.source.git // ""' "$recipe_file" | sed -E 's|https://github.com/([^/]+/[^/]+).*|\1|')
    if [[ -z "$repo_url" ]]; then
        # Try source.url (for github-latest-release strategy)
        # Format: https://github.com/owner/repo/archive/refs/tags/...
        repo_url=$(yq -r '.source.url // ""' "$recipe_file" | sed -E 's|https://github.com/([^/]+/[^/]+)/archive/.*|\1|')
    fi
    if [[ -z "$repo_url" ]]; then
        echo "Error: Could not find repository URL in recipe"
        exit 1
    fi
    echo "$repo_url"
}

git_path_exists_at_rev() {
    local repo=$1
    local upstream_path=$2
    local rev=$3

    gh api "repos/${repo}/contents/${upstream_path}?ref=${rev}" --silent >/dev/null 2>&1
}

get_latest_git_rev() {
    local repo=$1
    local upstream_path
    upstream_path=$(yq -r '.context.upstream_path // ""' "$recipe_file")

    if [[ -z "$upstream_path" ]]; then
        gh api "repos/${repo}/commits/main" --jq '.sha'
        return
    fi

    # Monorepo skills should bump only when their own path changes. Repository
    # HEAD may advance many times a day for unrelated files.
    local candidate_rev
    while IFS= read -r candidate_rev; do
        if [[ -z "$candidate_rev" ]]; then
            continue
        fi
        if git_path_exists_at_rev "$repo" "$upstream_path" "$candidate_rev"; then
            echo "$candidate_rev"
            return
        fi

        local parent_rev
        while IFS= read -r parent_rev; do
            if git_path_exists_at_rev "$repo" "$upstream_path" "$parent_rev"; then
                echo "$parent_rev"
                return
            fi
        done < <(gh api "repos/${repo}/commits/${candidate_rev}" --jq '.parents[].sha')
    done < <(gh api "repos/${repo}/commits?path=${upstream_path}&per_page=10" --jq '.[].sha')

    echo "Error: Could not find a commit containing $upstream_path in $repo" >&2
    exit 1
}

update_git_main() {
    local repo
    repo=$(get_repo_from_recipe)

    echo "Fetching latest commit from $repo main branch..."
    local latest_rev
    latest_rev=$(get_latest_git_rev "$repo")

    local current_rev
    current_rev=$(yq -r '.source.rev // ""' "$recipe_file")

    echo "old-version=$current_rev" >> "${GITHUB_OUTPUT:-/dev/stdout}"
    echo "new-version=$latest_rev" >> "${GITHUB_OUTPUT:-/dev/stdout}"

    if [[ "$latest_rev" == "$current_rev" ]]; then
        echo "$skill_name is up to date (rev: $current_rev)"
    else
        echo "$skill_name needs update"
        echo "  current rev: $current_rev"
        echo "  latest rev:  $latest_rev"

        yq -i ".source.rev = \"$latest_rev\"" "$recipe_file"
        bump_patch_version
        echo "Updated $recipe_file"
    fi
}

update_github_latest_release() {
    local repo
    repo=$(get_repo_from_recipe)

    echo "Fetching latest release from $repo..."
    local latest_version
    latest_version=$(gh api "repos/${repo}/releases/latest" --jq '.tag_name' | sed 's/^v//')

    local current_version
    current_version=$(yq -r '.context.version // ""' "$recipe_file")

    echo "old-version=$current_version" >> "${GITHUB_OUTPUT:-/dev/stdout}"
    echo "new-version=$latest_version" >> "${GITHUB_OUTPUT:-/dev/stdout}"

    if [[ "$latest_version" == "$current_version" ]]; then
        echo "$skill_name is up to date (version: $current_version)"
    else
        echo "$skill_name needs update"
        echo "  current version: $current_version"
        echo "  latest version:  $latest_version"

        # Download the tarball to calculate new sha256
        local tarball_url="https://github.com/${repo}/archive/refs/tags/v${latest_version}.tar.gz"
        echo "Downloading $tarball_url to calculate sha256..."
        local new_sha256
        new_sha256=$(curl -sL "$tarball_url" | shasum -a 256 | cut -d' ' -f1)

        echo "  new sha256: $new_sha256"

        # Update version in context
        yq -i ".context.version = \"$latest_version\"" "$recipe_file"
        # Update sha256
        yq -i ".source.sha256 = \"$new_sha256\"" "$recipe_file"

        echo "Updated $recipe_file"
    fi
}

update_yolo() {
    local source_url
    source_url=$(yq -r '.source.url' "$recipe_file")

    echo "Fetching $source_url..."
    local new_sha256
    new_sha256=$(curl -sL "$source_url" | shasum -a 256 | cut -d' ' -f1)

    local current_sha256
    current_sha256=$(yq -r '.source.sha256' "$recipe_file")

    echo "old-version=$current_sha256" >> "${GITHUB_OUTPUT:-/dev/stdout}"
    echo "new-version=$new_sha256" >> "${GITHUB_OUTPUT:-/dev/stdout}"

    if [[ "$new_sha256" == "$current_sha256" ]]; then
        echo "$skill_name is up to date (sha256: $current_sha256)"
    else
        echo "$skill_name needs update"
        echo "  current sha256: $current_sha256"
        echo "  new sha256:     $new_sha256"

        yq -i ".source.sha256 = \"$new_sha256\"" "$recipe_file"
        bump_patch_version
        echo "Updated $recipe_file"
    fi
}

main() {
    if [[ $# -lt 2 ]]; then
        usage
    fi

    skill_name=$1
    strategy=$2
    recipe_file="recipes/${skill_name}/recipe.yaml"

    if [[ ! -f "$recipe_file" ]]; then
        echo "Error: Recipe file not found: $recipe_file"
        exit 1
    fi

    case "$strategy" in
        git-main) update_git_main ;;
        github-latest-release) update_github_latest_release ;;
        yolo) update_yolo ;;
        *)
            echo "Error: Unknown strategy '$strategy'"
            usage
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
