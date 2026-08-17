#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

push_tag=false
case "${1:-}" in
    "") ;;
    --push) push_tag=true ;;
    --help|-h)
        echo "Usage: $0 [--push]"
        echo "Create the annotated release tag for the committed VERSION.txt."
        echo "Use --push to push only that tag to origin."
        exit 0
        ;;
    *)
        echo "Unknown option: $1" >&2
        echo "Usage: $0 [--push]" >&2
        exit 2
        ;;
esac

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "Release tag creation requires a clean working tree." >&2
    exit 1
fi

current_branch=$(git symbolic-ref --quiet --short HEAD) || {
    echo "Release tags must be created from a named branch." >&2
    exit 1
}
if [[ "$current_branch" != "main" ]]; then
    echo "Release tags must be created from main, not $current_branch." >&2
    exit 1
fi

git fetch --quiet origin main --tags
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "HEAD is not synchronized with origin/main. Wait for the main CI run and update first." >&2
    exit 1
fi

version=$(tr -d '[:space:]' < VERSION.txt)
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "VERSION.txt must contain X.Y.Z, found: $version" >&2
    exit 1
fi
tag="v$version"

if git rev-parse --verify --quiet "refs/tags/$tag" >/dev/null; then
    echo "Release tag $tag already exists locally; refusing to overwrite it." >&2
    exit 1
fi

if git ls-remote --exit-code --quiet --tags origin "refs/tags/$tag" >/dev/null 2>&1; then
    echo "Release tag $tag already exists on origin; refusing to overwrite it." >&2
    exit 1
fi

git tag -a "$tag" -m "Release poemai-utils $version"
echo "Created annotated tag $tag."

if [[ "$push_tag" == true ]]; then
    git push origin "$tag"
    echo "Pushed $tag. The tag-only GitHub Actions workflow will publish it to PyPI."
else
    echo "To publish through GitHub Actions, push only this tag with: git push origin $tag"
fi
