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
        echo "Validate the committed VERSION.txt and create its annotated release tag."
        echo "Use --push to push only that tag to origin."
        exit 0
        ;;
    *)
        echo "Unknown option: $1" >&2
        echo "Usage: $0 [--push]" >&2
        exit 2
        ;;
esac

"$repo_root/release_preflight.sh"

version=$(tr -d '[:space:]' < VERSION.txt)
tag="v$version"

# Re-check immediately before creating the tag in case another process created it
# while the preflight tests were running.
if git rev-parse --verify --quiet "refs/tags/$tag" >/dev/null; then
    echo "Release tag $tag already exists locally; refusing to overwrite it." >&2
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
