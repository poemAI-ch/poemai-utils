#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

fail() {
    echo "Release preflight failed: $*" >&2
    exit 1
}

command -v python >/dev/null 2>&1 || fail "python is required"
python -m pytest --version >/dev/null 2>&1 || fail "pytest is required"
python -m build --version >/dev/null 2>&1 || fail "build is required; install the build package"
python -m twine --version >/dev/null 2>&1 || fail "twine is required; install the twine package"

if [[ -n "$(git status --porcelain)" ]]; then
    fail "the working tree must be clean; commit the versioned change first"
fi

current_branch=$(git symbolic-ref --quiet --short HEAD) || fail "detached HEAD is not supported"
[[ "$current_branch" == "main" ]] || fail "release tags must be created from main, not $current_branch"

git fetch --quiet origin main --tags || fail "unable to refresh origin/main and tags"
head_sha=$(git rev-parse HEAD)
origin_main_sha=$(git rev-parse origin/main)
[[ "$head_sha" == "$origin_main_sha" ]] || fail "HEAD is not synchronized with origin/main"

version=$(tr -d '[:space:]' < VERSION.txt)
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "VERSION.txt must contain X.Y.Z, found: $version"
tag="v$version"

if git rev-parse --verify --quiet "refs/tags/$tag" >/dev/null; then
    fail "local tag $tag already exists"
fi

if git ls-remote --exit-code --quiet --tags origin "refs/tags/$tag" >/dev/null 2>&1; then
    fail "remote tag $tag already exists"
fi

echo "Running unit tests..."
python -m pytest tests/unit -v --tb=short

echo "Building distributions..."
rm -rf build dist
python -m build

echo "Checking distributions..."
python -m twine check dist/*

echo "Release preflight passed for $tag ($version)."
