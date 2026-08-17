#! /bin/bash

set -euo pipefail

clean_at_start=true
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    clean_at_start=false
fi

isort  $(find src -name '*.py' )  ; black  $(find src  -name '*.py' )  ; isort  $(find tests -name '*.py') ; black  $(find tests -name '*.py')

# Check for unused imports (excluding simple.py files which are meant for import convenience)
echo "Checking for unused imports..."
if ! flake8 --select=F401 --exclude="src/*/__init__.py,src/*/simple.py" src/; then
    echo "❌ Found unused imports! Please remove them before committing."
    exit 1
fi
echo "✅ No unused imports found."

if ! flake8 --select=F401 --exclude="src/*/simple.py" --exclude "build/*" --exclude "lambda_runtimes/*" tests/; then
    echo "WARNING: Found unused imports in tests! Please remove them before committing."
else
    echo "✅ No unused imports found in tests."
fi



npx prettier --write  .github/workflows/*.yml
if ! remote_branch=$(git rev-parse --abbrev-ref --symbolic-full-name @{u}); then
    echo "Unable to determine the upstream branch. Configure an upstream before running precommit.sh."
    exit 1
fi

# Get the upstream version
if ! upstream_version=$(git show "$remote_branch":VERSION.txt); then
    echo "Unable to read VERSION.txt from $remote_branch."
    exit 1
fi

if [[ ! "$upstream_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Upstream VERSION.txt contains an invalid release version: $upstream_version"
    exit 1
fi

if [[ "$clean_at_start" == true ]]; then
    current_version=$(tr -d '[:space:]' < VERSION.txt)
    if [[ ! "$current_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Current VERSION.txt contains an invalid release version: $current_version"
        exit 1
    fi

    current_tag="v$current_version"
    tag_exists=false
    if git rev-parse --verify --quiet "refs/tags/$current_tag" >/dev/null; then
        tag_exists=true
    elif git ls-remote --exit-code --quiet --tags origin "refs/tags/$current_tag" >/dev/null 2>&1; then
        tag_exists=true
    fi

    if [[ "$tag_exists" == true ]]; then
        echo "Working tree is clean; VERSION.txt remains $current_version and tag $current_tag exists."
    else
        echo "WARNING: Working tree is clean; VERSION.txt remains $current_version, but tag $current_tag was not found locally or on origin."
    fi
    exit 0
fi

# Check if VERSION.txt has been modified (either in working directory or staged)
version_modified=false

# Check if VERSION.txt is different from upstream in working directory
if ! git diff --quiet "$remote_branch" -- VERSION.txt; then
    version_modified=true
fi

# Check if VERSION.txt is staged (different from HEAD)
if git diff --cached --quiet VERSION.txt; then
    # No staged changes
    :
else
    version_modified=true
fi

if [ "$version_modified" = true ]; then
    echo "----------------------------------------"
    echo "VERSION.txt has been manually modified - skipping auto-increment"
    echo "Current version: $(cat VERSION.txt)"
    echo "----------------------------------------"
else
    # Auto-increment version
    new_version="${upstream_version%.*}.$((${upstream_version##*.}+1))"
    echo "$new_version" > VERSION.txt
    echo "----------------------------------------"
    echo "VERSION.txt auto-incremented to $(cat VERSION.txt)"
    echo "----------------------------------------"
fi
