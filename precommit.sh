#! /bin/bash

isort  $(find src -name '*.py' )  ; black  $(find src  -name '*.py' )  ; isort  $(find tests -name '*.py') ; black  $(find tests -name '*.py')

# Check for unused imports (excluding simple.py files which are meant for import convenience)
echo "Checking for unused imports..."
flake8 --select=F401 --exclude="src/*/__init__.py,src/*/simple.py" src/
if [ $? -ne 0 ]; then
    echo "❌ Found unused imports! Please remove them before committing."
    exit 1
fi
echo "✅ No unused imports found."

flake8 --select=F401 --exclude="src/*/simple.py" --exclude "build/*" --exclude "lambda_runtimes/*" tests/
if [ $? -ne 0 ]; then
    echo "WARNING: Found unused imports in tests! Please remove them before committing."
else
    echo "✅ No unused imports found in tests."
fi



npx prettier --write  .github/workflows/*.yml
remote_branch=$(git rev-parse --abbrev-ref --symbolic-full-name @{u})

# Get the upstream version
upstream_version=$(git show "$remote_branch":VERSION.txt)

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