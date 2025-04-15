#! /bin/bash

isort  $(find src -name '*.py' )  ; black  $(find src  -name '*.py' )  ; isort  $(find tests -name '*.py') ; black  $(find tests -name '*.py')

npx prettier --write  .github/workflows/*.yml
remote_branch=$(git rev-parse --abbrev-ref --symbolic-full-name @{u})

v=$(git show "$remote_branch":VERSION.txt)
echo "${v%.*}.$((${v##*.}+1))" > VERSION.txt

echo "----------------------------------------"
echo "VERSION.txt updated to $(cat VERSION.txt)"
echo "----------------------------------------"