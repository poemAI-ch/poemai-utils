# Releasing poemai-utils

`VERSION.txt` is the single source of the package version. The normal
development workflow keeps the automatic patch-version increment in
`precommit.sh`.

## Release flow

1. Make the code changes.
2. Run `./precommit.sh`. If `VERSION.txt` has not been manually changed, it is
   incremented to the next patch version.
3. Commit and push the resulting versioned change to `main`.
4. Wait for the green `main` GitHub Actions test/build run.
5. Run `./create_release_tag.sh --push`.
6. Monitor the tag workflow in GitHub Actions and verify the version on PyPI.

The release tag helper does not run tests or build distributions locally. It
relies on the successful `main` CI run and only checks release metadata and
repository state before creating and optionally pushing the matching annotated
tag, for example `v3.2.8`.

When `precommit.sh` starts with a clean working tree, it does not increment the
version. It checks whether the current `VERSION.txt` has a matching local or
origin tag and emits a warning if that tag is missing. Make a code change before
running it when you want to prepare the next automatic patch version.

## Safety checks

The release tag helper requires a clean `main` checkout synchronized with
`origin/main`. It validates the `X.Y.Z` version and rejects an existing local
or remote tag.

The GitHub Actions `main` run is the release candidate check: it runs the unit
tests, builds the source and wheel distributions, and checks them with Twine
without publishing anything.

The GitHub Actions workflow runs tests and builds on `main` and on `v*` tags.
Only the tag path reaches the publish job. Before publishing, it requires the
tag version and `VERSION.txt` to match exactly. A push to `main` never publishes
to PyPI.

PyPI publication uses the tested build artifact and PyPI Trusted Publishing.
The repository’s PyPI Trusted Publisher must be configured for
`.github/workflows/publish-to-pypi.yml` and the `pypi` GitHub environment.
