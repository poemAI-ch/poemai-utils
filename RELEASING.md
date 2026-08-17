# Releasing poemai-utils

`VERSION.txt` is the single source of the package version. The normal
development workflow keeps the automatic patch-version increment in
`precommit.sh`.

## Release flow

1. Make the code changes.
2. Run `./precommit.sh`. If `VERSION.txt` has not been manually changed, it is
   incremented to the next patch version.
3. Commit and push the resulting versioned change to `main`.
4. Run `./release_preflight.sh` from the clean, synchronized `main` checkout.
5. Run `./create_release_tag.sh --push`.
6. Monitor the tag workflow in GitHub Actions and verify the version on PyPI.

The release helpers never run `precommit.sh`, change `VERSION.txt`, commit
files, or force-update tags. They create and optionally push only the matching
annotated tag, for example `v3.2.8`.

When `precommit.sh` starts with a clean working tree, it does not increment the
version. It checks whether the current `VERSION.txt` has a matching local or
origin tag and emits a warning if that tag is missing. Make a code change before
running it when you want to prepare the next automatic patch version.

## Safety checks

The preflight requires a clean `main` checkout synchronized with
`origin/main`. It validates the `X.Y.Z` version, rejects an existing local or
remote tag, runs the unit tests, builds the source and wheel distributions, and
checks them with Twine.

The GitHub Actions workflow runs only for `v*` tags. Before publishing, it
requires the tag version and `VERSION.txt` to match exactly. A push to `main`
never publishes to PyPI.

PyPI publication uses the tested build artifact and PyPI Trusted Publishing.
The repository’s PyPI Trusted Publisher must be configured for
`.github/workflows/publish-to-pypi.yml` and the `pypi` GitHub environment.
