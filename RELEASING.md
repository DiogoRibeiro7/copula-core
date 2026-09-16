# Releasing copula-core

Releases reach crates.io only through the [Release workflow](.github/workflows/release.yml),
triggered by a `vX.Y.Z` tag on `main`. Do not run `cargo publish` from a local machine.

A crates.io version is permanent: it can be yanked, but its number can never be
reused and its contents can never be changed.

## One-time setup

1. **crates.io trusted publishing.** On crates.io, open the `copula-core` crate
   settings, then **Trusted Publishing**, and add a GitHub publisher:
   - repository owner: `DiogoRibeiro7`
   - repository name: `copula-core`
   - workflow filename: `release.yml`
   - environment: `release`
2. **Require trusted publishing.** Once one release has gone out through the
   workflow, enable the crate setting that restricts publishing to trusted
   publishers. After that, an API token on a local machine can no longer publish
   this crate by accident.
3. **Release environment.** In GitHub, go to **Settings → Environments →
   `release`**. Add yourself as a required reviewer, and limit deployments to
   tags matching `v*`.
4. **Tokens.** Revoke any crates.io API token with publish scope that is no
   longer needed (<https://crates.io/settings/tokens>).

## Cutting a release

1. On a working branch off `develop`:
   - set `version` in `Cargo.toml`
   - in `CHANGELOG.md`, move the `[Unreleased]` entries under a new
     `## [X.Y.Z] - YYYY-MM-DD` heading and update the compare links at the bottom
   - open a pull request into `develop`
2. Open a pull request from `develop` into `main` and merge it once CI passes.
3. Tag the merge commit and push the tag:

   ```sh
   git switch main
   git pull --ff-only
   git tag -a vX.Y.Z -m "copula-core X.Y.Z"
   git push origin vX.Y.Z
   ```

4. The workflow checks that the tag matches `Cargo.toml`, the commit is on
   `main`, and `CHANGELOG.md` has a section for the version. It then runs the
   tests, builds the package, and waits for approval on the `release`
   environment. After approval it publishes to crates.io and creates the GitHub
   release from the changelog section.

## Choosing the version

- Before 1.0: breaking changes bump the minor version (`0.2.0 → 0.3.0`);
  everything else bumps the patch version.
- An MSRV increase is allowed in a minor release and must be listed in the
  changelog.
- Pre-releases use a suffix (`v0.3.0-rc.1`) and are marked as such on GitHub.

## If a bad version is published

1. Yank it: `cargo yank --version X.Y.Z`. Yanking stops new dependency resolution
   from choosing that version; projects with it in their lockfile keep building.
   `cargo yank --version X.Y.Z --undo` reverses it.
2. Release a fixed version through the normal process.
3. Record the yank in `CHANGELOG.md` (`## [X.Y.Z] - YYYY-MM-DD [YANKED]`).

Deleting a crate is possible only under the conditions in the
[crates.io policies](https://crates.io/policies), for example within 72 hours of
publication.
