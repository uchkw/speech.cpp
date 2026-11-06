# The optimization fork is built from source without binary updates

## Context

This fork maintains changes absent from upstream binaries and has no binary release channel. Installing an upstream
release can replace those changes while keeping the same executable name and version format.

## Decision

Build and update this fork from its checkout and pinned submodule. Binary installation and update checks are
unsupported, and commands show no release notices. Model catalogs and Hugging Face sources remain shared with upstream.

The alternatives were turned down:

- Upstream installers and update notices: they direct users to binaries that omit the fork's changes.
- Redirecting release URLs to the fork: there are no fork releases to install, and a binary channel also needs distinct
  cache and installation identities so that upstream binaries cannot be mistaken for fork builds.

## Consequences

CI builds and packages the platform artifacts, but release installation and updates remain explicitly untested until
this fork establishes a binary release channel with its own checks.
