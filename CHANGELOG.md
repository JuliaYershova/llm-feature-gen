# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Contributor documentation and GitHub issue templates.
- Cross-platform CI coverage for Linux, macOS, and Windows, with Python 3.9, 3.11, and 3.13 exercised in GitHub Actions.
- Explicit support documentation via `SUPPORT.md`, a docs support matrix, and PyPI classifiers for supported operating systems and Python versions.
- Lightweight offline integration and smoke tests covering discovery output artifacts, generation CSV output, local-provider compatibility, and optional parser behavior.
- Configurable `min_features` discovery parameter for text, tabular, image, and video workflows.

### Changed

- README contributor guidance now points to the contributing guide, changelog, and issue templates.
- Project metadata now makes platform and interpreter support claims explicit for publication and review.
- Optional text-parser failures now raise clearer dependency guidance for PDF, DOCX, and HTML inputs.
- Provider response handling now uses a shared feature-value contract so generation rejects error and invalid payloads consistently.

### Fixed

- Improved offline test coverage for higher-level workflows so support claims are backed by end-to-end checks, not only unit tests.
- Documented the required Jupyter kernel restart after notebook installs to avoid immediate `ModuleNotFoundError` imports.

## [0.1.8]

### Added

- Initial `0.1.8` release of `llm-feature-gen`.
