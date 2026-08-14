# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- `LLMFeatureTransformer`, a scikit-learn compatible transformer that turns raw texts into discovered feature columns inside a `Pipeline`. Install with the new `sklearn` extra: `pip install "llm-feature-gen[sklearn]"`.
- `num_frames` parameter exposed in the video generation pipeline.

### Changed

- Documentation leads with a scikit-learn quickstart: README, docs landing page, and the quickstart guide now open with a runnable pipeline before the two-step discover/generate workflow.

### Changed

- Feature CSVs are rewritten on each run instead of appended, so repeated runs no longer accumulate duplicate rows.
- Clearer error message when a thinking model spends its entire token budget on reasoning and returns an empty reply.
- README clarifications for feature discovery and generation usage.
- PyPI publish workflow step is idempotent.

## [0.1.13] - 2026-07-08

### Added

- Multiclass workflows: class-aware feature discovery (`discover_features_multiclass`, `MultiClassDiscoveryPromptBuilder`), multiclass generation, and the `run_multiclass_pipeline` orchestration over shared train/test schemas.
- Batched text generation with on-disk response caching: `generate_features_batch`, `generate_features_from_texts_cached`, and `BatchTextCache`.
- Configurable `min_features` discovery parameter for text, tabular, image, and video workflows.
- `num_classes` parameter so text discovery prompts can target more than two hidden categories.
- File discovery includes files in subdirectories.

### Changed

- Providers raise `ProviderResponseError` instead of returning error dictionaries; a shared feature-value contract rejects error and invalid payloads consistently.
- Stronger generation prompts for schema enums and raw JSON output.
- Empty or whitespace-only input files are skipped with a warning instead of aborting the whole run.

### Fixed

- Fail-fast on an invalid or missing discovery schema instead of generating against garbage.
- Generation circuit breaker aborts after repeated consecutive provider/output failures instead of wasting the rest of a long run.

## [0.1.12] - 2026-04-19

### Added

- `CODEOWNERS` file for review routing.

### Fixed

- `LocalProvider` raises errors properly instead of swallowing them; tests updated to match the new error handling.

### Changed

- Documentation polish and a documentation link in project metadata.

## [0.1.11] - 2026-04-09

### Added

- MkDocs documentation site with automated GitHub Pages deployment.
- Cross-platform CI: Linux, macOS, and Windows, with Python 3.9, 3.11, and 3.13 exercised in GitHub Actions.
- Canonical end-to-end text-to-tabular example with an offline replay mode and a smoke test that verifies it in CI.
- Explicit support documentation via `SUPPORT.md`, a docs support matrix, and PyPI classifiers for supported operating systems and Python versions.

## [0.1.10] - 2026-04-09

### Added

- Expanded documentation: quickstart, tutorial notebook, and refreshed usage docs, including the required Jupyter kernel restart after notebook installs.
- Coverage reporting in CI with a Codecov badge.

### Changed

- Project metadata updates in `pyproject.toml`.
- Optional text-parser failures now raise clearer dependency guidance for PDF, DOCX, and HTML inputs.

## [0.1.9] - 2026-04-09

### Added

- CI test workflow in GitHub Actions with an end-to-end smoke test and full offline test coverage of discovery artifacts, generation CSV output, local-provider compatibility, and optional parser behavior.
- Contributor documentation and GitHub issue templates.

### Changed

- README contributor guidance now points to the contributing guide, changelog, and issue templates.

## [0.1.8] - 2026-03-10

### Added

- Initial `0.1.8` release of `llm-feature-gen`.
