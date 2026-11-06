# Contributing to LLM Feature Gen

Thanks for helping improve LLM Feature Gen.

## Local Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

If you need support for optional file formats, install the extra runtime dependencies called out in [`README.md`](README.md).

## Running Tests

Run the full test suite with:

```bash
pytest
```

Useful variants:

```bash
pytest -vv
pytest src/tests/test_discovery.py
```

Tests use fake providers and temporary directories, so they do not require real OpenAI or Azure credentials.

## Pull Request Expectations

- Keep pull requests focused and easy to review.
- Add or update tests when behavior changes.
- Update docs, examples, and prompts when user-facing behavior, supported inputs, or outputs change.
- Add user-visible changes to [`CHANGELOG.md`](CHANGELOG.md) under `## [Unreleased]`.
- Include a short summary of what changed and how you validated it.

## Publishing to PyPI after merge

The next release is `0.1.14`, as declared in `pyproject.toml`. The publishing
workflow runs when a `v*` tag is pushed; merging a PR alone does not publish.

After this PR is merged and main's tests pass, tag the merged main commit:

```bash
git switch main
git pull --ff-only origin main
git tag -a v0.1.14 -m "Release 0.1.14"
git push origin v0.1.14
```

The [Publish to PyPI workflow](.github/workflows/python-publish.yml) verifies that
the tag matches the package version and points to a commit included in main.
It runs the tests, builds the source distribution and wheel, checks them with
Twine, and uploads using the existing `PYPI_API_TOKEN` repository secret.
Check that workflow's result and the [PyPI release page](https://pypi.org/project/llm-feature-gen/)
after pushing the tag. A tag/version mismatch or failed test/build prevents upload.

For later releases, update the version in `pyproject.toml`, document the changes
in `CHANGELOG.md`, and use the matching version tag after those changes merge.

## Reporting Issues

- Use the GitHub issue templates when opening bugs or feature requests.
- Check for existing issues before creating a new one.
- For bugs, include a minimal reproduction, package version, Python version, OS/environment details, provider configuration, and any relevant logs or traceback.

## Security and Sensitive Data

- Never post API keys, tokens, secrets, or private datasets in issues, pull requests, screenshots, or logs.
- Redact sensitive prompts, customer data, and environment variables before sharing output.
