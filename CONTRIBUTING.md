# Contributing to LLM Feature Gen

Thanks for helping improve LLM Feature Gen.

## Local Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

If you need support for optional file formats, install the extra runtime dependencies called out in [`README.md`](https://github.com/JuliaYershova/LLM-feature-gen/blob/main/README.md).

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
- Add user-visible changes to [`CHANGELOG.md`](https://github.com/JuliaYershova/LLM-feature-gen/blob/main/CHANGELOG.md) under `## [Unreleased]`.
- Include a short summary of what changed and how you validated it.

## Reporting Issues

- Use the GitHub issue templates when opening bugs or feature requests.
- Check for existing issues before creating a new one.
- For bugs, include a minimal reproduction, package version, Python version, OS/environment details, provider configuration, and any relevant logs or traceback.

## Security and Sensitive Data

- Never post API keys, tokens, secrets, or private datasets in issues, pull requests, screenshots, or logs.
- Redact sensitive prompts, customer data, and environment variables before sharing output.
