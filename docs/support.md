# Platform and Python Support

This page makes the package support claims explicit for users, reviewers, and downstream projects.

## Declared support

`llm-feature-gen` targets CPython `3.9+` and is intended to run on current Linux, macOS, and Windows environments.

| Dimension | Supported targets |
| --- | --- |
| Python | CPython 3.9, 3.11, 3.13 |
| Operating systems | Linux, macOS, Windows |

These claims are also encoded in the repository metadata through `requires-python` and Trove classifiers in `pyproject.toml`.

## CI coverage

The current GitHub Actions matrix exercises the following environments:

| Runner | Python versions | CI scope |
| --- | --- | --- |
| `ubuntu-latest` | 3.9, 3.11, 3.13 | Full test suite |
| `macos-latest` | 3.11 | Full test suite |
| `windows-latest` | 3.11 | Full test suite |

This means Linux has full multi-version coverage in CI, while macOS and Windows run the same full test suite on Python 3.11 to check for platform regressions.

## Notes and caveats

- Video audio extraction requires the `ffmpeg` system binary on every platform.
- PDF support requires `pypdf`.
- DOCX support requires `python-docx`.
- HTML support requires `beautifulsoup4`.
- XLSX support requires `openpyxl`.
- XLS support requires `xlrd`.
- Parquet support requires `pyarrow` or `fastparquet`.
- Other Python `3.9+` environments may work, but the combinations above are the ones currently declared and exercised in project metadata and CI.
