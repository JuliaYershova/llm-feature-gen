"""Pytest configuration and helpers.

Ensures the project package is importable when running tests directly from the
repository root without installing the package. We prepend the `src` directory
to `sys.path` so that `import LLM_feature_gen` works reliably.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def _ensure_src_on_syspath() -> None:
    # tests/ is at: <repo>/src/tests/, so `parents[1]` is the `src` directory
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_src_on_syspath()

# Provide lightweight stubs for optional heavy dependencies so test import works
# without requiring system ffmpeg or the ffmpeg-python package.
def _install_test_stubs() -> None:
    # Stub for the external 'ffmpeg' module (used inside utils.video)
    if 'ffmpeg' not in sys.modules:
        sys.modules['ffmpeg'] = ModuleType('ffmpeg')

    # Stub for LLM_feature_gen.utils.video to avoid importing real ffmpeg
    pkg_name = 'LLM_feature_gen.utils.video'
    if pkg_name not in sys.modules:
        mod = ModuleType(pkg_name)

        # Minimal stand-ins matching the signatures used by discover.py
        def extract_key_frames(video_path: str, frame_limit: int = 5):
            # Return an empty list by default; tests will monkeypatch as needed
            return []

        def transcribe_video(video_path: str):
            return ""

        mod.extract_key_frames = extract_key_frames  # type: ignore[attr-defined]
        mod.transcribe_video = transcribe_video      # type: ignore[attr-defined]
        sys.modules[pkg_name] = mod


_install_test_stubs()
