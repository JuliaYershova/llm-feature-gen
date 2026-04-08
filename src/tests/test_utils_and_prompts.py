from __future__ import annotations

import base64
import importlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import llm_feature_gen.prompts as prompts_mod
from llm_feature_gen.utils import text as text_mod


def load_actual_video_module():
    import sys

    sys.modules.pop("llm_feature_gen.utils.video", None)
    module = importlib.import_module("llm_feature_gen.utils.video")
    return importlib.reload(module)


def make_b64_image(color: int) -> str:
    img = Image.new("RGB", (8, 8), color=(color, color, color))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_prompts_load_prompt_and_missing():
    text = prompts_mod.load_prompt("image_discovery_prompt")
    assert isinstance(text, str) and text

    with pytest.raises(FileNotFoundError):
        prompts_mod.load_prompt("missing_prompt")


def test_text_utils_support_formats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    txt = tmp_path / "a.txt"
    txt.write_text("hello", encoding="utf-8")
    md = tmp_path / "a.md"
    md.write_text("markdown", encoding="utf-8")
    html = tmp_path / "a.html"
    html.write_text("<html><body><h1>Hi</h1></body></html>", encoding="utf-8")

    fake_pypdf = SimpleNamespace(
        PdfReader=lambda path: SimpleNamespace(
            pages=[SimpleNamespace(extract_text=lambda: "p1"), SimpleNamespace(extract_text=lambda: None)]
        )
    )
    fake_docx = SimpleNamespace(
        Document=lambda path: SimpleNamespace(
            paragraphs=[SimpleNamespace(text="d1"), SimpleNamespace(text=" "), SimpleNamespace(text="d2")]
        )
    )

    class FakeSoup:
        def __init__(self, text, parser):
            self.text = text

        def get_text(self, separator="\n"):
            return "Hi"

    fake_bs4 = SimpleNamespace(BeautifulSoup=FakeSoup)

    monkeypatch.setitem(__import__("sys").modules, "pypdf", fake_pypdf)
    monkeypatch.setitem(__import__("sys").modules, "docx", fake_docx)
    monkeypatch.setitem(__import__("sys").modules, "bs4", fake_bs4)

    assert text_mod.extract_text_from_file(txt) == ["hello"]
    assert text_mod.extract_text_from_file(md) == ["markdown"]
    assert text_mod.extract_text_from_file(tmp_path / "a.pdf") == ["p1", ""]
    assert text_mod.extract_text_from_file(tmp_path / "a.docx") == ["d1", "d2"]
    assert text_mod.extract_text_from_file(html) == ["Hi"]

    with pytest.raises(ValueError):
        text_mod.extract_text_from_file(tmp_path / "a.bin")


def test_video_utils_signature_and_downsample(monkeypatch: pytest.MonkeyPatch):
    video_mod = load_actual_video_module()
    arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    signature = video_mod._get_frame_signature(arr)
    assert signature.size > 0

    small = ["a", "b"]
    assert video_mod.downsample_batch(small, target_count=3) == small

    assert video_mod.downsample_batch(["not-base64", "still-bad"], target_count=1) == ["not-base64"]

    images = [make_b64_image(i * 20) for i in range(4)]
    monkeypatch.setattr(video_mod.cv2, "kmeans", lambda data, K, *args: (None, np.array([[0], [1], [2], [2]]), None))
    distinct = video_mod.downsample_batch(images, target_count=3)
    assert len(distinct) == 3

    monkeypatch.setattr(video_mod.cv2, "kmeans", lambda data, K, *args: (None, np.array([[0], [0], [0], [0]]), None))
    chosen = video_mod.downsample_batch(images, target_count=3)
    assert len(chosen) == 3

    monkeypatch.setattr(video_mod.cv2, "kmeans", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    fallback = video_mod.downsample_batch(images, target_count=2)
    assert len(fallback) == 2


def test_video_utils_extract_key_frames_and_audio(monkeypatch: pytest.MonkeyPatch):
    video_mod = load_actual_video_module()

    class FakeCap:
        def __init__(self, frames, opened=True):
            self.frames = list(frames)
            self.opened = opened

        def isOpened(self):
            return self.opened

        def get(self, prop):
            return 2

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self):
            return None

    monkeypatch.setattr(video_mod.cv2, "VideoCapture", lambda path: FakeCap([], opened=False))
    assert video_mod.extract_key_frames("bad.mp4") == []

    blurry_frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    monkeypatch.setattr(video_mod.cv2, "VideoCapture", lambda path: FakeCap(blurry_frames, opened=True))
    assert video_mod.extract_key_frames("empty.mp4", sharpness_threshold=999999) == []

    monkeypatch.setattr(video_mod.cv2, "Laplacian", lambda gray, typ: SimpleNamespace(var=lambda: 1000))
    monkeypatch.setattr(video_mod, "_get_frame_signature", lambda frame: np.array([1.0, 2.0], dtype=np.float32))

    skip_frames = [np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(2)]
    class SkipCap(FakeCap):
        def get(self, prop):
            return 4

    monkeypatch.setattr(video_mod.cv2, "VideoCapture", lambda path: SkipCap(skip_frames.copy(), opened=True))
    result = video_mod.extract_key_frames("skip.mp4", frame_limit=5)
    assert len(result) == 1

    frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(4)]
    monkeypatch.setattr(video_mod.cv2, "VideoCapture", lambda path: FakeCap(frames.copy(), opened=True))
    monkeypatch.setattr(video_mod.cv2, "kmeans", lambda data, K, *args: (None, np.array([[0], [1], [0], [1]]), None))
    result = video_mod.extract_key_frames("ok.mp4", frame_limit=2, max_resolution=16)
    assert len(result) == 2

    monkeypatch.setattr(video_mod.cv2, "VideoCapture", lambda path: FakeCap(frames.copy(), opened=True))
    monkeypatch.setattr(video_mod.cv2, "kmeans", lambda data, K, *args: (None, np.array([[0], [0], [0], [0]]), None))
    result = video_mod.extract_key_frames("cluster-gap.mp4", frame_limit=2, max_resolution=16)
    assert len(result) == 1

    captured = {}

    class FakeFfmpegChain:
        def output(self, path, **kwargs):
            captured["path"] = path
            return self

        def run(self, quiet=True, overwrite_output=True):
            return None

    monkeypatch.setattr(video_mod.ffmpeg, "input", lambda path: FakeFfmpegChain(), raising=False)
    monkeypatch.setattr(video_mod.time, "time", lambda: 123)
    monkeypatch.setattr(video_mod.os.path, "exists", lambda path: path == captured.get("path"))
    assert video_mod.extract_audio_track("video.mp4") == "temp_audio_video_123.wav"

    monkeypatch.setattr(video_mod.os.path, "exists", lambda path: False)
    assert video_mod.extract_audio_track("video.mp4") is None

    monkeypatch.setattr(video_mod.ffmpeg, "input", lambda path: (_ for _ in ()).throw(RuntimeError("boom")))
    assert video_mod.extract_audio_track("video.mp4") is None
