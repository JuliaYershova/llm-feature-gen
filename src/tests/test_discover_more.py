from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_feature_gen import discover as discover_mod


class TextProvider:
    def __init__(self) -> None:
        self.calls = []

    def text_features(self, text_list, prompt=None):
        self.calls.append({"texts": list(text_list), "prompt": prompt})
        return [{"proposed_features": [{"feature": "x"}]} for _ in text_list]


class ImageProvider:
    def __init__(self) -> None:
        self.calls = []

    def image_features(self, image_base64_list, prompt=None, as_set=False, extra_context=None):
        self.calls.append(
            {
                "images": list(image_base64_list),
                "prompt": prompt,
                "as_set": as_set,
                "extra_context": extra_context,
            }
        )
        return [{"proposed_features": [{"feature": "img"}]} for _ in image_base64_list] if not as_set else [{"proposed_features": [{"feature": "img"}]}]


def test_discover_texts_from_file_and_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    provider = TextProvider()

    text_file = tmp_path / "doc.txt"
    text_file.write_text("ignored", encoding="utf-8")
    monkeypatch.setattr(discover_mod, "extract_text_from_file", lambda path: ["one", "two"])

    result = discover_mod.discover_features_from_texts(text_file, provider=provider, as_set=True, output_dir=tmp_path / "out1")
    assert "proposed_features" in result
    assert provider.calls[0]["texts"] == ["one\n\n---\n\ntwo"]

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.txt").write_text("a", encoding="utf-8")
    (folder / "b.bin").write_text("b", encoding="utf-8")
    (folder / "nested").mkdir()

    def fake_extract(path: Path):
        if path.suffix == ".bin":
            raise ValueError("unsupported")
        return [path.stem]

    monkeypatch.setattr(discover_mod, "extract_text_from_file", fake_extract)
    result = discover_mod.discover_features_from_texts(folder, provider=provider, as_set=False, output_dir=tmp_path / "out2")
    assert isinstance(result, list)
    assert provider.calls[-1]["texts"] == ["a"]

    custom_file = tmp_path / "custom.txt"
    custom_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(discover_mod, "extract_text_from_file", lambda path: ["single"])
    discover_mod.discover_features_from_texts(
        custom_file,
        provider=provider,
        as_set=True,
        output_dir=tmp_path / "out3",
        output_filename="custom_text.json",
    )
    assert (tmp_path / "out3" / "custom_text.json").exists()


def test_discover_texts_error_paths_and_invalid_special_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(FileNotFoundError):
        discover_mod.discover_features_from_texts(tmp_path / "missing.txt", provider=TextProvider())

    with pytest.raises(ValueError):
        discover_mod.discover_features_from_texts([], provider=TextProvider())

    class FakePath:
        def __init__(self, raw):
            self.raw = raw

        def exists(self):
            return True

        def is_file(self):
            return False

        def is_dir(self):
            return False

        def __str__(self):
            return self.raw

    monkeypatch.setattr(discover_mod, "Path", FakePath)
    with pytest.raises(ValueError):
        discover_mod.discover_features_from_texts("weird", provider=TextProvider())


def test_discover_images_single_file_and_video_edge_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    img_path = tmp_path / "one.png"
    from PIL import Image
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(img_path)

    image_provider = ImageProvider()
    result = discover_mod.discover_features_from_images(img_path, provider=image_provider, as_set=True, output_dir=tmp_path / "imgout")
    assert "proposed_features" in result
    assert len(image_provider.calls[0]["images"]) == 1

    default_text_provider = TextProvider()
    monkeypatch.setattr(discover_mod, "OpenAIProvider", lambda: default_text_provider)
    discover_mod.discover_features_from_texts(["raw"], output_dir=tmp_path / "textout")
    assert default_text_provider.calls

    video_provider = ImageProvider()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    frames_map = {
        "good1.mp4": ["f1"] * 3,
        "good2.mp4": ["f2"] * 3,
    }

    def fake_extract_frames(path: str, frame_limit: int = 5):
        name = Path(path).name
        if name == "bad.mp4":
            raise RuntimeError("boom")
        return frames_map[name]

    def fake_extract_audio(path: str):
        if Path(path).name == "good2.mp4":
            raise RuntimeError("audio-bad")
        return str(audio_path)

    downsample_calls = []
    monkeypatch.setattr(discover_mod, "extract_key_frames", fake_extract_frames)
    monkeypatch.setattr(discover_mod, "extract_audio_track", fake_extract_audio)
    monkeypatch.setattr(discover_mod, "downsample_batch", lambda frames, target: downsample_calls.append((list(frames), target)) or frames[:target])
    monkeypatch.setattr(discover_mod.random, "sample", lambda seq, k: list(seq)[:k])

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    for name in ["bad.mp4", "good1.mp4", "good2.mp4"]:
        (video_dir / name).write_bytes(b"video")

    result = discover_mod.discover_features_from_videos(
        video_dir,
        provider=video_provider,
        as_set=False,
        output_dir=tmp_path / "vidout",
        max_videos_to_sample=2,
        max_total_frames_payload=2,
    )
    assert isinstance(result, list)
    assert video_provider.calls[0]["as_set"] is False
    assert downsample_calls

    list_provider = ImageProvider()
    monkeypatch.setattr(discover_mod, "OpenAIProvider", lambda: list_provider)
    monkeypatch.setattr(discover_mod, "extract_key_frames", lambda path, frame_limit=5: ["x"])
    monkeypatch.setattr(discover_mod, "extract_audio_track", lambda path: str(audio_path))
    result = discover_mod.discover_features_from_videos(
        [str(video_dir / "good1.mp4")],
        as_set=False,
        use_audio=False,
        output_dir=tmp_path / "vidout2",
    )
    assert isinstance(result, list)
    assert list_provider.calls[0]["images"] == ["x"]

    short_provider = ImageProvider()

    def short_transcribe(_path: str) -> str:
        return "short"

    short_provider.transcribe_audio = short_transcribe  # type: ignore[attr-defined]
    monkeypatch.setattr(discover_mod, "extract_key_frames", lambda path, frame_limit=5: ["x"])
    monkeypatch.setattr(discover_mod, "extract_audio_track", lambda path: str(audio_path))
    discover_mod.discover_features_from_videos(
        str(video_dir / "good1.mp4"),
        provider=short_provider,
        output_dir=tmp_path / "vidout3",
        output_filename="custom_video.json",
    )
    assert (tmp_path / "vidout3" / "custom_video.json").exists()


def test_discover_videos_warns_when_provider_has_no_transcriber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    provider = ImageProvider()
    video_path = tmp_path / "solo.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "solo.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(discover_mod, "extract_key_frames", lambda path, frame_limit=5: ["frame"])
    monkeypatch.setattr(discover_mod, "extract_audio_track", lambda path: str(audio_path))

    result = discover_mod.discover_features_from_videos(
        video_path,
        provider=provider,
        use_audio=True,
        output_dir=tmp_path / "warnout",
    )
    assert "proposed_features" in result


def test_discover_videos_handles_frame_errors_and_short_transcripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad_video = tmp_path / "bad.mp4"
    good_video = tmp_path / "good.mp4"
    bad_video.write_bytes(b"video")
    good_video.write_bytes(b"video")
    audio_path = tmp_path / "short.wav"
    audio_path.write_bytes(b"audio")

    def fake_extract_frames(path: str, frame_limit: int = 5):
        if Path(path).name == "bad.mp4":
            raise RuntimeError("frame failure")
        return ["good-frame"]

    monkeypatch.setattr(discover_mod, "extract_key_frames", fake_extract_frames)
    monkeypatch.setattr(discover_mod, "extract_audio_track", lambda path: str(audio_path))

    class ShortTranscriptProvider(ImageProvider):
        def transcribe_audio(self, audio_path: str) -> str:
            return "short"

    provider = ShortTranscriptProvider()

    result = discover_mod.discover_features_from_videos(
        [str(bad_video), str(good_video)],
        provider=provider,
        use_audio=True,
        as_set=False,
        output_dir=tmp_path / "shortcase",
    )

    assert isinstance(result, list)
    assert provider.calls[0]["images"] == ["good-frame"]
    assert provider.calls[0]["extra_context"] is None


def test_discover_tabular_supports_multiple_formats_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = tmp_path / "tabular"
    folder.mkdir()
    for name in ["a.csv", "b.xlsx", "c.parquet", "d.json", "e.bin"]:
        (folder / name).write_text("x", encoding="utf-8")

    calls = []

    def fake_read_csv(path, sep=","):
        calls.append(("csv", sep))
        if sep == ",":
            raise ValueError("fallback")
        return pd.DataFrame({"text": ["c1", "c2"]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(pd, "read_excel", lambda path: pd.DataFrame({"text": ["x1"]}))
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.DataFrame({"text": ["p1"]}))
    monkeypatch.setattr(pd, "read_json", lambda path: pd.DataFrame({"text": ["j1"]}))

    captured = {}

    def fake_discover_texts(texts_or_file, prompt, provider, as_set, output_dir, output_filename):
        captured["texts"] = texts_or_file
        captured["output_filename"] = output_filename
        return {"ok": True}

    monkeypatch.setattr(discover_mod, "discover_features_from_texts", fake_discover_texts)

    result = discover_mod.discover_features_from_tabular(
        file_or_folder=folder,
        text_column="text",
        provider=TextProvider(),
        max_rows=3,
    )
    assert result == {"ok": True}
    assert captured["texts"] == ["c1", "c2", "x1"]
    assert captured["output_filename"] == "discovered_tabular_features.json"
    assert calls == [("csv", ","), ("csv", ";")]

    subdir = folder / "subdir"
    subdir.mkdir()
    monkeypatch.setattr(pd, "read_csv", lambda path, sep=",": pd.DataFrame({"text": ["kept"]}))
    result = discover_mod.discover_features_from_tabular(
        file_or_folder=folder,
        text_column="text",
        provider=TextProvider(),
    )
    assert result == {"ok": True}
    assert captured["texts"] == ["kept", "x1", "p1", "j1"]

    with pytest.raises(FileNotFoundError):
        discover_mod.discover_features_from_tabular(tmp_path / "missing.csv", text_column="text", provider=TextProvider())

    single = tmp_path / "single.csv"
    single.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(pd, "read_csv", lambda path, sep=",": pd.DataFrame({"other": ["x"]}))
    with pytest.raises(ValueError):
        discover_mod.discover_features_from_tabular(single, text_column="text", provider=TextProvider())

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "a.bin").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        discover_mod.discover_features_from_tabular(bad, text_column="text", provider=TextProvider())

    class FakePath:
        def __init__(self, raw):
            self.raw = raw

        def exists(self):
            return True

        def is_file(self):
            return False

        def is_dir(self):
            return False

        def __str__(self):
            return self.raw

    monkeypatch.setattr(discover_mod, "Path", FakePath)
    with pytest.raises(ValueError):
        discover_mod.discover_features_from_tabular("odd", text_column="text", provider=TextProvider())
