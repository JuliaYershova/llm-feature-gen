import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from PIL import Image

from LLM_feature_gen.discover import (
    discover_features_from_images,
    discover_features_from_videos,
)


class FakeProvider:
    """A minimal fake provider that mimics the interface used by discover.py.

    It does not perform any network calls. It only shapes the return value
    the way the real provider does, which is sufficient for unit testing
    the discovery workflow and file I/O.
    """

    def image_features(
        self,
        image_base64_list: List[str],
        prompt: Optional[str] = None,
        as_set: bool = False,
        extra_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if as_set:
            # Joint mode returns a single-item list with one dict
            return [
                {
                    "proposed_features": [
                        {"name": "size", "type": "numeric"},
                        {"name": "color", "type": "categorical"},
                    ],
                    "count": len(image_base64_list),
                    "has_context": extra_context is not None,
                }
            ]
        # Per-image mode returns one dict per image
        return [
            {"proposed_features": [{"name": f"feat_{i}", "type": "text"}]}
            for i in range(len(image_base64_list))
        ]


def _make_image(path: Path, size: tuple[int, int] = (16, 16), color=(128, 64, 32)) -> None:
    """Create a small RGB image file for tests."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = np.array(color, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path)


@pytest.fixture()
def tmp_images_dir(tmp_path: Path) -> Path:
    """Create a temporary directory populated with a few test images."""
    d = tmp_path / "images"
    d.mkdir()
    _make_image(d / "a.jpg")
    _make_image(d / "b.jpeg")
    _make_image(d / "c.png")
    return d


def test_images_from_folder_as_set_saves_and_returns_first(tmp_images_dir: Path, tmp_path: Path):
    """When as_set=True, the function should return the single dict from the list
    and save the same structure to disk.
    """
    provider = FakeProvider()
    out_dir = tmp_path / "out"

    result = discover_features_from_images(
        str(tmp_images_dir), provider=provider, as_set=True, output_dir=out_dir
    )

    # Return type: dict (first element of the single-item list)
    assert isinstance(result, dict)
    assert "proposed_features" in result
    assert result.get("count") == 3

    # File saved with default name
    out_file = out_dir / "discovered_features.json"
    assert out_file.exists()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert isinstance(saved, list) and len(saved) == 1
    assert saved[0].get("count") == 3


def test_images_from_list_per_image_returns_list_and_saves(tmp_images_dir: Path, tmp_path: Path):
    provider = FakeProvider()
    out_dir = tmp_path / "out"
    image_paths = [str(p) for p in sorted(tmp_images_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    result = discover_features_from_images(
        image_paths, provider=provider, as_set=False, output_dir=out_dir
    )

    # Return type: list with one item per image
    assert isinstance(result, list)
    assert len(result) == len(image_paths)
    assert all("proposed_features" in r for r in result)

    # File saved
    out_file = out_dir / "discovered_features.json"
    assert out_file.exists()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert isinstance(saved, list)
    assert len(saved) == len(image_paths)


def test_raises_on_no_images(tmp_path: Path):
    """An empty folder should raise a ValueError about no images to process."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        discover_features_from_images(str(empty_dir), provider=FakeProvider())


def test_raises_on_invalid_path(tmp_path: Path):
    """A non-existent path should raise FileNotFoundError early."""
    bogus = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_features_from_images(str(bogus), provider=FakeProvider())


def test_skips_unreadable_images_then_raises_if_none_loaded(tmp_path: Path):
    """If images cannot be opened, the function should end up raising RuntimeError."""
    d = tmp_path / "badimgs"
    d.mkdir()
    # Create files with image-like extensions but invalid content
    for name in ["x.jpg", "y.jpeg", "z.png"]:
        (d / name).write_bytes(b"not-an-image")

    with pytest.raises(RuntimeError):
        discover_features_from_images(str(d), provider=FakeProvider())


def test_custom_output_filename(tmp_images_dir: Path, tmp_path: Path):
    provider = FakeProvider()
    out_dir = tmp_path / "out"
    custom_name = "my_discovery.json"

    _ = discover_features_from_images(
        str(tmp_images_dir), provider=provider, as_set=True, output_dir=out_dir, output_filename=custom_name
    )

    out_file = out_dir / custom_name
    assert out_file.exists()


# ------------------------------
# Video discovery tests
# ------------------------------

def test_video_discovery_single_file_saves_with_derived_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """For a single video file, output name should be derived as features_<stem>.json.

    We monkeypatch frame extraction and transcription to avoid heavy deps.
    """

    # Create a fake video file
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")

    # Monkeypatch helpers used by the function
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example")

    def fake_extract_key_frames(_vp: str, frame_limit: int = 5):
        # Return a few base64-like placeholders
        return ["b64frame1", "b64frame2"]

    def fake_transcribe_video(_vp: str):
        return "this is a transcript with enough length to be included"

    from LLM_feature_gen import discover as discover_mod

    monkeypatch.setattr(discover_mod, "extract_key_frames", fake_extract_key_frames)
    monkeypatch.setattr(discover_mod, "transcribe_video", fake_transcribe_video)

    provider = FakeProvider()
    out_dir = tmp_path / "out"

    result = discover_features_from_videos(
        str(video_path), provider=provider, output_dir=out_dir, num_frames=3, use_audio=True
    )

    # Should return a dict (single joint call under the hood)
    assert isinstance(result, dict)
    assert result.get("count") == 2
    assert result.get("has_context") is True

    expected_name = out_dir / "features_sample.json"
    assert expected_name.exists()


def test_video_discovery_folder_sampling_and_no_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Folder with multiple videos: ensure it aggregates frames and skips audio when disabled."""
    folder = tmp_path / "videos"
    folder.mkdir()
    for name in ["a.mp4", "b.mov", "c.avi"]:
        (folder / name).write_bytes(b"fake")

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example")

    def fake_extract_key_frames(_vp: str, frame_limit: int = 5):
        return ["f1", "f2", "f3"]

    from LLM_feature_gen import discover as discover_mod
    monkeypatch.setattr(discover_mod, "extract_key_frames", fake_extract_key_frames)

    provider = FakeProvider()
    out_dir = tmp_path / "out"

    result = discover_features_from_videos(
        str(folder), provider=provider, output_dir=out_dir, num_frames=3, use_audio=False
    )

    # 3 videos * 3 frames each
    assert isinstance(result, dict)
    assert result.get("count") == 9
    assert result.get("has_context") is False

    # Output filename derived from folder name
    expected_name = out_dir / "features_videos.json"
    assert expected_name.exists()


def test_video_discovery_raises_on_invalid_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_features_from_videos(str(tmp_path / "nope.mp4"), provider=FakeProvider())


def test_video_discovery_raises_when_no_videos_in_folder(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        discover_features_from_videos(str(empty), provider=FakeProvider())


def test_video_discovery_raises_when_no_frames_extracted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"fake")

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example")

    def fake_extract_key_frames(_vp: str, frame_limit: int = 5):
        return []  # simulate failure to extract frames

    from LLM_feature_gen import discover as discover_mod
    monkeypatch.setattr(discover_mod, "extract_key_frames", fake_extract_key_frames)

    with pytest.raises(ValueError):
        discover_features_from_videos(str(video_path), provider=FakeProvider())