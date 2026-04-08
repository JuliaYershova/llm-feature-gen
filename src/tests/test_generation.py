from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from llm_feature_gen import generate as gen
from llm_feature_gen.utils import text as text_utils


class FakeProvider:
    def __init__(self) -> None:
        self.image_calls = []
        self.text_calls = []

    def image_features(self, image_base64_list, prompt=None, as_set=False, extra_context=None):
        self.image_calls.append(
            {
                "images": list(image_base64_list),
                "prompt": prompt,
                "as_set": as_set,
                "extra_context": extra_context,
            }
        )
        return [{"features": {"feat1": "img", "feat2": "common"}}]

    def text_features(self, text_list, prompt=None):
        self.text_calls.append({"texts": list(text_list), "prompt": prompt})
        if len(text_list) == 1 and "row-text" in text_list[0]:
            return [{"features": '{"feat1": "row-value"}'}]
        return [{"features": {"feat1": "txt", "feat2": "common"}}]

    def transcribe_audio(self, audio_path: str) -> str:
        return f"audio:{audio_path}"


def make_image(path: Path) -> None:
    Image.new("RGB", (10, 10), color=(100, 50, 20)).save(path)


def test_prepare_tabular_inputs_supports_formats_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("text;label\nhello;A\nworld;B\n", encoding="utf-8")

    calls = []

    def fake_read_csv(path, sep=","):
        calls.append(sep)
        if sep == ",":
            raise ValueError("fallback")
        return pd.DataFrame({"text": ["hello", "world"], "label": ["A", "B"]})

    monkeypatch.setattr(gen.pd, "read_csv", fake_read_csv)
    rows = gen._prepare_tabular_inputs(csv_path, "text", "label")
    assert rows == [{"text": "hello", "label": "A"}, {"text": "world", "label": "B"}]
    assert calls == [",", ";"]

    excel_path = tmp_path / "data.xlsx"
    parquet_path = tmp_path / "data.parquet"
    json_path = tmp_path / "data.json"
    monkeypatch.setattr(gen.pd, "read_excel", lambda path: pd.DataFrame({"text": ["x"]}))
    monkeypatch.setattr(gen.pd, "read_parquet", lambda path: pd.DataFrame({"text": ["y"]}))
    monkeypatch.setattr(gen.pd, "read_json", lambda path: pd.DataFrame({"text": ["z"]}))

    assert gen._prepare_tabular_inputs(excel_path, "text") == [{"text": "x"}]
    assert gen._prepare_tabular_inputs(parquet_path, "text") == [{"text": "y"}]
    assert gen._prepare_tabular_inputs(json_path, "text") == [{"text": "z"}]

    with pytest.raises(ValueError):
        gen._prepare_tabular_inputs(tmp_path / "data.bin", "text")

    monkeypatch.setattr(gen.pd, "read_json", lambda path: pd.DataFrame({"other": ["z"]}))
    with pytest.raises(ValueError):
        gen._prepare_tabular_inputs(json_path, "text")


def test_prepare_text_inputs_delegates_to_text_utils(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(text_utils, "extract_text_from_file", lambda path: ["chunk"])
    assert gen._prepare_text_inputs(Path("dummy.txt")) == ["chunk"]


def test_prepare_video_inputs_handles_audio_and_missing_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    provider = FakeProvider()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(gen, "extract_audio_track", lambda path: str(audio_path))
    monkeypatch.setattr(gen, "extract_key_frames", lambda path, frame_limit=6: ["frame1", "frame2"])

    removed = []
    monkeypatch.setattr(gen.os, "remove", lambda path: removed.append(path))

    frames, transcript = gen._prepare_video_inputs(tmp_path / "video.mp4", use_audio=True, provider=provider)
    assert frames == ["frame1", "frame2"]
    assert transcript == f"audio:{audio_path}"
    assert removed == [str(audio_path)]

    provider_no_audio = SimpleNamespace()
    audio_path_2 = tmp_path / "audio2.wav"
    audio_path_2.write_bytes(b"audio")
    monkeypatch.setattr(gen, "extract_audio_track", lambda path: str(audio_path_2))
    frames, transcript = gen._prepare_video_inputs(tmp_path / "video.mp4", use_audio=True, provider=provider_no_audio)
    assert transcript == "(Audio transcription not supported by provider)"

    monkeypatch.setattr(gen, "extract_audio_track", lambda path: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(gen, "extract_key_frames", lambda path, frame_limit=6: [])
    frames, transcript = gen._prepare_video_inputs(tmp_path / "video.mp4", use_audio=True, provider=provider)
    assert frames == []
    assert transcript is None

    monkeypatch.setattr(gen, "extract_key_frames", lambda path, frame_limit=6: ["frame3"])
    frames, transcript = gen._prepare_video_inputs(tmp_path / "video.mp4", use_audio=False, provider=provider)
    assert frames == ["frame3"]
    assert transcript is None


def test_prepare_image_inputs_and_helper_functions(tmp_path: Path):
    img_path = tmp_path / "img.png"
    make_image(img_path)
    b64_list, context = gen._prepare_image_inputs(img_path)
    assert len(b64_list) == 1
    assert context is None

    discovered_path = tmp_path / "disc.json"
    discovered_path.write_text(json.dumps([{"proposed_features": [{"feature": "feat1"}]}]), encoding="utf-8")
    assert gen.load_discovered_features(discovered_path)["proposed_features"][0]["feature"] == "feat1"

    discovered_path.write_text(json.dumps([{"feature": "feat1"}]), encoding="utf-8")
    assert gen.load_discovered_features(discovered_path) == {"proposed_features": [{"feature": "feat1"}]}

    discovered_path.write_text(json.dumps({"proposed_features": [{"feature": "feat1"}]}), encoding="utf-8")
    assert gen.load_discovered_features(discovered_path) == {"proposed_features": [{"feature": "feat1"}]}

    with pytest.raises(FileNotFoundError):
        gen.load_discovered_features(tmp_path / "missing.json")

    assert gen.parse_json_from_markdown("") == {}
    assert gen.parse_json_from_markdown("```json\n{\"x\": 1}\n```") == {"x": 1}
    assert gen.parse_json_from_markdown("```json\n{\"x\": 1}") == {"x": 1}
    assert gen.parse_json_from_markdown("not json") == {}

    prompt = gen._build_prompt_for_generation("Base", {"proposed_features": [{"feature": "f"}]})
    assert "DISOVERED_FEATURES_SPEC" in prompt

    out_dir = gen._ensure_output_dir(tmp_path / "nested" / "dir")
    assert out_dir.exists()

    assert gen._extract_feature_names({"proposed_features": [{"feature": "a"}, "b", {"ignored": "x"}]}) == ["a", "b"]
    assert gen._extract_feature_names([{"feature": "a"}]) == ["a"]
    assert gen._infer_feature_names_from_llm([{"a": 1, "b": 2}]) == ["a", "b"]
    assert gen._infer_feature_names_from_llm({"features": {"x": 1}}) == ["x"]
    assert gen._infer_feature_names_from_llm({"y": 2}) == ["y"]
    assert gen._infer_feature_names_from_llm("nope") == []


def test_assign_feature_values_from_folder_for_tabular_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "root"
    class_dir = root / "classA"
    class_dir.mkdir(parents=True)
    (class_dir / "rows.csv").write_text("text,label\nrow-text,L1\n", encoding="utf-8")

    monkeypatch.setattr(gen, "tqdm", lambda files, desc=None, unit=None: files)
    provider = FakeProvider()

    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classA",
        discovered_features={"proposed_features": [{"feature": "feat1"}, {"feature": "feat2"}]},
        provider=provider,
        output_dir=tmp_path / "out",
        text_column="text",
        label_column="label",
    )

    df = pd.read_csv(csv_path)
    assert list(df["Class"]) == ["L1"]
    assert list(df["feat1"]) == ["row-value"]
    assert list(df["feat2"]) == ["not given by LLM"]

    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classA",
        discovered_features={"proposed_features": [{"feature": "feat1"}, {"feature": "feat2"}]},
        provider=provider,
        output_dir=tmp_path / "out",
        text_column="text",
        label_column="label",
    )
    assert csv_path.exists()


def test_assign_feature_values_from_folder_for_modalities_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "root"
    class_dir = root / "classB"
    class_dir.mkdir(parents=True)
    for name in ["img.jpg", "clip.mp4", "note.txt", "skip.bin", "bad.md"]:
        (class_dir / name).write_bytes(b"x")

    monkeypatch.setattr(gen, "_prepare_image_inputs", lambda path: (["image-b64"], None))
    monkeypatch.setattr(gen, "_prepare_video_inputs", lambda path, use_audio, provider: (["video-b64"], "transcript"))

    def fake_prepare_text_inputs(path: Path):
        if path.name == "bad.md":
            raise RuntimeError("broken")
        return ["text body"]

    monkeypatch.setattr(gen, "_prepare_text_inputs", fake_prepare_text_inputs)
    monkeypatch.setattr(gen, "tqdm", None)

    provider = FakeProvider()
    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classB",
        discovered_features={"proposed_features": [{"feature": "feat1"}, {"feature": "feat2"}]},
        provider=provider,
        output_dir=tmp_path / "out",
        use_audio=True,
    )

    df = pd.read_csv(csv_path)
    assert sorted(df["File"].tolist()) == ["clip.mp4", "img.jpg", "note.txt"]
    assert set(df["feat1"]) == {"img", "txt"}


def test_assign_feature_values_from_folder_missing_class_and_feature_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(FileNotFoundError):
        gen.assign_feature_values_from_folder(
            folder_path=tmp_path,
            class_name="missing",
            discovered_features={},
            provider=FakeProvider(),
        )

    root = tmp_path / "root"
    class_dir = root / "classC"
    class_dir.mkdir(parents=True)
    (class_dir / "img.jpg").write_bytes(b"x")
    monkeypatch.setattr(gen, "_prepare_image_inputs", lambda path: (["image-b64"], None))

    class InferringProvider(FakeProvider):
        def image_features(self, image_base64_list, prompt=None, as_set=False, extra_context=None):
            return [{"features": {"inferred": "yes"}}]

    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classC",
        discovered_features={},
        provider=InferringProvider(),
        output_dir=tmp_path / "out",
    )
    df = pd.read_csv(csv_path)
    assert list(df["File"]) == ["img.jpg"]


def test_assign_feature_values_from_folder_covers_remaining_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "root"
    class_dir = root / "classD"
    class_dir.mkdir(parents=True)
    (class_dir / "rows.csv").write_text("text\nvalue\n", encoding="utf-8")
    (class_dir / "clip.mp4").write_bytes(b"x")
    (class_dir / "note.txt").write_text("body", encoding="utf-8")

    provider = FakeProvider()
    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classD",
        discovered_features={"proposed_features": [{"feature": "feat1"}]},
        provider=provider,
        output_dir=tmp_path / "out_missing",
    )
    df_missing = pd.read_csv(csv_path)
    assert list(df_missing["File"]) == ["note.txt"]

    monkeypatch.setattr(gen, "_prepare_video_inputs", lambda path, use_audio, provider: ([], None))
    monkeypatch.setattr(gen, "_prepare_text_inputs", lambda path: ["text body"])

    class StringProvider(FakeProvider):
        def text_features(self, text_list, prompt=None):
            return [{"features": '{"feat1": "from-string"}'}]

    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classD",
        discovered_features={"proposed_features": [{"feature": "feat1"}]},
        provider=StringProvider(),
        output_dir=tmp_path / "out",
        text_column="text",
    )
    df = pd.read_csv(csv_path)
    assert list(df["feat1"]) == ["from-string", "from-string"]

    class DictProvider(FakeProvider):
        def text_features(self, text_list, prompt=None):
            return [{"features": {"feat1": "direct"}}]

    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classD",
        discovered_features={"proposed_features": [{"feature": "feat1"}]},
        provider=DictProvider(),
        output_dir=tmp_path / "out_direct",
        text_column="text",
    )
    df = pd.read_csv(csv_path)
    assert "direct" in df["feat1"].tolist()


def test_assign_feature_values_from_folder_text_only_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "root"
    class_dir = root / "classText"
    class_dir.mkdir(parents=True)
    (class_dir / "note.txt").write_text("body", encoding="utf-8")

    monkeypatch.setattr(gen, "_prepare_text_inputs", lambda path: ["chunk one", "chunk two"])
    monkeypatch.setattr(gen, "tqdm", None)

    provider = FakeProvider()
    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classText",
        discovered_features={"proposed_features": [{"feature": "feat1"}]},
        provider=provider,
        output_dir=tmp_path / "out",
    )
    df = pd.read_csv(csv_path)
    assert list(df["File"]) == ["note.txt"]
    assert provider.text_calls[0]["texts"] == ["chunk one\n\n---\n\nchunk two"]


def test_assign_feature_values_from_folder_unreachable_else_via_pathlike(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "root"
    class_dir = root / "classE"
    class_dir.mkdir(parents=True)

    class FlakyName:
        def __init__(self):
            self.calls = 0

        def __fspath__(self):
            self.calls += 1
            return "supported.jpg" if self.calls == 1 else "unsupported.xyz"

    monkeypatch.setattr(gen.os, "listdir", lambda path: [FlakyName()])
    csv_path = gen.assign_feature_values_from_folder(
        folder_path=root,
        class_name="classE",
        discovered_features={"proposed_features": [{"feature": "feat1"}]},
        provider=FakeProvider(),
        output_dir=tmp_path / "out",
    )
    df = pd.read_csv(csv_path)
    assert df.empty


def test_generate_features_and_wrappers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "root"
    (root / "c1").mkdir(parents=True)
    (root / "c2").mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    generated = {}

    def fake_load(path):
        assert str(path).endswith(".json")
        return {"proposed_features": [{"feature": "feat1"}]}

    def fake_assign(folder_path, class_name, discovered_features, provider, output_dir, use_audio, text_column, label_column):
        csv_path = Path(output_dir) / f"{class_name}.csv"
        pd.DataFrame([{"File": f"{class_name}.txt", "Class": class_name, "feat1": "x", "raw_llm_output": "{}"}]).to_csv(
            csv_path,
            index=False,
        )
        generated[class_name] = {
            "use_audio": use_audio,
            "text_column": text_column,
            "label_column": label_column,
        }
        return csv_path

    monkeypatch.setattr(gen, "load_discovered_features", fake_load)
    monkeypatch.setattr(gen, "assign_feature_values_from_folder", fake_assign)
    monkeypatch.setattr(gen, "OpenAIProvider", lambda: "default-provider")

    result = gen.generate_features(
        root_folder=root,
        discovered_features_path=tmp_path / "features.json",
        output_dir=output_dir,
        merge_to_single_csv=True,
        text_column="body",
        label_column="label",
    )

    assert set(result) == {"c1", "c2", "__merged__"}
    assert Path(result["__merged__"]).exists()
    assert generated["c1"]["text_column"] == "body"

    result = gen.generate_features(
        root_folder=root,
        discovered_features_path=tmp_path / "features.json",
        output_dir=output_dir,
        classes=["c1"],
        merge_to_single_csv=False,
    )
    assert set(result) == {"c1"}

    captured = []

    def fake_generate(*args, **kwargs):
        captured.append(kwargs["discovered_features_path"])
        return {"ok": "1"}

    monkeypatch.setattr(gen, "generate_features", fake_generate)
    assert gen.generate_features_from_tabular(root_folder=root) == {"ok": "1"}
    assert gen.generate_features_from_texts(root_folder=root) == {"ok": "1"}
    assert gen.generate_features_from_images(root_folder=root) == {"ok": "1"}
    assert gen.generate_features_from_videos(root_folder=root) == {"ok": "1"}
    assert gen.generate_features_from_tabular(root_folder=root, discovered_features_path="custom_tab.json") == {"ok": "1"}
    assert gen.generate_features_from_texts(root_folder=root, discovered_features_path="custom_text.json") == {"ok": "1"}
    assert gen.generate_features_from_images(root_folder=root, discovered_features_path="custom_image.json") == {"ok": "1"}
    assert gen.generate_features_from_videos(root_folder=root, discovered_features_path="custom_video.json", use_audio=False) == {"ok": "1"}
    assert captured == [
        "outputs/discovered_tabular_features.json",
        "outputs/discovered_text_features.json",
        "outputs/discovered_image_features.json",
        "outputs/discovered_video_features.json",
        "custom_tab.json",
        "custom_text.json",
        "custom_image.json",
        "custom_video.json",
    ]


def test_video_discovery_and_generation_wrapper_defaults_line_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from llm_feature_gen import discover as disc

    monkeypatch.chdir(tmp_path)
    videos_root = tmp_path / "videos"
    class_dir = videos_root / "classA"
    class_dir.mkdir(parents=True)
    video_file = class_dir / "sample.mp4"
    video_file.write_bytes(b"video")

    monkeypatch.setattr(disc, "extract_key_frames", lambda path, frame_limit=5: ["discover-frame"])
    monkeypatch.setattr(disc, "extract_audio_track", lambda path: None)
    monkeypatch.setattr(gen, "_prepare_video_inputs", lambda path, use_audio, provider: (["gen-frame"], None))
    monkeypatch.setattr(gen, "tqdm", None)

    class SmokeProvider:
        def image_features(self, image_base64_list, prompt=None, as_set=False, extra_context=None):
            if prompt and "DISOVERED_FEATURES_SPEC" in prompt:
                return [{"features": {"shape": "round"}}]
            return [{"proposed_features": [{"feature": "shape"}]}]

    provider = SmokeProvider()

    discovered = disc.discover_features_from_videos(
        videos_or_folder=str(class_dir),
        provider=provider,
        use_audio=False,
    )
    assert discovered["proposed_features"][0]["feature"] == "shape"
    assert (tmp_path / "outputs" / "discovered_video_features.json").exists()

    result = gen.generate_features_from_videos(
        root_folder=videos_root,
        provider=provider,
        use_audio=False,
    )
    generated_csv = Path(result["classA"])
    assert generated_csv.exists()
    df = pd.read_csv(generated_csv)
    assert list(df["shape"]) == ["round"]


def test_generate_module_can_fall_back_without_tqdm(monkeypatch: pytest.MonkeyPatch):
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reloaded = importlib.reload(gen)
    assert reloaded.tqdm is None
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(gen)
