#!/usr/bin/env python3
"""Same pipeline as text_to_tabular_pipeline.py, expressed as a scikit-learn model.

No folders, no CSVs on the way in: ``X`` is a list of strings and ``y`` is a
list of labels, exactly as any other sklearn estimator expects. The pipeline is
``fit`` on a training split and asked to ``predict`` the held-out one, then
scored properly by cross-validation against a TF-IDF baseline. ``transform``
exposes the interpretable table the classifier actually saw, written to
feature_table.csv.

The task is the classic hard pair from 20 Newsgroups, ``comp.sys.mac.hardware``
against ``comp.sys.ibm.pc.hardware``: two groups arguing about the same SCSI
chains and RAM upgrades in the same vocabulary.

    python3 examples/sklearn_pipeline.py --provider replay --check   # offline
    python3 examples/sklearn_pipeline.py --provider auto             # live LLM

The feature schema and the recorded model answers are both reused when they
exist and generated when they do not, so a repeated run costs nothing.
``--rediscover`` forces a new schema, and ``--record`` writes a live run's
answers to ``examples/expected/sklearn_pipeline/responses.json``, which is what
an offline run replays. Answers are keyed by post, not by a cache key derived
from the prompt, so editing a prompt never invalidates them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_feature_gen import batch as batch_module
from llm_feature_gen import BatchTextCache, LLMFeatureTransformer
from llm_feature_gen.generate import parse_json_from_markdown
from llm_feature_gen.multiclass import discover_features_multiclass
from llm_feature_gen.providers import LocalProvider, OpenAIProvider

EXAMPLE_ROOT = REPO_ROOT / "examples"
POSTS_CSV = EXAMPLE_ROOT / "data" / "newsgroups_hardware" / "posts.csv"
EXPECTED_DIR = EXAMPLE_ROOT / "expected" / "sklearn_pipeline"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "sklearn_pipeline"

SCHEMA_NAME = "discovered_features.json"
RESPONSES_NAME = "responses.json"
CACHE_NAME = "feature_cache.json"
FEATURE_TABLE_NAME = "feature_table.csv"
RESULT_NAME = "result.json"

DATASET = "20 Newsgroups: comp.sys.mac.hardware vs comp.sys.ibm.pc.hardware"
N_SPLITS = 5
TOTAL_STEPS = 4
TEST_SIZE = 0.25
RANDOM_STATE = 0
BATCH_SIZE = 8


class SharedProvider:
    """Hand the same provider to every cross-validation fold.

    ``cross_val_score`` clones the pipeline per fold, and ``sklearn.base.clone``
    deep-copies every constructor parameter. A provider holds an HTTP client
    with a thread lock, which cannot be deep-copied at all, so a live run would
    fail with ``cannot pickle '_thread.RLock'``. Sharing is the correct
    semantics anyway: a provider is a connection, not fitted state.
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def __deepcopy__(self, memo: Dict[int, Any]) -> "SharedProvider":
        memo[id(self)] = self
        return self

    def text_features(self, text_list: List[str], prompt: str | None = None) -> List[Dict[str, Any]]:
        return self.provider.text_features(text_list, prompt=prompt)

    @property
    def recorder(self) -> "RecordingProvider | None":
        """Expose the recording wrapper, when this run is making fixtures."""
        return self.provider if isinstance(self.provider, RecordingProvider) else None


class SharedCache(BatchTextCache):
    """Hand the same cache to every fold, for the same reason.

    Without this, each fold starts from an empty copy and re-sends the whole
    corpus: five folds, five times the calls. Entries are keyed by text and
    schema hash, so sharing them across folds leaks nothing — identical input
    always maps to identical output.
    """

    def __deepcopy__(self, memo: Dict[int, Any]) -> "SharedCache":
        memo[id(self)] = self
        return self


def feature_values(answer: Any) -> Dict[str, str] | None:
    """Pull the feature mapping out of one provider answer.

    Providers may hand back the values directly, wrapped under ``features``, or
    as a JSON string in a markdown fence. Recording has to understand all three,
    because a live run is paid for once and cannot be replayed to fix a miss.
    """
    if not isinstance(answer, dict):
        return None

    values = answer.get("features", answer)
    if isinstance(values, str):
        values = parse_json_from_markdown(values)
    return values if isinstance(values, dict) else None


class ReplayProvider:
    """Answer from the recorded responses instead of calling a model.

    Recording at the provider boundary, rather than through the library cache,
    is what keeps this example reproducible. The cache key folds in the prompt
    text and the provider settings, so editing a prompt silently invalidates
    every recorded answer; a post keyed by its own id stays valid.
    """

    def __init__(self, responses: Dict[str, Dict[str, str]], post_ids: Dict[str, str]) -> None:
        self.responses = responses
        self.post_ids = post_ids

    def text_features(self, text_list: List[str], prompt: str | None = None) -> List[Dict[str, Any]]:
        answers = []
        for text in text_list:
            post_id = self.post_ids.get(text)
            recorded = self.responses.get(post_id) if post_id else None
            if recorded is None:
                # Falling back to a live call would be worse: it would cost
                # money and quietly make an offline run unreproducible.
                raise RuntimeError(
                    f"No recorded answer for post {post_id or '<unknown text>'}. "
                    "Rerun with --provider auto --record."
                )
            answers.append({"features": dict(recorded)})
        return answers


class RecordingProvider:
    """Pass every call to a live provider and keep the answers.

    What is kept is what a later offline run replays, so the fixtures are
    always written from real responses and never hand-edited.
    """

    def __init__(self, provider: Any, post_ids: Dict[str, str]) -> None:
        self.provider = provider
        self.post_ids = post_ids
        self.recorded: Dict[str, Dict[str, str]] = {}

    def text_features(self, text_list: List[str], prompt: str | None = None) -> List[Dict[str, Any]]:
        answers = self.provider.text_features(text_list, prompt=prompt)
        for text, answer in zip(text_list, answers):
            post_id = self.post_ids.get(text)
            values = feature_values(answer)
            if post_id and values is not None:
                self.recorded[post_id] = values
        return answers

    def save(self, path: Path) -> None:
        """Write the recorded answers as the fixtures a replay run reads."""
        if not self.recorded:
            raise RuntimeError("Nothing was recorded; refusing to overwrite the fixtures.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.recorded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "local", "replay"],
        default="auto",
        help=(
            "Provider backend. 'auto' selects OpenAI/Azure when configured, "
            "otherwise a local OpenAI-compatible endpoint. 'replay' runs "
            "offline from the checked-in cache."
        ),
    )
    parser.add_argument(
        "--rediscover",
        action="store_true",
        help=(
            "Force a fresh feature schema even though one exists. The new schema "
            "invalidates every cached response, so all documents are re-sent."
        ),
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "Write this run's answers and artifacts into "
            "examples/expected/sklearn_pipeline/ so they become the checked-in fixtures."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated artifacts against the checked-in expected outputs.",
    )
    args = parser.parse_args()
    if args.record:
        if args.provider == "replay":
            parser.error("--record needs a live provider; --provider replay cannot produce fixtures.")
        args.output_dir = EXPECTED_DIR
    return args


def select_provider(provider_name: str, post_ids: Dict[str, str], record: bool) -> SharedProvider:
    """Pick a backend and wrap it so cross-validation can share one instance."""
    if provider_name == "replay":
        return SharedProvider(ReplayProvider(load_responses(), post_ids))

    live = live_provider(provider_name)
    return SharedProvider(RecordingProvider(live, post_ids) if record else live)


def live_provider(provider_name: str) -> Any:
    """Return the configured model backend, or explain what is missing."""
    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "local":
        return LocalProvider()

    if "AZURE_OPENAI_ENDPOINT" in os.environ or "OPENAI_API_KEY" in os.environ:
        return OpenAIProvider()
    if (
        "LOCAL_OPENAI_BASE_URL" in os.environ
        or "LOCAL_MODEL_TEXT" in os.environ
        or "LOCAL_MODEL_VISION" in os.environ
    ):
        return LocalProvider()

    raise EnvironmentError(
        "No provider configuration found. Set OpenAI/Azure credentials, set local "
        "OpenAI-compatible server variables, or run with --provider replay."
    )


def load_responses() -> Dict[str, Dict[str, str]]:
    """Read the recorded answers an offline run replays."""
    path = EXPECTED_DIR / RESPONSES_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No recorded answers at {path}, and an offline run cannot create them. "
            "Record them with --provider auto --record."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_schema(
    args: argparse.Namespace,
    provider: Any,
    discovery_texts: List[str],
    classes: List[str],
) -> Path:
    """Return the feature schema, generating one only when none exists.

    Discovery is told which classes it has to separate. Without the class
    names the model proposes features that merely describe a text — tone,
    formality, humour — and those split evenly across both newsgroups, leaving
    the classifier at chance. Passing the names is not leakage: you know your
    own labels at training time, and the discovery posts are held out of both
    train and test, so no post informs both the schema and the score.

    The schema is pinned rather than rediscovered per fit, because every fit and
    predict has to see the same columns. It is also reused across runs: rerunning
    discovery yields a different schema, which changes the cache key of every
    recorded response and forces the whole corpus to be sent again.
    """
    pinned = EXPECTED_DIR / SCHEMA_NAME
    if not args.rediscover:
        # The fixtures first, then whatever an earlier plain run produced, so a
        # repeated run neither pays for discovery again nor needs --record.
        for candidate in (pinned, args.output_dir / SCHEMA_NAME):
            if candidate.exists():
                step(1, f"Feature discovery {attention('skipped')}: reusing the schema at {candidate}")
                note("Pass --rediscover to derive a new one; every document is then re-sent")
                return candidate

    if args.provider == "replay":
        raise FileNotFoundError(
            f"No schema at {pinned}, and an offline run cannot create one. "
            "Record it with --provider auto --record."
        )

    if pinned.exists():
        step(
            1,
            f"Re-running feature discovery for classes {', '.join(classes)} over "
            f"{len(discovery_texts)} documents. The previous schema is replaced, "
            "which makes every cached response unusable: all documents will be re-sent",
        )
    else:
        step(
            1,
            f"No schema found; running feature discovery for classes {', '.join(classes)} "
            f"({len(discovery_texts)} documents)",
        )

    # --record points --output-dir at the fixtures; any other run keeps its
    # schema local, so the committed one is only replaced deliberately.
    discover_features_multiclass(
        discovery_texts,
        classes=classes,
        provider=provider,
        output_dir=args.output_dir,
        output_filename=SCHEMA_NAME,
    )
    return args.output_dir / SCHEMA_NAME


ORANGE = "\033[33m"
RESET = "\033[0m"


def attention(word: str) -> str:
    """Colour a word, but only when writing to a terminal.

    Piped or redirected output stays plain, so escape codes never end up in a
    log file or in the expected-output fixtures.
    """
    return f"{ORANGE}{word}{RESET}" if sys.stdout.isatty() else word


def resolve_cache(args: argparse.Namespace) -> SharedCache | None:
    """Return the cache a live run shares across folds, or ``None`` offline.

    Cross-validation asks for the same document once per fold. The cache keeps
    a live run from paying for each of them five times. An offline run needs no
    cache: every answer is already recorded, so it is left out and the example
    exercises the library exactly as a first-time user would.
    """
    if args.provider == "replay":
        return None
    # Scratch, never a fixture: the cache key folds in the prompt text, so a
    # committed cache would go stale the moment a prompt changed.
    return SharedCache(cache_file=DEFAULT_OUTPUT_DIR / CACHE_NAME)


def step(number: int, message: str) -> None:
    """Announce each stage, so a slow provider call is never silent."""
    print(f"[{number}/{TOTAL_STEPS}] {message}", flush=True)


def note(message: str) -> None:
    """Indent a follow-up line under the step it belongs to."""
    print(f"      {message}", flush=True)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    posts = pd.read_csv(POSTS_CSV)
    post_ids = dict(zip(posts["text"], posts["post_id"]))
    provider = select_provider(args.provider, post_ids, args.record)
    cache = resolve_cache(args)
    discovery_posts = posts[posts["split"] == "discovery"]

    n_model = int((posts["split"] == "model").sum())
    print(f"{DATASET}")
    print(f"{n_model} documents, {len(discovery_posts)} held out for feature discovery, provider: {args.provider}\n")

    schema = resolve_schema(
        args,
        provider,
        discovery_posts["text"].tolist(),
        sorted(discovery_posts["label"].unique().tolist()),
    )

    model_posts = posts[posts["split"] == "model"]
    texts = model_posts["text"].tolist()
    labels = model_posts["label"].tolist()

    if args.provider == "replay":
        note(f"Provider calls {attention('skipped')}: answers replayed from {EXPECTED_DIR / RESPONSES_NAME}")

    # ------------------------------------------------------------------ #
    # The whole integration. Everything above is loading and plumbing, and
    # everything below is an ordinary scikit-learn estimator: fit on the
    # training half, predict the held-out half, transform to read the table
    # the classifier saw.
    # ------------------------------------------------------------------ #
    model = make_pipeline(
        LLMFeatureTransformer(
            provider=provider,
            discovered_features=schema,
            batch_size=BATCH_SIZE,
            cache=cache,
        ),
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=1000),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    step(2, f"fit() on {len(X_train)} documents, predict() on the {len(X_test)} held out")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    holdout_accuracy = accuracy_score(y_test, predictions)
    note(f"Held-out accuracy {holdout_accuracy:.3f}")
    # ------------------------------------------------------------------ #

    # One split of 30 documents is a noisy estimate, so the reported number
    # comes from cross-validation: every document is predicted exactly once,
    # by a model that never saw it. Feature values depend only on the text of
    # a single post, never on the other rows, so generating them inside the
    # pipeline leaks nothing across folds.
    step(3, f"{N_SPLITS}-fold stratified cross-validation over all {len(texts)} documents")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, texts, labels, cv=cv, scoring="accuracy")

    step(4, "TF-IDF baseline over the same folds")
    baseline = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english"),
        LogisticRegression(max_iter=1000),
    )
    baseline_scores = cross_val_score(baseline, texts, labels, cv=cv, scoring="accuracy")

    # Refit on everything, then transform once more: the table written to disk
    # is the interpretable artifact, one readable row per post.
    model.fit(texts, labels)
    transformer = model.named_steps["llmfeaturetransformer"]
    feature_table = transformer.transform(texts)
    feature_table.insert(0, "label", labels)
    feature_table.insert(0, "post_id", model_posts["post_id"].tolist())
    # The document goes in the last column: every row can then be traced back to
    # the text its values were generated from, while the feature columns stay
    # readable when the table is opened in a spreadsheet.
    feature_table["text"] = texts
    table_path = args.output_dir / FEATURE_TABLE_NAME
    feature_table.to_csv(table_path, index=False)

    result = {
        "dataset": DATASET,
        "n_samples": len(texts),
        "holdout": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "accuracy": round(float(holdout_accuracy), 6),
        },
        "cv": {"n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE},
        "feature_names": list(transformer.get_feature_names_out()),
        "accuracy_mean": round(float(scores.mean()), 6),
        "accuracy_std": round(float(scores.std()), 6),
        "fold_accuracy": [round(float(score), 6) for score in scores],
        "tfidf_baseline_mean": round(float(baseline_scores.mean()), 6),
        "tfidf_baseline_std": round(float(baseline_scores.std()), 6),
    }
    result_path = args.output_dir / RESULT_NAME
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.record and provider.recorder is not None:
        provider.recorder.save(EXPECTED_DIR / RESPONSES_NAME)

    show(result)
    return {"feature_table": table_path, "result": result_path, "schema": schema}


def show(result: Dict[str, Any]) -> None:
    """Report the cross-validated accuracy against the baseline."""
    holdout = result["holdout"]
    print(f"\nHeld-out accuracy ({holdout['n_train']} train, {holdout['n_test']} test)")
    print(f"  Semantic variables  {holdout['accuracy']:.3f}")

    print(f"\nClassification accuracy ({result['n_samples']} documents, "
          f"{N_SPLITS}-fold stratified cross-validation)")
    print(f"  Semantic variables  {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}")
    print(f"  TF-IDF baseline     {result['tfidf_baseline_mean']:.3f} ± {result['tfidf_baseline_std']:.3f}")


def compare_with_expected(output_dir: Path) -> None:
    generated_table = pd.read_csv(output_dir / FEATURE_TABLE_NAME)
    expected_table = pd.read_csv(EXPECTED_DIR / FEATURE_TABLE_NAME)
    if not generated_table.equals(expected_table):
        raise AssertionError(f"Generated feature table does not match {EXPECTED_DIR / FEATURE_TABLE_NAME}")

    generated = json.loads((output_dir / RESULT_NAME).read_text(encoding="utf-8"))
    expected = json.loads((EXPECTED_DIR / RESULT_NAME).read_text(encoding="utf-8"))
    if generated["feature_names"] != expected["feature_names"]:
        raise AssertionError("Generated feature names do not match the expected schema")

    # Accuracies allow a small tolerance: solver output can wobble in the last
    # digits across BLAS builds without anything being wrong.
    for key in ("accuracy_mean", "tfidf_baseline_mean"):
        if abs(generated[key] - expected[key]) > 1e-3:
            raise AssertionError(f"{key} drifted: {generated[key]} vs {expected[key]}")

    if abs(generated["holdout"]["accuracy"] - expected["holdout"]["accuracy"]) > 1e-3:
        raise AssertionError(
            f"holdout accuracy drifted: {generated['holdout']['accuracy']} "
            f"vs {expected['holdout']['accuracy']}"
        )


class QuietStdout:
    """Drop the library's routine per-batch cache messages.

    ``generate_features_batch`` reports a cache-hit count for every fold, which
    buries the actual report. Filtering happens per line, so warnings and batch
    failures still reach the terminal — only the routine chatter is dropped.
    """

    NOISE = ("Cache hits:", "Saved batch results to")

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.pending = ""

    def write(self, text: str) -> int:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if not line.startswith(self.NOISE):
                self.stream.write(line + "\n")
        return len(text)

    def isatty(self) -> bool:
        return self.stream.isatty()

    def flush(self) -> None:
        if self.pending and not self.pending.startswith(self.NOISE):
            self.stream.write(self.pending)
            self.pending = ""
        self.stream.flush()


def hide_empty_progress_bars(quiet: bool = False) -> None:
    """Drop the progress bar on passes that send nothing.

    ``quiet`` drops it entirely, which is what an offline run wants: replaying
    a recorded answer is instant, so a bar for it is pure noise.

    ``generate_features_batch`` builds a tqdm bar before it knows whether any
    text still needs generating, so a fully cached pass prints an empty
    ``Batch generation: 0batch`` line on stderr. The library is left untouched:
    it resolves its module-level ``_tqdm`` at call time, so wrapping that name
    here is enough to skip the bar when there is no work, and keep it when
    documents really are being sent.
    """
    # getattr, not attribute access: if the library ever renames this, the
    # example should quietly keep its old output rather than crash.
    original = getattr(batch_module, "_tqdm", None)
    if original is None:
        return

    def progress(iterable: Any, **kwargs: Any) -> Any:
        if quiet or len(iterable) == 0:
            return iterable
        return original(iterable, **kwargs)

    batch_module._tqdm = progress


def main() -> int:
    args = parse_args()
    hide_empty_progress_bars(quiet=args.provider == "replay")
    with redirect_stdout(QuietStdout(sys.stdout)):
        paths = run_pipeline(args)

    print(f"\nArtifacts written to {paths['result'].parent}")
    if args.check:
        compare_with_expected(args.output_dir)
        print("Generated artifacts match the checked-in expected outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
