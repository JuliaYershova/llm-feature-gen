# Reproducibility

LLM-backed feature generation has two sources of nondeterminism: the model's sampling and the provider's serving stack. This page lists what the package pins down for you, what it cannot pin, and what to record when you publish results.

## What the package controls

- **Temperature.** Both built-in providers default to `temperature=0.0` for every call.
- **The schema artifact.** Discovery writes the feature schema to a JSON file. Generation only reads that file — so once a schema is saved, every later generation run scores against identical feature definitions. Commit the JSON next to your experiment code and the most volatile step of the pipeline is out of the loop.
- **Video sampling.** When a folder holds more videos than `max_videos_to_sample`, the subset is random; pass `random_seed` to make it repeatable.
- **Audit trail.** Every generated CSV keeps the raw provider response in the `raw_llm_output` column, so any suspicious feature value can be traced back to what the model actually returned.
- **Response caching.** [`BatchTextCache`](api/batch.md) stores responses keyed by text content and schema hash. Re-runs replay cached responses byte-for-byte instead of asking the model again.

## What the package cannot control

- **Provider-side nondeterminism.** Hosted APIs can return different completions for identical requests at temperature 0 — model updates, serving changes, and floating-point nondeterminism are outside any client's control. Record the model identifier and the date of your runs.
- **Discovery variability.** Two discovery runs may propose different (equally reasonable) schemas. Do not re-discover inside an experiment loop; discover once, review, and pin the JSON.
- **Local backend variance.** Local servers differ in tokenizer, context length, JSON-mode support, and multimodal formatting. Local models also vary more than hosted APIs in their ability to return strict JSON — prefer instruction-tuned models with reliable JSON output, and keep discovery batches within the model's context window.

## What to report in a paper or benchmark

For hosted providers: the provider, model name, and run dates, plus the package version (`llm_feature_gen.__version__`).

For local backends: the server (Ollama, vLLM, LM Studio), model names and versions or hashes, quantization, hardware, and decoding settings.

In both cases, publish the discovered schema JSON and, where licensing allows, the generated CSVs — together they make the downstream modeling fully repeatable without any LLM access.

## A fully offline reference run

The repository's [end-to-end example](examples.md) has a `--provider replay` mode that replays checked-in provider responses. It exists so the example is verifiable in CI with no credentials, and it doubles as a template for shipping reproducible artifacts alongside your own experiments.
