# When to Use It

`llm-feature-gen` occupies a specific spot among the ways to get unstructured data in front of a model. This page states what that spot is, and when another approach serves you better.

## The trade-off it makes

Text embeddings, CLIP vectors, and similar dense representations carry more raw signal than a dozen named features, and after setup they cost almost nothing per row. What they cannot give you is a column named `urgency` with the value `high` that a domain expert, an auditor, or a three-level decision tree can act on directly. This package trades some predictive ceiling for exactly that: features you can read, question, and reuse.

| Approach | Output | Readable columns | Marginal cost per row | Feeds classical models |
| --- | --- | --- | --- | --- |
| Embeddings (sentence transformers, CLIP) | dense vector | no | near zero | yes, but opaque |
| Zero-shot LLM classification | a label | no — one opaque decision | one LLM call | no features produced |
| Manual feature engineering | named features | yes | human time | yes |
| `llm-feature-gen` | named features | yes | LLM call(s) | yes |

Against manual feature engineering, the difference is who writes the schema: here the LLM proposes it from your data in minutes, and you keep the veto — the schema is a JSON file you can edit before generating a single value.

Against zero-shot classification, the difference is what survives the call: instead of a bare label, you keep a tabular dataset. You can train any model on it, measure feature importance, hand it to a colleague, or re-model it later without paying the LLM again.

## Good fits

- Small-to-medium datasets (hundreds to low tens of thousands of items) where interpretability matters: clinical notes, support tickets, product reviews, curated image sets.
- Settings where a domain expert must be able to challenge the model — the feature table is the shared language.
- Multimodal datasets that should end up in one consistent tabular form: the same two-step workflow covers text, images, video, and tabular text columns.
- Teaching and research on interpretable pipelines, where "what did the model see" needs a concrete answer.

## Use something else when

- You need millions of rows or low-latency inference — per-row LLM calls will dominate your budget; embeddings win.
- The task is purely perceptual (fine-grained image similarity, speaker identification) — dense representations capture what named features cannot.
- You only ever need the final label and never the intermediate evidence — direct LLM classification is one step shorter.
- Your data cannot leave your infrastructure and no local model is capable enough — although a local [provider](provider-configuration.md) through Ollama or vLLM is often sufficient for feature-value generation.

## Two properties worth knowing before you commit

**Schema stability.** Discovery is a generative step: two runs can propose different features. Treat the schema JSON as a versioned artifact — discover once, review, commit the file, and reuse it for every generation run. See [reproducibility](reproducibility.md).

**Cost scales with rows × runs.** Generation makes provider calls for every input. Caching ([batch workflows](advanced-workflows.md)) removes the cost of repeated runs over the same texts, but a fresh dataset always pays once.
