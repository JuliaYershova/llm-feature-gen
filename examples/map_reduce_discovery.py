"""Run exhaustive discovery with an offline toy provider or a configured LLM."""

import argparse
import json
from pathlib import Path

from llm_feature_gen import discover_features_from_texts
from llm_feature_gen.providers import LocalProvider, OpenAIProvider


class OfflineProvider:
    cache_identity = {"example": "map_reduce_discovery", "version": 1}

    def text_features(self, texts, prompt=None, **kwargs):
        request = json.loads(texts[0])
        if request["mode"] == "reduce":
            return [{"merge_groups": []}]
        ids = [item["item_id"] for item in request["items"]]
        features = [{"feature": "topic", "description": "Observed topic",
                     "possible_values": ["ordinary", "rare"], "example_item_ids": ids}]
        rare = [item["item_id"] for item in request["items"] if "rare" in item["text"]]
        if rare:
            features.append({"feature": "rare_detail", "description": "Presence of an unusual explicit detail",
                             "possible_values": ["present", "absent"], "example_item_ids": rare})
        return [{"processed_item_ids": ids, "proposed_features": features}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["offline", "local", "openai"], default="offline")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/map_reduce_demo"))
    args = parser.parse_args()
    provider = {"offline": OfflineProvider, "local": LocalProvider, "openai": OpenAIProvider}[args.provider]()
    schema = discover_features_from_texts(
        ["Routine invoice request", "Routine account question", "Routine tracking request",
         "Routine billing question", "A rare unusual detail in the final input"],
        provider=provider, strategy="map_reduce", batch_size=2, shuffle_seed=None,
        min_features=2, output_dir=args.output_dir, checkpoint_dir=args.output_dir / "checkpoints",
    )
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
