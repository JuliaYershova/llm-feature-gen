import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def build_multiclass_discovery_prompt(class_samples: dict) -> str:
    n_classes = len(class_samples)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {n_classes}")

    class_list_str = ", ".join(f'"{c}"' for c in class_samples)
    lines = [
        f"You are a feature engineering assistant. There are {n_classes} text classes: {class_list_str}.",
        "",
        "Discover a compact set of binary (Yes/No) features that together uniquely distinguish",
        "each class from all others. Requirements:",
        "- Each feature must be answerable Yes or No for any text.",
        "- Prefer surface-observable patterns over vague semantics.",
        "- Aim for 5-15 features that collectively separate ALL classes.",
        "- Avoid features that are true (or false) for every class.",
        "",
        "## Class examples",
    ]
    for cn, samples in class_samples.items():
        lines.append(f"\n### Class: {cn}")
        for i, s in enumerate(samples[:3], 1):
            lines.append(f"  {i}. {s}")
    lines += [
        "",
        "Return ONLY a JSON array of feature name strings, no explanation, no markdown fences.",
    ]
    return "\n".join(lines)


class TestMulticlassDiscovery(unittest.TestCase):

    def test_two_classes(self):
        samples = {"a": ["text1", "text2"], "b": ["text3", "text4"]}
        prompt = build_multiclass_discovery_prompt(samples)
        self.assertIn('"a"', prompt)
        self.assertIn('"b"', prompt)

    def test_three_classes(self):
        samples = {"x": ["t1"], "y": ["t2"], "z": ["t3"]}
        prompt = build_multiclass_discovery_prompt(samples)
        self.assertIn("3 text classes", prompt)

    def test_one_class_raises(self):
        with self.assertRaises(ValueError):
            build_multiclass_discovery_prompt({"only": ["t1", "t2"]})

    def test_all_class_names_in_prompt(self):
        names = ["alpha", "beta", "gamma", "delta"]
        samples = {n: [f"example from {n}"] for n in names}
        prompt = build_multiclass_discovery_prompt(samples)
        for n in names:
            self.assertIn(n, prompt)

    def test_prompt_is_string(self):
        samples = {"a": ["t1"], "b": ["t2"]}
        self.assertIsInstance(build_multiclass_discovery_prompt(samples), str)

    def test_examples_appear_in_prompt(self):
        samples = {"cls1": ["unique_example_xyz"], "cls2": ["different_example_abc"]}
        prompt = build_multiclass_discovery_prompt(samples)
        self.assertIn("unique_example_xyz", prompt)
        self.assertIn("different_example_abc", prompt)


if __name__ == "__main__":
    unittest.main()
