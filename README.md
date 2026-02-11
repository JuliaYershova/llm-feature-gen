# LLM_feature_gen



**LLM Feature Gen** is a Python library for **discovering and generating interpretable features** from unstructured data using Large Language Models (LLMs).  
The library provides high-level utilities for:
- Discovering human-interpretable features from sets of images,
- Integrating prompts and model outputs into structured JSON representations,
- - Generating new feature representations automatically from raw multimodal data,
e.g., creating structured tables for downstream models,


---

## Module: `discover`

The `discover` module focuses on **feature discovery** — identifying interpretable, discriminative visual or textual properties using an LLM.

Supported Data Types
- Images (.jpg, .png)
- Text documents (.txt, .pdf, .docx, .md, .html)
- Tabular datasets (.csv, .xlsx, .parquet, .json)
- Videos (.mp4)

### ✅ What it does
Given a folder of images and a prompt, the library:
1. Converts each image into Base64 format,  
2. Sends them to an LLM,  
3. Receives a structured JSON response describing the discovered features,  
4. Automatically saves the output to a  JSON file in `outputs/`.

---

## 📂 Project Structure
```text
LLM_feature_gen/
├─ src/
│  └─ LLM_feature_gen/
│     ├─ __init__.py
│     ├─ discover.py                # High-level orchestration for feature discovery
│     ├─ generate.py                # Feature value generation
│     ├─ providers/
│     │   ├─ openai_provider.py     # OpenAI / Azure OpenAI API wrapper
│     │   └─ local_provider.py      # Local LLM wrapper
│     ├─ prompts/
│     │   ├─ image_discovery_prompt.txt
│     │   ├─ text_discovery_prompt.txt
│     │   ├─ image_generation_prompt.txt
│     │   └─ text_generation_prompt.txt
│     ├─ utils/
│     │   ├─ image.py               # Image → base64 conversion
│     │   ├─ video.py               # Video frame and audio extraction
│     │   └─ text.py                # Text extraction (txt, pdf, docx, etc.)
│     └─ tests/
│        └─ test_discover.py
├─ outputs/                         # Automatically generated feature JSONs
├─ pyproject.toml
└─ README.md
```

---

## ⚙️ Installation

Clone or download the repository, then install in editable mode:

```bash
pip install -e .
```

## 🔑 Environment Setup for OpenAI API

Create a .env file in the project root

##  Example: Discover Features from Images
```python
from LLM_feature_gen.discover import discover_features_from_images
# Folder with your example images
image_folder = "discover_images"

# Run feature discovery
result = discover_features_from_images(
    image_paths_or_folder=image_folder,
    as_set=True,  # analyze all images jointly
)

print(result)
```
This will:
- Read all .jpg/.png images from discover_images/
- the default prompt (prompts/image_discovery_prompt.txt)
- Send them to your LLM provider
- Save the results to outputs/discovered_image_features.json

Example saved JSON:
```json
{
  "proposed_features": [
    {
      "feature": "has visible handle",
      "description": "Some objects include handles, others do not.",
      "possible_values": ["present", "absent"]
    },
    {
      "feature": "color tone",
      "description": "Images vary between metallic and earthy color palettes.",
      "possible_values": ["metallic", "matte", "bright", "dark"]
    }
  ]
}
```

##  Example: Discover Features from Texts
```python
from LLM_feature_gen.discover import discover_features_from_texts

# Folder with text documents (txt, pdf, docx, md, html)
text_folder = "discover_texts"

# Run feature discovery
result = discover_features_from_texts(
    texts_or_file=text_folder,
    as_set=True,  # analyze all texts jointly
)

print(result)
```

This will:
- Load all supported text files from discover_texts/,
- Extract raw text automatically,
- Use the default text discovery prompt,
- Send them to your LLM provider,
- Save the results to outputs/discovered_text_features.json.

Example saved JSON:
```json
{
  "proposed_features": [
    {
      "feature": "presence_of_personal_experience",
      "description": "Some texts describe personal experiences or reflections, while others are more impersonal or instructional.",
      "possible_values": ["present", "absent"]
    },
    {
      "feature": "level_of_subjectivity",
      "description": "Texts vary in how subjective or opinion-based they are compared to neutral or factual descriptions.",
      "possible_values": ["highly subjective", "moderately subjective", "objective"]
    },
    {
      "feature": "use_of_first_person_perspective",
      "description": "Some texts use first-person pronouns indicating a personal perspective, while others do not.",
      "possible_values": ["first person", "third person or impersonal"]
    },
    {
      "feature": "presence_of_explicit_goal_or_intent",
      "description": "Texts may explicitly state an intended goal, motivation, or purpose behind actions or descriptions.",
      "possible_values": ["goal stated", "goal not stated"]
    }
  ]
}
```

##  Example: Discover Features from Tabular Data
```python
from LLM_feature_gen.discover import discover_features_from_tabular

# Folder with tabular files (.csv, .xlsx, .parquet, .json)
tabular_folder = "discover_tabular"

# Run feature discovery
result = discover_features_from_tabular(
    texts_or_file=tabular_folder,
    as_set=True,  # analyze all texts jointly
    text_column="text",   # required: column containing raw text
)

print(result)
```

This will:
1. Load all supported tabular files from the folder discover_tabular/
2. Extract the specified text_column
3. Apply the standard text discovery prompt
4. Save the output to outputs/discovered_tabular_features.json.

Example saved JSON:
```json
{
    "proposed_features": [
      {
        "feature": "overall sentiment",
        "description": "The texts differ in expressing positive or negative feelings about the subject, which can separate favorable from unfavorable opinions.",
        "possible_values": [
          "positive",
          "negative"
        ]
      },
      {
        "feature": "focus on emotional impact",
        "description": "Some texts emphasize emotional responses or feelings evoked, distinguishing those that highlight emotional engagement from those that do not.",
        "possible_values": [
          "emotional emphasis",
          "neutral or critical tone"
        ]
      },
      {
        "feature": "mention of specific artistic elements",
        "description": "Certain texts reference particular components like acting, soundtrack, or visuals, which can differentiate detailed critiques from more general statements.",
        "possible_values": [
          "acting",
          "story/plot",
          "soundtrack",
          "visuals",
          "dialogue",
          "character development",
          "none"
        ]
      }
      }
```