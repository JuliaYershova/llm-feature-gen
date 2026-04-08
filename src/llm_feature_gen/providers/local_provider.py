# src/llm_feature_gen/providers/local_provider.py
from __future__ import annotations

import os
import json
import time
import re
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

# OpenAI SDK (used as client for Local endpoints)
import openai
from openai import OpenAI, BadRequestError

# Optional: Local Whisper
try:
    from faster_whisper import WhisperModel

    HAS_LOCAL_WHISPER = True
except ImportError:
    HAS_LOCAL_WHISPER = False

load_dotenv()


class LocalProvider:
    """
    Thin adapter around OpenAI-compatible LOCAL endpoints.
        Supports:
        - Ollama
        - vLLM
        - LM Studio
        - Any OpenAI-compatible local server

    - Reads configuration from .env:
        LOCAL_OPENAI_BASE_URL
        LOCAL_OPENAI_API_KEY
        LOCAL_MODEL_TEXT
        LOCAL_MODEL_VISION
        LOCAL_WHISPER_MODEL_SIZE
        LOCAL_WHISPER_DEVICE

    - Two entry points:
        image_features(image_base64_list, prompt=None, deployment_name=None, feature_gen=False, as_set=False)
        text_features(text_list, prompt=None, deployment_name=None, feature_gen=False)
        transcribe_audio(audio_path)

    - Returns a list of dicts (one per input item) in the usual case.
      If `as_set=True`, returns a list with a single dict corresponding to the joint call.

    Provider is configured via environment variables.
    """

    def __init__(
            self,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            default_text_model: Optional[str] = None,
            default_vision_model: Optional[str] = None,
            max_retries: int = 5,
            temperature: float = 0.0,
            max_tokens: int = 2048,
    ) -> None:

        # -------------------------------------------------
        # LOCAL CONFIGURATION
        # -------------------------------------------------
        self.base_url = base_url or os.getenv(
            "LOCAL_OPENAI_BASE_URL",
            "http://localhost:11434/v1",
        )

        # Local servers usually ignore API key but SDK requires one
        self.api_key = api_key or os.getenv(
            "LOCAL_OPENAI_API_KEY",
            "ollama",
        )

        self.text_model = default_text_model or os.getenv(
            "LOCAL_MODEL_TEXT",
            "llama3",
        )

        self.vision_model = default_vision_model or os.getenv(
            "LOCAL_MODEL_VISION",
            "llava",
        )

        # -------------------------------------------------
        # CLIENT INITIALIZATION
        # -------------------------------------------------
        self.client: OpenAI = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            # Increased timeout for local batch processing (video frames)
            timeout=300.0,
        )

        # -------------------------------------------------
        # COMMON CONFIGURATION
        # -------------------------------------------------
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        # -------------------------------------------------
        # LOCAL WHISPER CONFIGURATION
        # -------------------------------------------------
        self.whisper_model_size = os.getenv(
            "LOCAL_WHISPER_MODEL_SIZE",
            "base",
        )
        self._whisper_model: Optional[WhisperModel] = None

    # -----------------------
    # Low-level helper
    # -----------------------
    def _extract_json(self, text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Attempts to extract valid JSON from a conversational response.
        Handles markdown code blocks and extra conversational text.
        Supports both Dict {...} and List [...] structures.
        """
        text = text.strip()

        # 1. Try direct load (best case)
        try:
            return json.loads(text)
        except:
            pass

        # 2. Try finding a markdown code block (capture {..} or [..])
        # Modified regex to support both object and array return types
        code_block = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except:
                pass

        # 3. Try finding outer braces or brackets (fallback)
        start_brace, end_brace = text.find("{"), text.rfind("}")
        start_bracket, end_bracket = text.find("["), text.rfind("]")

        # Determine which structure starts first
        candidates = []
        if start_brace != -1 and end_brace > start_brace:
            candidates.append((start_brace, end_brace))
        if start_bracket != -1 and end_bracket > start_bracket:
            candidates.append((start_bracket, end_bracket))

        if candidates:
            # Pick the candidate that starts earliest in the text
            start, end = min(candidates, key=lambda x: x[0])
            try:
                return json.loads(text[start: end + 1])
            except:
                pass

        return None

    def _chat_json(
            self,
            deployment_name: str,
            system_prompt: str,
            user_content: List[Dict[str, Any]],
            json_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request and tries to parse JSON from the reply.
        Falls back to {"features": "..."} if parsing fails.
        If `json_mode=True`, enforces strict JSON output (supported by Ollama/vLLM).
        Retries on RateLimitError with exponential backoff.
        """

        if json_mode and "JSON" not in system_prompt:
            system_prompt += " Respond in strict JSON format."

        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        backoff = 2
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )

                text = resp.choices[0].message.content or ""

                try:
                    return json.loads(text)
                except Exception:
                    extracted = self._extract_json(text)
                    if extracted:
                        # Wrap lists in a dict if expected interface requires it,
                        # but returning raw structure is usually safer here.
                        if isinstance(extracted, list):
                            return {"features": extracted}
                        return extracted
                    return {"features": text}

            except BadRequestError as e:
                # Fallback if model doesn't support json_mode
                if json_mode and "json_object" in str(e):
                    json_mode = False
                    continue
                return {"error": str(e)}

            except openai.RateLimitError:
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return {"error": "Rate limit exceeded."}

            except Exception as e:
                return {"error": str(e)}

        return {"error": "Unknown failure"}

    # -----------------------
    # Public APIs
    # -----------------------
    def image_features(
            self,
            image_base64_list: List[str],
            prompt: Optional[str] = None,
            deployment_name: Optional[str] = None,
            feature_gen: bool = False,
            as_set: bool = False,
            extra_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        For each base64 image, ask the LLM to extract features.

        - If as_set=False (default): behaves as before — one request per image.
        - If as_set=True: sends ALL images in ONE request.
        """
        deployment = deployment_name or self.vision_model

        # fallback/default prompt
        base_prompt = prompt or "Extract meaningful features from this image for tabular dataset construction."

        # System prompt
        system_prompt = "You are a feature extraction assistant for images."
        if feature_gen:
            system_prompt = (
                "You are a feature extraction assistant for images. "
                "Respond in strict JSON with keys as feature names and values as concise strings."
            )

        use_json_mode = True

        def build_content(txt_prompt, b64_imgs, context_txt=None):
            # Put images first for better compatibility with VLM models
            content = []
            for img_b64 in b64_imgs:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })

            final_text = txt_prompt
            if context_txt:
                final_text += f"\n\nADDITIONAL CONTEXT (AUDIO TRANSCRIPT):\n{context_txt}\n\nAnalyze the visual frames below taking the transcript into account:"

            content.append({"type": "text", "text": final_text})
            return content

        # ----------------------------
        # NEW JOINT MODE
        # ----------------------------
        if as_set or extra_context:
            user_content = build_content(base_prompt, image_base64_list, extra_context)
            out = self._chat_json(deployment, system_prompt, user_content, json_mode=use_json_mode)
            return [out]

        results: List[Dict[str, Any]] = []
        for img_b64 in image_base64_list:
            user_content = build_content(base_prompt, [img_b64], None)
            out = self._chat_json(deployment, system_prompt, user_content, json_mode=use_json_mode)
            results.append(out)

        return results

    def text_features(
            self,
            text_list: List[str],
            prompt: Optional[str] = None,
            deployment_name: Optional[str] = None,
            feature_gen: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        For each text, ask the LLM to extract features.
        If `feature_gen=True`, a JSON-only system prompt is enforced.
        """
        results: List[Dict[str, Any]] = []
        deployment = deployment_name or self.text_model

        # base prompt if none provided
        base_prompt = prompt or "Extract meaningful features from this text for tabular dataset construction."

        system_prompt = base_prompt
        if feature_gen:
            system_prompt = (
                "You are a feature extraction assistant for text documents. "
                "You provide output in a structured JSON format and do NOT provide explanations.\n"
                "{\n"
                '  "<feature1_name>": "<value1>",\n'
                '  "<feature2_name>": "<value2>",\n'
                '  "<feature3_name>": "<value3>",\n'
                '  "<feature4_name>": "<value4>",\n'
                '  "<feature5_name>": "<value5>"\n'
                "}\n"
                "If more than one value applies, pick the most important.\n"
                "GENERATE ALL PRESENTED FEATURES!\n"
            )
            if prompt:
                system_prompt += str(prompt)

        # We generally want JSON mode for feature extraction tasks locally
        use_json_mode = True

        for txt in text_list:
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": txt}]
            out = self._chat_json(deployment, system_prompt, user_content, json_mode=use_json_mode)
            results.append(out)

        return results

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribes audio file using local Faster-Whisper.
        """
        if not HAS_LOCAL_WHISPER:
            return "Error: faster-whisper not installed."

        # Lazy loading
        if self._whisper_model is None:
            device = os.getenv("LOCAL_WHISPER_DEVICE", "auto")
            device = "cuda" if device == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            try:
                self._whisper_model = WhisperModel(
                    self.whisper_model_size,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception as e:
                return f"Whisper initialization failed: {e}"

        try:
            segments, _ = self._whisper_model.transcribe(audio_path, beam_size=5)
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            return f"Transcription failed: {e}"
