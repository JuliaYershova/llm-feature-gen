# src/llm_feature_gen/providers/openai_provider.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from ..contracts import ProviderResponseError

# OpenAI SDK (Azure)
import openai
from openai import OpenAI, AzureOpenAI

load_dotenv()


FEATURE_DISCOVERY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposed_features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "description": {"type": "string"},
                    "possible_values": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["feature", "description", "possible_values"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["proposed_features"],
    "additionalProperties": False,
}


class OpenAIProvider:
    """
    Thin adapter around  OpenAI (Azure or personal) for feature discovery/generation.
        Supports:
        - Azure OpenAI
        - Personal / private OpenAI API

    - Reads credentials from .env:
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_API_VERSION
        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_GPT41_DEPLOYMENT_NAME  (default deployment/model name)

    - Two entry points:
        image_features(image_base64_list, prompt=None, deployment_name=None, feature_gen=False, as_set=False)
        text_features(text_list, prompt=None, deployment_name=None, feature_gen=False)

    - Returns a list of dicts (one per input item) in the usual case.
      If `as_set=True`, returns a list with a single dict corresponding to the joint call.

    Provider is auto-detected from environment variables.
    """

    supports_response_schema = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_deployment_name: Optional[str] = None,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_completion_tokens: int = 2048,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        default_audio_model: Optional[str] = None,
    ) -> None:
        if max_tokens is not None and max_tokens != max_completion_tokens:
            raise ValueError("Pass only one of max_completion_tokens or max_tokens.")

        # -------------------------------------------------
        # detect whether we are using Azure or not
        # -------------------------------------------------
        self.is_azure = bool(
            endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # -------------------------------------------------
        # AZURE OPENAI
        # -------------------------------------------------
        if self.is_azure:
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
            self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

            # renamed internally to default_model (deployment == model id)
            self.default_model = (
                    default_deployment_name
                    or os.getenv("AZURE_OPENAI_GPT41_DEPLOYMENT_NAME")
            )

            if not (self.api_key and self.api_version and self.endpoint):
                raise EnvironmentError(
                    "Missing Azure OpenAI .env vars: AZURE_OPENAI_API_KEY, "
                    "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT"
                )

            # AzureOpenAI client (new SDK style)
            self.client: AzureOpenAI = openai.AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )

            self.audio_model = (
                    default_audio_model
                    or os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT")
            )

        # -------------------------------------------------
        # PERSONAL / PRIVATE OPENAI
        # -------------------------------------------------
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.default_model = (
                    default_deployment_name  # reuse same parameter
                    or os.getenv("OPENAI_MODEL")
            )

            if not self.api_key:
                raise EnvironmentError("Missing OPENAI_API_KEY")

            if not self.default_model:
                raise EnvironmentError("Missing OPENAI_MODEL")

            # personal OpenAI client
            self.client: OpenAI = OpenAI(api_key=self.api_key)

            self.audio_model = (
                    default_audio_model
                    or os.getenv("OPENAI_AUDIO_MODEL")
                    or "whisper-1"
            )

        # -------------------------------------------------
        # Common configuration
        # -------------------------------------------------
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_completion_tokens = max_tokens if max_tokens is not None else max_completion_tokens
        self.max_tokens = self.max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self._completion_token_parameter = "max_completion_tokens"

    # -----------------------
    # Low-level helper
    # -----------------------
    def _token_limit_fallback(self, current_parameter: str, exc: Exception) -> Optional[str]:
        bad_request_error = getattr(openai, "BadRequestError", None)
        if bad_request_error is None or not isinstance(exc, bad_request_error):
            return None

        message = str(exc).lower()
        if "max_tokens" not in message or "max_completion_tokens" not in message:
            return None
        if current_parameter == "max_tokens":
            return "max_completion_tokens"
        if current_parameter == "max_completion_tokens":
            return "max_tokens"
        return None

    def _uses_reasoning_effort(self) -> bool:
        effort = getattr(self, "reasoning_effort", None)
        return effort is not None and str(effort).lower() != "none"

    def _validate_feature_discovery_payload(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ProviderResponseError("Feature discovery response must be a JSON object.")

        extra_keys = set(payload) - {"proposed_features"}
        if extra_keys:
            raise ProviderResponseError(
                f"Feature discovery response has unexpected keys: {sorted(extra_keys)}"
            )

        proposed_features = payload.get("proposed_features")
        if not isinstance(proposed_features, list):
            raise ProviderResponseError("Feature discovery response must contain a proposed_features list.")

        allowed_feature_keys = {"feature", "description", "possible_values"}
        for index, feature in enumerate(proposed_features):
            if not isinstance(feature, dict):
                raise ProviderResponseError(f"Feature at index {index} must be an object.")

            extra_feature_keys = set(feature) - allowed_feature_keys
            if extra_feature_keys:
                raise ProviderResponseError(
                    f"Feature at index {index} has unexpected keys: {sorted(extra_feature_keys)}"
                )

            missing_keys = allowed_feature_keys - set(feature)
            if missing_keys:
                raise ProviderResponseError(
                    f"Feature at index {index} is missing keys: {sorted(missing_keys)}"
                )

            if not isinstance(feature["feature"], str):
                raise ProviderResponseError(f"Feature name at index {index} must be a string.")
            if not isinstance(feature["description"], str):
                raise ProviderResponseError(f"Feature description at index {index} must be a string.")
            if not isinstance(feature["possible_values"], list) or not all(
                isinstance(value, str) for value in feature["possible_values"]
            ):
                raise ProviderResponseError(
                    f"Feature possible_values at index {index} must be a list of strings."
                )

    def _create_chat_completion(self, deployment_name: str, kwargs: Dict[str, Any]) -> Any:
        token_parameter = getattr(self, "_completion_token_parameter", "max_completion_tokens")
        token_limit = getattr(self, "max_completion_tokens", getattr(self, "max_tokens", 2048))
        request_kwargs = {**kwargs, token_parameter: token_limit}
        try:
            return self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            fallback = self._token_limit_fallback(token_parameter, exc)
            if not fallback:
                raise
            request_kwargs.pop(token_parameter)
            request_kwargs[fallback] = token_limit
            self._completion_token_parameter = fallback
            return self.client.chat.completions.create(**request_kwargs)

    def _chat_json(
        self,
        deployment_name: str, #  meaning: deployment (Azure) OR model (OpenAI)
        system_prompt: str,
        user_content: List[Dict[str, Any]],
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request and tries to parse JSON from the reply.
        Falls back to {"features": "..."} if parsing fails.
        Retries on RateLimitError with exponential backoff.
        """

        if json_mode and "JSON" not in system_prompt:
            system_prompt += " Respond in strict JSON format."

        kwargs = {}
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "feature_discovery",
                    "schema": response_schema,
                    "strict": True,
                },
            }
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._uses_reasoning_effort():
            kwargs["reasoning_effort"] = self.reasoning_effort

        backoff = 2
        for attempt in range(self.max_retries):
            try:
                request = {
                    "model": deployment_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    **kwargs,
                }
                if not self._uses_reasoning_effort():
                    request["temperature"] = self.temperature

                resp = self._create_chat_completion(deployment_name, request)
                text = resp.choices[0].message.content
                try:
                    parsed = json.loads(text)
                except Exception:
                    # Not strict JSON—wrap it so callers have something consistent
                    if response_schema is not None:
                        raise ProviderResponseError("Invalid JSON response for requested schema.")
                    return {"features": text}
                if response_schema is FEATURE_DISCOVERY_SCHEMA:
                    self._validate_feature_discovery_payload(parsed)
                return parsed
            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise ProviderResponseError("Rate limit exceeded. Please try again later.")
            except Exception as e:
                raise ProviderResponseError(str(e)) from e

        raise ProviderResponseError("Unknown failure: unable to get response.")

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
        system_prompt: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        For each base64 image, ask the LLM to extract features.

        - If as_set=False (default): behaves as before — one request per image,
          returns a list of dicts.
        - If as_set=True: sends ALL images in ONE request (for comparative / discovery
          prompts) and returns a list with a single dict.

        `feature_gen=True` can be used to enforce a strict JSON schema prompt on the system side.
        """
        deployment = deployment_name or self.default_model

        # fallback/default prompt
        base_prompt = prompt or "Extract meaningful features from this image for tabular dataset construction."

        # System prompt
        resolved_system_prompt = system_prompt or "You are a feature extraction assistant for images."
        if feature_gen and system_prompt is None:
            resolved_system_prompt = (
                "You are a feature extraction assistant for images. "
                "Respond in strict JSON with keys as feature names and values as concise strings."
            )

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
            # one message with many images
            user_content = build_content(base_prompt, image_base64_list, extra_context)
            out = self._chat_json(
                deployment,
                resolved_system_prompt,
                user_content,
                json_mode=True,
                response_schema=response_schema,
            )
            return [out]

        results: List[Dict[str, Any]] = []
        for img_b64 in image_base64_list:
            user_content = build_content(base_prompt, [img_b64], None)
            out = self._chat_json(
                deployment,
                resolved_system_prompt,
                user_content,
                json_mode=True,
                response_schema=response_schema,
            )
            results.append(out)

        return results

    def text_features(
        self,
        text_list: List[str],
        prompt: Optional[str] = None,
        deployment_name: Optional[str] = None,
        feature_gen: bool = False,
        system_prompt: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        For each text, ask the LLM to extract features.
        If `feature_gen=True`, a JSON-only system prompt is enforced and your custom prompt
        is appended (preserving your colleagues’ behavior).
        """
        results: List[Dict[str, Any]] = []
        deployment = deployment_name or self.default_model

        # base prompt if none provided
        base_prompt = prompt or "Extract meaningful features from this text for tabular dataset construction."

        resolved_system_prompt = system_prompt or base_prompt
        if feature_gen and system_prompt is None:
            resolved_system_prompt = (
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
                resolved_system_prompt += str(prompt)

        for txt in text_list:
            user_text = f"{base_prompt}\n\nTEXT:\n{txt}" if system_prompt else txt
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
            out = self._chat_json(
                deployment,
                resolved_system_prompt,
                user_content,
                json_mode=True,
                response_schema=response_schema,
            )
            results.append(out)

        return results

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribes audio file using OpenAI Whisper (Cloud).
        """

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        if self.is_azure and not self.audio_model:
            raise EnvironmentError(
                "Missing AZURE_OPENAI_WHISPER_DEPLOYMENT for Azure audio transcription."
            )

        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.audio_model,
                    file=audio_file,
                )

            return transcript.text

        except openai.RateLimitError as e:
            raise e

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e
