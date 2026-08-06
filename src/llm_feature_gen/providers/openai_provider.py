# src/llm_feature_gen/providers/openai_provider.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from ..contracts import ProviderResponseError, explain_empty_reply

# OpenAI SDK (Azure)
import openai
from openai import OpenAI, AzureOpenAI

load_dotenv()


class OpenAIProvider:
    """Adapter around the OpenAI or Azure OpenAI API for discovery and generation.

    The provider exposes three methods used by every discovery and generation
    helper: ``image_features``, ``text_features``, and ``transcribe_audio``.
    Each returns a list of dictionaries, one per input item — or a
    single-element list when a joint (``as_set=True``) call is made.

    Azure mode is selected automatically when an Azure endpoint is configured
    (via the ``endpoint`` argument or the ``AZURE_OPENAI_ENDPOINT`` environment
    variable); otherwise the standard OpenAI API is used. Any argument left as
    ``None`` falls back to the corresponding environment variable, typically
    loaded from a ``.env`` file in the working directory.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_deployment_name: Optional[str] = None,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        default_audio_model: Optional[str] = None,
    ) -> None:
        """Configure the client from arguments or environment variables.

        Args:
            api_key: API key. Falls back to ``AZURE_OPENAI_API_KEY`` (Azure
                mode) or ``OPENAI_API_KEY``.
            api_version: Azure API version. Falls back to
                ``AZURE_OPENAI_API_VERSION``. Ignored outside Azure mode.
            endpoint: Azure resource endpoint. Setting it (or
                ``AZURE_OPENAI_ENDPOINT``) switches the provider to Azure mode.
            default_deployment_name: Chat deployment (Azure) or model name
                (OpenAI). Falls back to ``AZURE_OPENAI_GPT41_DEPLOYMENT_NAME``
                or ``OPENAI_MODEL``.
            max_retries: Retries on rate-limit errors, with exponential
                backoff.
            temperature: Sampling temperature for all chat calls.
            max_tokens: Completion token limit for all chat calls.
            default_audio_model: Transcription deployment/model. Falls back to
                ``AZURE_OPENAI_WHISPER_DEPLOYMENT`` (Azure, required) or
                ``OPENAI_AUDIO_MODEL`` (default ``whisper-1``).

        Raises:
            EnvironmentError: If a required credential is neither passed nor
                present in the environment — in Azure mode the key, version,
                endpoint, and audio deployment; otherwise ``OPENAI_API_KEY``
                and ``OPENAI_MODEL``.

        Example:
            ```python
            provider = OpenAIProvider(temperature=0.0, max_tokens=4096)
            ```
        """

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

            if not self.audio_model:
                raise EnvironmentError(
                    "Missing AZURE_OPENAI_WHISPER_DEPLOYMENT for Azure audio transcription."
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
        self.max_tokens = max_tokens

    # -----------------------
    # Low-level helper
    # -----------------------
    def _chat_json(
        self,
        deployment_name: str, #  meaning: deployment (Azure) OR model (OpenAI)
        system_prompt: str,
        user_content: List[Dict[str, Any]],
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request and tries to parse JSON from the reply.
        Falls back to {"features": "..."} if parsing fails.
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
                message = resp.choices[0].message
                text = message.content or ""

                # An empty reply has nothing to parse; say what happened
                # instead of wrapping the emptiness and passing it on.
                if not text.strip():
                    raise ProviderResponseError(
                        explain_empty_reply(resp, message, deployment_name, self.max_tokens)
                    )
                try:
                    return json.loads(text)
                except Exception:
                    # Not strict JSON—wrap it so callers have something consistent
                    return {"features": text}
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
    ) -> List[Dict[str, Any]]:
        """Extract features from base64-encoded images.

        Args:
            image_base64_list: Base64-encoded JPEG payloads.
            prompt: Prompt sent with the images. Defaults to a generic
                feature-extraction instruction.
            deployment_name: Override the default chat deployment/model.
            feature_gen: Enforce the strict JSON feature-value system prompt
                used during generation.
            as_set: Send all images in one request (for comparative discovery)
                instead of one request per image.
            extra_context: Optional text appended to the prompt, for example
                an audio transcript for video frames.

        Returns:
            One dictionary per image, or a single-element list for a joint
            (``as_set=True``) call.

        Raises:
            ProviderResponseError: If the provider returns an empty reply,
                stays rate-limited after all retries, or fails the request.
        """
        deployment = deployment_name or self.default_model

        # fallback/default prompt
        base_prompt = prompt or "Extract meaningful features from this image for tabular dataset construction."

        # System prompt
        system_prompt = "You are a feature extraction assistant for images."
        if feature_gen:
            system_prompt = (
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
            out = self._chat_json(deployment, system_prompt, user_content, json_mode=True)
            return [out]

        results: List[Dict[str, Any]] = []
        for img_b64 in image_base64_list:
            user_content = build_content(base_prompt, [img_b64], None)
            out = self._chat_json(deployment, system_prompt, user_content, json_mode=True)
            results.append(out)

        return results

    def text_features(
        self,
        text_list: List[str],
        prompt: Optional[str] = None,
        deployment_name: Optional[str] = None,
        feature_gen: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract features from raw texts, one request per text.

        Args:
            text_list: Raw input texts.
            prompt: Prompt used as the system instruction. Defaults to a
                generic feature-extraction instruction.
            deployment_name: Override the default chat deployment/model.
            feature_gen: Enforce the strict JSON feature-value system prompt
                used during generation; a custom ``prompt`` is appended to it.

        Returns:
            One dictionary per input text.

        Raises:
            ProviderResponseError: If the provider returns an empty reply,
                stays rate-limited after all retries, or fails the request.
        """
        results: List[Dict[str, Any]] = []
        deployment = deployment_name or self.default_model

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

        for txt in text_list:
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": txt}]
            out = self._chat_json(deployment, system_prompt, user_content, json_mode=True)
            results.append(out)

        return results

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe an audio file with the configured Whisper model.

        Args:
            audio_path: Path to the audio file.

        Returns:
            The transcribed text.

        Raises:
            FileNotFoundError: If ``audio_path`` does not exist.
            RuntimeError: If the transcription request fails.
        """

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

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
