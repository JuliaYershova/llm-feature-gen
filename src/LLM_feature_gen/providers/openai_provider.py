# src/LLM_feature_gen/providers/openai_provider.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# OpenAI SDK (Azure)
import openai
from openai import OpenAI, AzureOpenAI

load_dotenv()


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

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_deployment_name: Optional[str] = None,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:

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
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request and tries to parse JSON from the reply.
        Falls back to {"features": "..."} if parsing fails.
        Retries on RateLimitError with exponential backoff.
        """
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
                )
                text = resp.choices[0].message.content
                try:
                    return json.loads(text)
                except Exception:
                    # Not strict JSON—wrap it so callers have something consistent
                    return {"features": text}
            except openai.RateLimitError:
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return {"error": "Rate limit exceeded. Please try again later."}
            except Exception as e:
                return {"error": str(e)}

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
        system_prompt = "You are a feature extraction assistant for images."
        if feature_gen:
            system_prompt = (
                "You are a feature extraction assistant for images. "
                "Respond in strict JSON with keys as feature names and values as concise strings."
            )

        def build_content(txt_prompt, b64_imgs, context_txt=None):
            content = [{"type": "text", "text": txt_prompt}]

            if context_txt:
                content.append({
                    "type": "text",
                    "text": f"\n\nADDITIONAL CONTEXT (AUDIO TRANSCRIPT):\n{context_txt}\n\nAnalyze the visual frames below taking the transcript into account:"
                })

            for img_b64 in b64_imgs:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            return content

        # ----------------------------
        # NEW JOINT MODE
        # ----------------------------
        if as_set or extra_context:
            # one message with many images
            user_content = build_content(base_prompt, image_base64_list, extra_context)
            out = self._chat_json(deployment, system_prompt, user_content)
            return [out]

        results: List[Dict[str, Any]] = []
        for img_b64 in image_base64_list:
            user_content = build_content(base_prompt, [img_b64], None)
            out = self._chat_json(deployment, system_prompt, user_content)
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
        If `feature_gen=True`, a JSON-only system prompt is enforced and your custom prompt
        is appended (preserving your colleagues’ behavior).
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
            out = self._chat_json(deployment, system_prompt, user_content)
            results.append(out)

        return results