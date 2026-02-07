import os
import io
import json
import base64
from typing import Any, Dict, List, Optional
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, pipeline

class LocalProvider:
    """
    Provider for local LLM inference using Hugging Face Transformers.
    Supports both text-only models and vision-language models (VLM) for image features.
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        """
        Initialize the local model.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
                                For vision tasks, use a vision-capable model (e.g. 'llava-hf/llava-1.5-7b-hf' or 'microsoft/Phi-3-vision-128k-instruct').
                                For text tasks, use a text model (e.g. 'microsoft/Phi-3-mini-4k-instruct').
            device: 'cuda', 'cpu', or 'mps'. If None, auto-detects.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
        """
        self.model_name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # We load models lazily or on init? Let's load on init for now to fail fast,
        # or we could make it lazy to avoid massive VRAM usage if not used.
        # For simplicity in this agent context, let's load components, but handle distinct logic for VLM vs Text in methods.

        # However, a single provider instance might be used for just one type of task or both.
        # If the user passes a text model and tries to process images, it will fail or we need a way to handle it.
        # Given the "LocalProvider" name, we usually expect one model loaded.

        # We'll assume the user instantiates this with the correct model for the task,
        # OR we can try to be smart. For now, we will maintain a generic pipeline text-generation
        # and a separate handling for VLM if the model supports it.

        print(f"Loading local model: {self.model_name} on {self.device}...")

        self.processor = None
        self.model = None
        self.tokenizer = None

        # Heuristic to detect VLM vs Text model might be complex, so we'll trust the loaded classes.
        # We will try to load as a VLM first if it looks like one, otherwise text.
        # But `AutoModelForCausalLM` is for text, `AutoModelForVision2Seq` or similar for VLM.
        # Actually `AutoModelForCausalLM` is often used for both in recent transformers versions (like Phi-3 vision).
        # Let's try standard AutoModelForCausalLM and AutoProcessor.

        try:
            # Common pattern for newer VLMs and Text models
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            except OSError:
                # This is likely a text-only model if no processor is found
                self.processor = None

        except Exception as e:
            # Fallback or specific handling could go here.
            print(f"Error loading model: {e}")
            raise e

    def _generate(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        """
        Internal generation method handling both text and multimodal inputs.
        """
        # Create inputs
        if self.processor and images:
            # Multimodal case
            # Note: Prompt formatting depends heavily on the model (Phi-3, LLaVA, etc.).
            # We will use a generic chat template if available, or simple concatenation.

            messages = [
                {"role": "user", "content": f"<|image_1|>\n{prompt}"}
            ]
            # Note: The <|image_1|> tag is specific to some models (like Phi-3 Vision).
            # LLaVA uses different logic. Transformers `apply_chat_template` is best if supported.

            # Best effort generic formatting for chat structure using tokenizer/processor
            if hasattr(self.processor, "apply_chat_template"):
                # Ideally we construct the prompt that the model expects
                # For this generic implementation, we might stick to raw text prompt if chat template fails
                pass

            # Construct inputs
            inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

        else:
            # Text-only case
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        # We use explicit arguments for generation
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=(self.temperature > 0),
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        # We need to slice off the input tokens to extract only the new generation
        input_len = inputs.input_ids.shape[1]
        response_ids = generate_ids[:, input_len:]
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """
        Attempt to parse JSON from the raw response string.
        """
        # Simple heuristic to find JSON-like structure
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return {"features": response.strip()}
        except Exception:
            return {"features": response.strip()}

    def image_features(
        self,
        image_base64_list: List[str],
        prompt: Optional[str] = None,
        deployment_name: Optional[str] = None, # Ignored for local, unless used to switch loaded model (not implemented)
        feature_gen: bool = False,
        as_set: bool = False,
        extra_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate features from images.
        """
        if not self.processor:
            raise RuntimeError("The loaded model does not appear to support vision (no PretrainedProcessor found).")

        base_prompt = prompt or "Extract meaningful features from this image."
        system_instruction = ""
        if feature_gen:
             system_instruction = "Respond in strict JSON format. "

        final_prompt = f"{system_instruction}{base_prompt}"
        if extra_context:
            final_prompt += f"\nContext: {extra_context}"

        results = []

        # Convert base64 to PIL Images
        pil_images = []
        for b64 in image_base64_list:
            image_data = base64.b64decode(b64)
            pil_images.append(Image.open(io.BytesIO(image_data)).convert("RGB"))

        if as_set:
            # Process all images together
            # Note: Many local models struggle with multi-image inputs unless specifically trained (like Phi-3-vision).
            # We will attempt to feed them all if the processor supports list of images.
            try:
                # We need to constructing a prompt that references multiple images if possible,
                # but standard apply_chat_template handles this best.
                # For simplicity in this "thin adapter":
                # We will concatenate them? Or just pass the list.
                # Most standard `processor(text=..., images=list_of_images)` works.

                 # Adjust prompt for multi-image
                multi_img_prompt = final_prompt
                # Some models need explicit tags per image <|image_1|> <|image_2|> ...
                # We'll just try to pass the semantic prompt and hope the processor handles the tokens.

                response_text = self._generate(multi_img_prompt, pil_images)
                results.append(self._parse_json(response_text))
            except Exception as e:
                return [{"error": f"Failed to process combined images: {str(e)}"}]
        else:
            # One by one
            for img in pil_images:
                try:
                    response_text = self._generate(final_prompt, [img])
                    results.append(self._parse_json(response_text))
                except Exception as e:
                    results.append({"error": str(e)})

        return results

    def text_features(
        self,
        text_list: List[str],
        prompt: Optional[str] = None,
        deployment_name: Optional[str] = None,
        feature_gen: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate features from text.
        """
        base_prompt = prompt or "Extract meaningful features."
        system_instruction = ""
        if feature_gen:
            system_instruction = (
                "You are a feature extraction assistant. "
                "Respond in strict JSON format.\n"
            )

        results = []
        for text in text_list:
            full_prompt = f"{system_instruction}{base_prompt}\n\nInput text:\n{text}"
            try:
                response_text = self._generate(full_prompt)
                results.append(self._parse_json(response_text))
            except Exception as e:
                results.append({"error": str(e)})

        return results

