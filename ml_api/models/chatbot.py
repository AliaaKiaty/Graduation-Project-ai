"""
Chatbot Model - Llama 3 8B with Arabic LoRA adapters inference
"""
import time
from typing import Optional, Dict, Any, Tuple

from ..models.loader import ModelManager


class ChatbotEngine:
    """
    Engine for Arabic chatbot inference using Llama 3 8B with LoRA adapters.
    Uses lazy loading to conserve GPU memory until first request.
    """

    def __init__(self):
        self._model_manager = ModelManager()
        self._model = None
        self._tokenizer = None

    def _ensure_model_loaded(self) -> bool:
        """
        Ensure the chatbot model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        if self._model is not None and self._tokenizer is not None:
            return True

        result = self._model_manager.get_chatbot_model()
        if result is None:
            return False

        self._model, self._tokenizer = result
        return True

    def is_model_loaded(self) -> bool:
        """Check if chatbot model is loaded."""
        return self._model_manager.is_model_loaded("chatbot_model")

    def get_model_status(self) -> Dict[str, str]:
        """Get the loading status of chatbot models."""
        status = self._model_manager.get_status()
        return status.get("chatbot", {})

    def generate_response(
        self,
        message: str,
        max_tokens: int = 256,
        temperature: float = 0.4,
        system_prompt: str = "اجب علي الاتي بالعربي فقط."
    ) -> Dict[str, Any]:
        """
        Generate a response to an Arabic message.

        Args:
            message: User's input message
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            system_prompt: System prompt for the model

        Returns:
            Dictionary containing response and metadata

        Raises:
            ValueError: If model is not loaded or message is empty
        """
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        if not self._ensure_model_loaded():
            raise ValueError("Chatbot model is not loaded. Check GPU availability and model paths.")

        start_time = time.time()

        try:
            import torch

            # Format messages using chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            # Apply chat template
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self._model.device)

            # Define termination tokens
            terminators = [
                self._tokenizer.eos_token_id,
                self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Generate response
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self._tokenizer.eos_token_id
                )

            # Decode response (skip input tokens)
            response_tokens = outputs[0][input_ids.shape[-1]:]
            response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)

            generation_time = time.time() - start_time
            tokens_generated = len(response_tokens)

            return {
                "response": response_text.strip(),
                "tokens_generated": tokens_generated,
                "generation_time_ms": int(generation_time * 1000),
                "model": "llama-3-8b-arabic",
                "temperature": temperature,
                "max_tokens": max_tokens
            }

        except Exception as e:
            raise ValueError(f"Error generating response: {str(e)}")

    def unload_model(self) -> bool:
        """
        Unload the chatbot model to free GPU memory.

        Returns:
            True if model was unloaded, False otherwise
        """
        if self._model is None:
            return False

        self._model_manager.unload_model("chatbot_model")
        self._model_manager.unload_model("chatbot_tokenizer")
        self._model = None
        self._tokenizer = None

        return True


# Singleton instance
_chatbot_engine: Optional[ChatbotEngine] = None


def get_chatbot_engine() -> ChatbotEngine:
    """Get the singleton chatbot engine instance."""
    global _chatbot_engine
    if _chatbot_engine is None:
        _chatbot_engine = ChatbotEngine()
    return _chatbot_engine
