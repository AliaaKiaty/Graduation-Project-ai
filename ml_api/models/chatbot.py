"""
Chatbot Model - Arabic assistant powered by Z.AI (GLM-5)
Falls back to 503 if ZAI_API_KEY is not configured.
"""
import time
from typing import Optional, Dict, Any

import httpx

from .. import config


# System prompt for the Arabic e-commerce assistant
_SYSTEM_PROMPT = (
    "أنت مساعد ذكي ومفيد لمنصة تسوق إلكترونية. "
    "مهمتك مساعدة المستخدمين في:\n"
    "- البحث عن المنتجات والإجابة على أسئلتهم\n"
    "- تقديم توصيات المنتجات المناسبة لاحتياجاتهم\n"
    "- الإجابة عن استفسارات التسوق والطلبات والشحن\n"
    "- مقارنة المنتجات وتقديم النصائح الشرائية\n\n"
    "تجيب دائماً باللغة العربية بأسلوب واضح ومختصر ومهني. "
    "إذا كان السؤال بالإنجليزية، أجب باللغة العربية."
)


class ChatbotEngine:
    """
    Arabic chatbot engine backed by Z.AI (GLM-5).
    No GPU or local model files required — calls the Z.AI REST API.
    """

    def is_model_loaded(self) -> bool:
        """Z.AI is always 'ready' as long as the API key is configured."""
        return bool(config.ZAI_API_KEY)

    def get_model_status(self) -> Dict[str, str]:
        status = "loaded" if config.ZAI_API_KEY else "not_configured"
        return {"llama_base": status, "lora_adapter": status}

    def generate_response(
        self,
        message: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """
        Generate a response via the Z.AI chat completions API.

        Args:
            message: User's message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt (defaults to Arabic e-commerce assistant)

        Returns:
            Dict with response text and metadata
        """
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        if not config.ZAI_API_KEY:
            raise ValueError(
                "Chatbot model is not loaded. ZAI_API_KEY environment variable is not set."
            )

        start_time = time.time()

        payload = {
            "model": config.ZAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            # Disable chain-of-thought thinking output for glm-4.7-flash
            "thinking": {"type": "disabled"},
        }

        headers = {
            "Authorization": f"Bearer {config.ZAI_API_KEY}",
            "Content-Type": "application/json",
            "Accept-Language": "ar,en",
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{config.ZAI_API_BASE}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            generation_time = time.time() - start_time
            message = data["choices"][0]["message"]

            # GLM thinking models: final answer is in `content`, chain-of-thought in `reasoning_content`
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""

            # Handle list-format content (multimodal)
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )

            # If content is all reasoning/thinking (GLM-4.7-flash style), extract final answer.
            # The final answer follows the thinking block, often after a blank line or "---".
            response_text = content.strip()
            if reasoning:
                # reasoning_content has the thinking; content has the final answer
                response_text = content.strip() or reasoning.strip()

            tokens_generated = data.get("usage", {}).get("completion_tokens", len(response_text.split()))

            return {
                "response": response_text.strip(),
                "tokens_generated": tokens_generated,
                "generation_time_ms": int(generation_time * 1000),
                "model": config.ZAI_MODEL,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        except httpx.HTTPStatusError as e:
            raise ValueError(f"Z.AI API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise ValueError(f"Error generating response: {str(e)}")

    def unload_model(self) -> bool:
        """No local model to unload — always returns False."""
        return False


# Singleton instance
_chatbot_engine: Optional[ChatbotEngine] = None


def get_chatbot_engine() -> ChatbotEngine:
    """Get the singleton chatbot engine instance."""
    global _chatbot_engine
    if _chatbot_engine is None:
        _chatbot_engine = ChatbotEngine()
    return _chatbot_engine
