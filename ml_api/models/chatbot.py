"""
Chatbot Model - OpenRouter-backed inference (gpt-oss-120b by default)
"""
import re
import time
from typing import Optional, Dict, Any, List

import httpx
from sqlalchemy.orm import Session
from sqlalchemy import func

from .. import config
from ..models.db_models import Product, ProductCategory


ARABIC_CHAR_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")

BUY_INTENT_KEYWORDS_EN = (
    "buy", "purchase", "want", "looking for", "need", "shop", "shopping",
    "recommend", "suggest", "show me", "find me", "get me",
)
BUY_INTENT_KEYWORDS_AR = (
    "اشتري", "أشتري", "شراء", "أبحث", "ابحث", "أريد", "اريد", "أحتاج",
    "احتاج", "اقترح", "رشح", "ابي", "أبي", "عايز", "عاوز", "دور", "دلني",
)

SYSTEM_PROMPT_AR = (
    "أنت مساعد ذكي ومفيد لمنصة تسوق إلكترونية. "
    "مهمتك مساعدة المستخدمين في:\n"
    "- البحث عن المنتجات والإجابة على أسئلتهم\n"
    "- تقديم توصيات المنتجات المناسبة لاحتياجاتهم\n"
    "- الإجابة عن استفسارات التسوق والطلبات والشحن\n"
    "- مقارنة المنتجات وتقديم النصائح الشرائية\n\n"
    "أجب باللغة العربية الفصحى المبسطة وبأسلوب واضح ومختصر ومهني."
)
SYSTEM_PROMPT_EN = (
    "You are a helpful assistant for an e-commerce store. "
    "Help users with:\n"
    "- Finding products and answering their questions\n"
    "- Recommending products that match their needs\n"
    "- Answering shopping, order, and shipping inquiries\n"
    "- Comparing products and giving buying advice\n\n"
    "Always reply in clear, concise, professional English."
)


def detect_language(text: str) -> str:
    """Return 'ar' if the message contains Arabic characters, otherwise 'en'."""
    return "ar" if ARABIC_CHAR_RE.search(text or "") else "en"


def _has_buy_intent(message: str, language: str) -> bool:
    lowered = message.lower()
    keywords = BUY_INTENT_KEYWORDS_AR if language == "ar" else BUY_INTENT_KEYWORDS_EN
    return any(kw in lowered for kw in keywords)


def _match_category(message: str, categories: List[ProductCategory]) -> Optional[ProductCategory]:
    """Find the longest category name that appears as a substring in the message."""
    lowered = message.lower()
    best: Optional[ProductCategory] = None
    best_len = 0
    for cat in categories:
        name = (cat.Name or "").strip().lower()
        if not name:
            continue
        if name in lowered and len(name) > best_len:
            best = cat
            best_len = len(name)
    return best


def find_suggested_product(db: Session, message: str, language: str) -> Optional[Dict[str, Any]]:
    """
    If the user shows buying intent and mentions a known category,
    return a random product from that category.
    """
    if not _has_buy_intent(message, language):
        return None

    categories = db.query(ProductCategory).all()
    category = _match_category(message, categories)
    if category is None:
        return None

    product = (
        db.query(Product)
        .filter(Product.CategoryId == category.Id)
        .order_by(func.random())
        .first()
    )
    if product is None:
        return None

    return {
        "product_id": product.Id,
        "product_name": product.Name,
        "image_url": product.ImageUrl,
        "price": float(product.Price) if product.Price is not None else None,
        "category_id": category.Id,
        "category_name": category.Name,
    }


class ChatbotEngine:
    """
    Chatbot inference engine backed by OpenRouter's OpenAI-compatible API
    (default model: openai/gpt-oss-120b).
    """

    def __init__(self):
        self._api_key: Optional[str] = config.OPENROUTER_API_KEY
        self._base_url: str = config.OPENROUTER_BASE_URL.rstrip("/")
        self._model_id: str = config.OPENROUTER_MODEL
        self._timeout: float = config.OPENROUTER_TIMEOUT_SECONDS

    def _ensure_model_loaded(self) -> bool:
        """OpenRouter is remote — 'loaded' just means the API key is configured."""
        return bool(self._api_key)

    def is_model_loaded(self) -> bool:
        return self._ensure_model_loaded()

    def get_model_status(self) -> Dict[str, str]:
        ready = "configured" if self._ensure_model_loaded() else "missing_api_key"
        return {
            "provider": "openrouter",
            "model": self._model_id,
            "llama_base": ready,
            "lora_adapter": "n/a",
        }

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if config.OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = config.OPENROUTER_HTTP_REFERER
        if config.OPENROUTER_X_TITLE:
            headers["X-Title"] = config.OPENROUTER_X_TITLE
        return headers

    def generate_response(
        self,
        message: str,
        max_tokens: int = 1024,
        temperature: float = 0.4,
        system_prompt: Optional[str] = None,
        language: Optional[str] = None,
        suggested_product: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response via the OpenRouter chat completions API.
        Replies in the user's language (Arabic or English) and, when a product
        suggestion is provided, weaves it naturally into the reply.
        """
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        if not self._ensure_model_loaded():
            raise ValueError(
                "Chatbot is not configured: set OPENROUTER_API_KEY to enable chat."
            )

        if language is None:
            language = detect_language(message)
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT_AR if language == "ar" else SYSTEM_PROMPT_EN

        if suggested_product:
            if language == "ar":
                system_prompt += (
                    f"\nلدينا اقتراح للمستخدم: المنتج \"{suggested_product['product_name']}\" "
                    f"من فئة \"{suggested_product.get('category_name', '')}\". "
                    "اذكره في ردك بشكل طبيعي وشجع المستخدم على تجربته."
                )
            else:
                system_prompt += (
                    f"\nWe have a suggestion for the user: the product \"{suggested_product['product_name']}\" "
                    f"from the \"{suggested_product.get('category_name', '')}\" category. "
                    "Mention it naturally in your reply and encourage the user to check it out."
                )

        start_time = time.time()

        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:500] if e.response is not None else str(e)
            raise ValueError(f"OpenRouter request failed ({e.response.status_code}): {detail}")
        except httpx.HTTPError as e:
            raise ValueError(f"OpenRouter request error: {e}")

        try:
            response_text = data["choices"][0]["message"]["content"] or ""
            if isinstance(response_text, list):
                response_text = " ".join(
                    part.get("text", "") for part in response_text if isinstance(part, dict)
                )
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"Unexpected OpenRouter response: {data}")

        usage = data.get("usage") or {}
        tokens_generated = int(usage.get("completion_tokens") or 0)
        generation_time = time.time() - start_time

        return {
            "response": response_text.strip(),
            "tokens_generated": tokens_generated,
            "generation_time_ms": int(generation_time * 1000),
            "model": data.get("model") or self._model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "language": language,
            "suggested_product": suggested_product,
        }

    def unload_model(self) -> bool:
        """No-op for the remote OpenRouter backend; kept for API compatibility."""
        return False


# Singleton instance
_chatbot_engine: Optional[ChatbotEngine] = None


def get_chatbot_engine() -> ChatbotEngine:
    """Get the singleton chatbot engine instance."""
    global _chatbot_engine
    if _chatbot_engine is None:
        _chatbot_engine = ChatbotEngine()
    return _chatbot_engine
