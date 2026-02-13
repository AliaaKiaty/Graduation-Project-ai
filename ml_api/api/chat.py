"""
Chat API router
Provides endpoints for Arabic chatbot inference using Llama 3 with LoRA
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Annotated

from ..auth import get_current_user, TokenUser
from ..models.chatbot import get_chatbot_engine, ChatbotEngine
from ..schemas.chat import ChatRequest, ChatResponse, ChatModelStatus
from ..limiter import limiter
from .. import config

router = APIRouter()


def get_engine() -> ChatbotEngine:
    """Get the chatbot engine instance."""
    return get_chatbot_engine()


@router.post("/message", response_model=ChatResponse)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def chat_message(
    request: Request,
    chat_request: ChatRequest,
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    engine: ChatbotEngine = Depends(get_engine),
) -> ChatResponse:
    """
    Generate a response to an Arabic message using Llama 3 with LoRA adapters.

    Requires authentication.

    The model is loaded on first request (lazy loading) to conserve GPU memory.
    If the model is not available, a 503 Service Unavailable error is returned.

    Args:
        chat_request: Chat request containing message and generation parameters

    Returns:
        Generated response with metadata
    """
    try:
        result = engine.generate_response(
            message=chat_request.message,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature
        )

        return ChatResponse(
            input_message=chat_request.message,
            response=result["response"],
            model=result["model"],
            tokens_generated=result["tokens_generated"],
            generation_time_ms=result["generation_time_ms"]
        )

    except ValueError as e:
        error_msg = str(e)
        if "not loaded" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Chatbot service not available: {error_msg}"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )


@router.get("/status", response_model=ChatModelStatus)
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def chat_status(
    request: Request,
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    engine: ChatbotEngine = Depends(get_engine),
) -> ChatModelStatus:
    """
    Get the current status of the chatbot model.

    Requires authentication.

    Returns:
        Model loading status information
    """
    model_status = engine.get_model_status()

    return ChatModelStatus(
        llama_base=model_status.get("llama_base", "unknown"),
        lora_adapter=model_status.get("lora_adapter", "unknown"),
        is_ready=engine.is_model_loaded()
    )


@router.post("/load")
@limiter.limit(config.RATE_LIMIT_LOGIN)
async def load_model(
    request: Request,
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    engine: ChatbotEngine = Depends(get_engine),
):
    """
    Explicitly load the chatbot model.

    Requires authentication.

    This endpoint can be used to pre-load the model before sending chat requests.
    Returns the model status after loading attempt.
    """
    try:
        # Trigger model loading
        if engine._ensure_model_loaded():
            return {
                "message": "Model loaded successfully",
                "status": engine.get_model_status()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to load chatbot model. Check GPU availability and model paths."
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )


@router.post("/unload")
@limiter.limit(config.RATE_LIMIT_LOGIN)
async def unload_model(
    request: Request,
    current_user: Annotated[TokenUser, Depends(get_current_user)],
    engine: ChatbotEngine = Depends(get_engine),
):
    """
    Unload the chatbot model to free GPU memory.

    Requires authentication.

    This endpoint can be used to free GPU memory when the chatbot is not needed.
    """
    if engine.unload_model():
        return {
            "message": "Model unloaded successfully",
            "status": engine.get_model_status()
        }
    else:
        return {
            "message": "Model was not loaded",
            "status": engine.get_model_status()
        }
