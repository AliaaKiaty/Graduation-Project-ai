"""
Pydantic schemas for chat API
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class ChatRequest(BaseModel):
    """Request schema for chat message endpoint"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's message in Arabic"
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (higher = more creative)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "ما هي عاصمة مصر؟",
                "max_tokens": 256,
                "temperature": 0.4
            }
        }
    )


class ChatResponse(BaseModel):
    """Response schema for chat message endpoint"""
    input_message: str = Field(..., description="Original user message")
    response: str = Field(..., description="Generated response in Arabic")
    model: str = Field(..., description="Model identifier used for generation")
    tokens_generated: int = Field(..., ge=0, description="Number of tokens generated")
    generation_time_ms: int = Field(..., ge=0, description="Generation time in milliseconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_message": "ما هي عاصمة مصر؟",
                "response": "عاصمة مصر هي القاهرة، وهي أكبر مدينة في مصر والعالم العربي.",
                "model": "llama-3-8b-arabic",
                "tokens_generated": 45,
                "generation_time_ms": 1250
            }
        }
    )


class ChatModelStatus(BaseModel):
    """Schema for chatbot model status"""
    llama_base: str = Field(..., description="Status of base Llama model")
    lora_adapter: str = Field(..., description="Status of LoRA adapter")
    is_ready: bool = Field(..., description="Whether the model is ready for inference")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "llama_base": "loaded",
                "lora_adapter": "loaded",
                "is_ready": True
            }
        }
    )
