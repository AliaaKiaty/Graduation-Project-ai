"""
Main FastAPI application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from datetime import datetime
from . import config
from .database import init_db
from .limiter import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces deprecated @app.on_event decorators.
    """
    # Startup
    init_db()
    print("Database initialized")

    # Initialize admin user
    try:
        from .database import SessionLocal
        from .auth.models import User
        from .auth.security import get_password_hash

        db = SessionLocal()
        try:
            # Check if admin user exists, create if not
            existing_admin = db.query(User).filter(User.username == config.ADMIN_USERNAME).first()
            if not existing_admin:
                admin_user = User(
                    username=config.ADMIN_USERNAME,
                    email=config.ADMIN_EMAIL,
                    hashed_password=get_password_hash(config.ADMIN_PASSWORD),
                    is_admin=True,
                    is_active=True
                )
                db.add(admin_user)
                db.commit()
                print(f"Created admin user: {config.ADMIN_USERNAME}")
            else:
                print(f"Admin user already exists: {config.ADMIN_USERNAME}")
        finally:
            db.close()
    except Exception as e:
        print(f"Warning: Failed to initialize admin user: {e}")

    # Load models (eager loading for recommendation and image models)
    try:
        from .models.loader import ModelManager
        manager = ModelManager()

        print("Loading recommendation models...")
        manager.load_recommendation_models()
        print("Recommendation models loaded")

        print("Loading image models...")
        manager.load_image_models()
        print("Image models loaded")

        print("Note: Chatbot model will be loaded on first request (lazy loading)")
    except Exception as e:
        print(f"Warning: Failed to load models: {e}")
        print("API will start but models may not be available")

    yield  # Application runs here

    # Shutdown (cleanup if needed)
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request size to prevent abuse"""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"Request too large. Max size: {config.MAX_UPLOAD_SIZE_MB}MB"
            }
        )
    return await call_next(request)


# Health check endpoint
@app.get("/health")
@limiter.limit(config.RATE_LIMIT_GENERAL)
async def health_check(request: Request):
    """
    Health check endpoint to verify API status and model availability
    """
    gpu_info = {
        "available": False,
        "memory_used_gb": 0,
        "memory_total_gb": 0
    }

    # Try to get GPU info (torch may not be installed)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["memory_used_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            gpu_info["memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        pass  # torch not installed, GPU info unavailable

    # Try to get model status
    try:
        from .models.loader import ModelManager
        manager = ModelManager()
        model_status = manager.get_status()
    except Exception as e:
        model_status = {"error": str(e)}

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models": model_status,
        "gpu": gpu_info
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ML Models API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
from .auth.router import router as auth_router
from .api.recommendation import router as recommendation_router
from .api.image import router as image_router
from .api.chat import router as chat_router

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(recommendation_router, prefix="/recommend", tags=["Recommendation"])
app.include_router(image_router, prefix="/image", tags=["Image"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ml_api.main:app", host="0.0.0.0", port=8000, reload=True)
