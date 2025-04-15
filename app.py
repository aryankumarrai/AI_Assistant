import logging
import os
from fastapi import FastAPI, Request, HTTPException, Depends
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
cache_dir = Path("/cache")
cache_dir.mkdir(exist_ok=True, parents=True)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
API_KEY = os.getenv("API_KEY", "default-secret-key")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")

# Model loading
try:
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise RuntimeError("Model initialization failed")

# Cache setup (example using simple dict)
response_cache = {}

async def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.get("/")
async def root():
    return {
        "status": "ready",
        "model": MODEL_NAME,
        "endpoints": {
            "/generate": "POST {message: str, max_length: int, temperature: float}",
            "/health": "GET"
        }
    }

@app.post("/generate")
@limiter.limit("5/minute")
async def generate(
    request: Request,
    message: str,
    max_length: Optional[int] = 100,
    temperature: Optional[float] = 0.7,
    auth: bool = Depends(verify_api_key)
):
    try:
        # Check cache
        cache_key = f"{message}-{max_length}-{temperature}"
        if cache_key in response_cache:
            return {"response": response_cache[cache_key], "source": "cache"}
            
        # Generate new response
        inputs = tokenizer.encode(message, return_tensors="pt")
        
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            do_sample=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Cache response
        response_cache[cache_key] = response
        
        return {
            "response": response,
            "status": "success",
            "model": MODEL_NAME,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature
            }
        }
    
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(429, "Prompt too long. Reduce max_length.")
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(500, "Generation failed")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "cache_size": len(response_cache)
    }