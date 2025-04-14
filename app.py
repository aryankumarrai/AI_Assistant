import logging
from fastapi import FastAPI, Request, HTTPException
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create cache directory
cache_dir = Path("/cache")
cache_dir.mkdir(exist_ok=True, parents=True)

# Load model with explicit cache
try:
    logger.info("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise RuntimeError("Model initialization failed")

@app.get("/")
async def root():
    return {"status": "ready", "model": "gpt2"}

@app.post("/generate")
async def generate(request: Request):
    try:
        data = await request.json()
        user_input = data.get("message", "").strip()
        
        if not user_input:
            raise HTTPException(400, "Empty input")

        inputs = tokenizer.encode(user_input, return_tensors="pt")
        
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7
        )
        
        return {
    "response": tokenizer.decode(outputs[0], skip_special_tokens=True),
    "status": "success"
}
    
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(429, "Prompt too long")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, "Generation failed")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}