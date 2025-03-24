import os
import logging
import time
from fastapi import FastAPI, Request, HTTPException
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model references
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        logger.info("🚦 Starting model initialization...")
        start_time = time.time()
        
        logger.info("⏳ Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        
        logger.info("⌛ Loading model weights...")
        model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        
        logger.info(f"✅ Model loaded in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        raise RuntimeError("Model initialization failed")

@app.post('/generate')
async def generate(request: Request):
    try:
        data = await request.json()
        user_input = data.get('message', '')
        
        if not user_input.strip():
            raise HTTPException(status_code=400, detail="Empty input")
        
        # Tokenization
        inputs = tokenizer.encode(user_input, return_tensors='pt')
        
        # Generation
        start_time = time.time()
        outputs = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True
        )
        
        # Decoding
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated response in {time.time() - start_time:.2f}s")
        return {'reply': response}
        
    except torch.cuda.OutOfMemoryError:
        logger.error("⚠️ CUDA Out of Memory")
        raise HTTPException(429, "Server busy, try simpler input")
    except Exception as e:
        logger.error(f"🚨 Generation error: {str(e)}")
        raise HTTPException(500, "Generation failed")

@app.get('/health')
async def health_check():
    return {"status": "healthy", "model": "gpt2-large"}

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
