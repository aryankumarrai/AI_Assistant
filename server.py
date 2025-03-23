from fastapi import FastAPI, Request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

# Load the GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.post('/generate')
async def generate(request: Request):
    data = await request.json()
    user_input = data.get('message', '')

    # Tokenize the input|
    inputs = tokenizer.encode(user_input, return_tensors='pt')

    # Generate a response
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {'reply': response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)