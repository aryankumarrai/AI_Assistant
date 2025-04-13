FROM python:3.9-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download models during build
RUN python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]