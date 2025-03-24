# Use official Python image
FROM python:3.9-slim@sha256:2cd6b4a1bdd1b6a0edec90c6b0e1869c7a4c1cdd680d2d5d5d8d2e1d2b15d3c

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model
RUN python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; \
    GPT2Tokenizer.from_pretrained('gpt2-large'); \
    GPT2LMHeadModel.from_pretrained('gpt2-large')"

COPY . .

# Cloud Run required settings
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
