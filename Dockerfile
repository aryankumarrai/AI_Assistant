FROM python:3.9-slim

# System setup
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache setup
RUN mkdir -p /cache && chmod 777 /cache

WORKDIR /app

# Dependency installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application setup
COPY . .

# Runtime configuration
ENV PORT=7860
EXPOSE ${PORT}
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]