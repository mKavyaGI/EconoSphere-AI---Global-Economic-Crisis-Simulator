FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y --no-install-recommends curl gcc && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependencies definitions
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv sync --no-dev

# Copy application code
COPY . /app

# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
