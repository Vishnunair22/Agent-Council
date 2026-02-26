FROM python:3.11-slim

# Install system dependencies including curl to download UV
RUN apt-get update && apt-get install -y curl gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install uv

WORKDIR /app

# Copy the dependency lockfiles first
COPY pyproject.toml uv.lock README.md ./

# Synchronize dependencies (creating a virtual environment)
RUN uv sync

# Copy the rest of the application
COPY . .

# Expose API port
EXPOSE 8000

# Start horizontal scaling UVicorn API worker
CMD ["uv", "run", "python", "scripts/run_api.py"]
