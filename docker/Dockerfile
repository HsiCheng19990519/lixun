FROM python:3.13-slim

# Install system dependencies and uv (Python package manager)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    CHROMA_TELEMETRY_ENABLED=0 \
    POSTHOG_DISABLE=1 \
    ANONYMIZED_TELEMETRY=false

WORKDIR /app

# Install Python dependencies first (better cache reuse)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY devmate ./devmate
COPY mcp_server ./mcp_server
COPY scripts ./scripts
COPY docs ./docs
COPY main.py legacy_main.py README.md ./

# Default command can be overridden by docker-compose
CMD ["python", "-m", "devmate.cli"]
