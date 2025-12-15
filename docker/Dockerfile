FROM m.daocloud.io/docker.io/library/python:3.13-slim-bookworm

# Install system dependencies and uv (Python package manager)
RUN set -eux; \
    # Switch Debian APT sources to a China mirror for faster downloads (works for both debian.sources and sources.list)
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i \
            -e 's|https\?://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
            -e 's|https\?://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
            /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e 's|https\?://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
            -e 's|https\?://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
            /etc/apt/sources.list; \
    fi; \
    apt-get update -o Acquire::Retries=10 -o Acquire::http::Timeout=30; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    # Install uv from PyPI (via China mirror) to avoid GitHub Releases download in the standalone installer
    python -m pip install --no-cache-dir --retries 10 --timeout 60 -i https://pypi.tuna.tsinghua.edu.cn/simple "uv==0.9.17"; \
    uv --version

ENV PATH="/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    # uv package index mirror
    UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple/" \
    # Networking: give uv more time and retries on slow links
    UV_HTTP_TIMEOUT=60 \
    UV_HTTP_RETRIES=8 \
    # Concurrency: avoid too many parallel downloads (can be counterproductive behind proxies)
    UV_CONCURRENT_DOWNLOADS=8 \
    UV_CONCURRENT_INSTALLS=8 \
    # Cache directory (used with a BuildKit cache mount below)
    UV_CACHE_DIR="/root/.cache/uv" \
    CHROMA_TELEMETRY_ENABLED=0 \
    POSTHOG_DISABLE=1 \
    ANONYMIZED_TELEMETRY=false

WORKDIR /app

# Install Python dependencies first (better cache reuse)
# Install Python dependencies first (better cache reuse)
COPY pyproject.toml uv.lock ./
# Install dependencies using the checked-in lockfile (avoid re-locking in Docker, which can fail on private/missing packages)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --verbose
# If it still looks "stuck", temporarily switch to trace logs (very noisy):
# RUN --mount=type=cache,target=/root/.cache/uv RUST_LOG=uv=debug uv sync --frozen --no-dev --verbose

# Copy application code
COPY devmate ./devmate
COPY mcp_server ./mcp_server
COPY scripts ./scripts
COPY docs ./docs
COPY main.py legacy_main.py README.md ./

# Default command can be overridden by docker-compose
CMD ["python", "-m", "devmate.cli"]
