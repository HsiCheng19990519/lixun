FROM python:3.13-slim

ENV PATH="/root/.local/bin:${PATH}"

RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml README.md .env.example ./
COPY src ./src
COPY docs ./docs

RUN uv pip install -e .

CMD ["devmate", "I want to build a site showing nearby hiking trails"]
