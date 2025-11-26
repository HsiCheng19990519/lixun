# Internal FastAPI Guidelines

- Prefer async endpoints and use Pydantic models for request validation.
- When building small project scaffolds, include `pyproject.toml`, `Dockerfile`, and `docker-compose.yml`.
- Document environment variables such as `AI_BASE_URL`, `API_KEY`, and `TAVILY_API_KEY`.
- Use LangSmith tracing during demos to capture agent behavior.
