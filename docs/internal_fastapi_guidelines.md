## Internal FastAPI Guidelines

1) Project layout  
- Keep `app/main.py` (or `app/__init__.py`) as the entrypoint; routers under `app/api`.  
- Split `schemas` (pydantic models), `services` (business logic), `db` (clients), `config` (settings).  
- Avoid putting DB or HTTP clients in global scope; use dependency injection.

2) Request/response models  
- All endpoints MUST declare `response_model` and use pydantic models for body/query params.  
- Enable `model_config = {"extra": "forbid"}` on public request models to reject unknown fields.  
- Return DTOs, not ORM objects, to avoid leaking internals.

3) Error handling  
- Use `HTTPException` with consistent `detail` shape: `{"message": "...", "code": "ERR_*"}`.  
- Add a global exception handler for unexpected errors to return 500 with trace id.

4) Validation & security  
- Validate user input early; strip or reject HTML in text fields if not needed.  
- Require CORS config for browser clients; default to `allow_credentials=False` unless necessary.  
- For auth, prefer JWT with short-lived access + refresh; pass tokens via `Authorization: Bearer`.

5) Observability & logging  
- Add request ID middleware; log `method`, `path`, `status`, `latency_ms`, `trace_id`.  
- Keep application logs structured (JSON or key=value) and write to a rotating file.

6) Testing  
- Provide `tests/` with at least one integration test using `httpx.AsyncClient(app=...)`.  
- Mock external services (DB/search) to keep tests deterministic.

7) Documentation  
- Tag routes; set `openapi_url` and `docs_url` (can be disabled via config in prod).  
- Provide minimal examples in `response_model` docstrings or `examples` metadata.
