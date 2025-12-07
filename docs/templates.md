## Web Project Template Notes

- Structure: keep `src/` for frontend assets, `api/` for backend samples, `tests/` for smoke cases.
- Entry points: provide `main.py` for backend demo or `index.html` for static demo.
- Config: include a `.env.example` showing required variables; always read model name, API keys, and URLs from config/env.
- Logging: default to `logs/` with rotation.
- Dependencies: use `uv` + `pyproject.toml`; avoid `requirements.txt`.

## Hiking Trails Site (example hints)

- Mention “nearby hiking trails” and prompt the user to allow geolocation.
- Use placeholder data or an open API (e.g., OpenRouteService); document API choices in README.
- Frontend should list trail name, difficulty, distance, and a GPX download link/placeholder.
- Prefer clean HTML/CSS/JS separation over inline styles.
