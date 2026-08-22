# JobBuddy API — production image for Fly.io.
#
# Single stage on purpose. The project installs from wheels only (psycopg is
# pulled in as psycopg[binary], so there is no libpq/compiler step), which means a
# builder stage would copy the same .venv across for no reproducibility gain and
# one more thing to keep in sync.
FROM python:3.12-slim-bookworm

# Pinned to the uv the lockfile was produced with; :latest would make the image
# non-reproducible for the sake of saving one edit.
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies before source, so editing a Python file does not invalidate the
# install layer. --frozen fails loudly if uv.lock and pyproject.toml disagree,
# which is what makes the build reproducible rather than merely repeatable.
#
# The project has no [build-system], so uv treats it as a non-package project and
# installs dependencies only. --no-install-project states that rather than relying
# on it, and keeps the build working if a build backend is added later.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Only what the service imports at runtime. Not tests/, evals/, scripts/,
# frontend/, supabase/, docs/ — and never .env (see .dockerignore).
COPY src/ ./src/

# The two deployment diagnostics, run by hand via `fly ssh console`. The rest of
# scripts/ is excluded in .dockerignore.
COPY scripts/ ./scripts/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# The existing entrypoint, not a second startup path. It already reads Fly's PORT
# (falling back to settings.PORT), binds settings.HOST = 0.0.0.0, and applies the
# Windows-only event-loop policy — which is a no-op here and stays untouched.
# Running it as a script puts /app/src on sys.path[0], which is how imports like
# `from core import settings` resolve without installing the project.
CMD ["python", "src/run_service.py"]
