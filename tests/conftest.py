from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="session")
def pg_engine():
    """Ephemeral PostgreSQL (pgvector) via testcontainers. Skips cleanly when
    Docker/testcontainers is unavailable so the offline suite still runs."""
    try:
        from testcontainers.postgres import PostgresContainer
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"testcontainers not installed: {exc}")

    from newsroom.db.base import init_db, make_engine

    try:
        container = PostgresContainer("pgvector/pgvector:pg16", driver="psycopg")
        container.start()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot start postgres container (is Docker running?): {exc}")

    engine = make_engine(container.get_connection_url())
    try:
        init_db(engine)
        yield engine
    finally:
        engine.dispose()
        container.stop()
