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


@pytest.fixture(autouse=True)
def _clean_db(request):
    """Isolate pg tests: truncate every table after each one, so global counts
    start clean and ids are predictable. The container is session-scoped for
    speed; this gives per-test isolation without restarting it. Offline tests
    never touch pg_engine (this is a no-op unless the test is @pytest.mark.pg)."""
    is_pg = request.node.get_closest_marker("pg") is not None
    engine = request.getfixturevalue("pg_engine") if is_pg else None   # resolve before yield
    yield
    if engine is None:
        return
    from sqlalchemy import text

    from newsroom.db.base import Base

    tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
    if tables:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
