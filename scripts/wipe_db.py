"""Повне очищення бази для чистого старту нового прогону (dev-утиліта, не міграція).

Скидає схему `public` цілком і відбудовує її заново через init_db (create_all +
розширення pgvector + додаткові колонки) — стан ідентичний свіжому розгортанню.
Дані НЕ зберігаються: це навмисно, для збору нових даних з нуля.

Запобіжники:
  * працює ЛИШЕ проти localhost/127.0.0.1 (щоб не знести хмарну БД), інакше треба --force;
  * без --yes — сухий прогін: лише показує ціль і що буде зроблено, нічого не чіпає.

Запуск:
    python -m scripts.wipe_db              # сухий прогін — покаже ціль
    python -m scripts.wipe_db --yes        # реально очистити localhost-базу
    python -m scripts.wipe_db --yes --force  # дозволити не-localhost (обережно!)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _load_db_url() -> str:
    url = os.environ.get("NEWSROOM_DATABASE_URL")
    if url:
        return url
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("NEWSROOM_DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("NEWSROOM_DATABASE_URL не задано (ні в оточенні, ні в .env)")


def _target(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    host = parts.hostname or "?"
    db = (parts.path or "").lstrip("/") or "?"
    return host, db


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="реально виконати (інакше сухий прогін)")
    ap.add_argument("--force", action="store_true", help="дозволити не-localhost ціль")
    args = ap.parse_args()

    url = _load_db_url()
    host, db = _target(url)
    print(f"Ціль: host={host}  db={db}")

    if host not in ("localhost", "127.0.0.1", "::1") and not args.force:
        raise SystemExit(f"ВІДМОВА: ціль не localhost ({host}). Додай --force, якщо це справді навмисно.")

    if not args.yes:
        print("Сухий прогін. Буде виконано (з --yes):")
        print("  DROP SCHEMA public CASCADE;  CREATE SCHEMA public;  + init_db (таблиці, pgvector, колонки)")
        print("Дані буде ПОВНІСТЮ видалено. Додай --yes, щоб очистити.")
        return 0

    from sqlalchemy import text

    from newsroom.db.base import init_db, make_engine

    engine = make_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO CURRENT_USER"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
    print("Схему public скинуто й відтворено.")

    init_db(engine)
    print("init_db виконано (таблиці + pgvector + додаткові колонки).")

    # підтвердження чистоти
    with engine.connect() as conn:
        has_vec = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')")).scalar()
        counts = {}
        for tbl in ("sources", "items", "events", "stories", "decisions", "llm_calls", "publications"):
            try:
                counts[tbl] = conn.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
            except Exception:  # noqa: BLE001 — таблиці може не бути в цій версії схеми
                counts[tbl] = "—"
    print(f"pgvector: {has_vec}")
    print("Кількість рядків: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("Готово — база чиста, можна запускати бота.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
