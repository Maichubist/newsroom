from __future__ import annotations

import logging

from newsroom.logsetup import RESERVED_LOGRECORD_KEYS, bind


def test_bind_suffixes_only_reserved_keys():
    out = bind(created=1, name="x", source_id=5, total=9)
    assert out == {"created_": 1, "name_": "x", "source_id": 5, "total": 9}
    assert "created" in RESERVED_LOGRECORD_KEYS and "name" in RESERVED_LOGRECORD_KEYS


def test_bound_extra_does_not_crash_logger():
    log = logging.getLogger("test.bind")
    # Passing reserved keys raw raises KeyError in LogRecord creation; bind() fixes it.
    log.info("collect done", extra=bind(created=3, updated=1, name="Ukrinform", msg="x"))
