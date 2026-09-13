"""Supervision bot — the human's controls (architecture §10).

The service chat's notices carry buttons; this bot polls Telegram for the button
presses (callback queries) and acts:

  * retract:<id> — pull a published post: delete the channel message and mark the
    publication retracted (works even while publishing is halted);
  * stop / resume — flip the publishing stop button (system_state).

The dispatch (parse callback_data -> action -> DB write) is pure/DB and tested;
the getUpdates polling loop is thin network glue (`# pragma: no cover`). Every
action is journaled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from newsroom.publishers.gate import set_publishing_stopped

log = logging.getLogger("newsroom.publishers.bot")


@dataclass(frozen=True)
class CallbackAction:
    kind: str                 # retract | stop | resume | unknown
    publication_id: int | None = None


def parse_callback(data: str | None) -> CallbackAction:
    """Parse callback_data into an action. Unknown/garbage -> kind='unknown'."""
    if not data:
        return CallbackAction("unknown")
    data = data.strip()
    if data == "stop":
        return CallbackAction("stop")
    if data == "resume":
        return CallbackAction("resume")
    if data.startswith("retract:"):
        _, _, raw = data.partition(":")
        try:
            return CallbackAction("retract", publication_id=int(raw))
        except (TypeError, ValueError):
            return CallbackAction("unknown")
    return CallbackAction("unknown")


class SupervisionBot:
    def __init__(self, session_factory, telegram, *, admin_chat_id: int | None = None,
                 console=None, charter_version: str = "0.2"):
        self.sf = session_factory
        self.telegram = telegram
        # admin_chat_id + console enable text /commands from the admin chat only.
        self.admin_chat_id = admin_chat_id
        self.console = console
        self.charter_version = charter_version

    def handle(self, action: CallbackAction) -> str:
        """Perform one action, return a short result label for the callback answer."""
        if action.kind == "stop":
            self._set_stop(True, "supervisor stop button")
            return "Публікацію зупинено"
        if action.kind == "resume":
            self._set_stop(False, "supervisor resume")
            return "Публікацію відновлено"
        if action.kind == "retract" and action.publication_id is not None:
            return self._retract(action.publication_id)
        return "Невідома дія"

    def _set_stop(self, stopped: bool, reason: str) -> None:
        with self.sf() as s:
            set_publishing_stopped(s, stopped, reason=reason)
            s.commit()

    def _retract(self, publication_id: int) -> str:
        from newsroom.models import Decision, Publication

        with self.sf() as s:
            pub = s.get(Publication, publication_id)
            if pub is None:
                return "Публікацію не знайдено"
            if pub.status != "published":
                return f"Публікація не активна ({pub.status})"
            deleted = False
            if pub.channel_ref:
                try:
                    deleted = self.telegram.delete_message(self.telegram.active_chat_id, int(pub.channel_ref))
                except (TypeError, ValueError):
                    deleted = False
            pub.status = "retracted"
            s.add(Decision(
                entity_type="publication", entity_id=str(publication_id), stage="publish",
                decision="retracted", reason="supervisor recall",
                details={"channel_ref": pub.channel_ref, "channel_deleted": deleted},
                charter_version=self.charter_version,
            ))
            s.commit()
        return "Відкликано" if deleted else "Позначено відкликаним (повідомлення не видалено)"

    def handle_update(self, update: dict) -> bool:
        """Dispatch one Telegram update (callback button or admin text command).
        Returns True if it was handled."""
        callback = update.get("callback_query")
        if callback:
            action = parse_callback((callback.get("data") or ""))
            result = self.handle(action)
            cq_id = callback.get("id")
            if cq_id:
                try:
                    self.telegram.answer_callback_query(cq_id, text=result)
                except Exception:  # noqa: BLE001
                    log.warning("answer_callback_query failed")
            return True
        return self._handle_message(update.get("message") or {})

    def _handle_message(self, message: dict) -> bool:
        """Dispatch a text /command — ONLY from the admin chat (the trust boundary)."""
        if self.console is None or self.admin_chat_id is None:
            return False
        chat_id = ((message.get("chat") or {}).get("id"))
        if chat_id != self.admin_chat_id:
            return False                      # ignore everyone but the admin
        reply = self.console.handle(message.get("text"))
        if reply is None:
            return False                      # not a command
        try:
            self.telegram.send_text(reply, chat_id=self.admin_chat_id, disable_preview=True)
        except Exception:  # noqa: BLE001 — a failed reply must not kill the poll loop
            log.warning("admin reply failed")
        return True

    async def poll_forever(self, *, stop=None, timeout: int = 25) -> None:  # pragma: no cover - network
        import asyncio

        offset: int | None = None
        while not (stop and stop.is_set()):
            try:
                updates = await asyncio.to_thread(
                    self.telegram.get_updates, offset=offset, timeout=timeout,
                    allowed_updates=["callback_query", "message"],
                )
                for update in updates:
                    offset = int(update.get("update_id", 0)) + 1
                    self.handle_update(update)
            except Exception:
                log.exception("supervision poll failed")
                await asyncio.sleep(3.0)
