from __future__ import annotations

from newsroom.collectors.telegram_import import DiscoveredChannel, render_source_lines
from newsroom.config.sources import load_sources


def test_renders_new_public_channels_only():
    discovered = [
        DiscoveredChannel(username="uanews", title="UA News", channel_id=1),
        DiscoveredChannel(username="known", title="Known", channel_id=2),
        DiscoveredChannel(username=None, title="Private", channel_id=3),  # no username -> skipped
    ]
    lines = render_source_lines(discovered, existing_handles=["@known"], origin="ua", tier="media")
    assert len(lines) == 1
    assert '"@uanews"' in lines[0]
    assert "kind: telegram" in lines[0] and "origin: ua" in lines[0] and "tier: media" in lines[0]


def test_dedup_is_case_insensitive_within_batch():
    discovered = [
        DiscoveredChannel(username="Dup", title="A", channel_id=1),
        DiscoveredChannel(username="dup", title="B", channel_id=2),
    ]
    assert len(render_source_lines(discovered, existing_handles=[])) == 1


def test_rendered_lines_are_loader_valid(tmp_path):
    """The strongest check: what we render must parse back through the real
    loader, including a title with quotes, a colon and braces."""
    discovered = [DiscoveredChannel(username="uanews", title='Новини "АТО": все {тут}', channel_id=1)]
    lines = render_source_lines(discovered, existing_handles=[])

    p = tmp_path / "sources.yaml"
    p.write_text("sources:\n" + "\n".join(lines) + "\n", encoding="utf-8")

    (s,) = load_sources(p)
    assert s.kind == "telegram"
    assert s.handle_or_url == "@uanews"
    assert s.origin == "ua" and s.tier == "media"
    assert s.name == 'Новини "АТО": все {тут}'
