"""Compact serialization for SLM input."""

from __future__ import annotations

from html import escape

from kivi_memory.common.schemas import GroundingContext, MemoryEpisode


def format_memory_episode(episode: MemoryEpisode) -> str:
    """Format a validated episode into compact, unambiguous text."""

    attrs = [f'id="{escape(episode.episode_id, quote=True)}"']
    if episode.timezone is not None:
        attrs.append(f'timezone="{escape(episode.timezone, quote=True)}"')
    if episode.locale is not None:
        attrs.append(f'locale="{escape(episode.locale, quote=True)}"')

    parts: list[str] = [f"<episode {' '.join(attrs)}>"]
    for message in episode.messages:
        message_attrs = [message.message_id, message.role.value.lower(), str(message.timestamp)]
        parts.extend(
            [
                "",
                f"[{']['.join(message_attrs)}]",
                escape(message.text, quote=False),
            ]
        )

    parts.extend(["", "</episode>"])

    if episode.tool_context:
        parts.extend(["", "<tool_context>"])
        for item in episode.tool_context:
            source = f"[source={item.source_message_id}]" if item.source_message_id else "[source=null]"
            parts.extend([f"[{item.tool_name}]{source}", escape(item.content, quote=False)])
        parts.append("</tool_context>")

    grounding_lines = _format_grounding_lines(episode.grounding_context)
    if grounding_lines:
        parts.extend(["", "<grounding>", *grounding_lines, "</grounding>"])

    return "\n".join(parts)

def _format_grounding_lines(grounding: GroundingContext | None) -> list[str]:
    if grounding is None:
        return []

    data = grounding.model_dump(exclude_none=True)
    return [f"{key}={escape(str(value), quote=False)}" for key, value in data.items()]
