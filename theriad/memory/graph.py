"""Graphify-style graph projection of the memory file.

Builds a node/link structure suitable for force-directed rendering:
- memory nodes (one per entry, labeled with a truncated text preview)
- tag nodes (one per distinct tag; memories sharing a tag cluster around it)
- source nodes (one per provider the memories came from)

Links: memory→tag and memory→source. Tags shared across providers become the
bridges that make cross-provider structure visible — the point of the view.
"""
from __future__ import annotations

from .store import MemoryEntry

MAX_LABEL_CHARS = 60


def build_memory_graph(entries: list[MemoryEntry]) -> dict:
    nodes: list[dict] = []
    links: list[dict] = []
    seen_tags: set[str] = set()
    seen_sources: set[str] = set()

    for e in entries:
        label_source = e.title or e.conversation_title or e.text
        label = label_source if len(label_source) <= MAX_LABEL_CHARS else label_source[: MAX_LABEL_CHARS - 1] + "…"
        nodes.append({
            "id": e.id,
            "label": label,
            "type": "memory",
            "source": e.source,
            "tags": e.tags,
        })

        source_id = f"source:{e.source}"
        if e.source not in seen_sources:
            seen_sources.add(e.source)
            nodes.append({"id": source_id, "label": e.source, "type": "source"})
        links.append({"source": e.id, "target": source_id})

        for tag in e.tags:
            # Skip tags that duplicate the source name; the source node covers it.
            if tag == e.source or tag == "imported":
                continue
            tag_id = f"tag:{tag}"
            if tag not in seen_tags:
                seen_tags.add(tag)
                nodes.append({"id": tag_id, "label": f"#{tag}", "type": "tag"})
            links.append({"source": e.id, "target": tag_id})

    return {
        "nodes": nodes,
        "links": links,
        "counts": {
            "memories": len(entries),
            "tags": len(seen_tags),
            "sources": len(seen_sources),
        },
    }
