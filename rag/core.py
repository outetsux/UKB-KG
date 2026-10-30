"""Shared embedding, path, answer and file helpers."""

import json
import os
import re
from pathlib import Path


class PathRetrievalError(RuntimeError):
    """A failed path query; an empty successful query is not an error."""


def select_entities(mentions, selected):
    """Match without case sensitivity and retain the original mention text."""
    selected = {entity.lower() for entity in selected}
    return [mention for mention in mentions if mention.lower() in selected]


def masked_mean_pool(hidden_states, attention_mask):
    """Use the same pooling as the graph-node encoder."""
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def path_steps(path):
    """Serialize each relationship in traversal order, preserving its direction."""
    nodes = list(path.nodes)
    relationships = list(path.relationships)
    if len(nodes) != len(relationships) + 1:
        raise ValueError("Invalid path node/relationship counts")
    steps = []
    for left, rel, right in zip(nodes, relationships, nodes[1:]):
        if rel.start_node == left and rel.end_node == right:
            forward = True
        elif rel.start_node == right and rel.end_node == left:
            forward = False
        else:
            raise ValueError("Relationship endpoints do not match path nodes")
        steps.append({
            "start": left["name"], "relation": rel["name"],
            "end": right["name"], "forward": forward,
        })
    return steps


def format_path(steps):
    if not steps:
        return ""
    text = steps[0]["start"]
    previous = steps[0]["start"]
    for step in steps:
        if step["start"] != previous:
            raise ValueError("Disconnected path steps")
        if step["forward"]:
            text += f" -({step['relation']})-> {step['end']}"
        else:
            text += f" <-({step['relation']})- {step['end']}"
        previous = step["end"]
    return "[" + text + "]"


def unique_triples(triples):
    """Keep the first occurrence of each directed (head, relation, tail)."""
    seen = set()
    result = []
    for triple in triples:
        key = tuple(triple[:3])
        if key not in seen:
            result.append(triple)
            seen.add(key)
    return result


def valid_answer(answer, dataset):
    labels = {"yes", "no", "maybe"} if dataset == "pubmedqa" else {"yes", "no"}
    return isinstance(answer, str) and answer.strip().lower() in labels


def parse_answer(response, dataset):
    """Accept one complete answer tag; reject missing/ambiguous/invalid labels."""
    if not isinstance(response, str):
        return None
    matches = re.findall(r"<ans>(.*?)</ans>", response, flags=re.DOTALL | re.IGNORECASE)
    if len(matches) != 1:
        return None
    answer = matches[0].strip().lower()
    return answer if valid_answer(answer, dataset) else None


def read_json(path):
    """Load saved progress, or start with an empty dictionary."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    """Replace a JSON file atomically, preserving the last complete checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=4, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
