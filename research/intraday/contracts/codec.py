"""Canonical JSON codec for events.

Driven by ``dataclasses.fields`` and resolved type hints, so new event types
need no codec changes. Decoding follows the annotation, never the value.
Values are passed through as parsed; the event's own validation is the
authority (no silent int/float coercion). Classes are looked up only via
``EVENT_TYPES``.
"""
from __future__ import annotations

import json
import types
import typing
from dataclasses import fields
from datetime import datetime
from typing import Any, Mapping, Union, get_args, get_origin

from research.intraday.contracts.events import EVENT_TYPES, SCHEMA_VERSION, Event


def _is_event_type(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, Event)


def _encode(value: Any) -> Any:
    if isinstance(value, Event):
        return to_record(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def to_record(event: Event) -> dict[str, object]:
    """Event -> JSON-ready dict: {"type", "v", **fields}."""
    record: dict[str, object] = {"type": type(event).__name__, "v": SCHEMA_VERSION}
    for f in fields(event):
        record[f.name] = _encode(getattr(event, f.name))
    return record


def _decode(tp: Any, value: Any) -> Any:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        if value is None:
            return None
        non_none = [a for a in get_args(tp) if a is not type(None)]
        return _decode(non_none[0], value) if len(non_none) == 1 else value
    if tp is datetime:
        return datetime.fromisoformat(value)
    if origin is tuple:
        elem = get_args(tp)[0]
        return tuple(_decode(elem, v) for v in value)
    if _is_event_type(tp):
        return from_record(value)
    return value


def from_record(record: Mapping[str, object]) -> Event:
    """Dict -> Event. ValueError on a non-mapping, unknown type, schema mismatch, or bad keys."""
    if not isinstance(record, Mapping):
        raise ValueError(f"event record must be a Mapping, got {type(record).__name__}")
    name = record.get("type")
    cls = EVENT_TYPES.get(name) if isinstance(name, str) else None
    if cls is None:
        raise ValueError(f"unknown event type {name!r}")
    version = record.get("v")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{name}: schema version {version!r} != supported {SCHEMA_VERSION}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    body = {k: v for k, v in record.items() if k not in ("type", "v")}
    extra, missing = set(body) - set(names), set(names) - set(body)
    if extra or missing:
        raise ValueError(
            f"{name}: extra keys {sorted(extra)}, missing keys {sorted(missing)}")
    try:
        return cls(**{n: _decode(hints[n], body[n]) for n in names})
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"{name}: cannot decode record: {exc}") from exc


def canonical_json(record: Mapping[str, object]) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, no NaN/inf."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
