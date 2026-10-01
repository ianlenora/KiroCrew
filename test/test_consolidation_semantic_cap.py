"""The ``## Current Semantic Memory`` block of a consolidation prompt is bounded.

Every consolidation pass hands the model the active semantic table so it can
update or delete the keys it already holds instead of minting near-duplicates.
The table only grows, and before this cap the block was serialised whole on
both the history pass and the preference-only pass: one reporter measured
1,977 rows rendering to about 2 million characters, 97% of the prompt, and the
next pass failed as too large to send. The chat path caps the same data at
``semantic_cap``; this pins the consolidation-side cap.

The cap bounds the PROMPT only. The write-side snapshot the consolidator hands
its writers still carries the whole table, so update-versus-create and the
revision checks keep seeing every row.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from kiro_crew.history import HistoryConsolidator
from kiro_crew.vector_memory_constants import _SEMANTIC_PROMPT_CAP_PER_CONSOLIDATION

_HEADING = "\n\n## Current Semantic Memory\n"
_NEXT_HEADING = "\n\n## Conversation to Process\n"
# The notice is the one line a model reads to learn the table is partial.
_OMISSION_LINE = re.compile(r"\[Context budget: omitted (\d+) of (\d+) semantic rows\b.*\]")


def _row(index: int, value_len: int, day: int) -> dict:
    # Keys sort unlike their age on purpose: ``key`` order is what the block
    # renders in, ``updated_at`` is what decides which rows survive the cap.
    return {
        "key": f"project.row_{(index * 7919) % 10_000:05d}",
        "value_json": json.dumps(f"v{index}:" + "x" * value_len),
        "confidence": 0.9,
        "source": "consolidation:test",
        "created_at": f"2026-01-{day:02d}T00:00:00",
        "updated_at": f"2026-01-{day:02d}T00:00:00",
        "is_deleted": 0,
    }


def _store_with(rows: list[dict]) -> MagicMock:
    vector_store = MagicMock()
    vector_store.algorithm_version = "v1"
    # ``get_all_semantic`` reads ``ORDER BY key``; the fake keeps that contract.
    vector_store.get_all_semantic.return_value = [
        dict(r) for r in sorted(rows, key=lambda r: r["key"])
    ]
    return vector_store


def _consolidator(vector_store: MagicMock) -> HistoryConsolidator:
    log = MagicMock()
    log.snapshot_for_consolidation.return_value = (
        [{"role": "user", "content": "hi"}],
        1,
        0,
    )
    log.get_metadata.return_value = {}
    log.get_metadata_status.return_value = ({}, True)
    log.consolidation_retry_state.return_value = (0, 0.0)
    memory = MagicMock()
    memory.read_preferences.return_value = ""
    memory.read_projects.return_value = ""
    return HistoryConsolidator(
        log=log,
        memory=memory,
        sessions=None,
        vector_store=vector_store,
        migrated=True,
    )


async def _prompt_for(rows: list[dict], *, include_history: bool) -> tuple[str, dict]:
    """Run one pass and return the prompt plus the writer's keyword arguments."""
    c = _consolidator(_store_with(rows))
    captured: dict = {}

    async def fake_llm(prompt: str, *, memory_store: str = "", session_key: str = "") -> dict:
        captured["prompt"] = prompt
        return {"semantic": []}

    def fake_write(result, key, vector_store=None, **kwargs):
        captured["write_kwargs"] = kwargs

    with (
        patch.object(c, "_call_llm", side_effect=fake_llm),
        patch.object(c, "_write_structured_memory", side_effect=fake_write),
    ):
        await c._consolidate("k", include_history=include_history)
    assert "prompt" in captured, "the pass must have issued a prompt"
    return captured["prompt"], captured.get("write_kwargs", {})


def _block_of(prompt: str) -> str:
    start = prompt.index(_HEADING) + len(_HEADING)
    end = prompt.index(_NEXT_HEADING, start)
    return prompt[start:end]


def _rendered(rows: list[dict]) -> str:
    """What the block looked like before the cap: every row, key order, indent=1."""
    return json.dumps(
        [
            {"key": r["key"], "value_json": r["value_json"], "confidence": r["confidence"]}
            for r in sorted(rows, key=lambda r: r["key"])
        ],
        indent=1,
    )


def _oversized_table() -> list[dict]:
    # ~1,000 characters per rendered row, 120 rows: about twice the cap, so a
    # bounded block must drop rows while plenty still fit under it.
    rows = [_row(i, 950, day=1 + i % 28) for i in range(120)]
    assert len(_rendered(rows)) > _SEMANTIC_PROMPT_CAP_PER_CONSOLIDATION
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("include_history", [False, True], ids=["prefs-only", "history"])
async def test_oversized_table_is_bounded_and_says_so(include_history: bool) -> None:
    rows = _oversized_table()
    prompt, write_kwargs = await _prompt_for(rows, include_history=include_history)
    block = _block_of(prompt)

    match = _OMISSION_LINE.search(block)
    table = block[: match.start()].rstrip("\n") if match else block
    assert len(table) <= _SEMANTIC_PROMPT_CAP_PER_CONSOLIDATION, (
        f"the semantic block is {len(table)} chars, over the "
        f"{_SEMANTIC_PROMPT_CAP_PER_CONSOLIDATION}-char consolidation cap"
    )
    assert match, "an over-cap table must carry the one-line omission notice"
    omitted, total = int(match.group(1)), int(match.group(2))
    assert total == len(rows)
    assert 0 < omitted < total

    kept = json.loads(table)
    assert len(kept) == total - omitted
    # The same shape the model was reading before: key, value_json, confidence.
    assert all(set(entry) == {"key", "value_json", "confidence"} for entry in kept)
    # Rendered in key order like the unbounded block, chosen newest first like
    # the chat path's ``semantic_cap`` does without a query.
    assert [entry["key"] for entry in kept] == sorted(entry["key"] for entry in kept)
    kept_keys = {entry["key"] for entry in kept}
    # Newest first; within one timestamp the key order breaks the tie.
    by_recency = sorted(rows, key=lambda r: r["key"])
    by_recency.sort(key=lambda r: r["updated_at"], reverse=True)
    newest_kept = [r["key"] for r in by_recency if r["key"] in kept_keys]
    assert newest_kept == [
        r["key"] for r in by_recency[: len(kept)]
    ], "the rows that survive the cap must be the most recently updated ones"

    # The cap is a prompt bound only: the writers still see the whole table.
    assert len(write_kwargs["snapshot"]) == len(rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_history", [False, True], ids=["prefs-only", "history"])
async def test_table_under_the_cap_renders_byte_identical(include_history: bool) -> None:
    rows = [_row(i, 40, day=1 + i % 28) for i in range(30)]
    assert len(_rendered(rows)) <= _SEMANTIC_PROMPT_CAP_PER_CONSOLIDATION

    prompt, write_kwargs = await _prompt_for(rows, include_history=include_history)

    assert _block_of(prompt) == _rendered(rows)
    assert "[Context budget:" not in prompt
    assert len(write_kwargs["snapshot"]) == len(rows)


@pytest.mark.asyncio
async def test_empty_table_still_renders_the_empty_list() -> None:
    prompt, _ = await _prompt_for([], include_history=False)
    assert _block_of(prompt) == "[]"
    assert "[Context budget:" not in prompt
