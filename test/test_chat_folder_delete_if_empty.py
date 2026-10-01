"""``DELETE /api/chat/folders/{id}?if_empty=true`` deletes only an empty folder.

This is the mode the ``chat_folder_delete`` MCP tool sends. It must never unfile
a session or lift a subfolder: occupancy is decided in the same locked step that
removes the folder, so a session filed after the caller's own read is still seen.
"""

from __future__ import annotations

from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_folder_app, _make_state

from kiro_crew.dashboard import chat_folders


async def _create(client: TestClient, name: str, parent_id: str = "") -> dict[str, Any]:
    body: dict[str, Any] = {"name": name}
    if parent_id:
        body["parent_id"] = parent_id
    resp = await client.post("/api/chat/folders", json=body)
    assert resp.status in (200, 201), await resp.text()
    return await resp.json()


def _ids(state: Any) -> set[str]:
    return {f["id"] for f in state._folders}


class TestIfEmptyDelete:
    @pytest.mark.asyncio
    async def test_an_empty_folder_is_deleted(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            folder = await _create(client, "Empty")
            resp = await client.delete(f"/api/chat/folders/{folder['id']}?if_empty=true")
            assert resp.status == 200
        assert folder["id"] not in _ids(state)

    @pytest.mark.asyncio
    async def test_a_live_session_keeps_the_folder(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            folder = await _create(client, "Work")
            slot = state.get_or_create_slot("filed")
            slot.folder_id = folder["id"]
            resp = await client.delete(f"/api/chat/folders/{folder['id']}?if_empty=true")
            assert resp.status == 409
            body = await resp.json()
        assert body["code"] == "folder_not_empty"
        assert folder["id"] in _ids(state)
        assert slot.folder_id == folder["id"], "the empty-only delete unfiled a session"

    @pytest.mark.asyncio
    async def test_a_subfolder_keeps_the_folder(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            parent = await _create(client, "Parent")
            child = await _create(client, "Child", parent["id"])
            resp = await client.delete(f"/api/chat/folders/{parent['id']}?if_empty=true")
            assert resp.status == 409
        assert parent["id"] in _ids(state)
        kept = next(f for f in state._folders if f["id"] == child["id"])
        assert kept["parent_id"] == parent["id"], "the empty-only delete lifted a subfolder"

    @pytest.mark.asyncio
    async def test_an_archived_session_keeps_the_folder(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            folder = await _create(client, "Old")
            monkeypatch.setattr(
                chat_folders, "_folder_history_counts", lambda _state: {folder["id"]: 1}
            )
            resp = await client.delete(f"/api/chat/folders/{folder['id']}?if_empty=true")
            assert resp.status == 409
            body = await resp.json()
        assert body["code"] == "folder_not_empty"
        assert "1" not in body["error"], "the refusal must not carry a count"
        assert folder["id"] in _ids(state)

    @pytest.mark.asyncio
    async def test_a_session_filed_during_the_archive_scan_is_still_seen(
        self, tmp_path, monkeypatch
    ) -> None:
        """The live-slot check runs after the scan's await, under the store lock.

        A pre-check before that await would pass here, and the delete would
        then unfile the session that landed.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            folder = await _create(client, "Racy")
            slot = state.get_or_create_slot("late")

            def _scan_while_a_session_is_filed(_state: Any) -> dict[str, int]:
                slot.folder_id = folder["id"]
                return {}

            monkeypatch.setattr(
                chat_folders, "_folder_history_counts", _scan_while_a_session_is_filed
            )
            resp = await client.delete(f"/api/chat/folders/{folder['id']}?if_empty=true")
            assert resp.status == 409
        assert folder["id"] in _ids(state)
        assert slot.folder_id == folder["id"]

    @pytest.mark.asyncio
    async def test_a_folder_gone_before_the_lock_is_not_found(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            folder = await _create(client, "Vanishing")

            def _scan_while_it_is_deleted(_state: Any) -> dict[str, int]:
                state._folders[:] = [f for f in state._folders if f["id"] != folder["id"]]
                return {}

            monkeypatch.setattr(chat_folders, "_folder_history_counts", _scan_while_it_is_deleted)
            resp = await client.delete(f"/api/chat/folders/{folder['id']}?if_empty=true")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_without_the_flag_a_full_folder_is_still_deleted(
        self, tmp_path, monkeypatch
    ) -> None:
        """The person's own sidebar delete keeps unfiling, as it always did."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            folder = await _create(client, "Full")
            slot = state.get_or_create_slot("filed")
            slot.folder_id = folder["id"]
            resp = await client.delete(f"/api/chat/folders/{folder['id']}")
            assert resp.status == 200
        assert folder["id"] not in _ids(state)
        assert slot.folder_id == ""
