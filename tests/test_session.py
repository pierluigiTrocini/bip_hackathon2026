import json
import os
import tempfile

from src.agent.session import SessionManager
from src.agent import config as cfg


def _isolated_session_manager(tmpdir: str) -> SessionManager:
    cfg.SESSION_PATH = os.path.join(tmpdir, "session.json")
    return SessionManager()


def test_detect_previous_session_returns_none_if_no_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = _isolated_session_manager(tmpdir)
        assert sm.detect_previous_session() is None


def test_create_new_generates_unique_session_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = _isolated_session_manager(tmpdir)
        s1 = sm.create_new("prompt A")
        s2 = sm.create_new("prompt B")
        assert s1["session_id"] != s2["session_id"]


def test_save_is_atomic():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = _isolated_session_manager(tmpdir)
        session = sm.create_new("test")
        session["cycle"] = 42
        sm.save(session)
        # Verify no .tmp file remains
        assert not os.path.exists(cfg.SESSION_PATH + ".tmp")
        # Verify saved content is valid
        with open(cfg.SESSION_PATH) as f:
            data = json.load(f)
        assert data["cycle"] == 42


def test_resume_sets_status_active():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = _isolated_session_manager(tmpdir)
        session = sm.create_new("test")
        sm.mark_paused(session)
        # Re-read from disk
        with open(cfg.SESSION_PATH) as f:
            paused = json.load(f)
        assert paused["status"] == "paused"
        resumed = sm.resume(paused)
        assert resumed["status"] == "active"
