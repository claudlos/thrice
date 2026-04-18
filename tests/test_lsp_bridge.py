"""Tests for ``lsp_bridge``.

Since we don't want to require pyright / rust-analyzer on the test
runner, the integration tests spawn a **tiny fake LSP server** (a small
Python script bundled inside the test file as a string).  The fake
server implements just enough of the protocol to drive the bridge's
``start → definition → references → shutdown`` happy path.
"""
from __future__ import annotations

import io
import os
import sys
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from lsp_bridge import (  # noqa: E402
    Location,
    LspBridge,
    LspError,
    SymbolInfo,
    _encode_frame,
    _path_to_uri,
    _read_frame,
    _uri_to_path,
)

# ---------------------------------------------------------------------------
# Pure unit tests: framing, URI conversion
# ---------------------------------------------------------------------------

class TestFraming:
    def test_encode_produces_content_length_header(self):
        frame = _encode_frame({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert frame.startswith(b"Content-Length: ")
        # body length matches.
        length_header = frame.split(b"\r\n")[0]
        length = int(length_header.split(b": ")[1])
        body = frame.split(b"\r\n\r\n", 1)[1]
        assert len(body) == length

    def test_roundtrip(self):
        msg = {"jsonrpc": "2.0", "id": 42, "method": "foo", "params": {"x": 1}}
        frame = _encode_frame(msg)
        parsed = _read_frame(io.BytesIO(frame))
        assert parsed == msg

    def test_eof_returns_none(self):
        assert _read_frame(io.BytesIO(b"")) is None


class TestUri:
    def test_path_to_uri_and_back(self, tmp_path):
        p = tmp_path / "file.py"
        p.write_text("")
        uri = _path_to_uri(str(p))
        assert uri.startswith("file://")
        back = _uri_to_path(uri)
        assert os.path.normpath(back) == os.path.normpath(str(p))

    def test_non_file_uri_passthrough(self):
        assert _uri_to_path("stdio:foo") == "stdio:foo"


# ---------------------------------------------------------------------------
# Fake-server integration: start() happy path
# ---------------------------------------------------------------------------

_FAKE_SERVER_SRC = textwrap.dedent(r'''
    """Tiny fake LSP server for lsp_bridge tests."""
    import json, sys, os, re

    # Record client-facing events so tests can assert on them.  The log
    # file path is pulled from LSP_FAKE_LOG in the process env.
    _LOG = os.environ.get("LSP_FAKE_LOG")

    def _log(entry):
        if not _LOG:
            return
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def read_frame():
        headers = {}
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            if b":" in line:
                k, _, v = line.partition(b":")
                headers[k.strip().decode("ascii").lower()] = v.strip().decode("ascii")
        n = int(headers.get("content-length", "0"))
        body = sys.stdin.buffer.read(n)
        return json.loads(body.decode("utf-8"))

    def write(obj):
        body = json.dumps(obj).encode("utf-8")
        sys.stdout.buffer.write(b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n")
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()

    while True:
        msg = read_frame()
        if msg is None:
            break
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            write({"jsonrpc": "2.0", "id": mid, "result": {"capabilities": {}}})
        elif method == "initialized":
            pass
        elif method == "textDocument/didOpen":
            td = msg.get("params", {}).get("textDocument", {})
            _log({"event": "didOpen",
                  "uri": td.get("uri"),
                  "version": td.get("version"),
                  "text": td.get("text")})
        elif method == "textDocument/didChange":
            td = msg.get("params", {}).get("textDocument", {})
            changes = msg.get("params", {}).get("contentChanges", [])
            text = changes[0].get("text") if changes else None
            _log({"event": "didChange",
                  "uri": td.get("uri"),
                  "version": td.get("version"),
                  "text": text})
        elif method == "textDocument/definition":
            uri = msg["params"]["textDocument"]["uri"]
            write({"jsonrpc": "2.0", "id": mid, "result": [{
                "uri": uri,
                "range": {"start": {"line": 10, "character": 0},
                          "end":   {"line": 10, "character": 5}},
            }]})
        elif method == "textDocument/references":
            uri = msg["params"]["textDocument"]["uri"]
            write({"jsonrpc": "2.0", "id": mid, "result": [
                {"uri": uri, "range": {"start": {"line": 1, "character": 0},
                                        "end":   {"line": 1, "character": 3}}},
                {"uri": uri, "range": {"start": {"line": 7, "character": 0},
                                        "end":   {"line": 7, "character": 3}}},
            ]})
        elif method == "textDocument/hover":
            write({"jsonrpc": "2.0", "id": mid,
                   "result": {"contents": {"value": "hello from fake server"}}})
        elif method == "textDocument/documentSymbol":
            uri = msg["params"]["textDocument"]["uri"]
            write({"jsonrpc": "2.0", "id": mid, "result": [
                {"name": "MyFunc", "kind": 12,
                 "range": {"start": {"line": 0, "character": 0},
                           "end":   {"line": 5, "character": 0}},
                 "selectionRange": {"start": {"line": 0, "character": 4},
                                     "end":   {"line": 0, "character": 10}}},
            ]})
        elif method == "textDocument/rename":
            uri = msg["params"]["textDocument"]["uri"]
            new = msg["params"]["newName"]
            write({"jsonrpc": "2.0", "id": mid, "result": {
                "changes": {uri: [{
                    "range": {"start": {"line": 3, "character": 4},
                              "end":   {"line": 3, "character": 10}},
                    "newText": new,
                }]},
            }})
        elif method == "shutdown":
            write({"jsonrpc": "2.0", "id": mid, "result": None})
        elif method == "exit":
            sys.exit(0)
        else:
            if mid is not None:
                write({"jsonrpc": "2.0", "id": mid,
                       "error": {"code": -32601,
                                  "message": f"method not found: {method}"}})
''')


@pytest.fixture
def fake_server(tmp_path):
    """Write the fake server script and return its argv."""
    script = tmp_path / "fake_lsp.py"
    script.write_text(_FAKE_SERVER_SRC)
    return [sys.executable, str(script)]


@pytest.fixture
def started_bridge(fake_server, tmp_path):
    (tmp_path / "foo.py").write_text("def hello():\n    return 1\n")
    bridge = LspBridge(fake_server, root=str(tmp_path),
                       init_timeout=5.0, request_timeout=5.0)
    bridge.start()
    yield bridge
    bridge.shutdown()


class TestLifecycle:
    def test_start_and_shutdown(self, fake_server, tmp_path):
        bridge = LspBridge(fake_server, root=str(tmp_path),
                           init_timeout=5.0, request_timeout=5.0)
        bridge.start()
        bridge.shutdown()

    def test_context_manager(self, fake_server, tmp_path):
        with LspBridge(fake_server, root=str(tmp_path),
                       init_timeout=5.0, request_timeout=5.0) as br:
            assert br._proc is not None

    def test_missing_server_raises(self, tmp_path):
        with pytest.raises(LspError):
            LspBridge(["does-not-exist-lsp-binary-xyz"], root=str(tmp_path)).start()


class TestDefinition:
    def test_returns_locations(self, started_bridge, tmp_path):
        f = str(tmp_path / "foo.py")
        locs = started_bridge.definition(f, line=0, character=4)
        assert locs and isinstance(locs[0], Location)
        assert locs[0].start_line == 10


class TestReferences:
    def test_returns_multiple(self, started_bridge, tmp_path):
        f = str(tmp_path / "foo.py")
        refs = started_bridge.references(f, line=0, character=4)
        assert len(refs) == 2
        assert {r.start_line for r in refs} == {1, 7}


class TestHover:
    def test_extracts_string(self, started_bridge, tmp_path):
        f = str(tmp_path / "foo.py")
        text = started_bridge.hover(f, line=0, character=4)
        assert text and "hello from fake" in text


class TestDocumentSymbols:
    def test_returns_symbols(self, started_bridge, tmp_path):
        f = str(tmp_path / "foo.py")
        syms = started_bridge.document_symbols(f)
        assert syms and isinstance(syms[0], SymbolInfo)
        assert syms[0].name == "MyFunc"


class TestRename:
    def test_returns_edits(self, started_bridge, tmp_path):
        f = str(tmp_path / "foo.py")
        changes = started_bridge.rename(f, line=0, character=4, new_name="howdy")
        assert changes
        # file-path key, list of edits.
        (path, edits), = changes.items()
        assert edits[0][4] == "howdy"


class TestLocationHelpers:
    def test_location_path(self, tmp_path):
        p = tmp_path / "x.py"
        p.write_text("")
        loc = Location(uri=_path_to_uri(str(p)),
                       start_line=0, start_char=0, end_line=1, end_char=2)
        assert os.path.normpath(loc.path) == os.path.normpath(str(p))


# ---------------------------------------------------------------------------
# File-sync regression tests for _sync_file (P1 fix).
#
# The fake server records didOpen / didChange notifications to a log file
# when LSP_FAKE_LOG is set in its environment.  Because start() forks the
# server process directly, we install the log path on os.environ for the
# duration of these tests.
# ---------------------------------------------------------------------------

import json as _json


def _read_log(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(_json.loads(line))
    return out


@pytest.fixture
def logged_bridge(fake_server, tmp_path, monkeypatch):
    log_path = tmp_path / "lsp_events.log"
    monkeypatch.setenv("LSP_FAKE_LOG", str(log_path))
    target = tmp_path / "foo.py"
    target.write_text("def hello():\n    return 1\n")
    bridge = LspBridge(fake_server, root=str(tmp_path),
                       init_timeout=5.0, request_timeout=5.0)
    bridge.start()
    try:
        yield bridge, target, log_path
    finally:
        bridge.shutdown()


class TestSyncFile:
    def test_first_query_sends_did_open(self, logged_bridge):
        bridge, target, log_path = logged_bridge
        bridge.definition(str(target), line=0, character=4)
        events = _read_log(str(log_path))
        opens = [e for e in events if e["event"] == "didOpen"]
        assert len(opens) == 1
        assert opens[0]["version"] == 1
        assert "def hello" in opens[0]["text"]

    def test_repeat_query_without_edit_does_not_resend(self, logged_bridge):
        bridge, target, log_path = logged_bridge
        bridge.definition(str(target), line=0, character=4)
        bridge.definition(str(target), line=0, character=4)
        events = _read_log(str(log_path))
        assert sum(1 for e in events if e["event"] == "didOpen") == 1
        assert sum(1 for e in events if e["event"] == "didChange") == 0

    def test_edit_triggers_did_change_with_bumped_version(self, logged_bridge):
        bridge, target, log_path = logged_bridge
        bridge.definition(str(target), line=0, character=4)
        target.write_text("def hello():\n    return 2  # edited\n")
        bridge.definition(str(target), line=0, character=4)
        events = _read_log(str(log_path))
        changes = [e for e in events if e["event"] == "didChange"]
        assert len(changes) == 1
        assert changes[0]["version"] == 2
        assert "return 2" in changes[0]["text"]

    def test_rename_sends_current_contents(self, logged_bridge):
        bridge, target, log_path = logged_bridge
        bridge.definition(str(target), line=0, character=4)
        target.write_text("def goodbye():\n    return 3\n")
        bridge.rename(str(target), line=0, character=4, new_name="howdy")
        events = _read_log(str(log_path))
        changes = [e for e in events if e["event"] == "didChange"]
        assert changes, "rename after edit must sync current contents"
        assert "goodbye" in changes[-1]["text"]

    def test_shutdown_clears_open_file_state(self, logged_bridge):
        bridge, target, _log_path = logged_bridge
        bridge.definition(str(target), line=0, character=4)
        assert bridge._open_files
        bridge.shutdown()
        assert bridge._open_files == {}
