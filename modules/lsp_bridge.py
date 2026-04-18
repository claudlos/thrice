"""Language-Server-Protocol bridge for Hermes Agent (Thrice).

Talks to a running LSP server over stdio JSON-RPC 2.0, so the agent can
use proper symbol-level navigation (definition, references, hover,
rename, document-symbols) instead of grepping.

The implementation is intentionally minimal:

- Line-oriented JSON-RPC framing (``Content-Length: N\\r\\n\\r\\n{...}``).
- Synchronous request/response; notifications are best-effort.
- No dynamic feature negotiation beyond the capabilities the agent
  actually needs - request a single call, return the structured result,
  close cleanly.

Tested against ``pyright`` out of the box; ``pylsp``, ``rust-analyzer``,
``gopls`` and ``tsserver`` follow the same protocol and should work with
their respective ``SERVERS`` entries.

Typical usage::

    from lsp_bridge import LspBridge

    bridge = LspBridge.pyright(root="/path/to/project")
    bridge.start()
    try:
        locs = bridge.definition("src/app.py", line=42, character=10)
    finally:
        bridge.shutdown()

There is no async story here; the bridge is meant to serve a handful of
queries at a time from a synchronous agent loop.  For high-volume use,
wrap an instance per worker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


JSONRPC_VERSION = "2.0"

# Servers we have built-in recipes for.  Each value is a template argv;
# callers can override by passing ``cmd=[...]`` to ``LspBridge(...)``.
SERVERS: Dict[str, List[str]] = {
    "pyright":         ["pyright-langserver", "--stdio"],
    "pylsp":           ["pylsp"],
    "rust-analyzer":   ["rust-analyzer"],
    "gopls":           ["gopls"],
    "typescript":      ["typescript-language-server", "--stdio"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Location:
    """An LSP ``Location`` - a file URI + a range.  We keep the range
    simplified as ``(start_line, start_char, end_line, end_char)``."""

    uri: str
    start_line: int
    start_char: int
    end_line: int
    end_char: int

    @property
    def path(self) -> str:
        """The underlying filesystem path, parsed from the ``file://`` URI."""
        return _uri_to_path(self.uri)

    def format_short(self) -> str:
        return f"{self.path}:{self.start_line + 1}:{self.start_char + 1}"


@dataclass(frozen=True)
class SymbolInfo:
    """An LSP ``DocumentSymbol`` - name, kind, range."""

    name: str
    kind: int        # LSP enum: 12=Function, 5=Class, 6=Method, ...
    location: Location
    container: Optional[str] = None


@dataclass
class LspError(Exception):
    """Raised on JSON-RPC error responses or protocol failures."""

    code: int
    message: str

    def __str__(self) -> str:
        return f"LSP error {self.code}: {self.message}"


# ---------------------------------------------------------------------------
# JSON-RPC framing
# ---------------------------------------------------------------------------

def _encode_frame(obj: Dict[str, Any]) -> bytes:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _read_frame(stream) -> Optional[Dict[str, Any]]:
    """Read one JSON-RPC frame off ``stream``; return None on EOF."""
    # Read headers.
    headers: Dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        if b":" in line:
            key, _, value = line.partition(b":")
            headers[key.strip().decode("ascii").lower()] = value.strip().decode("ascii")
    length_s = headers.get("content-length")
    if not length_s:
        return None
    length = int(length_s)
    body = stream.read(length)
    if len(body) < length:
        return None
    return json.loads(body.decode("utf-8"))


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class LspBridge:
    """Synchronous JSON-RPC client for an LSP server spawned over stdio."""

    def __init__(
        self,
        cmd: Sequence[str],
        root: str,
        *,
        language_id: str = "python",
        init_timeout: float = 10.0,
        request_timeout: float = 15.0,
        extra_init_options: Optional[Dict[str, Any]] = None,
    ):
        self.cmd = list(cmd)
        self.root = os.path.abspath(root)
        self.language_id = language_id
        self.init_timeout = init_timeout
        self.request_timeout = request_timeout
        self.extra_init_options = extra_init_options or {}
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 1
        self._pending: Dict[int, "queue.Queue[Dict[str, Any]]"] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stopped = False
        # abs_path -> (version, sha256 of last synced text).  Tracks the
        # contents the server has actually seen so _sync_file can decide
        # whether to fire didOpen, didChange, or no-op.
        self._open_files: Dict[str, Tuple[int, str]] = {}

    # -- Named constructors ----------------------------------------------

    @classmethod
    def pyright(cls, root: str, **kw) -> "LspBridge":
        return cls(SERVERS["pyright"], root=root, language_id="python", **kw)

    @classmethod
    def pylsp(cls, root: str, **kw) -> "LspBridge":
        return cls(SERVERS["pylsp"], root=root, language_id="python", **kw)

    @classmethod
    def rust(cls, root: str, **kw) -> "LspBridge":
        return cls(SERVERS["rust-analyzer"], root=root, language_id="rust", **kw)

    # -- Lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Spawn the server, complete the ``initialize`` handshake."""
        if self._proc is not None:
            return
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.root, bufsize=0,
            )
        except FileNotFoundError as exc:
            raise LspError(code=-1, message=f"LSP server not on PATH: {self.cmd[0]}") from exc

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name=f"lsp-reader-{os.getpid()}", daemon=True,
        )
        self._reader_thread.start()
        # Drain stderr continuously so a chatty server can't fill its OS-level
        # pipe buffer and deadlock the reader thread.
        self._stderr_thread = threading.Thread(
            target=self._stderr_drain, name=f"lsp-stderr-{os.getpid()}", daemon=True,
        )
        self._stderr_thread.start()

        init_params = {
            "processId": os.getpid(),
            "rootUri": _path_to_uri(self.root),
            "capabilities": {
                "textDocument": {
                    "definition":       {"linkSupport": False},
                    "references":       {"dynamicRegistration": False},
                    "hover":            {"contentFormat": ["plaintext"]},
                    "documentSymbol":   {"hierarchicalDocumentSymbolSupport": False},
                    "rename":           {"prepareSupport": False},
                },
                "workspace": {},
            },
            "initializationOptions": self.extra_init_options,
        }
        self._call("initialize", init_params, timeout=self.init_timeout)
        self._notify("initialized", {})

    def shutdown(self) -> None:
        """Close the server cleanly (best-effort)."""
        if self._proc is None:
            return
        try:
            self._call("shutdown", None, timeout=3.0)
            self._notify("exit", None)
        except Exception:
            pass
        self._stopped = True
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3.0)
        except Exception:
            if self._proc.poll() is None:
                self._proc.kill()
        self._proc = None
        # Clear tracked open-file state so a reused bridge starts clean.
        self._open_files.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.shutdown()
        return False

    # -- High-level queries ----------------------------------------------

    def definition(self, file: str, line: int, character: int) -> List[Location]:
        """``textDocument/definition`` - returns a list (may be empty)."""
        self._sync_file(file)
        params = self._text_doc_position(file, line, character)
        result = self._call("textDocument/definition", params) or []
        return [_loc_from_lsp(r) for r in _as_list(result)]

    def references(self, file: str, line: int, character: int,
                   include_decl: bool = True) -> List[Location]:
        """``textDocument/references`` - returns all call sites."""
        self._sync_file(file)
        params = self._text_doc_position(file, line, character)
        params["context"] = {"includeDeclaration": include_decl}
        result = self._call("textDocument/references", params) or []
        return [_loc_from_lsp(r) for r in _as_list(result)]

    def hover(self, file: str, line: int, character: int) -> Optional[str]:
        """``textDocument/hover`` - returns the plaintext contents or None."""
        self._sync_file(file)
        params = self._text_doc_position(file, line, character)
        result = self._call("textDocument/hover", params)
        if not result:
            return None
        contents = result.get("contents")
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            return contents.get("value")
        if isinstance(contents, list):
            parts = []
            for c in contents:
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, dict) and "value" in c:
                    parts.append(c["value"])
            return "\n".join(parts) if parts else None
        return None

    def document_symbols(self, file: str) -> List[SymbolInfo]:
        """``textDocument/documentSymbol`` - top-level + class members."""
        self._sync_file(file)
        params = {"textDocument": {"uri": _path_to_uri(file)}}
        result = self._call("textDocument/documentSymbol", params) or []
        return [_symbol_from_lsp(s, file) for s in _as_list(result)]

    def rename(
        self,
        file: str,
        line: int,
        character: int,
        new_name: str,
    ) -> Dict[str, List[Tuple[int, int, int, int, str]]]:
        """``textDocument/rename`` - returns a map ``{file: [edits]}``.

        Each edit is a tuple ``(start_line, start_char, end_line, end_char,
        new_text)``.  The client is responsible for applying them.
        """
        self._sync_file(file)
        params = self._text_doc_position(file, line, character)
        params["newName"] = new_name
        result = self._call("textDocument/rename", params)
        if not result:
            return {}
        changes = result.get("changes") or {}
        out: Dict[str, List[Tuple[int, int, int, int, str]]] = {}
        for uri, edits in changes.items():
            path = _uri_to_path(uri)
            out.setdefault(path, [])
            for edit in edits:
                r = edit["range"]
                out[path].append((
                    r["start"]["line"], r["start"]["character"],
                    r["end"]["line"],   r["end"]["character"],
                    edit["newText"],
                ))
        return out

    # -- Low-level plumbing ----------------------------------------------

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _call(self, method: str, params: Any, *, timeout: Optional[float] = None) -> Any:
        rid = self._next_request_id()
        fut: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._pending[rid] = fut
        msg = {"jsonrpc": JSONRPC_VERSION, "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        try:
            resp = fut.get(timeout=timeout if timeout is not None else self.request_timeout)
        except queue.Empty:
            self._pending.pop(rid, None)
            raise LspError(
                code=-32001,
                message=f"LSP request {method} timed out after "
                        f"{timeout or self.request_timeout:.1f}s",
            ) from None
        if "error" in resp:
            err = resp["error"]
            raise LspError(code=int(err.get("code", -32000)),
                           message=str(err.get("message", "unknown")))
        return resp.get("result")

    def _notify(self, method: str, params: Any) -> None:
        msg = {"jsonrpc": JSONRPC_VERSION, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def _send(self, msg: Dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise LspError(code=-1, message="LSP process not running")
        frame = _encode_frame(msg)
        try:
            self._proc.stdin.write(frame)
            self._proc.stdin.flush()
        except Exception as exc:
            raise LspError(code=-1, message=f"send failed: {exc!s}") from exc

    def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while not self._stopped:
            try:
                msg = _read_frame(self._proc.stdout)
            except Exception:
                break
            if msg is None:
                break
            if "id" in msg and "method" not in msg:
                rid = int(msg["id"])
                fut = self._pending.pop(rid, None)
                if fut is not None:
                    try:
                        fut.put_nowait(msg)
                    except queue.Full:
                        pass
            # Server->client notifications and requests are ignored (we don't
            # implement workspace/configuration, publishDiagnostics, etc).

    def _stderr_drain(self) -> None:
        """Continuously read and discard stderr so the pipe can't fill.

        LSP servers may log verbosely to stderr; if no one reads it, the
        kernel-level pipe buffer (~64 KB on Linux) fills and the server
        blocks on write, which then stalls the reader thread too.
        Draining here keeps both sides flowing.
        """
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            for _ in iter(self._proc.stderr.readline, b""):
                if self._stopped:
                    break
        except Exception:
            pass

    def _text_doc_position(self, file: str, line: int, character: int) -> Dict[str, Any]:
        return {
            "textDocument": {"uri": _path_to_uri(file)},
            "position":     {"line": line, "character": character},
        }

    def _sync_file(self, file: str) -> None:
        """Synchronise ``file``'s on-disk contents with the LSP server.

        - First visit: send ``textDocument/didOpen`` with version 1.
        - Content changed since last sync: bump the version and send
          ``textDocument/didChange`` with a full-document replacement.
        - Content unchanged: no-op.

        Using a content hash instead of mtime-or-marker semantics keeps
        the bridge correct across tools that touch files without changing
        bytes, and across filesystems with coarse mtime resolution.
        """
        abs_path = os.path.abspath(file)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            raise LspError(code=-1, message=f"open failed: {exc!s}") from exc

        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        previous = self._open_files.get(abs_path)

        if previous is None:
            version = 1
            self._open_files[abs_path] = (version, digest)
            self._notify("textDocument/didOpen", {
                "textDocument": {
                    "uri": _path_to_uri(abs_path),
                    "languageId": self.language_id,
                    "version": version,
                    "text": text,
                },
            })
            return

        prev_version, prev_digest = previous
        if prev_digest == digest:
            return

        version = prev_version + 1
        self._open_files[abs_path] = (version, digest)
        self._notify("textDocument/didChange", {
            "textDocument": {
                "uri": _path_to_uri(abs_path),
                "version": version,
            },
            # Full-document sync; we don't negotiate incremental sync
            # capabilities, so we always replace the whole buffer.
            "contentChanges": [{"text": text}],
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_to_uri(p: str) -> str:
    return Path(p).resolve().as_uri()


def _uri_to_path(uri: str) -> str:
    if not uri.startswith("file://"):
        return uri
    from urllib.parse import unquote, urlparse
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    # Windows: URIs look like ``file:///C:/path``; strip leading slash.
    if os.name == "nt" and path.startswith("/") and len(path) > 3 and path[2] == ":":
        path = path[1:]
    return path


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _loc_from_lsp(loc: Dict[str, Any]) -> Location:
    rng = loc.get("range") or {}
    s = rng.get("start", {})
    e = rng.get("end", {})
    return Location(
        uri=loc.get("uri", ""),
        start_line=int(s.get("line", 0)),
        start_char=int(s.get("character", 0)),
        end_line=int(e.get("line", 0)),
        end_char=int(e.get("character", 0)),
    )


def _symbol_from_lsp(s: Dict[str, Any], file_fallback: str) -> SymbolInfo:
    if "location" in s:  # SymbolInformation
        return SymbolInfo(
            name=s["name"],
            kind=int(s.get("kind", 0)),
            location=_loc_from_lsp(s["location"]),
            container=s.get("containerName"),
        )
    # DocumentSymbol -> synthesize a Location from the range.
    rng = s.get("range") or {}
    sp = rng.get("start", {})
    ep = rng.get("end", {})
    return SymbolInfo(
        name=s["name"],
        kind=int(s.get("kind", 0)),
        location=Location(
            uri=_path_to_uri(file_fallback),
            start_line=int(sp.get("line", 0)),
            start_char=int(sp.get("character", 0)),
            end_line=int(ep.get("line", 0)),
            end_char=int(ep.get("character", 0)),
        ),
    )


__all__ = [
    "LspBridge",
    "LspError",
    "Location",
    "SymbolInfo",
    "SERVERS",
]
