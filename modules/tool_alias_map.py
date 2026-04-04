"""
Tool Name Alias Map — Instant resolution of common model hallucinations.

Models frequently hallucinate tool names from their training data (bash, grep,
cat, edit, ls, find, file_read, etc.). Instead of burning 2-3 iterations on
fuzzy matching or error recovery, this module provides a static alias map
that resolves hallucinated names to Hermes tool names BEFORE any fuzzy matching.

Integration point: run_agent.py, _repair_tool_call() (~line 2499)
Apply this BEFORE the fuzzy matching logic.

Usage:
    from new_files.tool_alias_map import resolve_alias

    resolved = resolve_alias("bash")
    # Returns "terminal"
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Static alias map ───────────────────────────────────────────────────────
# Keys are lowercase. Values are canonical Hermes tool names.
# 90+ entries covering the most common hallucinations observed in production.
TOOL_ALIASES: dict[str, str] = {
    # ── Terminal / Shell (15 aliases) ──
    "bash": "terminal",
    "shell": "terminal",
    "execute": "terminal",
    "run": "terminal",
    "exec": "terminal",
    "command": "terminal",
    "run_command": "terminal",
    "execute_command": "terminal",
    "run_terminal_cmd": "terminal",
    "shell_exec": "terminal",
    "subprocess": "terminal",
    "system": "terminal",
    "sh": "terminal",
    "zsh": "terminal",
    "cmd": "terminal",

    # ── File Reading (12 aliases) ──
    "cat": "read_file",
    "file_read": "read_file",
    "read": "read_file",
    "view_file": "read_file",
    "open_file": "read_file",
    "get_file": "read_file",
    "show_file": "read_file",
    "file_content": "read_file",
    "read_text_file": "read_file",
    "head": "read_file",
    "tail": "read_file",
    "less": "read_file",

    # ── File Searching / Grep (14 aliases) ──
    "grep": "search_files",
    "rg": "search_files",
    "ripgrep": "search_files",
    "find": "search_files",
    "search": "search_files",
    "find_files": "search_files",
    "file_search": "search_files",
    "search_file": "search_files",
    "grep_search": "search_files",
    "codebase_search": "search_files",
    "code_search": "search_files",
    "search_code": "search_files",
    "find_in_files": "search_files",
    "ag": "search_files",  # the silver searcher

    # ── Listing Files (8 aliases) ──
    "ls": "search_files",
    "list_files": "search_files",
    "list_dir": "search_files",
    "list_directory": "search_files",
    "dir": "search_files",
    "tree": "search_files",
    "list_folder": "search_files",
    "directory_listing": "search_files",

    # ── File Editing / Patching (14 aliases) ──
    "edit": "patch",
    "edit_file": "patch",
    "file_edit": "patch",
    "modify_file": "patch",
    "replace": "patch",
    "sed": "patch",
    "awk": "patch",
    "str_replace_editor": "patch",
    "replace_in_file": "patch",
    "file_str_replace": "patch",
    "insert_code": "patch",
    "apply_diff": "patch",
    "update_file": "patch",
    "str_replace": "patch",

    # ── File Writing (8 aliases) ──
    "write": "write_file",
    "create_file": "write_file",
    "save_file": "write_file",
    "file_write": "write_file",
    "new_file": "write_file",
    "touch": "write_file",
    "write_text_file": "write_file",
    "overwrite_file": "write_file",

    # ── Web Browsing (9 aliases) ──
    "browse": "web_browse",
    "browser": "web_browse",
    "open_url": "web_browse",
    "fetch_url": "web_browse",
    "curl": "web_browse",
    "wget": "web_browse",
    "http_get": "web_browse",
    "fetch": "web_browse",
    "http_request": "web_browse",

    # ── Web Search (5 aliases) ──
    "google": "web_search",
    "search_web": "web_search",
    "internet_search": "web_search",
    "bing": "web_search",
    "duckduckgo": "web_search",

    # ── Process Management (5 aliases) ──
    "bg_process": "process",
    "background": "process",
    "manage_process": "process",
    "background_process": "process",
    "ps": "process",

    # ── Memory (5 aliases) ──
    "remember": "memory",
    "save_memory": "memory",
    "note": "memory",
    "store_memory": "memory",
    "recall": "memory",

    # ── Delegation (5 aliases) ──
    "delegate": "delegate_task",
    "spawn_agent": "delegate_task",
    "subagent": "delegate_task",
    "sub_agent": "delegate_task",
    "fork_task": "delegate_task",

    # ── Vision (6 aliases) ──
    "analyze_image": "vision_analyze",
    "view_image": "vision_analyze",
    "image_analysis": "vision_analyze",
    "ocr": "vision_analyze",
    "describe_image": "vision_analyze",
    "read_image": "vision_analyze",
}

# ─── Reverse index: canonical name -> list of aliases ───────────────────────
_REVERSE_INDEX: dict[str, list[str]] = {}
for _alias, _target in TOOL_ALIASES.items():
    _REVERSE_INDEX.setdefault(_target, []).append(_alias)


def resolve_alias(name: str) -> Optional[str]:
    """Resolve a hallucinated tool name to a canonical Hermes tool name.

    Returns the resolved name if found in the alias map, None otherwise.
    Case-insensitive lookup. Logs when an alias is resolved.

    Args:
        name: The tool name to resolve (possibly hallucinated).

    Returns:
        Canonical tool name if alias found, None otherwise.
    """
    if not name:
        return None

    key = name.lower().strip()
    resolved = TOOL_ALIASES.get(key)

    if resolved is not None:
        logger.info(
            "Tool alias resolved: '%s' -> '%s'",
            name,
            resolved,
        )

    return resolved


def get_alias_suggestions(tool_name: str) -> list[str]:
    """Get candidate Hermes tool names for a given (possibly hallucinated) name.

    Checks exact alias match first, then partial substring matching.

    Args:
        tool_name: The tool name to look up.

    Returns:
        List of canonical tool names that might be what was intended.
    """
    resolved = resolve_alias(tool_name)
    if resolved:
        return [resolved]

    # Partial match: substring search
    lower_name = tool_name.lower().strip()
    matches = set()
    for alias, target in TOOL_ALIASES.items():
        if lower_name in alias or alias in lower_name:
            matches.add(target)
    return sorted(matches)


def get_aliases_for_tool(tool_name: str) -> list[str]:
    """Get all known aliases that map to a given canonical tool name.

    Args:
        tool_name: A canonical Hermes tool name (e.g., "terminal").

    Returns:
        List of alias names that resolve to this tool.
    """
    return list(_REVERSE_INDEX.get(tool_name, []))


def get_all_canonical_tools() -> list[str]:
    """Return all unique canonical tool names referenced in the alias map."""
    return sorted(set(TOOL_ALIASES.values()))
