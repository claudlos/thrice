"""Compatibility shim: ``skill_index_cache`` is the canonical import name.

The real implementation lives at ``modules/agent/skill_index_cache.py``
because the installer places it under ``<hermes>/agent/``.  This top-level
module re-exports the public API so tests and callers can use either
``import skill_index_cache`` or ``from agent.skill_index_cache import ...``.
"""
from agent.skill_index_cache import *  # noqa: F401,F403
from agent.skill_index_cache import (  # noqa: F401 (explicit re-export)
    SkillIndexCache,
    get_default_cache,
    reset_default_cache,
)
