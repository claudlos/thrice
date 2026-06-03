"""Compatibility shim: ``skill_index_cache`` is the canonical import name.

The real implementation lives at ``modules/agent/skill_index_cache.py``
because the installer places it under ``<hermes>/agent/``.  This top-level
module re-exports the public API so tests and callers can use either
``import skill_index_cache`` or ``from agent.skill_index_cache import ...``.
"""

from importlib import util
from pathlib import Path

try:
    from agent.skill_index_cache import *  # type: ignore # noqa: F401,F403
    from agent.skill_index_cache import (  # type: ignore # noqa: F401
        SkillIndexCache,
        get_default_cache,
        reset_default_cache,
    )
except ImportError:
    impl_path = Path(__file__).resolve().parent / "agent" / "skill_index_cache.py"
    spec = util.spec_from_file_location("_thrice_agent_skill_index_cache", impl_path)
    if spec is None or spec.loader is None:
        raise
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    SkillIndexCache = module.SkillIndexCache
    get_default_cache = module.get_default_cache
    reset_default_cache = module.reset_default_cache

    for name in getattr(module, "__all__", ()):
        globals()[name] = getattr(module, name)

    __all__ = getattr(
        module,
        "__all__",
        ("SkillIndexCache", "get_default_cache", "reset_default_cache"),
    )
