"""
Adaptive Context Compression Threshold — Scale by model context window.

Current behavior: fixed compression threshold (e.g., 60%) regardless of model.
This means 200K-context models compress too early, wasting capacity, while
32K models may compress too late, causing truncation.

This module scales the compression trigger threshold based on the model's
context window size:
    200K+ tokens -> 75% (compress at 150K)
    128K tokens  -> 65% (compress at ~83K)
    64K tokens   -> 60% (compress at ~38K)
    32K tokens   -> 50% (compress at 16K)

Integration point: agent/context_compressor.py (~line 68)

Usage:
    from new_files.adaptive_compression import get_compression_threshold

    threshold = get_compression_threshold(model_context_size=200000)
    # Returns 0.75

    # Or look up by model name:
    threshold = get_compression_threshold(model_name="claude-sonnet-4-20250514")
    # Returns 0.75 (auto-detected 200K context)
"""

from typing import Optional

# ─── Context size -> compression threshold brackets ─────────────────────────
# Sorted descending by context_size. First matching bracket wins.
_THRESHOLD_BRACKETS: list[tuple[int, float]] = [
    (200_000, 0.75),   # 200K+ context -> compress at 75%
    (128_000, 0.65),   # 128K context  -> compress at 65%
    (64_000,  0.60),   # 64K context   -> compress at 60%
    (32_000,  0.50),   # 32K context   -> compress at 50%
    (16_000,  0.45),   # 16K context   -> compress at 45%
    (0,       0.40),   # Fallback      -> compress at 40%
]

# ─── Known model context sizes ──────────────────────────────────────────────
# Maps model name patterns to context window sizes (in tokens).
# Checked via substring match (case-insensitive).
_KNOWN_MODELS: dict[str, int] = {
    # Anthropic
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,

    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-0125": 128_000,
    "gpt-4-1106": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1-preview": 128_000,
    "o1-mini": 128_000,
    "o1": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,

    # Google
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-pro": 32_768,

    # Meta
    "llama-3.1-405b": 128_000,
    "llama-3.1-70b": 128_000,
    "llama-3.1-8b": 128_000,
    "llama-3-70b": 8_192,
    "llama-3-8b": 8_192,

    # Mistral
    "mistral-large": 128_000,
    "mistral-medium": 32_000,
    "mistral-small": 32_000,
    "mixtral-8x22b": 65_536,
    "mixtral-8x7b": 32_768,

    # DeepSeek
    "deepseek-v3": 128_000,
    "deepseek-r1": 128_000,
    "deepseek-coder": 128_000,

    # Cohere
    "command-r-plus": 128_000,
    "command-r": 128_000,

    # Qwen
    "qwen-2.5-72b": 128_000,
    "qwen-2.5-coder": 128_000,
}

# ─── Custom overrides (set at runtime) ──────────────────────────────────────
_custom_overrides: dict[str, float] = {}


def set_custom_threshold(model_name: str, threshold: float) -> None:
    """Set a custom compression threshold override for a specific model.

    Args:
        model_name: Model name (case-insensitive).
        threshold: Compression threshold as a fraction (0.0 to 1.0).

    Raises:
        ValueError: If threshold is outside valid range.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
    _custom_overrides[model_name.lower()] = threshold


def clear_custom_overrides() -> None:
    """Remove all custom threshold overrides."""
    _custom_overrides.clear()


def lookup_model_context_size(model_name: str) -> Optional[int]:
    """Look up the context window size for a known model.

    Uses substring matching against the known models table.

    Args:
        model_name: The model identifier string.

    Returns:
        Context window size in tokens, or None if model is unknown.
    """
    lower = model_name.lower()

    # Check longest matches first for specificity (e.g., "gpt-4o" before "gpt-4")
    sorted_models = sorted(_KNOWN_MODELS.keys(), key=len, reverse=True)
    for pattern in sorted_models:
        if pattern in lower:
            return _KNOWN_MODELS[pattern]

    return None


def _threshold_for_context_size(context_size: int) -> float:
    """Get the compression threshold for a given context window size."""
    for min_size, threshold in _THRESHOLD_BRACKETS:
        if context_size >= min_size:
            return threshold
    return _THRESHOLD_BRACKETS[-1][1]  # Fallback


def get_compression_threshold(
    model_context_size: Optional[int] = None,
    model_name: Optional[str] = None,
    default: float = 0.60,
) -> float:
    """Get the context compression threshold for a model.

    Resolution order:
    1. Custom override for model_name (if set via set_custom_threshold)
    2. Bracket lookup by model_context_size (if provided)
    3. Auto-detect context size from model_name via known models table
    4. Return default

    Args:
        model_context_size: Context window size in tokens (if known).
        model_name: Model identifier string (for auto-detection and overrides).
        default: Fallback threshold if nothing can be determined.

    Returns:
        Compression threshold as a fraction (e.g., 0.75 means compress at 75%).
    """
    # 1. Check custom overrides
    if model_name:
        lower = model_name.lower()
        if lower in _custom_overrides:
            return _custom_overrides[lower]

    # 2. Use explicit context size
    if model_context_size is not None and model_context_size > 0:
        return _threshold_for_context_size(model_context_size)

    # 3. Auto-detect from model name
    if model_name:
        detected_size = lookup_model_context_size(model_name)
        if detected_size is not None:
            return _threshold_for_context_size(detected_size)

    # 4. Fallback
    return default


def get_compression_trigger_tokens(
    model_context_size: int,
    model_name: Optional[str] = None,
) -> int:
    """Get the absolute token count at which compression should trigger.

    Convenience function that multiplies context size by threshold.

    Args:
        model_context_size: Total context window in tokens.
        model_name: Optional model name for override lookup.

    Returns:
        Number of tokens at which compression should be triggered.
    """
    threshold = get_compression_threshold(
        model_context_size=model_context_size,
        model_name=model_name,
    )
    return int(model_context_size * threshold)
