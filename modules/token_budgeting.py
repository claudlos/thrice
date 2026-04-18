"""
Predictive Token Budgeting (#18)

Predicts token usage and iteration counts for sessions based on task features.
Uses simple linear regression (no external dependencies) with heuristic fallbacks.
Logs session history for continuous improvement of predictions.

Classes:
    SessionFeatureExtractor - Extracts features from task descriptions
    BudgetPredictor - Predicts token/iteration budgets via linear regression
    BudgetAdvisor - High-level advice on budgets, costs, model selection
    SessionLogger - Logs session outcomes to ~/.hermes/budget_history.db
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """Numeric features extracted from a task + environment."""
    task_length: int = 0
    task_complexity: float = 0.0
    codebase_files: int = 0
    codebase_loc: int = 0
    available_tools: int = 0
    model_context_size: int = 128000
    estimated_turns: int = 1

    def to_list(self) -> List[float]:
        return [
            float(self.task_length),
            self.task_complexity,
            float(self.codebase_files),
            float(self.codebase_loc),
            float(self.available_tools),
            float(self.model_context_size),
            float(self.estimated_turns),
        ]


@dataclass
class BudgetPrediction:
    """Predicted resource usage for a session."""
    tokens: int = 0
    iterations: int = 1
    confidence: float = 0.0  # 0.0 – 1.0


@dataclass
class BudgetAdvice:
    """Human-friendly budget advice."""
    recommended_iterations: int = 10
    estimated_tokens: int = 50000
    estimated_cost: float = 0.0  # USD
    model_suggestion: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPLEXITY_KEYWORDS: Dict[str, float] = {
    # High complexity
    "refactor": 3.0, "migrate": 3.0, "redesign": 3.0, "architect": 3.0,
    "optimize": 2.5, "parallelize": 3.0, "distributed": 3.0,
    # Medium complexity
    "implement": 2.0, "create": 1.5, "build": 1.5, "integrate": 2.0,
    "debug": 2.0, "fix": 1.5, "test": 1.5, "add": 1.0,
    # Low complexity
    "update": 0.8, "rename": 0.5, "delete": 0.3, "list": 0.3,
    "read": 0.3, "check": 0.5, "format": 0.5, "lint": 0.5,
}

MODEL_CONTEXT_SIZES: Dict[str, int] = {
    "claude-opus-4-20250514": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-3-20250307": 200000,
    "gpt-4": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
}

# Cost per 1M tokens (input, output) in USD — approximate
MODEL_COSTS: Dict[str, Tuple[float, float]] = {
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-3-20250307": (0.25, 1.25),
    "gpt-4": (30.0, 60.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-3.5-turbo": (0.5, 1.5),
}

CHEAPER_ALTERNATIVES: Dict[str, List[str]] = {
    "claude-opus-4-20250514": ["claude-sonnet-4-20250514", "claude-haiku-3-20250307"],
    "claude-sonnet-4-20250514": ["claude-haiku-3-20250307"],
    "gpt-4": ["gpt-4o", "gpt-3.5-turbo"],
    "gpt-4o": ["gpt-3.5-turbo"],
}


# ---------------------------------------------------------------------------
# SessionFeatureExtractor
# ---------------------------------------------------------------------------

class SessionFeatureExtractor:
    """Extracts a FeatureVector from task description and environment info."""

    def extract(
        self,
        task_description: str,
        codebase_size: Optional[Dict[str, int]] = None,
        tool_count: int = 0,
        model: str = "claude-sonnet-4-20250514",
    ) -> FeatureVector:
        codebase_size = codebase_size or {}
        task_length = len(task_description)
        task_complexity = self._compute_complexity(task_description)
        codebase_files = codebase_size.get("files", 0)
        codebase_loc = codebase_size.get("loc", 0)
        model_context = MODEL_CONTEXT_SIZES.get(model, 128000)
        estimated_turns = self._estimate_turns(task_complexity, codebase_files, tool_count)

        return FeatureVector(
            task_length=task_length,
            task_complexity=task_complexity,
            codebase_files=codebase_files,
            codebase_loc=codebase_loc,
            available_tools=tool_count,
            model_context_size=model_context,
            estimated_turns=estimated_turns,
        )

    def _compute_complexity(self, text: str) -> float:
        text_lower = text.lower()
        words = re.findall(r"\w+", text_lower)
        score = 0.0
        matched = 0
        for word in words:
            if word in COMPLEXITY_KEYWORDS:
                score += COMPLEXITY_KEYWORDS[word]
                matched += 1
        # Base complexity from length
        length_factor = min(len(text) / 500.0, 3.0)
        if matched == 0:
            return length_factor
        return score / matched + length_factor

    def _estimate_turns(self, complexity: float, files: int, tools: int) -> int:
        base = max(1, int(complexity * 2))
        file_factor = min(files // 20, 10)
        tool_factor = min(tools // 3, 5)
        return base + file_factor + tool_factor


# ---------------------------------------------------------------------------
# Pure-Python Linear Regression
# ---------------------------------------------------------------------------

class SimpleLinearRegression:
    """Multivariate linear regression using ordinary least squares (normal equation).

    No external dependencies — pure Python + math stdlib.
    """

    def __init__(self) -> None:
        self.weights: Optional[List[float]] = None  # includes bias as last element
        self._n_features: int = 0

    def fit(self, X: List[List[float]], y: List[float], ridge: float = 1e-6) -> None:
        """Fit via normal equation: w = (X^T X + λI)^-1 X^T y.  X is augmented with 1s.

        Uses ridge regularization (default λ=1e-6) for numerical stability.
        """
        if len(X) < 2 or len(X) != len(y):
            raise ValueError("Need at least 2 samples and len(X)==len(y)")
        self._n_features = len(X[0])
        # Augment with bias column
        Xa = [row + [1.0] for row in X]
        n = len(Xa)
        m = len(Xa[0])

        # X^T X + ridge * I
        XtX = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                s = 0.0
                for k in range(n):
                    s += Xa[k][i] * Xa[k][j]
                XtX[i][j] = s
            XtX[i][i] += ridge  # Ridge regularization

        # X^T y
        Xty = [0.0] * m
        for i in range(m):
            s = 0.0
            for k in range(n):
                s += Xa[k][i] * y[k]
            Xty[i] = s

        # Invert XtX via Gauss-Jordan
        inv = self._invert(XtX)
        if inv is None:
            raise ValueError("Singular matrix — cannot fit")

        # w = inv * Xty
        self.weights = [0.0] * m
        for i in range(m):
            s = 0.0
            for j in range(m):
                s += inv[i][j] * Xty[j]
            self.weights[i] = s

    def predict(self, x: List[float]) -> float:
        if self.weights is None:
            raise RuntimeError("Model not fitted")
        xa = x + [1.0]
        return sum(w * v for w, v in zip(self.weights, xa, strict=True))

    def predict_batch(self, X: List[List[float]]) -> List[float]:
        return [self.predict(x) for x in X]

    @staticmethod
    def _invert(matrix: List[List[float]]) -> Optional[List[List[float]]]:
        """Gauss-Jordan matrix inversion."""
        n = len(matrix)
        # Augment with identity
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

        for col in range(n):
            # Partial pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                return None  # Singular
            for j in range(2 * n):
                aug[col][j] /= pivot
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

        return [row[n:] for row in aug]


# ---------------------------------------------------------------------------
# BudgetPredictor
# ---------------------------------------------------------------------------

class BudgetPredictor:
    """Predicts token and iteration budgets from features."""

    def __init__(self) -> None:
        self._token_model: Optional[SimpleLinearRegression] = None
        self._iter_model: Optional[SimpleLinearRegression] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, historical_sessions: List[Dict[str, Any]]) -> None:
        """Train on historical session data.

        Each session dict must contain:
            features: dict matching FeatureVector fields
            actual_tokens: int
            actual_iterations: int
        """
        if len(historical_sessions) < 3:
            return  # Not enough data; will use heuristic fallback

        X: List[List[float]] = []
        y_tokens: List[float] = []
        y_iters: List[float] = []

        for session in historical_sessions:
            fv = session.get("features", {})
            vec = FeatureVector(**{k: fv[k] for k in fv if k in FeatureVector.__dataclass_fields__})
            X.append(vec.to_list())
            y_tokens.append(float(session["actual_tokens"]))
            y_iters.append(float(session["actual_iterations"]))

        try:
            self._token_model = SimpleLinearRegression()
            self._token_model.fit(X, y_tokens)
            self._iter_model = SimpleLinearRegression()
            self._iter_model.fit(X, y_iters)
            self._trained = True
        except ValueError:
            self._trained = False

    def predict(self, features: FeatureVector) -> BudgetPrediction:
        """Predict budget. Uses trained model if available, else heuristic."""
        if self._trained and self._token_model and self._iter_model:
            return self._predict_trained(features)
        return self._predict_heuristic(features)

    def _predict_trained(self, features: FeatureVector) -> BudgetPrediction:
        x = features.to_list()
        tokens = max(1000, int(self._token_model.predict(x)))
        iters = max(1, int(self._iter_model.predict(x)))
        return BudgetPrediction(tokens=tokens, iterations=iters, confidence=0.7)

    def _predict_heuristic(self, features: FeatureVector) -> BudgetPrediction:
        """Heuristic fallback based on task length and tool count."""
        # Base tokens: ~500 tokens per 100 chars of task description
        base_tokens = (features.task_length / 100) * 500
        # Complexity multiplier
        complexity_mult = 1.0 + features.task_complexity * 0.5
        # Tool usage adds tokens
        tool_tokens = features.available_tools * 800
        # Codebase reading
        code_tokens = min(features.codebase_loc * 2, features.model_context_size * 0.3)

        total_tokens = int((base_tokens * complexity_mult + tool_tokens + code_tokens) * 1.2)
        total_tokens = max(2000, min(total_tokens, features.model_context_size * 5))

        iterations = features.estimated_turns
        iterations = max(1, min(iterations, 200))

        return BudgetPrediction(
            tokens=total_tokens,
            iterations=iterations,
            confidence=0.3,
        )


# ---------------------------------------------------------------------------
# BudgetAdvisor
# ---------------------------------------------------------------------------

class BudgetAdvisor:
    """High-level budget advice combining extraction, prediction, and cost estimation."""

    def __init__(self, predictor: Optional[BudgetPredictor] = None) -> None:
        self.extractor = SessionFeatureExtractor()
        self.predictor = predictor or BudgetPredictor()

    def suggest_budget(
        self,
        task: str,
        codebase: Optional[Dict[str, int]] = None,
        tools: int = 0,
        model: str = "claude-sonnet-4-20250514",
    ) -> BudgetAdvice:
        features = self.extractor.extract(task, codebase, tools, model)
        prediction = self.predictor.predict(features)

        # Estimate cost
        costs = MODEL_COSTS.get(model, (5.0, 15.0))
        # Assume 40% input, 60% output token split
        input_tokens = int(prediction.tokens * 0.4)
        output_tokens = int(prediction.tokens * 0.6)
        cost = (input_tokens / 1_000_000) * costs[0] + (output_tokens / 1_000_000) * costs[1]

        advice = BudgetAdvice(
            recommended_iterations=prediction.iterations,
            estimated_tokens=prediction.tokens,
            estimated_cost=round(cost, 4),
            model_suggestion=model,
        )

        # Add warnings
        warning = self.warn_if_likely_expensive(task, model, prediction)
        if warning:
            advice.warnings.append(warning)

        cheaper = self.suggest_cheaper_model(task, model)
        if cheaper:
            advice.model_suggestion = cheaper
            advice.warnings.append(f"Consider using {cheaper} for this task to reduce cost.")

        return advice

    def warn_if_likely_expensive(
        self,
        task: str,
        model: str = "claude-sonnet-4-20250514",
        prediction: Optional[BudgetPrediction] = None,
    ) -> Optional[str]:
        if prediction is None:
            features = self.extractor.extract(task, model=model)
            prediction = self.predictor.predict(features)

        costs = MODEL_COSTS.get(model, (5.0, 15.0))
        est_cost = (prediction.tokens / 1_000_000) * ((costs[0] + costs[1]) / 2)

        if est_cost > 1.0:
            return f"Warning: estimated cost ${est_cost:.2f} exceeds $1.00 threshold."
        if prediction.iterations > 50:
            return f"Warning: estimated {prediction.iterations} iterations — this may be a long-running task."
        return None

    def suggest_cheaper_model(self, task: str, current_model: str) -> Optional[str]:
        """Suggest a cheaper model if the task is simple enough."""
        features = self.extractor.extract(task, model=current_model)
        if features.task_complexity < 2.0 and features.task_length < 300:
            alternatives = CHEAPER_ALTERNATIVES.get(current_model, [])
            if alternatives:
                return alternatives[0]
        return None


# ---------------------------------------------------------------------------
# SessionLogger
# ---------------------------------------------------------------------------

class SessionLogger:
    """Logs session outcomes to SQLite for future training."""

    DEFAULT_DB_PATH = os.path.expanduser("~/.hermes/budget_history.db")

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self._ensure_db()

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    features TEXT,
                    actual_tokens INTEGER,
                    actual_iterations INTEGER,
                    outcome TEXT
                )
            """)
            conn.commit()

    def log_session(
        self,
        features: FeatureVector,
        actual_tokens: int,
        actual_iterations: int,
        outcome: str = "success",
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (timestamp, features, actual_tokens, actual_iterations, outcome) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), json.dumps(asdict(features)), actual_tokens, actual_iterations, outcome),
            )
            conn.commit()

    def get_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT features, actual_tokens, actual_iterations, outcome "
                "FROM sessions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        result = []
        for features_json, tokens, iters, outcome in rows:
            result.append({
                "features": json.loads(features_json),
                "actual_tokens": tokens,
                "actual_iterations": iters,
                "outcome": outcome,
            })
        return result

    def clear_history(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions")
            conn.commit()

    def session_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
