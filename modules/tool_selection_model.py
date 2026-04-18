"""
Probabilistic Tool Selection Model for Hermes.

Learns from past tool usage patterns to predict which tool is most appropriate
for a given user intent. Uses TF-IDF + logistic regression with a frequency-based
fallback for small datasets.

Usage:
    # Logging
    log = ToolSelectionLogger("tool_calls.db")
    log.log_tool_call("read the file", "read_file", {"path": "x.py"}, True, 0.3)

    # Training
    model = ToolSelectionModel(log)
    model.train()

    # Prediction
    selector = ToolSelector(model)
    suggestion = selector.suggest_tool("search for errors in logs", ["grep", "read_file", "search"])
"""

import json
import logging
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """A single logged tool call."""
    id: int
    timestamp: float
    intent: str
    tool_name: str
    args_json: str
    success: bool
    duration: float


class ToolSelectionLogger:
    """
    Logs tool calls to a SQLite database for later analysis and model training.
    """

    def __init__(self, db_path: str = "tool_calls.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the database table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    intent TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    args_json TEXT DEFAULT '{}',
                    success INTEGER NOT NULL DEFAULT 1,
                    duration REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_calls_tool
                ON tool_calls(tool_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_calls_timestamp
                ON tool_calls(timestamp)
            """)
            conn.commit()

    def log_tool_call(self, intent: str, tool_name: str,
                      args: Optional[Dict] = None,
                      success: bool = True, duration: float = 0.0) -> int:
        """
        Log a tool call. Returns the record ID.
        """
        args_json = json.dumps(args or {})
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO tool_calls (timestamp, intent, tool_name, args_json, success, duration) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), intent, tool_name, args_json, int(success), duration),
            )
            conn.commit()
            return cursor.lastrowid

    def get_all_records(self, limit: int = 10000) -> List[ToolCallRecord]:
        """Retrieve all records."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, intent, tool_name, args_json, success, duration "
                "FROM tool_calls ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [ToolCallRecord(*row) for row in rows]

    def get_record_count(self) -> int:
        """Get total number of records."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]

    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-tool statistics."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT tool_name,
                       COUNT(*) as total,
                       SUM(success) as successes,
                       AVG(duration) as avg_duration
                FROM tool_calls
                GROUP BY tool_name
                ORDER BY total DESC
            """).fetchall()
            return {
                row[0]: {
                    "total": row[1],
                    "successes": row[2],
                    "success_rate": row[2] / row[1] if row[1] > 0 else 0,
                    "avg_duration": row[3],
                }
                for row in rows
            }


class _TfIdf:
    """
    Minimal TF-IDF implementation (no sklearn dependency).
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        import re
        tokens = re.findall(r'[a-z0-9_]+', text.lower())
        return tokens

    def fit(self, documents: List[str]):
        """Build vocabulary and compute IDF from documents."""
        doc_freq: Counter = Counter()
        vocab_set: set = set()
        n_docs = len(documents)

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1
                vocab_set.add(token)

        # Build vocabulary mapping
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(vocab_set))}

        # Compute IDF: log((1 + n) / (1 + df)) + 1  (smooth IDF)
        self.idf = {}
        for token, df in doc_freq.items():
            self.idf[token] = math.log((1 + n_docs) / (1 + df)) + 1

        self._fitted = True

    def transform(self, documents: List[str]) -> List[Dict[str, float]]:
        """Transform documents to TF-IDF sparse vectors (as dicts)."""
        if not self._fitted:
            raise RuntimeError("TfIdf not fitted yet")

        results = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1

            vec = {}
            for token, count in tf.items():
                if token in self.vocabulary:
                    tfidf = (count / total) * self.idf.get(token, 1.0)
                    vec[token] = tfidf
            results.append(vec)
        return results

    def fit_transform(self, documents: List[str]) -> List[Dict[str, float]]:
        self.fit(documents)
        return self.transform(documents)


class _LogisticClassifier:
    """
    Minimal multi-class logistic regression (one-vs-rest) using SGD.
    No external dependencies.
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 50):
        self.lr = learning_rate
        self.epochs = epochs
        self.classes: List[str] = []
        self.weights: Dict[str, Dict[str, float]] = {}  # class -> {feature: weight}
        self.biases: Dict[str, float] = {}

    def _sigmoid(self, x: float) -> float:
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def _dot(self, vec: Dict[str, float], weights: Dict[str, float]) -> float:
        total = 0.0
        for k, v in vec.items():
            if k in weights:
                total += v * weights[k]
        return total

    def fit(self, X: List[Dict[str, float]], y: List[str]):
        """Train one-vs-rest classifiers."""
        self.classes = sorted(set(y))
        if len(self.classes) < 2:
            self.classes = list(set(y))

        for cls in self.classes:
            weights: Dict[str, float] = {}
            bias = 0.0
            binary_y = [1.0 if label == cls else 0.0 for label in y]

            for _ in range(self.epochs):
                for vec, target in zip(X, binary_y, strict=True):
                    pred = self._sigmoid(self._dot(vec, weights) + bias)
                    error = pred - target
                    # Update weights
                    for feature, value in vec.items():
                        weights[feature] = weights.get(feature, 0.0) - self.lr * error * value
                    bias -= self.lr * error

            self.weights[cls] = weights
            self.biases[cls] = bias

    def predict_proba(self, vec: Dict[str, float]) -> Dict[str, float]:
        """Return probability estimates for each class."""
        scores = {}
        for cls in self.classes:
            scores[cls] = self._sigmoid(
                self._dot(vec, self.weights.get(cls, {})) + self.biases.get(cls, 0.0)
            )

        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores

    def predict(self, vec: Dict[str, float]) -> str:
        """Predict the most likely class."""
        probs = self.predict_proba(vec)
        return max(probs, key=probs.get)


class ToolSelectionModel:
    """
    Learns to predict which tool to use based on intent text.
    Uses TF-IDF + logistic regression, with frequency-based fallback.
    """

    def __init__(self, log: Optional[ToolSelectionLogger] = None):
        self.logger = log
        self.tfidf = _TfIdf()
        self.classifier = _LogisticClassifier()
        self._trained = False
        self._frequency_ranking: List[Tuple[str, float]] = []
        self._min_samples_used = 100

    def train(self, min_samples: int = 100) -> bool:
        """
        Build model from logged data.
        Returns True if ML model was trained, False if using frequency fallback.
        """
        self._min_samples_used = min_samples

        if not self.logger:
            logger.warning("No logger configured, cannot train")
            return False

        records = self.logger.get_all_records(limit=50000)
        if not records:
            logger.warning("No training data available")
            return False

        # Build frequency ranking (always available as fallback)
        tool_counts = Counter(r.tool_name for r in records if r.success)
        total = sum(tool_counts.values()) or 1
        self._frequency_ranking = [
            (tool, count / total) for tool, count in tool_counts.most_common()
        ]

        # Only train ML model if we have enough samples
        successful = [r for r in records if r.success]
        if len(successful) < min_samples:
            logger.info("Only %d samples (need %d), using frequency fallback",
                        len(successful), min_samples)
            self._trained = False
            return False

        # Train TF-IDF + classifier
        intents = [r.intent for r in successful]
        labels = [r.tool_name for r in successful]

        X = self.tfidf.fit_transform(intents)
        self.classifier.fit(X, labels)
        self._trained = True
        logger.info("Model trained on %d samples, %d tools",
                     len(successful), len(set(labels)))
        return True

    def predict(self, intent: str,
                available_tools: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Predict tool probabilities for the given intent.
        Returns list of (tool_name, probability) sorted by probability descending.
        """
        if self._trained:
            vec = self.tfidf.transform([intent])[0]
            probs = self.classifier.predict_proba(vec)

            # Filter to available tools if specified
            if available_tools:
                probs = {k: v for k, v in probs.items() if k in available_tools}

            # Normalize
            total = sum(probs.values()) or 1
            ranked = sorted(
                [(k, v / total) for k, v in probs.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked

        # Frequency fallback
        if available_tools:
            filtered = [(t, p) for t, p in self._frequency_ranking if t in available_tools]
        else:
            filtered = self._frequency_ranking

        if not filtered:
            # Uniform distribution over available tools
            if available_tools:
                p = 1.0 / len(available_tools)
                return [(t, p) for t in available_tools]
            return []

        # Normalize filtered frequencies
        total = sum(p for _, p in filtered) or 1
        return [(t, p / total) for t, p in filtered]

    @property
    def is_trained(self) -> bool:
        return self._trained


class ToolSelector:
    """
    Integration layer: uses the model to suggest tools and provide stats.
    """

    def __init__(self, model: ToolSelectionModel):
        self.model = model

    def suggest_tool(self, user_message: str,
                     available_tools: List[str]) -> Optional[str]:
        """
        Suggest the best tool for the user message.
        Returns None if confidence is too low.
        """
        if not available_tools:
            return None

        predictions = self.model.predict(user_message, available_tools)
        if not predictions:
            return None

        top_tool, top_prob = predictions[0]

        # Only suggest if we have reasonable confidence
        # With many tools, even 0.2 can be significant
        threshold = max(0.15, 1.0 / (len(available_tools) * 1.5))
        if top_prob >= threshold:
            return top_tool

        return None

    def should_override_model_choice(self, model_choice: str,
                                     our_prediction: str,
                                     confidence: float = 0.0) -> bool:
        """
        Determine if our prediction should override the LLM's tool choice.
        Conservative: only override with high confidence on clear mistakes.
        """
        if model_choice == our_prediction:
            return False

        # Only override with very high confidence
        return confidence > 0.85

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        stats: Dict[str, Any] = {
            "model_trained": self.model.is_trained,
            "has_frequency_data": bool(self.model._frequency_ranking),
        }

        if self.model.logger:
            stats["total_records"] = self.model.logger.get_record_count()
            stats["tool_stats"] = self.model.logger.get_tool_stats()

        if self.model._frequency_ranking:
            stats["top_tools"] = self.model._frequency_ranking[:10]

        return stats
