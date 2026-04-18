"""Tests for tool_selection_model.py — Probabilistic Tool Selection."""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))
from tool_selection_model import (
    ToolCallRecord,
    ToolSelectionLogger,
    ToolSelectionModel,
    ToolSelector,
    _LogisticClassifier,
    _TfIdf,
)

# Synthetic data: intent patterns mapped to expected tools
SYNTHETIC_DATA = [
    # read_file patterns
    ("read the contents of main.py", "read_file"),
    ("show me the file src/utils.py", "read_file"),
    ("what's in config.yaml", "read_file"),
    ("open the readme file", "read_file"),
    ("display the source code", "read_file"),
    ("cat the log file", "read_file"),
    ("view the configuration", "read_file"),
    ("print the contents of index.html", "read_file"),
    # search patterns
    ("find all TODO comments in the codebase", "search"),
    ("search for error handling", "search"),
    ("grep for import statements", "search"),
    ("look for references to UserModel", "search"),
    ("find where the function is defined", "search"),
    ("search the project for deprecated calls", "search"),
    ("find all files matching pattern", "search"),
    ("locate the database configuration", "search"),
    # write_file patterns
    ("create a new file called server.py", "write_file"),
    ("write the test to tests/test_main.py", "write_file"),
    ("save this content to output.txt", "write_file"),
    ("generate a dockerfile", "write_file"),
    ("create the configuration file", "write_file"),
    ("write a new module", "write_file"),
    ("make a new script", "write_file"),
    ("add a new python file", "write_file"),
    # terminal patterns
    ("run the test suite", "terminal"),
    ("execute npm install", "terminal"),
    ("build the project", "terminal"),
    ("start the development server", "terminal"),
    ("install the dependencies", "terminal"),
    ("compile the code", "terminal"),
    ("run pytest", "terminal"),
    ("execute the migration", "terminal"),
    # patch patterns
    ("fix the bug on line 42", "patch"),
    ("replace the old function name", "patch"),
    ("update the import statement", "patch"),
    ("change the variable name", "patch"),
    ("modify the existing code", "patch"),
    ("edit the function to add error handling", "patch"),
    ("refactor the class method", "patch"),
    ("update the return value", "patch"),
]

AVAILABLE_TOOLS = ["read_file", "search", "write_file", "terminal", "patch"]


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_tool_calls.db")


@pytest.fixture
def logger_with_data(db_path):
    """Logger pre-populated with synthetic data."""
    log = ToolSelectionLogger(db_path)
    # Add enough data for ML training (repeat patterns with variation)
    random.seed(42)
    for _ in range(5):  # 5 * 40 = 200 records
        for intent, tool in SYNTHETIC_DATA:
            # Add slight variation
            variations = [intent, intent.upper(), f"please {intent}", f"can you {intent}"]
            chosen = random.choice(variations)
            success = random.random() > 0.1  # 90% success rate
            duration = random.uniform(0.1, 2.0)
            log.log_tool_call(chosen, tool, {"test": True}, success, duration)
    return log


class TestToolSelectionLogger:
    def test_create_and_log(self, db_path):
        log = ToolSelectionLogger(db_path)
        rid = log.log_tool_call("read file x", "read_file", {"path": "x.py"}, True, 0.5)
        assert rid >= 1

    def test_get_records(self, db_path):
        log = ToolSelectionLogger(db_path)
        log.log_tool_call("intent1", "tool1", {}, True, 0.1)
        log.log_tool_call("intent2", "tool2", {}, False, 0.2)

        records = log.get_all_records()
        assert len(records) == 2
        assert all(isinstance(r, ToolCallRecord) for r in records)

    def test_record_count(self, db_path):
        log = ToolSelectionLogger(db_path)
        for i in range(5):
            log.log_tool_call(f"intent {i}", "tool", {}, True, 0.1)
        assert log.get_record_count() == 5

    def test_tool_stats(self, db_path):
        log = ToolSelectionLogger(db_path)
        for _ in range(10):
            log.log_tool_call("read", "read_file", {}, True, 0.1)
        for _ in range(5):
            log.log_tool_call("write", "write_file", {}, True, 0.2)
        log.log_tool_call("write fail", "write_file", {}, False, 0.3)

        stats = log.get_tool_stats()
        assert "read_file" in stats
        assert stats["read_file"]["total"] == 10
        assert stats["read_file"]["success_rate"] == 1.0
        assert stats["write_file"]["total"] == 6

    def test_empty_db(self, db_path):
        log = ToolSelectionLogger(db_path)
        assert log.get_record_count() == 0
        assert log.get_all_records() == []
        assert log.get_tool_stats() == {}


class TestTfIdf:
    def test_basic_transform(self):
        tfidf = _TfIdf()
        docs = ["hello world", "hello python", "world python code"]
        vecs = tfidf.fit_transform(docs)
        assert len(vecs) == 3
        # Each vector should have some entries
        for v in vecs:
            assert len(v) > 0

    def test_different_docs_different_vectors(self):
        tfidf = _TfIdf()
        docs = ["read the file", "run the tests", "search for errors"]
        vecs = tfidf.fit_transform(docs)
        # Vectors should differ
        assert vecs[0] != vecs[1]
        assert vecs[1] != vecs[2]

    def test_empty_document(self):
        tfidf = _TfIdf()
        tfidf.fit(["hello world"])
        vecs = tfidf.transform([""])
        assert vecs[0] == {}


class TestLogisticClassifier:
    def test_simple_classification(self):
        clf = _LogisticClassifier(learning_rate=0.5, epochs=100)
        # Simple separable data
        X = [
            {"feature_a": 1.0, "feature_b": 0.0},
            {"feature_a": 0.9, "feature_b": 0.1},
            {"feature_a": 0.0, "feature_b": 1.0},
            {"feature_a": 0.1, "feature_b": 0.9},
        ]
        y = ["A", "A", "B", "B"]
        clf.fit(X, y)

        # Should predict correctly on training data
        pred = clf.predict({"feature_a": 1.0, "feature_b": 0.0})
        assert pred == "A"
        pred = clf.predict({"feature_a": 0.0, "feature_b": 1.0})
        assert pred == "B"

    def test_predict_proba_sums_to_one(self):
        clf = _LogisticClassifier()
        X = [{"a": 1.0}, {"b": 1.0}]
        y = ["X", "Y"]
        clf.fit(X, y)

        probs = clf.predict_proba({"a": 0.5, "b": 0.5})
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


class TestToolSelectionModel:
    def test_train_with_enough_data(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        result = model.train(min_samples=100)
        assert result is True
        assert model.is_trained is True

    def test_train_with_insufficient_data(self, db_path):
        log = ToolSelectionLogger(db_path)
        # Add only a few records
        for i in range(5):
            log.log_tool_call(f"intent {i}", "tool1", {}, True, 0.1)

        model = ToolSelectionModel(log)
        result = model.train(min_samples=100)
        assert result is False
        assert model.is_trained is False

    def test_predict_with_trained_model(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        model.train(min_samples=50)

        predictions = model.predict("read the file contents", AVAILABLE_TOOLS)
        assert len(predictions) > 0
        # Should be sorted by probability descending
        probs = [p for _, p in predictions]
        assert probs == sorted(probs, reverse=True)
        # Probabilities should sum to ~1
        assert abs(sum(probs) - 1.0) < 0.05

    def test_predict_with_frequency_fallback(self, db_path):
        log = ToolSelectionLogger(db_path)
        # Add just a few records
        for _ in range(10):
            log.log_tool_call("read stuff", "read_file", {}, True, 0.1)
        for _ in range(5):
            log.log_tool_call("search", "search", {}, True, 0.1)

        model = ToolSelectionModel(log)
        model.train(min_samples=1000)  # Force fallback

        predictions = model.predict("anything", ["read_file", "search"])
        assert len(predictions) == 2
        # read_file should rank higher (more frequent)
        assert predictions[0][0] == "read_file"

    def test_predict_filters_available_tools(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        model.train(min_samples=50)

        predictions = model.predict("read the file", ["read_file", "terminal"])
        tools = [t for t, _ in predictions]
        assert all(t in ["read_file", "terminal"] for t in tools)

    def test_predict_no_training(self, db_path):
        log = ToolSelectionLogger(db_path)
        model = ToolSelectionModel(log)
        # No training at all
        predictions = model.predict("something", ["tool1", "tool2"])
        assert len(predictions) == 2
        # Should be uniform
        for _, prob in predictions:
            assert abs(prob - 0.5) < 0.01

    def test_model_accuracy_on_clear_patterns(self, logger_with_data):
        """The model should get clear-cut patterns right."""
        model = ToolSelectionModel(logger_with_data)
        model.train(min_samples=50)

        test_cases = [
            ("read the source file", "read_file"),
            ("search for the function definition", "search"),
            ("create a new python file", "write_file"),
            ("run the tests", "terminal"),
            ("fix the bug in the code", "patch"),
        ]

        correct = 0
        for intent, expected in test_cases:
            preds = model.predict(intent, AVAILABLE_TOOLS)
            if preds and preds[0][0] == expected:
                correct += 1

        # Should get at least 3/5 right
        assert correct >= 3, f"Only got {correct}/5 correct"


class TestToolSelector:
    def test_suggest_tool(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        model.train(min_samples=50)
        selector = ToolSelector(model)

        suggestion = selector.suggest_tool("read the file", AVAILABLE_TOOLS)
        assert suggestion is None or suggestion in AVAILABLE_TOOLS

    def test_suggest_tool_empty_tools(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        selector = ToolSelector(model)
        assert selector.suggest_tool("anything", []) is None

    def test_should_override_same_choice(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        selector = ToolSelector(model)
        assert selector.should_override_model_choice("read_file", "read_file") is False

    def test_should_override_low_confidence(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        selector = ToolSelector(model)
        assert selector.should_override_model_choice("read_file", "search", 0.5) is False

    def test_should_override_high_confidence(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        selector = ToolSelector(model)
        assert selector.should_override_model_choice("read_file", "search", 0.9) is True

    def test_get_stats(self, logger_with_data):
        model = ToolSelectionModel(logger_with_data)
        model.train(min_samples=50)
        selector = ToolSelector(model)

        stats = selector.get_stats()
        assert "model_trained" in stats
        assert "total_records" in stats
        assert "tool_stats" in stats
        assert stats["model_trained"] is True
        assert stats["total_records"] > 0

    def test_get_stats_untrained(self, db_path):
        log = ToolSelectionLogger(db_path)
        model = ToolSelectionModel(log)
        selector = ToolSelector(model)

        stats = selector.get_stats()
        assert stats["model_trained"] is False
