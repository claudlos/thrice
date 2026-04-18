"""Tests for Predictive Token Budgeting (#18)."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from token_budgeting import (
    BudgetAdvisor,
    BudgetPrediction,
    BudgetPredictor,
    FeatureVector,
    SessionFeatureExtractor,
    SessionLogger,
    SimpleLinearRegression,
)

# ---------------------------------------------------------------------------
# FeatureVector
# ---------------------------------------------------------------------------

class TestFeatureVector:
    def test_defaults(self):
        fv = FeatureVector()
        assert fv.task_length == 0
        assert fv.task_complexity == 0.0
        assert fv.model_context_size == 128000

    def test_to_list_length(self):
        fv = FeatureVector(task_length=100, task_complexity=2.5)
        lst = fv.to_list()
        assert len(lst) == 7
        assert lst[0] == 100.0
        assert lst[1] == 2.5

    def test_to_list_all_floats(self):
        fv = FeatureVector(task_length=50, codebase_files=10)
        assert all(isinstance(v, float) for v in fv.to_list())


# ---------------------------------------------------------------------------
# SessionFeatureExtractor
# ---------------------------------------------------------------------------

class TestSessionFeatureExtractor:
    def setup_method(self):
        self.extractor = SessionFeatureExtractor()

    def test_basic_extraction(self):
        fv = self.extractor.extract("implement a REST API", tool_count=5)
        assert fv.task_length == len("implement a REST API")
        assert fv.available_tools == 5
        assert fv.task_complexity > 0

    def test_complexity_high_keywords(self):
        fv_high = self.extractor.extract("refactor and redesign the entire architecture")
        fv_low = self.extractor.extract("rename a variable")
        assert fv_high.task_complexity > fv_low.task_complexity

    def test_complexity_no_keywords(self):
        fv = self.extractor.extract("hello world xyz")
        # Should still get length-based complexity
        assert fv.task_complexity > 0

    def test_codebase_size(self):
        fv = self.extractor.extract("fix bug", codebase_size={"files": 100, "loc": 50000})
        assert fv.codebase_files == 100
        assert fv.codebase_loc == 50000

    def test_model_context_size(self):
        fv = self.extractor.extract("test", model="gpt-3.5-turbo")
        assert fv.model_context_size == 16385

    def test_unknown_model(self):
        fv = self.extractor.extract("test", model="unknown-model-v99")
        assert fv.model_context_size == 128000  # default

    def test_estimated_turns_increases_with_complexity(self):
        fv_simple = self.extractor.extract("list files")
        fv_complex = self.extractor.extract(
            "refactor and migrate the distributed architecture",
            codebase_size={"files": 200, "loc": 100000},
            tool_count=15,
        )
        assert fv_complex.estimated_turns > fv_simple.estimated_turns


# ---------------------------------------------------------------------------
# SimpleLinearRegression
# ---------------------------------------------------------------------------

class TestSimpleLinearRegression:
    def test_simple_fit_predict(self):
        # y = 2*x + 1
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        model = SimpleLinearRegression()
        model.fit(X, y)
        pred = model.predict([6.0])
        assert abs(pred - 13.0) < 0.01

    def test_multivariate(self):
        # y = x1 + 2*x2
        X = [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 3.0]]
        y = [3.0, 4.0, 5.0, 9.0]
        model = SimpleLinearRegression()
        model.fit(X, y)
        pred = model.predict([2.0, 2.0])
        assert abs(pred - 6.0) < 0.1

    def test_too_few_samples(self):
        model = SimpleLinearRegression()
        with pytest.raises(ValueError):
            model.fit([[1.0]], [1.0])

    def test_predict_before_fit(self):
        model = SimpleLinearRegression()
        with pytest.raises(RuntimeError):
            model.predict([1.0])

    def test_predict_batch(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [2.0, 4.0, 6.0, 8.0]
        model = SimpleLinearRegression()
        model.fit(X, y)
        preds = model.predict_batch([[5.0], [10.0]])
        assert abs(preds[0] - 10.0) < 0.1
        assert abs(preds[1] - 20.0) < 0.1


# ---------------------------------------------------------------------------
# BudgetPredictor
# ---------------------------------------------------------------------------

class TestBudgetPredictor:
    def test_heuristic_fallback(self):
        predictor = BudgetPredictor()
        assert not predictor.is_trained
        fv = FeatureVector(task_length=200, task_complexity=2.0, available_tools=5, estimated_turns=4)
        pred = predictor.predict(fv)
        assert pred.tokens > 0
        assert pred.iterations >= 1
        assert pred.confidence == 0.3  # heuristic confidence

    def test_train_and_predict(self):
        predictor = BudgetPredictor()
        sessions = [
            {"features": {"task_length": 100, "task_complexity": 1.0, "codebase_files": 10,
                          "codebase_loc": 1000, "available_tools": 3, "model_context_size": 128000,
                          "estimated_turns": 3},
             "actual_tokens": 5000, "actual_iterations": 3},
            {"features": {"task_length": 200, "task_complexity": 2.0, "codebase_files": 20,
                          "codebase_loc": 5000, "available_tools": 5, "model_context_size": 128000,
                          "estimated_turns": 6},
             "actual_tokens": 15000, "actual_iterations": 7},
            {"features": {"task_length": 500, "task_complexity": 3.0, "codebase_files": 50,
                          "codebase_loc": 20000, "available_tools": 10, "model_context_size": 200000,
                          "estimated_turns": 12},
             "actual_tokens": 50000, "actual_iterations": 15},
            {"features": {"task_length": 300, "task_complexity": 2.5, "codebase_files": 30,
                          "codebase_loc": 10000, "available_tools": 7, "model_context_size": 128000,
                          "estimated_turns": 8},
             "actual_tokens": 25000, "actual_iterations": 10},
        ]
        predictor.train(sessions)
        assert predictor.is_trained

        fv = FeatureVector(task_length=250, task_complexity=2.0, codebase_files=25,
                          codebase_loc=8000, available_tools=6, model_context_size=128000,
                          estimated_turns=7)
        pred = predictor.predict(fv)
        assert pred.tokens >= 1000
        assert pred.iterations >= 1
        assert pred.confidence == 0.7

    def test_train_insufficient_data(self):
        predictor = BudgetPredictor()
        predictor.train([{"features": {}, "actual_tokens": 100, "actual_iterations": 1}])
        assert not predictor.is_trained

    def test_heuristic_minimum_tokens(self):
        predictor = BudgetPredictor()
        fv = FeatureVector(task_length=1, task_complexity=0.0, estimated_turns=1)
        pred = predictor.predict(fv)
        assert pred.tokens >= 2000

    def test_heuristic_caps_at_context(self):
        predictor = BudgetPredictor()
        fv = FeatureVector(
            task_length=10000, task_complexity=5.0,
            codebase_loc=1000000, available_tools=50,
            model_context_size=128000, estimated_turns=100,
        )
        pred = predictor.predict(fv)
        assert pred.tokens <= 128000 * 5


# ---------------------------------------------------------------------------
# BudgetAdvisor
# ---------------------------------------------------------------------------

class TestBudgetAdvisor:
    def setup_method(self):
        self.advisor = BudgetAdvisor()

    def test_suggest_budget_basic(self):
        advice = self.advisor.suggest_budget("implement user login", tools=5)
        assert advice.recommended_iterations >= 1
        assert advice.estimated_tokens > 0
        assert advice.estimated_cost >= 0

    def test_suggest_budget_with_codebase(self):
        advice = self.advisor.suggest_budget(
            "refactor the database layer",
            codebase={"files": 100, "loc": 50000},
            tools=10,
            model="claude-opus-4-20250514",
        )
        assert advice.estimated_tokens > 0
        assert advice.estimated_cost > 0

    def test_warn_if_likely_expensive_cheap(self):
        warning = self.advisor.warn_if_likely_expensive("rename variable")
        assert warning is None

    def test_warn_if_likely_expensive_costly(self):
        # Create a prediction that would be expensive
        pred = BudgetPrediction(tokens=5_000_000, iterations=100, confidence=0.5)
        warning = self.advisor.warn_if_likely_expensive("big task", prediction=pred)
        assert warning is not None
        assert "$" in warning or "iterations" in warning

    def test_suggest_cheaper_model_simple_task(self):
        result = self.advisor.suggest_cheaper_model("list files", "claude-opus-4-20250514")
        assert result is not None
        assert "sonnet" in result or "haiku" in result

    def test_suggest_cheaper_model_complex_task(self):
        result = self.advisor.suggest_cheaper_model(
            "refactor and migrate the entire distributed architecture for maximum performance " * 3,
            "claude-opus-4-20250514",
        )
        assert result is None  # too complex for cheaper model

    def test_suggest_cheaper_no_alternative(self):
        result = self.advisor.suggest_cheaper_model("list files", "gpt-3.5-turbo")
        assert result is None

    def test_warnings_in_advice(self):
        # Simple task on expensive model should suggest cheaper
        advice = self.advisor.suggest_budget("list files", tools=1, model="claude-opus-4-20250514")
        assert len(advice.warnings) > 0


# ---------------------------------------------------------------------------
# SessionLogger
# ---------------------------------------------------------------------------

class TestSessionLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_budget.db")
        self.logger = SessionLogger(db_path=self.db_path)

    def test_log_and_count(self):
        fv = FeatureVector(task_length=100, task_complexity=1.5)
        self.logger.log_session(fv, actual_tokens=5000, actual_iterations=3)
        assert self.logger.session_count() == 1

    def test_log_multiple(self):
        for i in range(5):
            fv = FeatureVector(task_length=100 * (i + 1))
            self.logger.log_session(fv, actual_tokens=1000 * (i + 1), actual_iterations=i + 1)
        assert self.logger.session_count() == 5

    def test_get_history(self):
        fv = FeatureVector(task_length=200, task_complexity=2.0, available_tools=5)
        self.logger.log_session(fv, actual_tokens=10000, actual_iterations=5, outcome="success")
        history = self.logger.get_history()
        assert len(history) == 1
        assert history[0]["actual_tokens"] == 10000
        assert history[0]["features"]["task_length"] == 200

    def test_clear_history(self):
        fv = FeatureVector()
        self.logger.log_session(fv, actual_tokens=100, actual_iterations=1)
        assert self.logger.session_count() == 1
        self.logger.clear_history()
        assert self.logger.session_count() == 0

    def test_history_limit(self):
        for i in range(10):
            fv = FeatureVector(task_length=i)
            self.logger.log_session(fv, actual_tokens=i * 100, actual_iterations=1)
        history = self.logger.get_history(limit=3)
        assert len(history) == 3

    def test_roundtrip_with_predictor(self):
        """Log sessions, retrieve, and use for training."""
        for i in range(5):
            fv = FeatureVector(
                task_length=100 * (i + 1), task_complexity=float(i),
                codebase_files=10 * i, codebase_loc=1000 * i,
                available_tools=i + 1, model_context_size=128000,
                estimated_turns=i + 2,
            )
            self.logger.log_session(fv, actual_tokens=5000 * (i + 1), actual_iterations=i + 2)

        history = self.logger.get_history()
        assert len(history) == 5

        predictor = BudgetPredictor()
        predictor.train(history)
        # May or may not train successfully depending on data quality
        # Just verify no crash
        fv = FeatureVector(task_length=300, task_complexity=2.0, available_tools=3,
                          model_context_size=128000, estimated_turns=5)
        pred = predictor.predict(fv)
        assert pred.tokens > 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_workflow(self):
        """End-to-end: extract -> predict -> advise -> log."""
        extractor = SessionFeatureExtractor()
        predictor = BudgetPredictor()
        advisor = BudgetAdvisor(predictor)

        # Get advice
        advice = advisor.suggest_budget(
            "implement OAuth2 authentication flow",
            codebase={"files": 50, "loc": 20000},
            tools=8,
            model="claude-sonnet-4-20250514",
        )
        assert advice.recommended_iterations >= 1
        assert advice.estimated_tokens > 0

        # Log the session
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "integration_test.db")
        logger = SessionLogger(db_path=db_path)
        features = extractor.extract(
            "implement OAuth2 authentication flow",
            codebase_size={"files": 50, "loc": 20000},
            tool_count=8,
        )
        logger.log_session(features, actual_tokens=30000, actual_iterations=8)
        assert logger.session_count() == 1

    def test_multiple_models_comparison(self):
        advisor = BudgetAdvisor()
        models = ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-haiku-3-20250307"]
        costs = []
        for model in models:
            advice = advisor.suggest_budget("build a REST API", tools=5, model=model)
            costs.append(advice.estimated_cost)
        # Opus should be most expensive, Haiku cheapest
        assert costs[0] >= costs[2]
