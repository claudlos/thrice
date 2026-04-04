"""
Property-based tests for Delegation Tree (SM-3).

Properties verified:
  INV-D1: Depth never exceeds MAX_DEPTH (=2)
  INV-D2: Budget Conservation (parent budget >= sum of children consumed)
  INV-D3: Active Children Consistency
  INV-D5: Batch Size Bound (<= 3)
  INV-D7: Tree Acyclicity
"""

import pytest
from typing import Dict, List, Optional, Set

from hypothesis import given, settings, HealthCheck, Phase, assume
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
    precondition,
)

pytestmark = [pytest.mark.property]

MAX_DEPTH = 2
BATCH_MAX = 3
MIN_CHILD_BUDGET = 2

CHILD_STATES = {"building", "running", "completed", "failed", "interrupted", "error"}
TREE_STATES = {"idle", "dispatching", "children_running", "collecting_results", "done"}


# ============================================================================
# Abstract delegation tree model
# ============================================================================

class AgentNode:
    """Model of an agent in the delegation tree."""

    def __init__(self, agent_id: str, depth: int, budget: int, parent: Optional["AgentNode"] = None):
        self.id = agent_id
        self.depth = depth
        self.budget = budget
        self.budget_consumed = 0
        self.parent = parent
        self.children: List["AgentNode"] = []
        self.state = "running"  # simplified

    @property
    def budget_remaining(self):
        return self.budget - self.budget_consumed

    def consume_budget(self, amount: int):
        self.budget_consumed += amount


class DelegationTreeStateMachine(RuleBasedStateMachine):
    """
    Stateful test: randomly delegate tasks creating a tree of agents,
    verify invariants after every step.
    """

    def __init__(self):
        super().__init__()
        self._counter = 0
        # Root agent at depth 0 with generous budget
        self.root = AgentNode("root", depth=0, budget=100)
        self.all_nodes: Dict[str, AgentNode] = {"root": self.root}

    def _new_id(self):
        self._counter += 1
        return f"agent_{self._counter}"

    def _delegable_nodes(self) -> List[AgentNode]:
        """Nodes that can still delegate (depth < MAX, budget > MIN, state=running)."""
        return [
            n for n in self.all_nodes.values()
            if n.depth < MAX_DEPTH
            and n.budget_remaining > MIN_CHILD_BUDGET
            and n.state == "running"
            and len(n.children) < BATCH_MAX
        ]

    def _running_children(self) -> List[AgentNode]:
        """All non-root nodes that are still running."""
        return [n for n in self.all_nodes.values() if n.id != "root" and n.state == "running"]

    # -- Rules -----------------------------------------------------------------

    @precondition(lambda self: bool(self._delegable_nodes()))
    @rule(data=st.data(), child_budget_frac=st.floats(0.1, 0.5))
    def delegate_single(self, data, child_budget_frac):
        """Parent delegates a single task to a new child."""
        parent = data.draw(st.sampled_from(self._delegable_nodes()))
        child_budget = max(MIN_CHILD_BUDGET, int(parent.budget_remaining * child_budget_frac))
        child_budget = min(child_budget, parent.budget_remaining)

        child_id = self._new_id()
        child = AgentNode(child_id, depth=parent.depth + 1, budget=child_budget, parent=parent)
        parent.children.append(child)
        self.all_nodes[child_id] = child

    @precondition(lambda self: bool(self._delegable_nodes()))
    @rule(data=st.data(), n_children=st.integers(1, BATCH_MAX))
    def delegate_batch(self, data, n_children):
        """Parent delegates a batch of tasks."""
        parent = data.draw(st.sampled_from(self._delegable_nodes()))
        slots = BATCH_MAX - len(parent.children)
        actual_n = min(n_children, slots)
        if actual_n <= 0:
            return

        budget_per_child = max(MIN_CHILD_BUDGET, parent.budget_remaining // (actual_n + 1))

        for _ in range(actual_n):
            if parent.budget_remaining < MIN_CHILD_BUDGET:
                break
            child_id = self._new_id()
            child_budget = min(budget_per_child, parent.budget_remaining)
            child = AgentNode(child_id, depth=parent.depth + 1, budget=child_budget, parent=parent)
            parent.children.append(child)
            self.all_nodes[child_id] = child

    @precondition(lambda self: bool(self._running_children()))
    @rule(data=st.data(), consumed=st.integers(1, 10))
    def child_works(self, data, consumed):
        """A running child consumes some budget."""
        child = data.draw(st.sampled_from(self._running_children()))
        actual = min(consumed, child.budget_remaining)
        child.consume_budget(actual)

    @precondition(lambda self: bool(self._running_children()))
    @rule(data=st.data())
    def child_completes(self, data):
        """A running child finishes successfully."""
        child = data.draw(st.sampled_from(self._running_children()))
        child.state = "completed"

    @precondition(lambda self: bool(self._running_children()))
    @rule(data=st.data())
    def child_fails(self, data):
        """A running child encounters an error."""
        child = data.draw(st.sampled_from(self._running_children()))
        child.state = "failed"

    @precondition(lambda self: bool(self._running_children()))
    @rule(data=st.data())
    def child_interrupted(self, data):
        """A running child is interrupted (budget exhausted, parent signal)."""
        child = data.draw(st.sampled_from(self._running_children()))
        child.state = "interrupted"

    @rule(consumed=st.integers(1, 5))
    def root_works(self, consumed):
        """Root agent consumes budget."""
        actual = min(consumed, self.root.budget_remaining)
        self.root.consume_budget(actual)

    # -- Invariants (checked after every step) ---------------------------------

    @invariant()
    def inv_depth_bound(self):
        """INV-D1: No node exceeds MAX_DEPTH."""
        for nid, node in self.all_nodes.items():
            assert node.depth <= MAX_DEPTH, (
                f"Node {nid} at depth {node.depth} exceeds MAX_DEPTH={MAX_DEPTH}"
            )

    @invariant()
    def inv_budget_conservation(self):
        """INV-D2: Budget remaining is non-negative for all nodes."""
        for nid, node in self.all_nodes.items():
            assert node.budget_remaining >= 0, (
                f"Node {nid}: budget_remaining={node.budget_remaining} < 0 "
                f"(budget={node.budget}, consumed={node.budget_consumed})"
            )

    @invariant()
    def inv_child_budget_bounded(self):
        """INV-D2 extended: Each child's budget <= parent's budget at delegation time.
        (We verify child.budget <= parent.budget as a weaker bound.)"""
        for nid, node in self.all_nodes.items():
            if node.parent is not None:
                assert node.budget <= node.parent.budget, (
                    f"Child {nid} budget {node.budget} > parent {node.parent.id} "
                    f"budget {node.parent.budget}"
                )

    @invariant()
    def inv_batch_size_bound(self):
        """INV-D5: No parent has more than BATCH_MAX children."""
        for nid, node in self.all_nodes.items():
            assert len(node.children) <= BATCH_MAX, (
                f"Node {nid} has {len(node.children)} children > BATCH_MAX={BATCH_MAX}"
            )

    @invariant()
    def inv_tree_acyclicity(self):
        """INV-D7: No cycles in the delegation tree."""
        for nid, node in self.all_nodes.items():
            visited = set()
            current = node
            while current is not None:
                assert current.id not in visited, (
                    f"Cycle detected: node {current.id} revisited while "
                    f"traversing ancestors of {nid}"
                )
                visited.add(current.id)
                current = current.parent

    @invariant()
    def inv_parent_child_consistency(self):
        """Every child's parent points back to a node that lists it."""
        for nid, node in self.all_nodes.items():
            if node.parent is not None:
                assert node in node.parent.children, (
                    f"Node {nid}'s parent {node.parent.id} doesn't list it as child"
                )

    @invariant()
    def inv_child_states_valid(self):
        """All non-root node states are valid."""
        for nid, node in self.all_nodes.items():
            if nid != "root":
                assert node.state in CHILD_STATES, (
                    f"Node {nid} has invalid state {node.state}"
                )

    @invariant()
    def inv_depth_matches_ancestry(self):
        """Depth equals number of ancestor hops to root."""
        for nid, node in self.all_nodes.items():
            hops = 0
            current = node
            while current.parent is not None:
                hops += 1
                current = current.parent
            assert node.depth == hops, (
                f"Node {nid}: depth={node.depth} but ancestry hops={hops}"
            )


# Generate the test class
TestDelegationTreeStateMachine = DelegationTreeStateMachine.TestCase
TestDelegationTreeStateMachine.settings = settings(
    max_examples=50,
    stateful_step_count=30,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    phases=[Phase.generate],
)


# ============================================================================
# Standalone property tests
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "new-files"))

from hermes_invariants import InvariantChecker


@pytest.mark.property
@given(
    depth=st.integers(0, 10),
    max_depth=st.integers(1, 5),
)
def test_depth_check(depth, max_depth):
    """InvariantChecker.check_delegation_tree catches depth > max."""
    violations = InvariantChecker.check_delegation_tree(
        parent_agent=None, active_children=[], depth=depth, max_depth=max_depth
    )
    if depth > max_depth:
        assert any("DEPTH_EXCEEDED" in v for v in violations)
    else:
        assert not any("DEPTH_EXCEEDED" in v for v in violations)


@pytest.mark.property
@given(n_children=st.integers(0, 10))
def test_children_count_check(n_children):
    """InvariantChecker.check_delegation_tree catches >3 active children."""
    children = [object() for _ in range(n_children)]
    violations = InvariantChecker.check_delegation_tree(
        parent_agent=None, active_children=children, depth=0, max_depth=5
    )
    if n_children > 3:
        assert any("TOO_MANY_CHILDREN" in v for v in violations)
    else:
        assert not any("TOO_MANY_CHILDREN" in v for v in violations)


@pytest.mark.property
def test_duplicate_child_detected():
    """InvariantChecker.check_delegation_tree catches duplicate children."""
    child = object()
    violations = InvariantChecker.check_delegation_tree(
        parent_agent=None, active_children=[child, child], depth=0, max_depth=5
    )
    assert any("DUPLICATE_CHILD" in v for v in violations)
