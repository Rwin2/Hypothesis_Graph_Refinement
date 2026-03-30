"""Core data structures for Hypothesis Graph Refinement."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class NodeType(Enum):
    """Supported node types in the hypothesis graph."""

    OBSERVED = "observed"
    HYPOTHESIS = "hypothesis"
    FALSIFIED = "falsified"


class NodeStatus(Enum):
    """Lifecycle states for graph nodes."""

    ACTIVE = "active"
    PENDING = "pending"
    INVALIDATED = "invalidated"


@dataclass
class SemanticDistribution:
    """Probability distribution over candidate semantic categories."""

    categories: List[str]
    probabilities: np.ndarray
    top_k: int = 3
    entropy: float = 0.0

    def __post_init__(self):
        probs = np.asarray(self.probabilities, dtype=float)
        if probs.size == 0:
            self.probabilities = probs
            self.entropy = 0.0
            return

        total = probs.sum()
        if total > 0:
            probs = probs / total
        self.probabilities = probs

        valid_probs = probs[probs > 0]
        if valid_probs.size > 0:
            self.entropy = float(-np.sum(valid_probs * np.log(valid_probs + 1e-10)))

    def get_top_k(self, k: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
        """Return the highest-probability categories and their scores."""

        if len(self.probabilities) == 0:
            return [], np.array([])

        k = min(k or self.top_k, len(self.probabilities))
        top_indices = np.argsort(self.probabilities)[-k:][::-1]
        top_categories = [self.categories[i] for i in top_indices]
        top_probs = self.probabilities[top_indices]
        return top_categories, top_probs

    def get_expected_semantic_score(self, target_category: str) -> float:
        """Return the probability assigned to a target category."""

        if target_category in self.categories:
            idx = self.categories.index(target_category)
            return float(self.probabilities[idx])
        return 0.0


@dataclass
class CognitiveDependency:
    """Directed dependency between two nodes."""

    parent_id: str
    child_id: str
    dependency_type: str
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)

    def is_strong_dependency(self, threshold: float = 0.8) -> bool:
        return self.confidence >= threshold


@dataclass
class HypothesisNode:
    """Unified representation for observed and inferred nodes."""

    node_id: str
    node_type: NodeType
    status: NodeStatus = NodeStatus.ACTIVE
    position: np.ndarray = None
    orientation: Optional[np.ndarray] = None
    semantic_dist: Optional[SemanticDistribution] = None
    observed_class: Optional[str] = None
    confidence: float = 0.0
    associated_object: Optional[object] = None
    feature: Optional[torch.Tensor] = None
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    expected_observation: Optional[Dict] = None
    actual_observation: Optional[Dict] = None
    semantic_residual: float = 0.0
    falsification_reason: Optional[str] = None
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    visit_count: int = 0

    def mark_falsified(self, reason: str, residual: float):
        self.node_type = NodeType.FALSIFIED
        self.status = NodeStatus.INVALIDATED
        self.falsification_reason = reason
        self.semantic_residual = residual
        self.last_update_time = time.time()

    def upgrade_to_observed(self, observed_class: str, confidence: float):
        self.node_type = NodeType.OBSERVED
        self.observed_class = observed_class
        self.confidence = confidence
        self.last_update_time = time.time()
        self.visit_count += 1


class HypothesisGraph:
    """Graph container for hypothesis generation and falsification."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.nodes: Dict[str, HypothesisNode] = {}
        self.dependencies: Dict[str, List[CognitiveDependency]] = {}
        self.hypothesis_nodes: Set[str] = set()
        self.observed_nodes: Set[str] = set()
        self.falsified_nodes: Set[str] = set()
        self.stats = {
            "total_nodes_created": 0,
            "hypothesis_nodes_created": 0,
            "hypothesis_nodes_verified": 0,
            "hypothesis_nodes_falsified": 0,
            "cascade_deletions": 0,
            "total_dependencies": 0,
        }

        self.semantic_residual_threshold = cfg.get("semantic_residual_threshold", 0.5)
        self.min_dependency_confidence = cfg.get("min_dependency_confidence", 0.6)
        self.enable_cascade_deletion = cfg.get("enable_cascade_deletion", True)

    def add_node(self, node: HypothesisNode) -> str:
        node_id = node.node_id
        self.nodes[node_id] = node

        if node.node_type == NodeType.HYPOTHESIS:
            self.hypothesis_nodes.add(node_id)
            self.stats["hypothesis_nodes_created"] += 1
        elif node.node_type == NodeType.OBSERVED:
            self.observed_nodes.add(node_id)
        elif node.node_type == NodeType.FALSIFIED:
            self.falsified_nodes.add(node_id)

        self.stats["total_nodes_created"] += 1
        return node_id

    def add_dependency(self, dependency: CognitiveDependency):
        parent_id = dependency.parent_id
        child_id = dependency.child_id
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise ValueError(f"Dependency nodes not found: {parent_id} -> {child_id}")

        self.dependencies.setdefault(parent_id, []).append(dependency)

        if parent_id not in self.nodes[child_id].parent_ids:
            self.nodes[child_id].parent_ids.append(parent_id)
        if child_id not in self.nodes[parent_id].child_ids:
            self.nodes[parent_id].child_ids.append(child_id)

        self.stats["total_dependencies"] += 1

    def verify_hypothesis_node(self, node_id: str, actual_observation: Dict) -> bool:
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]
        if node.node_type != NodeType.HYPOTHESIS:
            raise ValueError(f"Node {node_id} is not a hypothesis node")

        node.actual_observation = actual_observation
        semantic_residual = self._compute_semantic_residual(node, actual_observation)
        node.semantic_residual = semantic_residual

        if semantic_residual < self.semantic_residual_threshold:
            predicted_class = node.semantic_dist.categories[
                np.argmax(node.semantic_dist.probabilities)
            ]
            node.upgrade_to_observed(predicted_class, 1.0 - semantic_residual)
            self.hypothesis_nodes.discard(node_id)
            self.observed_nodes.add(node_id)
            self.stats["hypothesis_nodes_verified"] += 1
            return True

        falsification_reason = actual_observation.get(
            "falsification_reason", "High semantic residual"
        )
        node.mark_falsified(falsification_reason, semantic_residual)
        self.hypothesis_nodes.discard(node_id)
        self.falsified_nodes.add(node_id)
        self.stats["hypothesis_nodes_falsified"] += 1

        if self.enable_cascade_deletion:
            deleted_count = self.cascade_delete(node_id)
            print(
                f"[Hypothesis Graph] Falsified node {node_id}, "
                f"cascade deleted {deleted_count} dependent nodes"
            )
        return False

    def _compute_semantic_residual(
        self, node: HypothesisNode, actual_observation: Dict
    ) -> float:
        if node.semantic_dist is None:
            return 1.0

        actual_class = actual_observation.get("semantic_class") or actual_observation.get(
            "observed_class"
        )
        if actual_class is None:
            return 1.0

        expected_prob = node.semantic_dist.get_expected_semantic_score(actual_class)
        return 1.0 - expected_prob

    def cascade_delete(self, falsified_node_id: str) -> int:
        deleted_nodes = set()
        queue = deque([falsified_node_id])

        while queue:
            current_id = queue.popleft()
            for dep in self.dependencies.get(current_id, []):
                child_id = dep.child_id
                if child_id in deleted_nodes or child_id not in self.nodes:
                    continue

                child_node = self.nodes[child_id]
                child_node.status = NodeStatus.INVALIDATED
                child_node.falsification_reason = (
                    f"Dependent on falsified node {current_id}"
                )
                deleted_nodes.add(child_id)
                queue.append(child_id)

        for node_id in deleted_nodes:
            self._remove_node(node_id)

        self.stats["cascade_deletions"] += len(deleted_nodes)
        return len(deleted_nodes)

    def _remove_node(self, node_id: str):
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        self.hypothesis_nodes.discard(node_id)
        self.observed_nodes.discard(node_id)
        self.falsified_nodes.discard(node_id)

        if node_id in self.dependencies:
            del self.dependencies[node_id]

        for parent_id in node.parent_ids:
            if parent_id in self.nodes and node_id in self.nodes[parent_id].child_ids:
                self.nodes[parent_id].child_ids.remove(node_id)
            if parent_id in self.dependencies:
                self.dependencies[parent_id] = [
                    dep for dep in self.dependencies[parent_id] if dep.child_id != node_id
                ]

        for child_id in node.child_ids:
            if child_id in self.nodes and node_id in self.nodes[child_id].parent_ids:
                self.nodes[child_id].parent_ids.remove(node_id)

        del self.nodes[node_id]

    def get_active_hypothesis_nodes(self) -> List[HypothesisNode]:
        return [
            self.nodes[node_id]
            for node_id in self.hypothesis_nodes
            if self.nodes[node_id].status == NodeStatus.ACTIVE
        ]

    def get_semantic_potential(self, node_id: str, target_category: str) -> float:
        if node_id not in self.nodes:
            return 0.0

        node = self.nodes[node_id]
        if node.node_type == NodeType.OBSERVED:
            return 1.0 if node.observed_class == target_category else 0.0
        if node.node_type == NodeType.HYPOTHESIS and node.semantic_dist is not None:
            return node.semantic_dist.get_expected_semantic_score(target_category)
        return 0.0

    def get_dependency_chain(self, node_id: str) -> List[str]:
        if node_id not in self.nodes:
            return []

        chain = [node_id]
        current_id = node_id
        while current_id in self.nodes:
            node = self.nodes[current_id]
            if not node.parent_ids:
                break
            parent_id = node.parent_ids[0]
            chain.insert(0, parent_id)
            current_id = parent_id
        return chain

    def get_statistics(self) -> Dict:
        active_hypothesis_count = sum(
            1
            for node_id in self.hypothesis_nodes
            if self.nodes[node_id].status == NodeStatus.ACTIVE
        )
        created = max(1, self.stats["hypothesis_nodes_created"])
        return {
            **self.stats,
            "current_total_nodes": len(self.nodes),
            "current_hypothesis_nodes": len(self.hypothesis_nodes),
            "current_observed_nodes": len(self.observed_nodes),
            "current_falsified_nodes": len(self.falsified_nodes),
            "active_hypothesis_nodes": active_hypothesis_count,
            "verification_rate": self.stats["hypothesis_nodes_verified"] / created,
            "falsification_rate": self.stats["hypothesis_nodes_falsified"] / created,
        }

    def prune_falsified_nodes(self):
        falsified_list = list(self.falsified_nodes)
        for node_id in falsified_list:
            self._remove_node(node_id)
        print(f"[Hypothesis Graph] Pruned {len(falsified_list)} falsified nodes")

    def visualize_dependency_tree(self, root_node_id: Optional[str] = None) -> str:
        if root_node_id is None:
            root_nodes = [
                node_id for node_id, node in self.nodes.items() if not node.parent_ids
            ]
        else:
            root_nodes = [root_node_id]

        lines = ["Hypothesis Graph Dependency Tree:", "=" * 50]
        for root_id in root_nodes:
            self._visualize_subtree(root_id, lines, indent=0)
        return "\n".join(lines)

    def _visualize_subtree(self, node_id: str, lines: List[str], indent: int):
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        prefix = "  " * indent + "├─ "
        node_info = f"{prefix}[{node.node_type.value}] {node_id}"

        if node.node_type == NodeType.HYPOTHESIS and node.semantic_dist is not None:
            top_cats, top_probs = node.semantic_dist.get_top_k(3)
            pairs = [f"{cat}:{prob:.2f}" for cat, prob in zip(top_cats, top_probs)]
            node_info += f" | Top-3: {pairs}"
        elif node.node_type == NodeType.OBSERVED:
            node_info += f" | Class: {node.observed_class} ({node.confidence:.2f})"
        elif node.node_type == NodeType.FALSIFIED:
            node_info += f" | Falsified: {node.falsification_reason}"

        lines.append(node_info)
        for child_id in node.child_ids:
            self._visualize_subtree(child_id, lines, indent + 1)


def create_hypothesis_node_from_frontier(
    frontier: object,
    semantic_dist: SemanticDistribution,
    parent_node_id: Optional[str] = None,
) -> HypothesisNode:
    node_id = f"hypothesis_frontier_{frontier.frontier_id}_{int(time.time() * 1000)}"
    hypothesis_node = HypothesisNode(
        node_id=node_id,
        node_type=NodeType.HYPOTHESIS,
        status=NodeStatus.ACTIVE,
        position=frontier.position,
        orientation=frontier.orientation,
        semantic_dist=semantic_dist,
        associated_object=frontier,
        feature=frontier.feature if hasattr(frontier, "feature") else None,
    )

    if parent_node_id is not None:
        hypothesis_node.parent_ids.append(parent_node_id)
    return hypothesis_node


def create_observed_node_from_snapshot(
    snapshot: object,
    observed_class: str,
    confidence: float,
) -> HypothesisNode:
    node_id = f"observed_snapshot_{snapshot.image}_{int(time.time() * 1000)}"
    return HypothesisNode(
        node_id=node_id,
        node_type=NodeType.OBSERVED,
        status=NodeStatus.ACTIVE,
        position=snapshot.position if hasattr(snapshot, "position") else None,
        observed_class=observed_class,
        confidence=confidence,
        associated_object=snapshot,
    )
