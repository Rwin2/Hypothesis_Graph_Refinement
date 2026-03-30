"""Semantic verification and falsification utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.eval_utils_gpt_aeqa import call_openai_api, encode_tensor2base64
from src.hypothesis_graph import HypothesisGraph, HypothesisNode, SemanticDistribution


@dataclass
class FalsificationReport:
    """Structured result for hypothesis verification."""

    node_id: str
    semantic_residual: float
    is_falsified: bool
    physical_cause: Optional[str]
    textual_explanation: str
    cascade_deleted_nodes: List[str]
    confidence: float


class SemanticCritic:
    """Validate hypothesis nodes against newly observed evidence."""

    def __init__(self, cfg: dict, hypothesis_graph: HypothesisGraph):
        self.cfg = cfg
        self.hypothesis_graph = hypothesis_graph
        self.semantic_residual_threshold = cfg.get("semantic_residual_threshold", 0.5)
        self.enable_vlm_diagnosis = cfg.get("enable_vlm_causality_diagnosis", True)
        self.min_falsification_confidence = cfg.get("min_falsification_confidence", 0.7)
        self.clip_model = None
        self.clip_preprocess = None
        self.stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "falsifications": 0,
            "hallucination_types": {},
        }

    def set_clip_model(self, clip_model, clip_preprocess):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

    def verify_hypothesis_node_arrival(
        self,
        hypothesis_node: HypothesisNode,
        actual_observation: Dict,
        scene_objects: Dict,
    ) -> FalsificationReport:
        del scene_objects
        self.stats["total_verifications"] += 1

        semantic_residual = self._compute_semantic_residual(
            hypothesis_node, actual_observation
        )
        is_falsified = semantic_residual > self.semantic_residual_threshold

        physical_cause = None
        textual_explanation = ""
        cascade_deleted_nodes: List[str] = []

        if is_falsified:
            if self.enable_vlm_diagnosis:
                physical_cause, textual_explanation = self._diagnose_falsification_cause(
                    hypothesis_node, actual_observation
                )
            else:
                physical_cause = None
                textual_explanation = "Semantic residual exceeds threshold"

            if physical_cause:
                self.stats["hallucination_types"][physical_cause] = (
                    self.stats["hallucination_types"].get(physical_cause, 0) + 1
                )

            cascade_deleted_nodes = self._get_cascade_deleted_node_ids(
                hypothesis_node.node_id
            )
            self.hypothesis_graph.nodes[hypothesis_node.node_id].mark_falsified(
                reason=textual_explanation,
                residual=semantic_residual,
            )
            self.hypothesis_graph.hypothesis_nodes.discard(hypothesis_node.node_id)
            self.hypothesis_graph.falsified_nodes.add(hypothesis_node.node_id)
            self.hypothesis_graph.stats["hypothesis_nodes_falsified"] += 1
            self.hypothesis_graph.cascade_delete(hypothesis_node.node_id)
            self.stats["falsifications"] += 1

            print(
                f"""
[Semantic Critic] FALSIFICATION DETECTED
  Node ID: {hypothesis_node.node_id}
  Semantic Residual: {semantic_residual:.3f}
  Physical Cause: {physical_cause}
  Explanation: {textual_explanation}
  Cascade Deleted: {len(cascade_deleted_nodes)} nodes
"""
            )
        else:
            predicted_class = hypothesis_node.semantic_dist.categories[
                np.argmax(hypothesis_node.semantic_dist.probabilities)
            ]
            hypothesis_node.upgrade_to_observed(
                observed_class=predicted_class,
                confidence=1.0 - semantic_residual,
            )
            self.hypothesis_graph.hypothesis_nodes.discard(hypothesis_node.node_id)
            self.hypothesis_graph.observed_nodes.add(hypothesis_node.node_id)
            self.hypothesis_graph.stats["hypothesis_nodes_verified"] += 1
            self.stats["successful_verifications"] += 1
            textual_explanation = f"Verified as {predicted_class}"

            print(
                f"""
[Semantic Critic] VERIFICATION PASSED
  Node ID: {hypothesis_node.node_id}
  Predicted: {predicted_class}
  Residual: {semantic_residual:.3f}
"""
            )

        return FalsificationReport(
            node_id=hypothesis_node.node_id,
            semantic_residual=semantic_residual,
            is_falsified=is_falsified,
            physical_cause=physical_cause,
            textual_explanation=textual_explanation,
            cascade_deleted_nodes=cascade_deleted_nodes,
            confidence=abs(semantic_residual - self.semantic_residual_threshold),
        )

    def _compute_semantic_residual(
        self, hypothesis_node: HypothesisNode, actual_observation: Dict
    ) -> float:
        if "semantic_class" in actual_observation:
            actual_class = actual_observation["semantic_class"]
            expected_prob = hypothesis_node.semantic_dist.get_expected_semantic_score(
                actual_class
            )
            residual_category = 1.0 - expected_prob
        else:
            residual_category = 0.5

        if self.clip_model is not None and hypothesis_node.feature is not None:
            residual_feature = self._compute_feature_residual(
                hypothesis_node.feature,
                actual_observation.get("rgb_image"),
            )
        else:
            residual_feature = 0.5

        residual_objects = self._compute_object_residual(
            hypothesis_node.semantic_dist,
            actual_observation.get("detected_objects", []),
        )

        w1 = self.cfg.get("residual_weight_category", 0.5)
        w2 = self.cfg.get("residual_weight_feature", 0.3)
        w3 = self.cfg.get("residual_weight_objects", 0.2)

        return float(w1 * residual_category + w2 * residual_feature + w3 * residual_objects)

    def _compute_feature_residual(
        self, expected_feature: torch.Tensor, actual_image: Optional[np.ndarray]
    ) -> float:
        if actual_image is None or self.clip_model is None or self.clip_preprocess is None:
            return 0.5

        try:
            from PIL import Image

            actual_pil = Image.fromarray(actual_image)
            actual_tensor = self.clip_preprocess(actual_pil).unsqueeze(0)
            with torch.no_grad():
                actual_feature = self.clip_model.encode_image(actual_tensor)
                actual_feature = F.normalize(actual_feature, dim=-1)

            expected_feature_norm = F.normalize(expected_feature.unsqueeze(0), dim=-1)
            similarity = F.cosine_similarity(
                expected_feature_norm, actual_feature, dim=-1
            )
            return 1.0 - float(similarity.item())
        except Exception as exc:
            print(f"[Semantic Critic] Error computing feature residual: {exc}")
            return 0.5

    def _compute_object_residual(
        self,
        semantic_dist: Optional[SemanticDistribution],
        detected_objects: List[str],
    ) -> float:
        if semantic_dist is None or not detected_objects:
            return 0.5

        room_object_map = {
            "bedroom": {"bed", "nightstand", "dresser", "wardrobe", "pillow"},
            "bathroom": {"toilet", "sink", "shower", "bathtub", "mirror"},
            "kitchen": {"stove", "oven", "refrigerator", "microwave", "sink"},
            "living_room": {"sofa", "tv", "coffee_table", "lamp"},
            "dining_room": {"dining_table", "chair"},
            "hallway": {"door", "painting", "console_table"},
            "closet": {"wardrobe", "shelf", "hanger"},
            "office": {"desk", "chair", "monitor", "keyboard"},
        }

        detected_set = {obj.lower() for obj in detected_objects}
        weighted_match = 0.0
        weight_total = 0.0

        for category, probability in zip(
            semantic_dist.categories, semantic_dist.probabilities
        ):
            expected_objects = room_object_map.get(category, set())
            if not expected_objects:
                continue
            overlap = len(detected_set & expected_objects)
            union = len(detected_set | expected_objects)
            match_score = overlap / max(union, 1)
            weighted_match += float(probability) * match_score
            weight_total += float(probability)

        if weight_total == 0:
            return 0.5
        return 1.0 - weighted_match / weight_total

    def _diagnose_falsification_cause(
        self, hypothesis_node: HypothesisNode, actual_observation: Dict
    ) -> Tuple[Optional[str], str]:
        fallback_cause = self.detect_hallucination_in_snapshot(actual_observation)
        fallback_explanation = (
            f"Potential falsification cause: {fallback_cause}"
            if fallback_cause
            else "The observed evidence does not match the predicted semantics."
        )

        actual_image = actual_observation.get("rgb_image")
        if actual_image is None:
            return fallback_cause, fallback_explanation

        predicted_class = hypothesis_node.semantic_dist.categories[
            np.argmax(hypothesis_node.semantic_dist.probabilities)
        ]
        detected_objects = actual_observation.get("detected_objects", [])
        actual_class = actual_observation.get("semantic_class")

        expected_image = None
        if isinstance(hypothesis_node.feature, torch.Tensor):
            expected_image = hypothesis_node.feature.detach().cpu().numpy()
        elif isinstance(hypothesis_node.feature, np.ndarray):
            expected_image = hypothesis_node.feature

        try:
            actual_img_base64 = encode_tensor2base64(actual_image)
            expected_img_base64 = (
                encode_tensor2base64(expected_image) if expected_image is not None else None
            )

            sys_prompt = """You diagnose why a predicted indoor room hypothesis was falsified.
Return JSON with keys "physical_cause" and "explanation".
Use one of: mirror_reflection, glass_window, painting_poster, occlusion,
lighting_artifact, layout_mismatch, semantic_mismatch, unknown."""

            user_prompt = f"""Predicted room type: {predicted_class}
Observed room type: {actual_class}
Detected objects: {', '.join(detected_objects) if detected_objects else 'None'}
Explain the most likely physical or perceptual cause of the mismatch."""

            contents = [(user_prompt, actual_img_base64)]
            if expected_img_base64 is not None:
                contents.append(("Predicted frontier evidence", expected_img_base64))

            response = call_openai_api(sys_prompt, contents)
            if response is None:
                return fallback_cause, fallback_explanation

            import json

            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                return fallback_cause, fallback_explanation

            payload = json.loads(response[start_idx:end_idx])
            return payload.get("physical_cause"), payload.get(
                "explanation", fallback_explanation
            )
        except Exception as exc:
            print(f"[Semantic Critic] Error in falsification diagnosis: {exc}")
            return fallback_cause, fallback_explanation

    def _get_cascade_deleted_node_ids(self, node_id: str) -> List[str]:
        deleted_nodes = []
        visited = set()
        queue = [node_id]

        while queue:
            current_id = queue.pop(0)
            for dep in self.hypothesis_graph.dependencies.get(current_id, []):
                child_id = dep.child_id
                if child_id in visited or child_id not in self.hypothesis_graph.nodes:
                    continue
                visited.add(child_id)
                deleted_nodes.append(child_id)
                queue.append(child_id)

        return deleted_nodes

    def detect_hallucination_in_snapshot(self, snapshot_data: Dict) -> Optional[str]:
        detected_objects = {
            obj.lower() for obj in snapshot_data.get("detected_objects", []) if obj
        }
        if detected_objects & {"mirror", "glass", "window"}:
            return "mirror_reflection"

        depth = snapshot_data.get("depth")
        if depth is not None:
            try:
                depth_std = float(np.std(depth))
                if depth_std > 2.0:
                    return "lighting_artifact"
            except Exception:
                pass

        inconsistent_pairs = [
            ({"bed"}, {"toilet", "shower", "bathtub"}),
            ({"stove", "oven"}, {"bed", "pillow"}),
        ]
        for lhs, rhs in inconsistent_pairs:
            if detected_objects & lhs and detected_objects & rhs:
                return "semantic_mismatch"

        return None

    def get_statistics(self) -> Dict:
        total = max(1, self.stats["total_verifications"])
        return {
            **self.stats,
            "verification_rate": self.stats["successful_verifications"] / total,
            "falsification_rate": self.stats["falsifications"] / total,
        }


class ExpectationViolationDetector:
    """Detect expectation-observation mismatch during navigation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.violation_threshold = cfg.get("expectation_violation_threshold", 0.6)

    def detect_violation(
        self, expected_state: Dict, observed_state: Dict
    ) -> Tuple[bool, float]:
        spatial_score = self._check_spatial_violation(expected_state, observed_state)
        semantic_score = self._check_semantic_violation(expected_state, observed_state)
        violation_score = float(np.mean([spatial_score, semantic_score]))
        return violation_score > self.violation_threshold, violation_score

    def _check_spatial_violation(self, expected: Dict, observed: Dict) -> float:
        scores = []

        if "navigable" in expected and "navigable" in observed:
            scores.append(1.0 if expected["navigable"] != observed["navigable"] else 0.0)

        if "distance" in expected and "distance" in observed:
            expected_distance = max(float(expected["distance"]), 1e-6)
            relative_error = abs(float(observed["distance"]) - expected_distance) / expected_distance
            scores.append(min(relative_error, 1.0))

        expected_obstacles = set(expected.get("obstacles", []))
        observed_obstacles = set(observed.get("obstacles", []))
        if expected_obstacles or observed_obstacles:
            intersection = len(expected_obstacles & observed_obstacles)
            union = len(expected_obstacles | observed_obstacles)
            scores.append(1.0 - intersection / max(union, 1))

        if "depth" in expected and "depth" in observed:
            try:
                mse = float(
                    np.mean(
                        (np.asarray(expected["depth"]) - np.asarray(observed["depth"])) ** 2
                    )
                )
                scores.append(min(mse / 10.0, 1.0))
            except Exception:
                pass

        return float(np.mean(scores)) if scores else 0.0

    def _check_semantic_violation(self, expected: Dict, observed: Dict) -> float:
        scores = []

        expected_room = expected.get("room_type")
        observed_room = observed.get("room_type")
        if expected_room is not None and observed_room is not None:
            scores.append(0.0 if expected_room == observed_room else 1.0)

        semantic_distribution = expected.get("semantic_distribution")
        if isinstance(semantic_distribution, SemanticDistribution) and observed_room:
            scores.append(
                1.0
                - semantic_distribution.get_expected_semantic_score(observed_room)
            )

        expected_objects = set(expected.get("expected_objects", []))
        detected_objects = set(observed.get("detected_objects", []))
        if expected_objects or detected_objects:
            unexpected_ratio = len(detected_objects - expected_objects) / max(
                len(detected_objects), 1
            )
            missing_ratio = len(expected_objects - detected_objects) / max(
                len(expected_objects), 1
            )
            scores.append(min(unexpected_ratio + missing_ratio, 1.0))

        if "expected_feature" in expected and "observed_feature" in observed:
            try:
                expected_feature = torch.as_tensor(expected["expected_feature"]).float()
                observed_feature = torch.as_tensor(observed["observed_feature"]).float()
                expected_feature = F.normalize(expected_feature, dim=-1)
                observed_feature = F.normalize(observed_feature, dim=-1)
                similarity = F.cosine_similarity(expected_feature, observed_feature, dim=-1)
                scores.append(1.0 - float(similarity.mean().item()))
            except Exception:
                pass

        incompatible_objects = {
            "bedroom": {"stove", "oven"},
            "bathroom": {"bed", "sofa"},
            "kitchen": {"bed", "bathtub"},
        }
        if observed_room in incompatible_objects:
            if detected_objects & incompatible_objects[observed_room]:
                scores.append(0.8)

        return float(np.mean(scores)) if scores else 0.0
