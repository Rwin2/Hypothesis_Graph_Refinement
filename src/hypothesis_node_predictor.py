"""Semantic prediction utilities for hypothesis nodes."""

from typing import Dict, List

import numpy as np
import torch

from src.eval_utils_gpt_aeqa import call_openai_api, encode_tensor2base64
from src.hypothesis_graph import SemanticDistribution


class HypothesisNodePredictor:
    """Predict semantic distributions for unexplored frontiers."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.common_room_categories = [
            "bedroom",
            "bathroom",
            "kitchen",
            "living_room",
            "dining_room",
            "hallway",
            "closet",
            "office",
            "garage",
            "laundry_room",
            "balcony",
            "staircase",
            "entryway",
            "pantry",
            "gym",
        ]
        self.enable_vlm_prediction = cfg.get("enable_vlm_hypothesis_prediction", True)
        self.top_k_categories = cfg.get("hypothesis_node_top_k_categories", 5)
        self.use_spatial_context = cfg.get("use_spatial_context", True)
        self.use_visual_cues = cfg.get("use_visual_cues", True)
        self.prediction_cache = {}

    def predict_semantic_distribution(
        self,
        frontier_image: np.ndarray,
        frontier_position: np.ndarray,
        spatial_context: Dict,
        observed_history: List[Dict],
    ) -> SemanticDistribution:
        del frontier_position  # Reserved for future use.
        if self.enable_vlm_prediction:
            return self._predict_with_vlm(
                frontier_image, spatial_context, observed_history
            )
        return self._predict_with_heuristics(spatial_context, observed_history)

    def _predict_with_vlm(
        self,
        frontier_image: np.ndarray,
        spatial_context: Dict,
        observed_history: List[Dict],
    ) -> SemanticDistribution:
        frontier_img_base64 = encode_tensor2base64(frontier_image)
        context_description = self._build_context_description(
            spatial_context, observed_history
        )

        sys_prompt = """You are an expert in indoor scene understanding and spatial reasoning.
Your task is to predict the semantic type of an unexplored area in a house, based on:
1. A frontier observation image showing the boundary between explored and unexplored areas
2. Spatial context information (current location, nearby objects, explored rooms)
3. Common-sense reasoning about house layouts

Return ONLY a JSON object with room types as keys and probabilities as values.
The probabilities must sum to 1.0."""

        user_prompt = f"""Predict the room type of the unexplored area shown in the frontier image.

Context information:
{context_description}

Candidate room types: {', '.join(self.common_room_categories)}
"""

        try:
            response = call_openai_api(sys_prompt, [(user_prompt, frontier_img_base64)])
            if response is None:
                print("[Hypothesis Predictor] VLM API failed, falling back to heuristics")
                return self._predict_with_heuristics(spatial_context, observed_history)

            import json

            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                print(
                    f"[Hypothesis Predictor] Failed to parse VLM response: {response}"
                )
                return self._predict_with_heuristics(spatial_context, observed_history)

            prediction_dict = json.loads(response[start_idx:end_idx])
            categories = list(prediction_dict.keys())
            probabilities = np.array([prediction_dict[cat] for cat in categories])

            if probabilities.sum() <= 0:
                return self._predict_with_heuristics(spatial_context, observed_history)

            probabilities = probabilities / probabilities.sum()
            top_k_indices = np.argsort(probabilities)[-self.top_k_categories :][::-1]
            top_categories = [categories[i] for i in top_k_indices]
            top_probs = probabilities[top_k_indices]

            semantic_dist = SemanticDistribution(
                categories=top_categories,
                probabilities=top_probs,
                top_k=self.top_k_categories,
            )
            print(
                f"[Hypothesis Predictor] VLM prediction: {list(zip(top_categories, top_probs))}"
            )
            return semantic_dist
        except Exception as exc:
            print(f"[Hypothesis Predictor] Error in VLM prediction: {exc}")
            return self._predict_with_heuristics(spatial_context, observed_history)

    def _predict_with_heuristics(
        self, spatial_context: Dict, observed_history: List[Dict]
    ) -> SemanticDistribution:
        nearby_objects = spatial_context.get("nearby_objects", [])
        current_room_type = spatial_context.get("current_room_type")

        probabilities = np.ones(len(self.common_room_categories), dtype=float)
        probabilities /= probabilities.sum()

        object_room_association = {
            "bed": {"bedroom": 0.7, "living_room": 0.1, "hallway": 0.1},
            "toilet": {"bathroom": 0.8, "hallway": 0.1},
            "shower": {"bathroom": 0.9},
            "bathtub": {"bathroom": 0.9},
            "sink": {"bathroom": 0.4, "kitchen": 0.4, "laundry_room": 0.1},
            "sofa": {"living_room": 0.6, "bedroom": 0.2, "office": 0.1},
            "dining_table": {"dining_room": 0.7, "kitchen": 0.2},
            "refrigerator": {"kitchen": 0.9},
            "stove": {"kitchen": 0.9},
            "oven": {"kitchen": 0.9},
            "desk": {"office": 0.5, "bedroom": 0.3},
            "chair": {
                "dining_room": 0.3,
                "office": 0.2,
                "living_room": 0.2,
                "bedroom": 0.1,
            },
            "wardrobe": {"bedroom": 0.7, "closet": 0.2},
            "tv": {"living_room": 0.6, "bedroom": 0.3},
        }

        for obj_class in nearby_objects:
            obj_class_lower = obj_class.lower()
            if obj_class_lower not in object_room_association:
                continue
            for room_type, boost in object_room_association[obj_class_lower].items():
                if room_type in self.common_room_categories:
                    idx = self.common_room_categories.index(room_type)
                    probabilities[idx] *= 1.0 + boost

        if current_room_type is not None:
            room_adjacency = {
                "bedroom": {"hallway": 0.5, "bathroom": 0.3, "closet": 0.2},
                "bathroom": {"hallway": 0.5, "bedroom": 0.3},
                "kitchen": {
                    "dining_room": 0.5,
                    "living_room": 0.3,
                    "hallway": 0.2,
                },
                "living_room": {
                    "hallway": 0.4,
                    "dining_room": 0.3,
                    "kitchen": 0.2,
                },
                "hallway": {
                    "bedroom": 0.3,
                    "bathroom": 0.2,
                    "living_room": 0.2,
                    "kitchen": 0.1,
                },
            }
            for adjacent_room, boost in room_adjacency.get(current_room_type, {}).items():
                if adjacent_room in self.common_room_categories:
                    idx = self.common_room_categories.index(adjacent_room)
                    probabilities[idx] *= 1.0 + boost

        num_explored_rooms = len(observed_history)
        if num_explored_rooms < 3:
            for room_type in ["bedroom", "living_room", "kitchen", "bathroom"]:
                if room_type in self.common_room_categories:
                    probabilities[self.common_room_categories.index(room_type)] *= 1.3
        else:
            for room_type in ["closet", "hallway", "pantry", "laundry_room"]:
                if room_type in self.common_room_categories:
                    probabilities[self.common_room_categories.index(room_type)] *= 1.2

        probabilities = probabilities / probabilities.sum()
        top_k_indices = np.argsort(probabilities)[-self.top_k_categories :][::-1]
        top_categories = [self.common_room_categories[i] for i in top_k_indices]
        top_probs = probabilities[top_k_indices]

        semantic_dist = SemanticDistribution(
            categories=top_categories,
            probabilities=top_probs,
            top_k=self.top_k_categories,
        )
        print(
            f"[Hypothesis Predictor] Heuristic prediction: {list(zip(top_categories, top_probs))}"
        )
        return semantic_dist

    def _build_context_description(
        self, spatial_context: Dict, observed_history: List[Dict]
    ) -> str:
        lines = []
        current_room = spatial_context.get("current_room_type", "Unknown")
        lines.append(f"Current location: {current_room}")

        nearby_objects = spatial_context.get("nearby_objects", [])
        if nearby_objects:
            lines.append(f"Nearby objects: {', '.join(nearby_objects[:10])}")

        explored_rooms = [hist.get("room_type", "Unknown") for hist in observed_history]
        unique_rooms = sorted(set(explored_rooms))
        lines.append(f"Explored rooms so far: {', '.join(unique_rooms)}")
        lines.append(f"Number of explored areas: {len(observed_history)}")
        return "\n".join(lines)

    def batch_predict_frontiers(
        self, frontiers: List, spatial_context: Dict, observed_history: List[Dict]
    ) -> Dict[int, SemanticDistribution]:
        predictions = {}
        for frontier in frontiers:
            if hasattr(frontier, "feature") and frontier.feature is not None:
                if isinstance(frontier.feature, torch.Tensor):
                    frontier_image = frontier.feature.cpu().numpy()
                    if frontier_image.dtype != np.uint8:
                        frontier_image = (frontier_image * 255).astype(np.uint8)
                else:
                    frontier_image = frontier.feature
            else:
                print(
                    f"[Hypothesis Predictor] Frontier {frontier.frontier_id} has no image, using heuristics only"
                )
                predictions[frontier.frontier_id] = self._predict_with_heuristics(
                    spatial_context, observed_history
                )
                continue

            predictions[frontier.frontier_id] = self.predict_semantic_distribution(
                frontier_image=frontier_image,
                frontier_position=frontier.position,
                spatial_context=spatial_context,
                observed_history=observed_history,
            )
        return predictions

    def update_prediction_with_new_observation(
        self,
        frontier_id: int,
        new_observation: Dict,
        previous_dist: SemanticDistribution,
    ) -> SemanticDistribution:
        del frontier_id, new_observation
        categories = previous_dist.categories.copy()
        probabilities = previous_dist.probabilities.copy()
        probabilities = probabilities / probabilities.sum()
        return SemanticDistribution(
            categories=categories,
            probabilities=probabilities,
            top_k=previous_dist.top_k,
        )


def compute_semantic_potential_score(
    hypothesis_node_semantic_dist: SemanticDistribution,
    target_category: str,
    distance: float,
    cfg: dict,
) -> float:
    semantic_score = hypothesis_node_semantic_dist.get_expected_semantic_score(
        target_category
    )
    distance_weight = cfg.get("distance_weight", 0.1)
    distance_penalty = np.exp(-distance_weight * distance)
    entropy_weight = cfg.get("entropy_weight", 0.05)
    entropy_bonus = entropy_weight * hypothesis_node_semantic_dist.entropy
    return float(semantic_score * distance_penalty + entropy_bonus)
