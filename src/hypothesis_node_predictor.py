"""Semantic prediction utilities for hypothesis nodes."""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from src.eval_utils_gpt_aeqa import call_openai_api, encode_tensor2base64
from src.hypothesis_graph import SemanticDistribution


# ---------------------------------------------------------------------------
# OPEN-VOCAB room-goal relevance via a caption-text embedder (vLLM :8002).
#
# Gated behind ``cfg.hypothesis.use_open_vocab_rooms``. A single process-wide
# embedder is shared so predictor instances and the module-level
# ``compute_semantic_potential_score`` reuse one cache. When the flag is off
# nothing here is ever touched and vanilla behavior is byte-for-byte unchanged.
# ---------------------------------------------------------------------------
class _TextEmbedder:
    """Cosine-similarity text embedder backed by the vLLM embeddings API.

    Caches embeddings by exact text so repeated room/goal strings cost nothing,
    and batches uncached texts into a single request.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8002/v1",
        model: str = "qwen3-emb-0.6b",
        api_key: str = "EMPTY",
    ):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Return {text: L2-normalized embedding} for ``texts`` (cached+batched)."""
        want = [t for t in dict.fromkeys(texts) if t]  # de-dup, drop empty
        missing = [t for t in want if t not in self._cache]
        if missing:
            resp = self._client.embeddings.create(model=self._model, input=missing)
            for t, item in zip(missing, resp.data):
                v = np.asarray(item.embedding, dtype=np.float32)
                n = float(np.linalg.norm(v))
                self._cache[t] = v / n if n > 0 else v
        return {t: self._cache[t] for t in want if t in self._cache}

    def similarity(self, a: str, b: str) -> float:
        embs = self.embed([a, b])
        if a not in embs or b not in embs:
            return 0.0
        return float(np.dot(embs[a], embs[b]))

    def max_similarity(self, candidates: List[str], goal_text: str) -> float:
        """Max cosine similarity between ``goal_text`` and any candidate string.

        Cosine on L2-normalized vectors lies in [-1, 1]; clamp to [0, 1] so the
        relevance term stays a non-negative "probability-like" weight.
        """
        cands = [c for c in candidates if c]
        if not cands or not goal_text:
            return 0.0
        embs = self.embed(cands + [goal_text])
        if goal_text not in embs:
            return 0.0
        g = embs[goal_text]
        best = 0.0
        for c in cands:
            if c in embs:
                best = max(best, float(np.dot(embs[c], g)))
        return max(0.0, best)


_SHARED_EMBEDDER: Optional[_TextEmbedder] = None


def _get_text_embedder(cfg: Optional[dict] = None) -> _TextEmbedder:
    """Lazily build (once) and return the process-wide text embedder."""
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        cfg = cfg or {}
        base_url = cfg.get(
            "embedding_base_url",
            os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8002/v1"),
        )
        model = cfg.get("embedding_model", "qwen3-emb-0.6b")
        _SHARED_EMBEDDER = _TextEmbedder(base_url=base_url, model=model)
    return _SHARED_EMBEDDER


# ---------------------------------------------------------------------------
# Qwen3-VL multimodal embedder (vLLM :8006) — FARM's own VL embedding server.
# Embeds text and images in one shared space; used by the v2 FARM-perception
# mode for image-goal relevance and the critic's feature residual. Payload
# shapes mirror FARM's SceneGraphRetriever / caption worker exactly.
# ---------------------------------------------------------------------------
class _VLEmbedder:
    _TEXT_PROMPT = (
        "Retrieve images of an object that matches this description. "
        "Focus on the object's category and visible attributes."
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8006/v1",
        model: str = "qwen3-vl-emb-2b",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._text_cache: Dict[str, np.ndarray] = {}
        self._image_cache: Dict[str, np.ndarray] = {}

    def _url(self) -> str:
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/embeddings"
        return f"{self._base_url}/v1/embeddings"

    def _post(self, content: list) -> Optional[np.ndarray]:
        import requests

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "encoding_format": "float",
        }
        try:
            resp = requests.post(self._url(), json=payload, timeout=30)
            if not resp.ok:
                return None
            rows = resp.json().get("data") or []
            vec = np.asarray(rows[0]["embedding"], dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(vec))
            return vec / n if n > 0 else vec
        except Exception:
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        if text not in self._text_cache:
            vec = self._post(
                [{"type": "text", "text": f"{self._TEXT_PROMPT}\nQuery: {text}"}]
            )
            if vec is None:
                return None
            self._text_cache[text] = vec
        return self._text_cache.get(text)

    @staticmethod
    def _to_data_uri(rgb: np.ndarray) -> Optional[str]:
        import base64
        from io import BytesIO

        from PIL import Image

        try:
            pil = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    def embed_image(self, rgb: np.ndarray, cache_key: Optional[str] = None):
        if cache_key is not None and cache_key in self._image_cache:
            return self._image_cache[cache_key]
        uri = self._to_data_uri(rgb)
        if uri is None:
            return None
        vec = self._post(
            [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "Represent the given image."},
            ]
        )
        if vec is not None and cache_key is not None:
            self._image_cache[cache_key] = vec
        return vec

    def embed_image_file(self, path: str) -> Optional[np.ndarray]:
        if path in self._image_cache:
            return self._image_cache[path]
        try:
            from PIL import Image

            rgb = np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            return None
        vec = self.embed_image(rgb)
        if vec is not None:
            self._image_cache[path] = vec
        return vec


_SHARED_VL_EMBEDDER: Optional[_VLEmbedder] = None


def _get_vl_embedder(cfg: Optional[dict] = None) -> _VLEmbedder:
    """Lazily build (once) and return the process-wide Qwen3-VL embedder."""
    global _SHARED_VL_EMBEDDER
    if _SHARED_VL_EMBEDDER is None:
        cfg = cfg or {}
        base_url = cfg.get(
            "vl_embedding_base_url",
            os.environ.get("VL_EMBED_BASE_URL", "http://127.0.0.1:8006/v1"),
        )
        model = cfg.get("vl_embedding_model", "qwen3-vl-emb-2b")
        _SHARED_VL_EMBEDDER = _VLEmbedder(base_url=base_url, model=model)
    return _SHARED_VL_EMBEDDER


def normalized_entropy(semantic_dist: SemanticDistribution) -> float:
    """Shannon entropy normalized by ln(K) to [0, 1] (0 when K <= 1).

    Open-vocab predictions emit a varying number of labels K per frontier;
    dividing by ln(K) puts every frontier's uncertainty on one scale.
    """
    k = len(semantic_dist.categories) if semantic_dist is not None else 0
    if k <= 1:
        return 0.0
    return float(semantic_dist.entropy) / float(np.log(k))


def expected_goal_relevance_from_vecs(
    categories: List[str],
    probabilities: np.ndarray,
    goal_vec: Optional[np.ndarray],
    embed_text_fn,
) -> float:
    """Sum_r P(r) * max(0, cos(E(room_r), goal_vec)) — the probability-weighted
    generalization of vanilla's Sum_r P(r) * 1[r == goal]."""
    if goal_vec is None or not categories:
        return 0.0
    total = 0.0
    for cat, prob in zip(categories, probabilities):
        v = embed_text_fn(cat)
        if v is None:
            continue
        total += float(prob) * max(0.0, float(np.dot(v, goal_vec)))
    return total


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
        # OPEN-VOCAB room prediction (default OFF -> original behavior exactly).
        self.use_open_vocab_rooms = bool(cfg.get("use_open_vocab_rooms", False))
        self._open_vocab_logged = False
        self._embedder = None
        if self.use_open_vocab_rooms:
            print(
                "[OPEN-VOCAB-ROOM] ENABLED: hypothesis room prediction is now "
                "free-text open-vocabulary (no fixed 15-room list); goal relevance "
                "uses embedding cosine similarity via the :8002 caption embedder"
            )
            self._open_vocab_logged = True

    def _get_embedder(self) -> _TextEmbedder:
        if self._embedder is None:
            self._embedder = _get_text_embedder(self.cfg)
        return self._embedder

    def room_goal_relevance(self, predicted_rooms, goal_text: str) -> float:
        """Embedding cosine relevance between predicted open-vocab room label(s)
        and the goal text. ``predicted_rooms`` may be a str, a list of strs, or a
        ``SemanticDistribution`` (its categories are used). Returns max cosine in
        [0, 1]; 0.0 if the goal text is empty (e.g. image goals -> neutral)."""
        if isinstance(predicted_rooms, SemanticDistribution):
            cands = list(predicted_rooms.categories)
        elif isinstance(predicted_rooms, str):
            cands = [predicted_rooms]
        else:
            cands = list(predicted_rooms)
        if not goal_text:
            return 0.0
        return self._get_embedder().max_similarity(cands, goal_text)

    # ---- v2 (use_farm_perception): probability-weighted relevance -------
    def expected_goal_relevance(
        self, semantic_dist: SemanticDistribution, goal_text: str
    ) -> float:
        """Sum_r P(r) * max(0, cos(E(r), E(goal_text))) via the :8002 text
        embedder — the open-vocab generalization of vanilla's P(goal room)."""
        if semantic_dist is None or not goal_text:
            return 0.0
        emb = self._get_embedder()
        embs = emb.embed(list(semantic_dist.categories) + [goal_text])
        goal_vec = embs.get(goal_text)
        return expected_goal_relevance_from_vecs(
            list(semantic_dist.categories),
            semantic_dist.probabilities,
            goal_vec,
            lambda t: embs.get(t),
        )

    def expected_goal_relevance_image(
        self, semantic_dist: SemanticDistribution, goal_image_path: str
    ) -> float:
        """Image-goal relevance: goal image and room labels embedded in the
        shared Qwen3-VL space (:8006), then the same probability-weighted sum."""
        if semantic_dist is None or not goal_image_path:
            return 0.0
        vl = _get_vl_embedder(self.cfg)
        goal_vec = vl.embed_image_file(goal_image_path)
        return expected_goal_relevance_from_vecs(
            list(semantic_dist.categories),
            semantic_dist.probabilities,
            goal_vec,
            vl.embed_text,
        )

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

        if self.use_open_vocab_rooms:
            # OPEN-VOCAB: no fixed candidate list; free-text room labels.
            sys_prompt = """You are an expert in indoor scene understanding and spatial reasoning.
Your task is to predict the semantic type of an unexplored area in a house, based on:
1. A frontier observation image showing the boundary between explored and unexplored areas
2. Spatial context information (current location, nearby objects, explored rooms)
3. Common-sense reasoning about house layouts

Predict the most likely room type(s) that lie BEYOND the frontier, using an OPEN
VOCABULARY: name the rooms in free text, as specifically as the evidence warrants
(e.g. "home office", "reading nook", "walk-in closet", "mudroom"). Do NOT restrict
yourself to any fixed list.

Return ONLY a JSON object mapping free-text room-type labels to probabilities,
e.g. {"home office": 0.6, "reading nook": 0.4}. The probabilities must sum to 1.0."""

            user_prompt = f"""Predict the room type(s) of the unexplored area shown in the frontier image.
Use open-vocabulary free-text labels (no fixed list).

Context information:
{context_description}
"""
        else:
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
            if self.use_open_vocab_rooms:
                print(
                    "[OPEN-VOCAB-ROOM] free-text room prediction: "
                    f"{list(zip(top_categories, [round(float(p), 3) for p in top_probs]))}"
                )
            else:
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
        unique_rooms = sorted(r for r in set(explored_rooms) if r is not None)
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
    goal_text: Optional[str] = None,
) -> float:
    """Exploration potential of a hypothesis node w.r.t. the goal.

    Vanilla (default): the goal-room relevance term is the probability the fixed
    predicted-category distribution assigns to ``target_category``.

    When ``cfg.use_open_vocab_rooms`` is True: the goal-room relevance term is the
    max embedding cosine similarity between the predicted (open-vocab, free-text)
    room labels and ``goal_text`` (falling back to ``target_category`` if
    ``goal_text`` is not supplied). This lets an open-vocab room like "home office"
    score highly for a "laptop" goal even though neither is in the fixed list.
    Travel-cost and entropy terms are unchanged.
    """
    if cfg.get("use_open_vocab_rooms", False):
        text = goal_text if goal_text else target_category
        if text:
            semantic_score = _get_text_embedder(cfg).max_similarity(
                list(hypothesis_node_semantic_dist.categories), text
            )
        else:
            # IMAGE goals: no text -> neutral relevance (fall back to vanilla term).
            semantic_score = hypothesis_node_semantic_dist.get_expected_semantic_score(
                target_category
            )
    else:
        semantic_score = hypothesis_node_semantic_dist.get_expected_semantic_score(
            target_category
        )
    distance_weight = cfg.get("distance_weight", 0.1)
    distance_penalty = np.exp(-distance_weight * distance)
    entropy_weight = cfg.get("entropy_weight", 0.05)
    entropy_bonus = entropy_weight * hypothesis_node_semantic_dist.entropy
    return float(semantic_score * distance_penalty + entropy_bonus)
