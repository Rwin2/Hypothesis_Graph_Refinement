"""REAL FARM scene-graph integration for HGR.

This module drives FARM's *actual* perception pipeline in-process
(``FARM-Project/src/scene_graph``): open-vocabulary YOLOE segmentation +
DINOv3 merge features + 3D Gaussian fusion + union-find correspondence +
async vLLM captioning + multimodal embeddings. HGR's own per-frame RGB-D
observations are fed straight into it, so the object graph HGR's hypothesis
reasoning reads is FARM's real ``scene_state`` (means / object_category /
object_caption / object_id / active), not HGR's ConceptGraph.

Only used when ``cfg.hypothesis.use_farm_graph`` is True. When the flag is
off this class is never instantiated in the hot path and vanilla behavior is
byte-for-byte unchanged.

Coordinate convention (mirrors FARM's ``scripts/habitat_online_sim.py``
``sensor_pose_opencv``): FARM expects ``T_world_cam`` in the OpenCV camera
convention (camera looks down +Z, +x right, +y down), with the rotation
``R_habitat @ diag(1,-1,-1)`` and the habitat sensor position as translation.
HGR's ``cam_pose`` (``scene.get_observation``) is ``R_habitat`` directly with
the habitat sensor position, so::

    T_world_cam_FARM = cam_pose_hgr @ diag(1, -1, -1, 1)

Because we feed FARM the ABSOLUTE habitat pose (no first-frame re-basing, unlike
``PipelineOrchestrator`` which applies ``relative_pose``), FARM's object means
come out directly in HGR's habitat world frame — the same frame HGR's frontier
queries use via ``habitat2voxel``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

try:  # torch is always present in HGR; keep the import defensive anyway.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# FARM's exact OpenGL->OpenCV mapping (habitat_online_sim.py _OPENGL_TO_OPENCV).
_OPENGL_TO_OPENCV = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
)


def _farm_T_world_cam(cam_pose_hgr: np.ndarray) -> np.ndarray:
    """HGR ``cam_pose`` -> FARM ``T_world_cam`` (OpenCV cam-to-world).

    Applies the OpenGL->OpenCV basis change to the rotation, exactly as FARM's
    ``sensor_pose_opencv`` does (``R = R_habitat @ _OPENGL_TO_OPENCV``). The
    translation (habitat sensor position) is unchanged.
    """
    T = np.asarray(cam_pose_hgr, dtype=np.float32).copy()
    T[:3, :3] = T[:3, :3] @ _OPENGL_TO_OPENCV
    return T


class FarmLiveGraph:
    """In-process driver of FARM's real perception pipeline.

    Keeps the interface HGR's ``_build_spatial_context`` already consumes:
    ``num_objects()`` and an ``objects`` list of dicts with ``"mean"`` (habitat
    world-frame centroid) and ``"object_category"``. Adds ``integrate_frame`` to
    push each HGR RGB-D frame through FARM, and ``query_nearby`` for radius
    queries in the habitat world frame.
    """

    def __init__(
        self,
        device: str = "cuda",
        feature_dim: int = 384,
        vllm_base_url: str = "http://127.0.0.1:8000/v1",
        caption_step_interval: int = 5,
        caption_start_step: int = 3,
        use_dino_features: bool = True,
        lazy: bool = True,
    ):
        self._device = device
        self._feature_dim = int(feature_dim)
        self._vllm_base_url = vllm_base_url
        self._use_dino_features = bool(use_dino_features)
        self._caption_step_interval = int(caption_step_interval)
        self._caption_start_step = int(caption_start_step)

        # Lazily-initialized FARM objects (heavy: loads YOLOE + DINOv3 + starts
        # the caption worker). Kept None until the first frame so importing this
        # module (e.g. when the flag is off) is free.
        self._segmenter = None
        self._scene_state = None
        self._caption_manager = None
        self._steps = None  # module handle for the pipeline step functions
        self._register_batch_images = None
        self._compute_caption_observations = None
        self._compute_detection_image_ids = None

        self._step_index = 0
        self._pending_caption_indices: List[int] = []
        self._enabled_logged = False
        self._initialized = False
        if not lazy:
            self._ensure_initialized()

    # ------------------------------------------------------------------
    # FARM pipeline lifecycle
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        # FARM's src must be importable (PYTHONPATH set by the launch command).
        from scene_graph.segmentation.yoloe import YOLOESegmenter
        from scene_graph.segmentation.dino import DINOFeaturesExtractor
        from scene_graph.map_update.models import initialize_scene_graph_state
        from scene_graph.captioning.services import CaptionManager
        from scene_graph.pipeline import steps as farm_steps
        from scene_graph.storage.image_save_worker import register_batch_images
        from scene_graph.captioning.crop_util import compute_caption_observations

        # FARM's caption worker reads VLLM_BASE_URL from the environment.
        os.environ.setdefault("VLLM_BASE_URL", self._vllm_base_url)

        dino_extractor = None
        if self._use_dino_features:
            # Auto-resolves the in-repo dinov3-vits16 backbone (feature_dim 384).
            dino_extractor = DINOFeaturesExtractor(
                model="dinov3-vits16", load_size=512, device=self._device
            )
        self._segmenter = YOLOESegmenter(
            model_id="yoloe-v8l",
            device=self._device,
            use_dino_features=self._use_dino_features,
            dino_extractor=dino_extractor,
        )
        self._feature_dim = int(self._segmenter.feature_dim)
        self._scene_state = initialize_scene_graph_state(
            self._feature_dim, self._segmenter.device
        )
        self._caption_manager = CaptionManager(
            scene_state=self._scene_state,
            enabled=True,
            caption_device="cuda:0",
            caption_server="vllm",
        )
        self._caption_manager.maybe_start_worker()
        try:
            self._caption_manager.warm_up_model()
        except Exception as exc:  # pragma: no cover - warm-up is best-effort
            LOGGER.warning("[FARM-REAL] caption warm-up failed: %s", exc)

        self._steps = farm_steps
        self._register_batch_images = register_batch_images
        self._compute_caption_observations = compute_caption_observations
        self._compute_detection_image_ids = farm_steps.compute_detection_image_ids
        self._initialized = True

    # ------------------------------------------------------------------
    # Per-frame integration
    # ------------------------------------------------------------------
    def integrate_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics,
        cam_pos_habitat: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> None:
        """Push one HGR RGB-D frame through FARM's real perception pipeline.

        Mirrors the per-batch step sequence FARM's ``PipelineOrchestrator`` /
        streaming mapper run, but feeds the ABSOLUTE habitat pose so object
        means land in HGR's habitat world frame.

        Args:
            rgb: (H, W, 3) uint8.
            depth: (H, W) float32 metric depth (meters).
            intrinsics: 3x3 or 4x4 camera intrinsics (fx, fy, cx, cy).
            cam_pos_habitat: HGR's 4x4 ``cam_pose`` from ``get_observation``.
            frame_idx: optional frame counter for periodic logging.
        """
        self._ensure_initialized()

        rgb_np = np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)
        depth_np = np.ascontiguousarray(np.asarray(depth), dtype=np.float32)
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]

        K = np.asarray(intrinsics, dtype=np.float32)
        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = K[:3, :3]

        T_wc = _farm_T_world_cam(np.asarray(cam_pos_habitat, dtype=np.float32))

        dev = self._segmenter.device
        color_t = torch.from_numpy(rgb_np).to(dev)
        depth_t = torch.from_numpy(depth_np).to(dev)
        K_t = torch.from_numpy(K4).to(dev)
        pose_t = torch.from_numpy(T_wc)

        steps = self._steps
        scene_state = self._scene_state
        cm = self._caption_manager

        # Register the frame FIRST (as the orchestrator does) so detection image
        # ids are valid for caption crops.
        batch_image_ids = self._register_batch_images(
            scene_state, [pose_t], camera_ids=["hgr"]
        )

        seg = steps.segment_and_transform(
            self._segmenter, [color_t], [depth_t], [K_t], [pose_t]
        )
        n_det = len(seg.get("means", []))
        detection_image_ids = self._compute_detection_image_ids(
            seg, batch_image_ids, n_det
        )

        rgb_obs = (
            self._compute_caption_observations(seg, [color_t]) if n_det else []
        )

        neighbors, _ = steps.find_neighbors_for_detections(seg, scene_state)
        det_idx, obj_idx = steps.resolve_correspondence(
            neighbors,
            scene_state["count"].shape[0],
            scene_state=scene_state,
            detection_image_ids=detection_image_ids,
            seg_outputs=seg,
        )
        update_info = steps.update_state_and_enqueue_captions(
            scene_state,
            seg,
            det_idx,
            obj_idx,
            rgb_obs,
            detection_image_ids,
            cm,
            self._pending_caption_indices,
        )

        # Per-frame detection -> FARM object mapping, for the FARM-only
        # perception mode (use_farm_perception). Mirrors the det_to_obj
        # reconstruction inside steps.update_state_and_enqueue_captions.
        frame_result = {"detections": [], "new_object_ids": []}
        if n_det:
            det_to_slot = [None] * n_det
            if det_idx.numel() > 0 and obj_idx.numel() > 0:
                det_idx_cpu = det_idx.detach().to("cpu", dtype=torch.long)
                obj_idx_cpu = obj_idx.detach().to("cpu", dtype=torch.long)
                for det_i, raw_obj_idx in enumerate(det_idx_cpu.tolist()):
                    if 0 <= int(raw_obj_idx) < obj_idx_cpu.numel():
                        det_to_slot[det_i] = int(obj_idx_cpu[int(raw_obj_idx)].item())
                new_det_indices = [
                    int(i) for i, v in enumerate(det_idx_cpu.tolist()) if v < 0
                ]
                new_object_slots = [
                    int(x)
                    for x in ((update_info or {}).get("new_object_indices", []) or [])
                ]
                for det_i, obj_i in zip(new_det_indices, new_object_slots):
                    if 0 <= det_i < len(det_to_slot):
                        det_to_slot[det_i] = int(obj_i)
            oid_t = scene_state.get("object_id")
            boxes = seg.get("boxes_xyxy")
            scores = seg.get("scores")
            masks = seg.get("masks")
            boxes_np = (
                boxes.detach().cpu().numpy()
                if boxes is not None and hasattr(boxes, "detach")
                else np.asarray(boxes) if boxes is not None else None
            )
            scores_np = (
                scores.detach().cpu().numpy()
                if scores is not None and hasattr(scores, "detach")
                else np.asarray(scores) if scores is not None else None
            )
            n_slots = int(oid_t.shape[0]) if oid_t is not None else 0
            for det_i in range(n_det):
                slot = det_to_slot[det_i]
                if slot is None or not (0 <= slot < n_slots):
                    continue
                ext_id = int(oid_t[slot].item())
                mask_np = None
                if masks is not None and det_i < len(masks):
                    m = masks[det_i]
                    mask_np = (
                        m.detach().cpu().numpy().astype(bool)
                        if hasattr(m, "detach")
                        else np.asarray(m).astype(bool)
                    )
                frame_result["detections"].append(
                    {
                        "object_id": ext_id,
                        "bbox_xyxy": (
                            boxes_np[det_i].astype(np.float32)
                            if boxes_np is not None and det_i < len(boxes_np)
                            else None
                        ),
                        "score": (
                            float(scores_np[det_i])
                            if scores_np is not None and det_i < len(scores_np)
                            else 1.0
                        ),
                        "mask": mask_np,
                    }
                )
            new_slots = [
                int(x) for x in ((update_info or {}).get("new_object_indices", []) or [])
            ]
            frame_result["new_object_ids"] = [
                int(oid_t[s].item()) for s in new_slots if 0 <= s < n_slots
            ]

        # Periodically enqueue pending captions + drain finished ones, exactly
        # like the orchestrator's caption cadence.
        should_process = (
            self._step_index >= self._caption_start_step
            and (self._step_index - self._caption_start_step)
            % self._caption_step_interval
            == 0
        )
        if should_process and self._pending_caption_indices:
            cm.enqueue_objects(self._pending_caption_indices)
            self._pending_caption_indices.clear()
        if should_process:
            cm.drain_results()
        self._step_index += 1

        if not self._enabled_logged:
            LOGGER.info(
                "[FARM-REAL] ENABLED: HGR frames build FARM's real scene graph "
                "(YOLOE open-vocab + DINOv3 + vLLM captions), feature_dim=%d",
                self._feature_dim,
            )
            self._enabled_logged = True

        if frame_idx is not None and frame_idx % 10 == 0:
            active = scene_state.get("active")
            n_active = int(active.sum().item()) if active is not None and active.numel() else 0
            caps = [c for c in (scene_state.get("object_caption") or []) if c]
            sample = caps[-2:] if caps else []
            LOGGER.info(
                "[FARM-REAL] frame %s: %d FARM objects (%d captioned); sample captions=%s",
                frame_idx, n_active, len(caps), sample,
            )

        return frame_result

    # ------------------------------------------------------------------
    # Read interface consumed by HGR's hypothesis reasoning
    # ------------------------------------------------------------------
    @property
    def objects(self) -> List[dict]:
        """FARM objects as ``{"object_id", "mean", "object_category"}`` dicts.

        ``object_category`` prefers FARM's captioned category, falling back to
        the caption text or the YOLOE label so a frontier query is never blank
        for a real object. Means are in HGR's habitat world frame.
        """
        if not self._initialized or self._scene_state is None:
            return []
        ss = self._scene_state
        means = ss.get("means")
        if means is None or means.shape[0] == 0:
            return []
        means_np = means.detach().cpu().numpy()
        active = ss.get("active")
        active_np = (
            active.detach().cpu().numpy()
            if active is not None and active.numel()
            else np.ones(means_np.shape[0], dtype=bool)
        )
        oid = ss.get("object_id")
        oid_np = (
            oid.detach().cpu().numpy()
            if oid is not None and oid.numel()
            else np.arange(means_np.shape[0])
        )
        cats = ss.get("object_category") or []
        caps = ss.get("object_caption") or []
        names = self._farm_class_names()
        class_ids = ss.get("class_ids")
        class_ids_np = (
            class_ids.detach().cpu().numpy()
            if class_ids is not None and class_ids.numel()
            else None
        )

        out: List[dict] = []
        for i in range(means_np.shape[0]):
            if not active_np[i]:
                continue
            m = means_np[i]
            if not np.all(np.isfinite(m)):
                continue
            cat = cats[i] if i < len(cats) else ""
            if not cat and i < len(caps) and caps[i]:
                cat = caps[i]
            if not cat and class_ids_np is not None and names is not None:
                cid = int(class_ids_np[i])
                if 0 <= cid < len(names):
                    cat = names[cid]
            if not cat:
                continue
            out.append(
                {
                    "object_id": int(oid_np[i]),
                    "mean": m.astype(np.float32),
                    "object_category": str(cat),
                }
            )
        return out

    def resolve_object_id(self, object_id: int) -> Optional[int]:
        """Follow FARM's ``id_redirect`` chain to the canonical external id.

        Returns None if the resolved object no longer exists or is inactive.
        """
        if not self._initialized or self._scene_state is None:
            return None
        ss = self._scene_state
        current = int(object_id)
        redirect = ss.get("id_redirect") or {}
        seen = set()
        while current in redirect and current not in seen:
            seen.add(current)
            current = int(redirect[current])
        oid = ss.get("object_id")
        active = ss.get("active")
        if oid is None or oid.numel() == 0:
            return None
        slots = (oid == current).nonzero()
        if slots.numel() == 0:
            return None
        slot = int(slots[0].item())
        if active is not None and active.numel() > slot and not bool(active[slot].item()):
            return None
        return current

    def objects_full(self) -> dict:
        """Canonical-external-id -> object info for the perception mirror.

        ``name`` prefers the FARM caption (open-vocab, the decision-prompt
        label), then the captioned category, then the raw YOLOE class name.
        """
        out = {}
        if not self._initialized or self._scene_state is None:
            return out
        ss = self._scene_state
        means = ss.get("means")
        if means is None or means.shape[0] == 0:
            return out
        means_np = means.detach().cpu().numpy()
        active = ss.get("active")
        active_np = (
            active.detach().cpu().numpy()
            if active is not None and active.numel()
            else np.ones(means_np.shape[0], dtype=bool)
        )
        oid = ss.get("object_id")
        oid_np = (
            oid.detach().cpu().numpy()
            if oid is not None and oid.numel()
            else np.arange(means_np.shape[0])
        )
        count = ss.get("count")
        count_np = (
            count.detach().cpu().numpy()
            if count is not None and count.numel()
            else np.ones(means_np.shape[0], dtype=np.int32)
        )
        cats = ss.get("object_category") or []
        caps = ss.get("object_caption") or []
        names = self._farm_class_names()
        class_ids = ss.get("class_ids")
        class_ids_np = (
            class_ids.detach().cpu().numpy()
            if class_ids is not None and class_ids.numel()
            else None
        )
        for i in range(means_np.shape[0]):
            if not active_np[i]:
                continue
            m = means_np[i]
            if not np.all(np.isfinite(m)):
                continue
            name = caps[i] if i < len(caps) and caps[i] else ""
            if not name and i < len(cats) and cats[i]:
                name = cats[i]
            if not name and class_ids_np is not None and names is not None:
                cid = int(class_ids_np[i])
                if 0 <= cid < len(names):
                    name = names[cid]
            if not name:
                name = "object"
            out[int(oid_np[i])] = {
                "object_id": int(oid_np[i]),
                "mean": m.astype(np.float32),
                "name": str(name),
                "num_detections": int(count_np[i]) if i < len(count_np) else 1,
            }
        return out

    def _farm_class_names(self) -> Optional[List[str]]:
        if self._segmenter is not None:
            names = getattr(self._segmenter, "names", None)
            if names:
                return list(names)
        return None

    def num_objects(self) -> int:
        if not self._initialized or self._scene_state is None:
            return 0
        active = self._scene_state.get("active")
        if active is None or active.numel() == 0:
            return 0
        return int(active.sum().item())

    def __len__(self) -> int:
        return self.num_objects()

    def query_nearby(self, position: np.ndarray, radius: float) -> List[str]:
        """Return distinct ``object_category`` of FARM objects within ``radius``
        of ``position`` (both in HGR's habitat world frame)."""
        position = np.asarray(position, dtype=np.float32).reshape(-1)
        cats: List[str] = []
        seen = set()
        for obj in self.objects:
            m = obj["mean"]
            dim = min(position.shape[0], m.shape[0])
            dist = float(np.linalg.norm(m[:dim] - position[:dim]))
            if dist < float(radius):
                cat = obj["object_category"]
                if cat not in seen:
                    seen.add(cat)
                    cats.append(cat)
        return cats

    def flush_captions(self, timeout: float = 10.0) -> None:
        """Drain any outstanding captions (call at episode end for completeness)."""
        if not self._initialized or self._caption_manager is None:
            return
        try:
            if self._pending_caption_indices:
                self._caption_manager.enqueue_objects(self._pending_caption_indices)
                self._pending_caption_indices.clear()
            self._caption_manager.wait_until_idle(timeout=timeout, poll_interval=0.05)
            self._caption_manager.drain_results()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("[FARM-REAL] flush_captions failed: %s", exc)
