"""VLM query helpers for Hypothesis Graph Refinement."""

import logging
from typing import Dict, Optional, Tuple, Union

from src.eval_utils_gpt_aeqa import explore_step
from src.scene_aeqa import Scene
from src.semantic_critic import SemanticCritic
from src.tsdf_planner import Frontier, SnapShot, TSDFPlanner


def query_vlm_for_response_with_hypothesis_refinement(
    question: str,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    semantic_critic: Optional[SemanticCritic] = None,
    verbose: bool = False,
) -> Optional[Tuple[Union[SnapShot, Frontier], str, int, Dict]]:
    del semantic_critic

    step_dict = {}
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    step_dict["snapshot_objects"] = {}
    step_dict["snapshot_imgs"] = {}
    for rgb_id, snapshot in scene.snapshots.items():
        step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
        step_dict["snapshot_imgs"][rgb_id] = scene.all_observations[rgb_id]

    step_dict["frontier_imgs"] = []
    step_dict["frontier_semantic_predictions"] = []
    for frontier in tsdf_planner.frontiers:
        step_dict["frontier_imgs"].append(frontier.feature)
        if hasattr(frontier, "semantic_dist") and frontier.semantic_dist is not None:
            top_cats, top_probs = frontier.semantic_dist.get_top_k(3)
            step_dict["frontier_semantic_predictions"].append(
                {
                    "predicted_categories": top_cats,
                    "probabilities": [float(p) for p in top_probs],
                    "entropy": float(frontier.semantic_dist.entropy),
                }
            )
        else:
            step_dict["frontier_semantic_predictions"].append(None)

    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    step_dict["question"] = question

    if cfg.get("use_hypothesis_node_semantics_in_prompt", True):
        step_dict["use_hypothesis_node_semantics"] = True

    outputs, snapshot_id_mapping, reason, n_filtered_snapshots = explore_step(
        step_dict, cfg, verbose=verbose
    )
    if outputs is None:
        logging.error("explore_step failed and returned None")
        return None

    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    try:
        target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
        logging.info(f"Prediction: {target_type}, {target_index}")
    except Exception:
        logging.info("Wrong output format, failed!")
        return None

    if target_type not in ["snapshot", "frontier"]:
        logging.info(f"Wrong target type: {target_type}, failed!")
        return None

    hypothesis_info = {
        "hypothesis_graph_stats": tsdf_planner.get_hypothesis_graph_statistics(),
        "hypothesis_node_used": False,
        "verification_report": None,
    }

    if target_type == "snapshot":
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(snapshot_id_mapping):
            logging.info(
                f"Target index can not match real objects: {target_index}, failed!"
            )
            return None

        target_index = snapshot_id_mapping[target_index]
        if target_index < 0 or target_index >= len(scene.snapshots):
            logging.info(
                f"Predicted snapshot target index out of range: {target_index}, failed!"
            )
            return None

        pred_target_snapshot = list(scene.snapshots.values())[target_index]
        logging.info(
            "Pred_target_class: "
            + " ".join(
                object_id_to_name[obj_id] for obj_id in pred_target_snapshot.cluster
            )
        )
        logging.info(f"Next choice Snapshot of {pred_target_snapshot.image}")
        return pred_target_snapshot, reason, n_filtered_snapshots, hypothesis_info

    target_index = int(target_index)
    if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
        logging.info(
            f"Predicted frontier target index out of range: {target_index}, failed!"
        )
        return None

    pred_target_frontier = tsdf_planner.frontiers[target_index]
    logging.info(f"Next choice: Frontier at {pred_target_frontier.position}")

    if (
        hasattr(pred_target_frontier, "semantic_dist")
        and pred_target_frontier.semantic_dist is not None
    ):
        top_cats, top_probs = pred_target_frontier.semantic_dist.get_top_k(3)
        logging.info(
            "Hypothesis Node Prediction: "
            + str(list(zip(top_cats, [f"{p:.2f}" for p in top_probs])))
        )
        hypothesis_info["hypothesis_node_used"] = True
        hypothesis_info["predicted_semantics"] = {
            "categories": top_cats,
            "probabilities": [float(p) for p in top_probs],
        }

    return pred_target_frontier, reason, n_filtered_snapshots, hypothesis_info


def verify_hypothesis_node_arrival(
    frontier: Frontier,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    semantic_critic: SemanticCritic,
    actual_rgb,
    actual_depth,
    detected_objects: list,
) -> Dict:
    if (
        not hasattr(frontier, "hypothesis_node_id")
        or frontier.hypothesis_node_id is None
    ):
        logging.warning(
            "Frontier has no associated hypothesis node, skipping verification"
        )
        return {"skipped": True, "reason": "No hypothesis node"}

    hypothesis_node_id = frontier.hypothesis_node_id
    hypothesis_graph = tsdf_planner.hypothesis_graph

    if hypothesis_node_id not in hypothesis_graph.nodes:
        logging.warning(
            f"Hypothesis node {hypothesis_node_id} not found in hypothesis graph"
        )
        return {"skipped": True, "reason": "Hypothesis node not found"}

    hypothesis_node = hypothesis_graph.nodes[hypothesis_node_id]
    actual_observation = {
        "rgb_image": actual_rgb,
        "depth": actual_depth,
        "detected_objects": detected_objects,
        "semantic_class": infer_room_type_from_objects(detected_objects)
        if detected_objects
        else None,
    }

    logging.info(f"[Phase II] Verifying Hypothesis Node {hypothesis_node_id}")
    verification_report = semantic_critic.verify_hypothesis_node_arrival(
        hypothesis_node=hypothesis_node,
        actual_observation=actual_observation,
        scene_objects=scene.objects,
    )

    phase_label = "FALSIFIED" if verification_report.is_falsified else "VERIFIED"
    log_fn = logging.warning if verification_report.is_falsified else logging.info
    log_fn(
        f"""
[Phase II] HYPOTHESIS NODE {phase_label}
  Node ID: {hypothesis_node_id}
  Semantic Residual: {verification_report.semantic_residual:.3f}
  Physical Cause: {verification_report.physical_cause}
  Explanation: {verification_report.textual_explanation}
  Cascade Deleted: {len(verification_report.cascade_deleted_nodes)} nodes
"""
    )

    return {
        "skipped": False,
        "hypothesis_node_id": hypothesis_node_id,
        "is_falsified": verification_report.is_falsified,
        "semantic_residual": verification_report.semantic_residual,
        "physical_cause": verification_report.physical_cause,
        "explanation": verification_report.textual_explanation,
        "cascade_deleted_count": len(verification_report.cascade_deleted_nodes),
        "cascade_deleted_nodes": verification_report.cascade_deleted_nodes,
    }


def infer_room_type_from_objects(detected_objects: list) -> Optional[str]:
    if not detected_objects:
        return None

    object_set = {obj.lower() for obj in detected_objects}
    if any(obj in object_set for obj in ["bed", "nightstand", "wardrobe"]):
        return "bedroom"
    if any(obj in object_set for obj in ["toilet", "sink", "shower", "bathtub"]):
        return "bathroom"
    if any(obj in object_set for obj in ["stove", "oven", "refrigerator"]):
        return "kitchen"
    if any(obj in object_set for obj in ["sofa", "tv", "coffee_table"]):
        return "living_room"
    if "dining_table" in object_set:
        return "dining_room"
    return None


def get_frontier_with_best_semantic_score(
    tsdf_planner: TSDFPlanner, target_category: str
) -> Optional[Frontier]:
    best_frontier = None
    best_score = float("-inf")
    for frontier in tsdf_planner.frontiers:
        if not hasattr(frontier, "semantic_dist") or frontier.semantic_dist is None:
            continue
        score = frontier.semantic_dist.get_expected_semantic_score(target_category)
        if score > best_score:
            best_score = score
            best_frontier = frontier
    return best_frontier
