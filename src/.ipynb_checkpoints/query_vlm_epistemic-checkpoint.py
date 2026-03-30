"""
VLM Query Integration with Epistemic Graph Refinement
集成认识论图修正的VLM查询模块

This module extends the VLM query functionality to support:
1. Phase I: Using Ghost Node semantic predictions in exploration decisions
2. Phase II: Verifying Ghost Nodes and triggering counterfactual pruning
"""

import logging
from typing import Tuple, Optional, Union, Dict

from src.eval_utils_gpt_aeqa import explore_step
from src.tsdf_planner import TSDFPlanner, SnapShot, Frontier
from src.scene_aeqa import Scene
from src.semantic_critic import SemanticCritic


def query_vlm_for_response_with_epistemic_refinement(
    question: str,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    semantic_critic: Optional[SemanticCritic] = None,
    verbose: bool = False,
) -> Optional[Tuple[Union[SnapShot, Frontier], str, int, Dict]]:
    """
    Enhanced VLM query with Epistemic Graph Refinement

    Returns:
        Tuple of (target, reason, n_filtered_snapshots, epistemic_info)
        - epistemic_info contains Ghost Node predictions and statistics
    """

    # prepare input for vlm
    step_dict = {}

    # prepare snapshots
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    step_dict["snapshot_objects"] = {}
    step_dict["snapshot_imgs"] = {}
    for rgb_id, snapshot in scene.snapshots.items():
        step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
        step_dict["snapshot_imgs"][rgb_id] = scene.all_observations[rgb_id]

    # Phase I: Prepare frontier with Ghost Node semantic predictions
    step_dict["frontier_imgs"] = []
    step_dict["frontier_semantic_predictions"] = []

    for frontier in tsdf_planner.frontiers:
        step_dict["frontier_imgs"].append(frontier.feature)

        # Add semantic prediction information
        if hasattr(frontier, 'semantic_dist') and frontier.semantic_dist is not None:
            top_cats, top_probs = frontier.semantic_dist.get_top_k(3)
            semantic_info = {
                "predicted_categories": top_cats,
                "probabilities": [float(p) for p in top_probs],
                "entropy": float(frontier.semantic_dist.entropy)
            }
            step_dict["frontier_semantic_predictions"].append(semantic_info)
        else:
            step_dict["frontier_semantic_predictions"].append(None)

    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare question
    step_dict["question"] = question

    # Phase I: Enhance VLM prompt with semantic predictions (if enabled)
    if cfg.get("use_ghost_node_semantics_in_prompt", True):
        step_dict["use_ghost_node_semantics"] = True

    # query vlm
    outputs, snapshot_id_mapping, reason, n_filtered_snapshots = explore_step(
        step_dict, cfg, verbose=verbose
    )

    if outputs is None:
        logging.error(f"explore_step failed and returned None")
        return None

    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    # parse returned results
    try:
        target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, failed!")
        return None

    if target_type not in ["snapshot", "frontier"]:
        logging.info(f"Wrong target type: {target_type}, failed!")
        return None

    epistemic_info = {
        "epistemic_graph_stats": tsdf_planner.get_epistemic_graph_statistics(),
        "ghost_node_used": False,
        "verification_report": None
    }

    if target_type == "snapshot":
        if int(target_index) < 0 or int(target_index) >= len(snapshot_id_mapping):
            logging.info(
                f"Target index can not match real objects: {target_index}, failed!"
            )
            return None
        target_index = snapshot_id_mapping[int(target_index)]
        logging.info(f"The index of target snapshot {target_index}")

        # get the target snapshot
        if target_index < 0 or target_index >= len(scene.snapshots):
            logging.info(
                f"Predicted snapshot target index out of range: {target_index}, failed!"
            )
            return None

        pred_target_snapshot = list(scene.snapshots.values())[target_index]
        logging.info(
            "Pred_target_class: "
            + str(
                " ".join(
                    [
                        object_id_to_name[obj_id]
                        for obj_id in pred_target_snapshot.cluster
                    ]
                )
            )
        )
        logging.info(f"Next choice Snapshot of {pred_target_snapshot.image}")

        return pred_target_snapshot, reason, n_filtered_snapshots, epistemic_info

    else:  # target_type == "frontier"
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, failed!"
            )
            return None

        target_point = tsdf_planner.frontiers[target_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_index]

        # Phase I: Log Ghost Node prediction
        if hasattr(pred_target_frontier, 'semantic_dist') and pred_target_frontier.semantic_dist is not None:
            top_cats, top_probs = pred_target_frontier.semantic_dist.get_top_k(3)
            logging.info(f"Ghost Node Prediction: {list(zip(top_cats, [f'{p:.2f}' for p in top_probs]))}")
            epistemic_info["ghost_node_used"] = True
            epistemic_info["predicted_semantics"] = {
                "categories": top_cats,
                "probabilities": [float(p) for p in top_probs]
            }

        return pred_target_frontier, reason, n_filtered_snapshots, epistemic_info


def verify_ghost_node_arrival(
    frontier: Frontier,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    semantic_critic: SemanticCritic,
    actual_rgb: any,
    actual_depth: any,
    detected_objects: list,
) -> Dict:
    """
    Phase II: Verify Ghost Node when agent arrives at a frontier

    This function implements the counterfactual causal pruning:
    1. Compare expected vs. actual observation
    2. Compute semantic residual
    3. If falsified, diagnose cause and trigger cascade deletion

    Args:
        frontier: The frontier (Ghost Node) the agent just arrived at
        scene: Current scene
        tsdf_planner: TSDF planner containing epistemic graph
        semantic_critic: Semantic critic for verification
        actual_rgb: Actual RGB observation at frontier
        actual_depth: Actual depth observation
        detected_objects: List of detected object classes

    Returns:
        Verification report dictionary
    """

    if not hasattr(frontier, 'ghost_node_id') or frontier.ghost_node_id is None:
        logging.warning(f"Frontier has no associated Ghost Node, skipping verification")
        return {"skipped": True, "reason": "No Ghost Node"}

    # Retrieve Ghost Node from epistemic graph
    ghost_node_id = frontier.ghost_node_id
    epistemic_graph = tsdf_planner.epistemic_graph

    if ghost_node_id not in epistemic_graph.nodes:
        logging.warning(f"Ghost Node {ghost_node_id} not found in epistemic graph")
        return {"skipped": True, "reason": "Ghost Node not found"}

    ghost_node = epistemic_graph.nodes[ghost_node_id]

    # Prepare actual observation
    actual_observation = {
        "rgb_image": actual_rgb,
        "depth": actual_depth,
        "detected_objects": detected_objects,
        "semantic_class": infer_room_type_from_objects(detected_objects) if detected_objects else None
    }

    # Phase II: Verify Ghost Node
    logging.info(f"[Phase II] Verifying Ghost Node {ghost_node_id}")

    verification_report = semantic_critic.verify_ghost_node_arrival(
        ghost_node=ghost_node,
        actual_observation=actual_observation,
        scene_objects=scene.objects
    )

    # Log verification results
    if verification_report.is_falsified:
        logging.warning(f"""
[Phase II] GHOST NODE FALSIFIED
  Node ID: {ghost_node_id}
  Semantic Residual: {verification_report.semantic_residual:.3f}
  Physical Cause: {verification_report.physical_cause}
  Explanation: {verification_report.textual_explanation}
  Cascade Deleted: {len(verification_report.cascade_deleted_nodes)} nodes
""")
    else:
        logging.info(f"""
[Phase II] GHOST NODE VERIFIED
  Node ID: {ghost_node_id}
  Semantic Residual: {verification_report.semantic_residual:.3f}
  Explanation: {verification_report.textual_explanation}
""")

    return {
        "skipped": False,
        "ghost_node_id": ghost_node_id,
        "is_falsified": verification_report.is_falsified,
        "semantic_residual": verification_report.semantic_residual,
        "physical_cause": verification_report.physical_cause,
        "explanation": verification_report.textual_explanation,
        "cascade_deleted_count": len(verification_report.cascade_deleted_nodes),
        "cascade_deleted_nodes": verification_report.cascade_deleted_nodes
    }


def infer_room_type_from_objects(detected_objects: list) -> Optional[str]:
    """
    Infer room type from detected objects using simple heuristics

    This is a simplified version - can be enhanced with VLM
    """
    if not detected_objects:
        return None

    object_set = set([obj.lower() for obj in detected_objects])

    # Room inference rules
    if any(obj in object_set for obj in ["bed", "nightstand", "wardrobe"]):
        return "bedroom"
    elif any(obj in object_set for obj in ["toilet", "sink", "shower", "bathtub"]):
        return "bathroom"
    elif any(obj in object_set for obj in ["stove", "oven", "refrigerator"]):
        return "kitchen"
    elif any(obj in object_set for obj in ["sofa", "tv", "coffee_table"]):
        return "living_room"
    elif any(obj in object_set for obj in ["dining_table"]):
        return "dining_room"
    else:
        return "unknown"


def get_frontier_with_best_semantic_score(
    frontiers: list,
    target_category: str,
    cfg: dict
) -> Optional[Frontier]:
    """
    Select the frontier with highest semantic potential score for target category

    This can be used as an alternative to VLM-based frontier selection
    """
    if not frontiers:
        return None

    best_frontier = None
    best_score = -float('inf')

    for frontier in frontiers:
        if hasattr(frontier, 'semantic_dist') and frontier.semantic_dist is not None:
            semantic_score = frontier.semantic_dist.get_expected_semantic_score(target_category)

            # You can also add distance penalty here
            # score = compute_semantic_potential_score(frontier.semantic_dist, target_category, distance, cfg)

            if semantic_score > best_score:
                best_score = semantic_score
                best_frontier = frontier

    return best_frontier
