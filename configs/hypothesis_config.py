"""
Hypothesis Graph Refinement Configuration

This file contains all configuration parameters for the Hypothesis Graph Refinement system.
"""

# ============================================================================
# Phase I: Hypothesis Node Semantic Prediction Configuration
# ============================================================================

HYPOTHESIS_NODE_PREDICTION_CONFIG = {
    # Enable/Disable hypothesis refinement
    "enable_hypothesis_refinement": True,

    # VLM-based semantic prediction
    "enable_vlm_hypothesis_prediction": True,  # Use VLM for semantic prediction (costly but accurate)
    "hypothesis_node_top_k_categories": 5,  # Number of top categories to keep in distribution

    # Context usage
    "use_spatial_context": True,  # Use nearby objects for prediction
    "use_visual_cues": True,  # Use frontier image for prediction

    # Semantic potential scoring (for path planning)
    "distance_weight": 0.1,  # Weight for distance penalty in semantic score
    "entropy_weight": 0.05,  # Weight for entropy bonus (exploration)
}


# ============================================================================
# Phase II: Semantic Critic & Counterfactual Pruning Configuration
# ============================================================================

SEMANTIC_CRITIC_CONFIG = {
    # Semantic residual threshold
    "semantic_residual_threshold": 0.5,  # Above this -> falsified

    # Residual computation weights
    "residual_weight_category": 0.4,  # Weight for category matching
    "residual_weight_feature": 0.3,  # Weight for CLIP feature similarity
    "residual_weight_objects": 0.3,  # Weight for object-level matching

    # VLM-based causality diagnosis
    "enable_vlm_causality_diagnosis": True,  # Use VLM to diagnose falsification cause

    # Falsification confidence
    "min_falsification_confidence": 0.7,  # Minimum confidence to trigger falsification
}


# ============================================================================
# Hypothesis Graph Configuration
# ============================================================================

HYPOTHESIS_GRAPH_CONFIG = {
    # Dependency tracking
    "min_dependency_confidence": 0.6,  # Minimum confidence for dependency creation

    # Cascade deletion
    "enable_cascade_deletion": True,  # Enable cascade deletion of dependent nodes

    # Expectation violation detection
    "expectation_violation_threshold": 0.6,  # Threshold for detecting violations
}


# ============================================================================
# Integration with VLM Query
# ============================================================================

VLM_QUERY_CONFIG = {
    # Use Hypothesis Node semantics in VLM prompt
    "use_hypothesis_node_semantics_in_prompt": True,

    # Enhanced frontier selection
    "use_semantic_potential_for_selection": False,  # Use semantic score instead of VLM
}


# ============================================================================
# Complete Configuration (merge all configs)
# ============================================================================

COMPLETE_HYPOTHESIS_CONFIG = {
    **HYPOTHESIS_NODE_PREDICTION_CONFIG,
    **SEMANTIC_CRITIC_CONFIG,
    **HYPOTHESIS_GRAPH_CONFIG,
    **VLM_QUERY_CONFIG,
}


# ============================================================================
# Preset Configurations
# ============================================================================

# Full system enabled (best performance but most costly)
FULL_SYSTEM_CONFIG = COMPLETE_HYPOTHESIS_CONFIG.copy()

# Heuristic-only (fast but less accurate)
HEURISTIC_ONLY_CONFIG = {
    **COMPLETE_HYPOTHESIS_CONFIG,
    "enable_vlm_hypothesis_prediction": False,
    "enable_vlm_causality_diagnosis": False,
}

# Lightweight (minimal overhead)
LIGHTWEIGHT_CONFIG = {
    **COMPLETE_HYPOTHESIS_CONFIG,
    "enable_hypothesis_refinement": False,
}

# Research mode (maximum logging and statistics)
RESEARCH_CONFIG = {
    **COMPLETE_HYPOTHESIS_CONFIG,
    "enable_detailed_logging": True,
    "save_hypothesis_graph_visualizations": True,
    "track_hallucination_statistics": True,
}


# ============================================================================
# Helper function to get configuration
# ============================================================================

def get_hypothesis_config(preset: str = "full") -> dict:
    """
    Get hypothesis configuration by preset name

    Args:
        preset: Configuration preset name
            - "full": Full system with all features
            - "heuristic": Heuristic-only (no VLM for prediction/diagnosis)
            - "lightweight": Minimal overhead (disabled)
            - "research": Maximum logging and statistics
            - "custom": Custom configuration (returns default full config)

    Returns:
        Configuration dictionary
    """
    presets = {
        "full": FULL_SYSTEM_CONFIG,
        "heuristic": HEURISTIC_ONLY_CONFIG,
        "lightweight": LIGHTWEIGHT_CONFIG,
        "research": RESEARCH_CONFIG,
        "custom": COMPLETE_HYPOTHESIS_CONFIG,
    }

    return presets.get(preset, FULL_SYSTEM_CONFIG).copy()
