# HGR: Hypothesis Graph Refinement

Hypothesis Graph Refinement (HGR) is a visual-language navigation and embodied question answering codebase built around frontier-based exploration, TSDF mapping, scene graph construction, and hypothesis-aware reasoning over unexplored space. The project in this directory contains evaluation pipelines for AEQA and GOAT-Bench together with the core modules used for mapping, frontier selection, semantic inference, and graph-based refinement.

## Overview

HGR combines several components already present in the project:

- TSDF-based geometric mapping and frontier extraction.
- ConceptGraph-based scene graph construction.
- VLM-guided frontier or snapshot selection.
- Hypothesis graph utilities for representing, scoring, and validating inferred unexplored regions.

The current evaluation runners are:

- `run_aeqa_evaluation.py` for AEQA.
- `run_goatbench_evaluation.py` for GOAT-Bench.

## Key Features

- Frontier-based navigation with volumetric TSDF fusion in [`src/tsdf_planner.py`](src/tsdf_planner.py).
- Scene graph construction and maintenance for both AEQA and GOAT-Bench under [`src/`](src/).
- Hypothesis graph data structures and verification utilities in [`src/hypothesis_graph.py`](src/hypothesis_graph.py), [`src/hypothesis_node_predictor.py`](src/hypothesis_node_predictor.py), and [`src/semantic_critic.py`](src/semantic_critic.py).
- Separate evaluation configurations for AEQA and GOAT-Bench in [`cfg/eval_aeqa.yaml`](cfg/eval_aeqa.yaml) and [`cfg/eval_goatbench.yaml`](cfg/eval_goatbench.yaml).

## Repository Structure

```text
.
├── README.md
├── run_aeqa_evaluation.py
├── run_goatbench_evaluation.py
├── cfg/
│   ├── concept_graph_default.yaml
│   ├── eval_aeqa.yaml
│   └── eval_goatbench.yaml
├── configs/
│   └── hypothesis_config.py
└── src/
    ├── tsdf_planner.py
    ├── scene_aeqa.py
    ├── scene_goatbench.py
    ├── query_vlm_aeqa.py
    ├── query_vlm_goatbench.py
    ├── query_vlm_hypothesis.py
    ├── hypothesis_graph.py
    ├── hypothesis_node_predictor.py
    ├── semantic_critic.py
    └── conceptgraph/
```

## Installation

An environment file is not included in this subdirectory, so setup is partly manual.

At minimum, the Python code imports and expects packages such as:

- `torch`
- `numpy`
- `omegaconf`
- `matplotlib`
- `scipy`
- `scikit-image`
- `scikit-learn`
- `habitat-sim`
- `open_clip`
- `ultralytics`
- `supervision`

Install the dependencies required by your local Habitat / OpenEQA / ConceptGraph setup before running the evaluation scripts.

## Data and Dependency Preparation

The provided YAML configs reference external data that is not included in this folder. Before running anything, review and update the following fields in the YAML files to match your machine:

- `scene_dataset_config_path`
- `scene_data_path`
- `questions_list_path` for AEQA
- `test_data_dir` for GOAT-Bench
- `concept_graph_config_path`

The default configs currently point to HM3D-style scene assets and benchmark data, for example:

- [`cfg/eval_aeqa.yaml`](cfg/eval_aeqa.yaml)
- [`cfg/eval_goatbench.yaml`](cfg/eval_goatbench.yaml)

Model names are also configured in YAML:

- `yolo_model_name`
- `sam_model_name`

The runners expect the corresponding weights to be available in your environment.

## Running Evaluation

### AEQA

```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 1.0
```

### GOAT-Bench

```bash
python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml --start_ratio 0.0 --end_ratio 1.0 --split 1
```

## Important Entry Points

- [`run_aeqa_evaluation.py`](run_aeqa_evaluation.py): AEQA evaluation driver.
- [`run_goatbench_evaluation.py`](run_goatbench_evaluation.py): GOAT-Bench evaluation driver.
- [`src/tsdf_planner.py`](src/tsdf_planner.py): TSDF mapping, frontier updates, navigation targets, and HGR integration hooks.
- [`src/query_vlm_hypothesis.py`](src/query_vlm_hypothesis.py): helper utilities for hypothesis-aware VLM querying and verification.
- [`configs/hypothesis_config.py`](configs/hypothesis_config.py): reusable HGR configuration presets for downstream integrations.

## Outputs, Logs, and Checkpoints

Both runners create `cfg.output_dir` as:

```text
results/<exp_name>/
```

The scripts also:

- copy the YAML config into the output directory,
- write a log file into the output directory,
- save per-episode artifacts under the experiment folder,
- aggregate evaluation results at the end of a run.

Exact artifact contents depend on the task and whether `save_visualization` is enabled.

## Notes on HGR Integration

This repository now uses the HGR naming scheme throughout the core hypothesis modules:

- `HGR`
- `Hypothesis Graph Refinement`
- `hypothesis_graph`
- `hypothesis_node`

The reusable HGR modules are present in `src/`, while the default AEQA and GOAT-Bench runners continue to use the project’s established evaluation flow. If you want to wire the standalone HGR helper modules directly into a custom runner, start from:

- [`src/hypothesis_graph.py`](src/hypothesis_graph.py)
- [`src/hypothesis_node_predictor.py`](src/hypothesis_node_predictor.py)
- [`src/semantic_critic.py`](src/semantic_critic.py)
- [`src/query_vlm_hypothesis.py`](src/query_vlm_hypothesis.py)

## Acknowledgement

The codebase is built upon OpenEQA, Explore-EQA, and ConceptGraph. We thank the authors for their great work.

## Citation

Details to be added.
