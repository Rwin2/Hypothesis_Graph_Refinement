<br/>
<p align="center">
  <h1 align="center">Hypothesis Graph Refinement: Hypothesis-Driven Exploration with Cascade Error Correction for Embodied Navigation</h1>
  <p align="center">
    <a href="#">Peixin Chen</a><sup>1,2*</sup>,
    <a href="#">Guoxi Zhang</a><sup>2*</sup>,
    <a href="#">Jianwei Ma</a><sup>1&dagger;</sup>,
    <a href="#">Qing Li</a><sup>2&dagger;</sup>
  </p>
  <p align="center">
    <sup>1</sup>Harbin Institute of Technology &nbsp;&nbsp; <sup>2</sup>Beijing Institute for General Artificial Intelligence (BIGAI)<br/>
    <sup>*</sup>Equal Contribution &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding Author
  </p>
  <p align="center">
    <a href="PROJECT_WEB_URL">
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href="ARXIV_URL">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href="PAPER_URL">
      <img src='https://img.shields.io/badge/Download-Paper-green?style=flat&logo=adobeacrobatreader&logoColor=green' alt='Download Paper'>
    </a>
  </p>
</p>

---

This is the official repository of **Hypothesis Graph Refinement (HGR)**: Hypothesis-Driven Exploration with Cascade Error Correction for Embodied Navigation.

![overview](assets/overview.png)

---

## Overview

Embodied agents navigating partially observed environments face two intertwined challenges: frontiers lack semantic cues for efficient exploration, and VLM-based predictions risk embedding errors that propagate through dependent hypotheses over long horizons.

**HGR** addresses both within a unified graph-based framework. It represents frontier predictions as revisable hypothesis nodes in a dependency-aware graph memory and introduces:

1. **Semantic Hypothesis Module** — estimates context-conditioned semantic distributions over frontiers and ranks exploration targets by goal relevance, travel cost, and uncertainty.
2. **Verification-Driven Cascade Correction** — compares on-site observations against predicted semantics and, upon mismatch, retracts the refuted node together with all its downstream dependents via the dependency DAG.

Unlike additive map-building, this allows the graph to *contract* by pruning erroneous subgraphs, keeping memory reliable throughout long episodes.

## Installation

Set up the conda environment (Linux, Python 3.9):

```bash
conda create -n hgr python=3.9 -y && conda activate hgr

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge -c aihabitat habitat-sim=0.2.5 headless faiss-cpu=1.7.4 -y
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2 -y

pip install omegaconf==2.3.0 open-clip-torch==2.26.1 ultralytics==8.2.31 supervision==0.21.0 opencv-python-headless==4.10.* \
  scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 hipart==1.0.4 openai==1.35.3 httpx==0.27.2
```

## Run Evaluation

### 1 - Preparations

#### Dataset

Please download the train and val split of [HM3D](https://aihabitat.org/datasets/hm3d-semantics/), and specify the path in `cfg/eval_aeqa.yaml` and `cfg/eval_goatbench.yaml`. For example, if your download path is `/your_path/hm3d/` that contains `/your_path/hm3d/train/` and `/your_path/hm3d/val/`, set `scene_data_path` in the config files to `/your_path/hm3d/`.

The test questions for A-EQA and GOAT-Bench are provided in the `data/` folder.

#### OpenAI API Setup

Set up the endpoint and API key for the OpenAI API in `src/const.py`.

### 2 - Run Evaluation on A-EQA

```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml
```

To split tasks across multiple runs, use `--start_ratio` and `--end_ratio`:

```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 0.5
```

Results from all splits will be automatically aggregated after the scripts finish.

### 3 - Run Evaluation on GOAT-Bench

```bash
python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml
```

Results will be saved and printed after the script finishes. You can split tasks similarly with `--start_ratio` and `--end_ratio`. Specify the episode to evaluate for each scene with `--split`.

### 4 - Save Visualization

The default evaluation config saves visualization results including topdown maps, egocentric views, memory snapshots, and frontier snapshots at each step. Although helpful, this may slow down evaluation. Set `save_visualization: false` in the config to disable it for large-scale runs.

## Repository Structure

```text
.
├── run_aeqa_evaluation.py          # A-EQA evaluation entry point
├── run_goatbench_evaluation.py     # GOAT-Bench evaluation entry point
├── cfg/
│   ├── eval_aeqa.yaml              # A-EQA evaluation config
│   ├── eval_goatbench.yaml         # GOAT-Bench evaluation config
│   └── concept_graph_default.yaml  # ConceptGraph SLAM parameters
├── configs/
│   └── hypothesis_config.py        # HGR configuration presets
└── src/
    ├── tsdf_planner.py             # TSDF mapping, frontier extraction, navigation
    ├── hypothesis_graph.py         # Hypothesis graph data structures and operations
    ├── hypothesis_node_predictor.py # Semantic distribution prediction for frontiers
    ├── semantic_critic.py          # Prediction residual test and cascade correction
    ├── query_vlm_hypothesis.py     # Hypothesis-aware VLM query utilities
    ├── scene_aeqa.py               # Scene management for A-EQA
    ├── scene_goatbench.py          # Scene management for GOAT-Bench
    ├── query_vlm_aeqa.py           # VLM query wrapper for A-EQA
    ├── query_vlm_goatbench.py      # VLM query wrapper for GOAT-Bench
    └── conceptgraph/               # ConceptGraph scene graph construction
```

## Acknowledgement

The codebase is built upon [OpenEQA](https://github.com/facebookresearch/open-eqa), [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa), and [ConceptGraph](https://github.com/concept-graphs/concept-graphs). We thank the authors for their great work. We also thank the authors of [3D-Mem](https://github.com/UMass-Foundation-Model/3D-Mem) for open-sourcing their project, which provided valuable reference for our repository organization and documentation.

## Citation

```bibtex
@article{chen2026hgr,
  title={Hypothesis Graph Refinement: Hypothesis-Driven Exploration with Cascade Error Correction for Embodied Navigation},
  author={Chen, Peixin and Zhang, Guoxi and Ma, Jianwei and Li, Qing},
  year={2026}
}
```
