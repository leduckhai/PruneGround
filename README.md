<div align="center">

<img src="https://arxiv.org/html/2606.31148/images/Framework_Icon.PNG" alt="PruneGround official icon — cube and scissors" width="80" />

### PruneGround: Plug-and-Play Spatial Pruning for 3D Visual Grounding

[![Venue](https://img.shields.io/badge/EMNLP%202026-FFD1C1?style=flat-square)](https://2026.emnlp.org/)
[![Oral](https://img.shields.io/badge/Oral-90EE90?style=flat-square)](https://2026.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv%3A2606.31148-B31B1B?logo=arxiv&logoColor=white&style=flat)](https://arxiv.org/abs/2606.31148)
[![GitHub PruneGround](https://img.shields.io/badge/GitHub-PruneGround-white?logo=github&logoColor=white&style=flat&labelColor=181717)](https://github.com/leduckhai/PruneGround)
[![Stars](https://img.shields.io/github/stars/leduckhai/PruneGround?style=social)](https://github.com/leduckhai/PruneGround)


**[Duc Cao Dinh](https://caodinhduc.github.io/)<sup><img src="assets/equal-contribution-star.svg" alt="Equal contribution" width="12" height="12" /></sup> · [Khai Le-Duc](https://github.com/leduckhai)<sup><img src="assets/equal-contribution-star.svg" alt="Equal contribution" width="12" height="12" /></sup> · [Florent Draye](https://www.fdraye.com/) · [Chris Ngo](https://sg.linkedin.com/in/chris-ngo-6ba2136/) · [Terry Jingchen Zhang](https://terry-zhang.notion.site/) · [Bernhard Schölkopf](https://is.mpg.de/en/people/bs) · [Zhijing Jin](https://zhijing-jin.com/home/)**

<sub>Knovel Engineering Lab · University of Toronto · Vector Institute</sub>  
<sub>MPI for Intelligent Systems, Tübingen · ELLIS Institute Tübingen · University of Oxford</sub>

<sub><img src="assets/equal-contribution-star.svg" alt="" width="11" height="11" /> Equal contribution</sub>

**Less clutter. More context. Better grounding.**

</div>

---

## ✨ Overview

**PruneGround** is a plug-and-play spatial pruning pipeline for **3D visual grounding**. Given a 3D scene and a natural-language description, it uses a vision-language model (VLM) to identify a promising region in a rendered top-down view, lifts that region back into 3D, and focuses subsequent reasoning on a smaller, more relevant scene.

It then renders **local multi-view observations** of the pruned region and reformulates the description into **target–anchor relations** to provide clearer context for downstream grounding.

> **Core idea:** Instead of asking a grounder to search the entire scene, first narrow down *where to look* and clarify *what to look for*.

### 💡 Motivation: Focus on What Matters (Figure 1)

A 3D scene can contain several objects of the same category. The language description often includes spatial cues—such as a relationship to a door or another nearby object—that make only part of the scene relevant. Figure 1 illustrates why concentrating on **language-relevant regions** can be more useful than reasoning over every candidate object.

<div align="center">
  <img src="https://arxiv.org/html/2606.31148/introduction_1.png" alt="Figure 1: A 3D scene and top-view illustrating how spatial descriptions identify different relevant regions" width="500" />
  <br />
  <sub>Figure 1 — Motivation for language-guided spatial attention.</sub>
</div>

## 🔍 Pipeline at a Glance

<div align="center">
  <img src="https://arxiv.org/html/2606.31148/main_pipeline.png" alt="Figure 2: PruneGround framework with LGSP, MCDR, and LLM-Grounder" width="1100" />
  <br />
  <sub>Figure 2 — Overview of the three-stage PruneGround framework from the paper.</sub>
</div>

The framework contains three complementary components:

1. **LGSP — Language-Guided Spatial Pruning:** Render global views, ask a frozen VLM to identify a language-relevant region, and retain the corresponding part of the 3D scene.
2. **MCDR — Multi-View-Conditioned Description Reformulation:** Use local views of the pruned scene to simplify the query into target–anchor relations and supplement it with visible spatial context.
3. **LLM-Grounder:** In the *paper's full pipeline*, a detection-pretrained spatial LLM is adapted for language-conditioned grounding using the pruned scene and reformulated description.

## 🏆 Benchmark Results

We evaluate PruneGround on **ScanRefer**, **Nr3D**, and **Sr3D**. Tables 1 and 2 reproduce the comparisons reported in the paper. All entries are **grounding accuracy (%)**; higher is better.

### Table 1 — ScanRefer

ScanRefer reports accuracy at two 3D intersection-over-union thresholds: **Acc@0.25** and **Acc@0.5**. The *Unique* subset contains a single object of the target class in each scene; *Multiple* contains same-class distractors. PruneGround achieves **63.8% Overall Acc@0.25** and **56.1% Overall Acc@0.5**, as reported in Table 1.

| Method | Venue | Unique @0.25 | Unique @0.5 | Multiple @0.25 | Multiple @0.5 | Overall @0.25 | Overall @0.5 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| ScanRefer | ECCV | 67.6 | 46.2 | 32.1 | 21.3 | 40.0 | 26.1 |
| MVT | CVPR | 77.7 | 66.5 | 31.9 | 25.3 | 40.8 | 33.3 |
| 3D-SPS | CVPR | 84.1 | 66.7 | 40.3 | 29.8 | 48.8 | 37.0 |
| EDA | CVPR | 85.8 | 68.6 | 49.1 | 37.6 | 54.6 | 42.3 |
| ConcreteNet | ECCV | 86.4 | 82.1 | 42.4 | 38.4 | 50.6 | 46.5 |
| Chat-Scene | NeurIPS | 89.6 | <u>82.5</u> | 47.8 | 42.9 | 55.5 | 50.2 |
| VPP-Net | CVPR | 86.1 | 67.1 | 50.3 | 39.0 | 55.7 | 43.3 |
| G³-LQ | CVPR | 88.6 | 73.3 | 50.2 | 39.7 | 56.0 | 44.7 |
| MA²TransVG | CVPR | 86.3 | 74.1 | 53.8 | 41.4 | 57.9 | 45.7 |
| MCLN | ECCV | 86.9 | 72.7 | 52.0 | 40.8 | 57.2 | 45.5 |
| PQ3D | ECCV | 86.7 | 78.3 | 51.5 | 46.2 | 57.0 | 51.2 |
| VGMamba | ICCV | **91.9** | 79.6 | 54.8 | <u>47.6</u> | 60.0 | <u>53.9</u> |
| PanoGrounder | arXiv | 84.3 | — | <u>55.3</u> | — | <u>61.0</u> | — |
| **PruneGround (Ours)** | — | <u>90.2</u> | **84.7** | **57.6** | **49.4** | **63.8** | **56.1** |

<sub>Bold: best result per column; underline: second-best result per column.</sub>

### Table 2 — Nr3D and Sr3D

Both benchmarks report **Acc@0.5 (3D IoU)**. *Easy* and *Hard* separate cases by the number of same-class distractors; *View-D* and *View-I* indicate view-dependent and view-independent expressions. For readability on GitHub, the paper's combined Table 2 is displayed below as two dataset-specific tables, with the **same methods and numbers**.

**Nr3D** — PruneGround obtains **75.1% Overall**, including **68.8% on Hard** examples.

| Method | Venue | Easy | Hard | View-D | View-I | Overall |
|:--|:--|--:|--:|--:|--:|--:|
| ReferIt3D | ECCV | 43.6 | 27.9 | 32.5 | 37.1 | 35.6 |
| ViL3DRel | NeurIPS | 70.2 | 57.4 | 62.0 | 64.5 | 64.4 |
| 3D-VisTA | ICCV | 65.9 | 49.4 | 53.7 | 59.4 | 57.5 |
| MIKASA | CVPR | 69.7 | 59.4 | 65.4 | 64.0 | 64.4 |
| G³-LQ | CVPR | — | 50.7 | — | — | 58.4 |
| MA²TransVG | CVPR | — | 57.6 | — | — | 65.2 |
| PQ3D | ICCV | 73.3 | 56.7 | 60.7 | 67.0 | 64.9 |
| LIBA | AAAI | — | 57.2 | 60.3 | — | 64.5 |
| ViewSRD | ICCV | 75.3 | 64.8 | 68.6 | 70.6 | 69.9 |
| VGMamba | ICCV | — | 61.4 | — | — | 68.3 |
| PanoGrounder | arXiv | <u>82.2</u> | <u>67.2</u> | <u>70.5</u> | <u>76.3</u> | <u>74.6</u> |
| **PruneGround (Ours)** | — | **84.6** | **68.8** | **71.2** | **76.6** | **75.1** |

**Sr3D** — PruneGround obtains **81.5% Overall**, including **76.5% on Hard** examples. The published comparison does **not** show the best result for PruneGround on the *View-D* subset (61.7%).

| Method | Venue | Easy | Hard | View-D | View-I | Overall |
|:--|:--|--:|--:|--:|--:|--:|
| ReferIt3D | ECCV | 44.7 | 31.5 | 39.2 | 40.8 | 40.8 |
| ViL3DRel | NeurIPS | 74.9 | 67.9 | 63.8 | 73.2 | 72.8 |
| 3D-VisTA | ICCV | 72.1 | 63.6 | 57.9 | 70.1 | 69.6 |
| MIKASA | CVPR | 78.6 | 67.3 | **70.4** | 75.4 | 75.2 |
| G³-LQ | CVPR | — | 66.3 | — | — | 73.1 |
| MA²TransVG | CVPR | — | 69.3 | — | — | 73.9 |
| PQ3D | ICCV | 78.8 | 68.2 | 51.5 | 76.7 | 75.6 |
| LIBA | AAAI | — | 70.2 | 61.7 | — | 75.8 |
| ViewSRD | ICCV | 78.3 | 70.6 | <u>69.0</u> | 76.2 | 76.0 |
| VGMamba | ICCV | — | <u>74.4</u> | — | — | <u>81.3</u> |
| PanoGrounder | arXiv | <u>81.3</u> | 74.2 | 60.5 | <u>80.0</u> | 79.1 |
| **PruneGround (Ours)** | — | **83.7** | **76.5** | 61.7 | **82.4** | **81.5** |

<sub>Bold: best result per column; underline: second-best result per column. Table 2 is split into Nr3D and Sr3D for readability.</sub>

### 🖼️ Qualitative Results (Figure 3)

Beyond benchmark scores, Figure 3 shows **what changes across the pipeline** on ScanRefer, Nr3D, and Sr3D: an input scene is spatially pruned, its original description is reformulated, and the grounder predicts a target box. In the reformulated-description column, **purple** marks the simplified text and **green** marks visually augmented information.

<div align="center">
  <img src="https://arxiv.org/html/2606.31148/images/qualitative.png" alt="Figure 3: Qualitative examples comparing original scenes, pruned scenes, reformulated descriptions, predictions and ground truth" width="1100" />
  <br />
  <sub>Figure 3 — Qualitative results from ScanRefer, Nr3D, and Sr3D, reproduced from the paper.</sub>
</div>

## 🚧 Release Status

According to the release checklist below, rendering, pruning, and reformulation are implemented; **downstream grounding is not yet marked as available**.

- [x] Top-down and oblique-view rendering
- [x] VLM-based top-view box prediction
- [x] 3D scene pruning
- [x] Local multi-view rendering (MCDR)
- [x] Description reformulation (MCDR)
- [ ] Downstream 3D grounding

## 🚀 Getting Started

### 1. Set up the environment

```bash
conda create -n pruneground python=3.10 -y
conda activate pruneground
pip install -r requirements.txt
```

If your GPU requires a different CUDA-compatible PyTorch build, reinstall `torch` and `torchvision` accordingly. For example, for CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Prepare ScanNet scenes

Our experiments use **ScanRefer**, **Sr3D**, and **Nr3D**, all based on **ScanNet**. Download ScanNet from the [official website](https://www.scan-net.org/) and place the following files for each scene in your data directory:

```text
examples/data/
├── scene0011_00_vh_clean_2.ply   # Textured mesh
├── scene0011_00.txt              # Contains the axisAlignment matrix
└── scene0011_00_instruction.txt  # Optional: example query
```

### 3. Render the scene

Generate one orthographic top-down view and four oblique RGB-D views:

```bash
python 'render_top&oblique_view.py' \
    --scene_id scene0011_00 \
    --data_dir examples/data \
    --output_dir examples/output
```

This creates:

```text
examples/output/scene0011_00/
├── topview.png
├── metadata.json
└── sideviews/
    ├── view_1_rgb.png
    ├── view_1_depth.png
    ├── ...
    ├── view_4_rgb.png
    └── view_4_depth.png
```

`metadata.json` records scene dimensions (`m`, `n`, `k`, in meters) and the top-view resolution (`w`, `h`, in pixels).

### 4. Predict, prune, and render local views (LGSP)

Run the VLM-based spatial pruning pipeline:

```bash
python LGSP.py \
    --scene_id scene0011_00 \
    --output_dir examples/output \
    --description "This is a single chair nearest the TV" \
    --prompt_style simple
```

**One command performs three steps:**

1. Predict a bounding box in the rendered top-down view.
2. Project the predicted box back into 3D and save the pruned scene.
3. Render two local RGB views of the pruned region for MCDR.

The script also prints the VLM's raw response and writes an annotated `topview_pred.png`.

**LGSP options**

| Flag | Default | Description |
|:--|:--|:--|
| `--model_name` | `Qwen/Qwen2.5-VL-3B-Instruct` | Hugging Face model ID; e.g., use the 7B variant. |
| `--prompt_style` | `full` | `full` uses the paper's Appendix C prompt with top-down and four oblique RGB-D views; `simple` uses a shorter top-down-only prompt. **`simple` is recommended for the 3B model.** |
| `--sideview_max_pixels` | `256*28*28` | Maximum pixels per oblique side view; reduce if you encounter CUDA OOM. The top-down image remains at its native resolution for bounding-box alignment. |
| `--max_new_tokens` | `512` | Maximum number of generated tokens. |
| `--data_dir` | `examples/data` | Directory containing ScanNet `.ply` and `.txt` files. |
| `--pruned_dir` | `examples/pruned` | Destination for pruned `.ply` files. |

> **Prompt tip:** The `full` setting follows the complete prompt described in Appendix C of the paper. For the 3B model, `simple` can be more reliable, since the longer prompt may lead it to predict a full-image box.

### 5. Reformulate the description (MCDR)

MCDR uses the two local images produced by LGSP to restructure the text query:

```bash
python MCDR.py \
    --scene_id scene0011_00 \
    --output_dir examples/output \
    --description "This is a single chair nearest the TV"
```

**Run `LGSP.py` first:** MCDR requires `local_views/local_x.png` and `local_views/local_y.png`.

The result is saved as:

```text
examples/output/scene0011_00/reformulation.json
```

**MCDR options**

| Flag | Default | Description |
|:--|:--|:--|
| `--model_name` | `Qwen/Qwen2.5-VL-3B-Instruct` | Hugging Face model ID. |
| `--max_new_tokens` | `768` | Maximum number of generated tokens. |
| `--sideview_max_pixels` | Unset | Optional pixel cap for local images to reduce GPU memory usage. |

## 🧩 Inside MCDR

`LGSP.py` renders the pruned region for MCDR from the **x- and y-axis directions** of its bounding box, choosing viewpoints closer to the original scene center so the cameras face outward. A slight elevation helps mitigate occlusion; depth maps are not retained. See **Section 3.3 and Appendix B (Figure 5)** of the paper.

The VLM then converts the input into **single-target, single-anchor** statements and supplements them with anchors visible in the local images. See **Appendix C (Figure 7)** for the reformulation prompt.

The resulting JSON contains:

| Field | Meaning |
|:--|:--|
| `target` | Name of the target object. |
| `anchors` | Anchor objects mentioned in the original description. |
| `anchors_visual` | Additional anchors seen in local views but not mentioned in the input. |
| `target_properties` / `augmented_target_properties` | Explicit and visually inferred target attributes. |
| `utterances` / `augmented_utterances` | Structured target–anchor statements. |
| `reformulated_description` | Flattened text intended for the downstream grounder. |
| `raw` | Unparsed VLM output. |

## 📁 Where to Find the Outputs

After the LGSP and MCDR commands, the key artifacts are:

```text
examples/
├── output/
│   └── scene0011_00/
│       ├── topview.png
│       ├── topview_pred.png
│       ├── metadata.json
│       ├── sideviews/
│       │   └── ...
│       ├── local_views/
│       │   ├── local_x.png
│       │   └── local_y.png
│       └── reformulation.json
└── pruned/
    └── scene0011_00.ply
```


## 📚 Citation

If you find PruneGround useful in your research, please consider citing our paper:

```bibtex
@article{dinh2026pruneground,
  title={PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding},
  author={Dinh, Duc Cao and Le-Duc, Khai and Draye, Florent and Ngo, Chris and Zhang, Terry Jingchen and Sch{\"o}lkopf, Bernhard and Jin, Zhijing},
  journal={arXiv preprint arXiv:2606.31148},
  year={2026}
}
```
---

<div align="center">

**Prune the scene. Refine the language. Focus the grounding.**

</div>
