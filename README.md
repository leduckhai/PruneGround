# PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

**EMNLP 2026 (Oral)**

Duc Cao Dinh*, Khai Le-Duc*, Florent Draye, Chris Ngo, Terry Jingchen Zhang, Bernhard Schölkopf, Zhijing Jin

## Status

- [x] Render guide (topview + oblique views)
- [x] Topview box prediction by VLM
- [x] Prune scene
- [x] Local multiview render (MCDR sideviews)
- [x] Description reformulation (MCDR)
- [ ] Grounding

## 1. Environment Setup

Create a conda environment and install the required dependencies:

```bash id="2ymklr"
conda create -n pruneground python=3.10 -y
conda activate pruneground
pip install -r requirements.txt
```

If your GPU requires a specific CUDA build of PyTorch (e.g. RTX 40/50 series needing CUDA 12.8), reinstall `torch` and `torchvision` after the step above:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 2. Data Preparation

All experiments are conducted on **ScanRefer**, **Sr3D**, and **Nr3D**, which are based on the **ScanNet** dataset.

Please make sure the ScanNet dataset is downloaded on your device before running the code. Refer to the official ScanNet website for download instructions:

Refer to https://www.scan-net.org/ to proceed with the download.

Each scene requires two files under `data_dir`:
- `{scene_id}_vh_clean_2.ply` — the textured mesh
- `{scene_id}.txt` — metadata file containing the `axisAlignment` matrix

## 3. Render Topview and Oblique Views

For each scene, generate the topview image and 4 oblique RGB-D side views:

```bash
python render_top\&oblique_view.py --scene_id scene0011_00 --data_dir examples/data --output_dir examples/output
```

This produces, under `examples/output/{scene_id}/`:
- `topview.png` — orthographic top-down rendering of the scene
- `sideviews/view_{1..4}_rgb.png` and `sideviews/view_{1..4}_depth.png` — oblique perspective RGB and depth images
- `metadata.json` — scene bounding box size (`m`, `n`, `k` in meters) and topview resolution (`w`, `h` in pixels)

## 4. Query with Qwen2.5-VL

Run the 2D bounding box prediction on a rendered scene (see Appendix C of the paper for the full prompt structure):

```bash id="zv0wtx"
python LGSP.py \
    --scene_id scene0011_00 \
    --output_dir examples/output \
    --description "This is a single chair nearest the TV" \
    --prompt_style simple
```

The description can be read from the per-scene instruction file (e.g. `examples/data/scene0011_00_instruction.txt`).

The script prints the model's raw output and saves an annotated topview at `examples/output/{scene_id}/topview_pred.png` with the predicted bounding box drawn on it.

**Useful flags:**

| Flag | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace model ID (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) |
| `--prompt_style` | `full` | `full`: Appendix C prompt with topview + 4 oblique RGB-D views + reasoning steps. `simple`: topview only with a minimal grounding prompt — recommended for the 3B model, which tends to return a full-image box on the longer `full` prompt |
| `--sideview_max_pixels` | `256*28*28` | Max pixels per oblique side-view image. Lower this if you hit CUDA OOM. Does not affect `topview.png` (kept at native resolution so `bbox_2d` coordinates stay aligned) |
| `--max_new_tokens` | `512` | Max tokens to generate |
| `--data_dir` | `examples/data` | Directory with raw ScanNet `.ply` and `.txt` files |
| `--pruned_dir` | `examples/pruned` | Output directory for pruned `.ply` files |

`LGSP.py` automatically runs the full pipeline end-to-end:
1. Predict a 2D bounding box on the topview image
2. Project the box back to 3D, prune the scene, save `{pruned_dir}/{scene_id}.ply`
3. Render 2 local RGB side-view images of the pruned region (see below)

## 5. Local Multi-View Rendering (MCDR)

After pruning, `LGSP.py` renders two local RGB side-view images of the pruned scene from the x- and y-axis directions of its bounding box (Sec. 3.3 of the paper). These views are used as visual context for the description reformulation step (MCDR).

Camera selection follows the paper (Appendix B, Fig. 5): for each axis, the camera position closer to the original scene center is chosen so the camera faces **outward** (away from the interior), minimising look-through-wall artefacts. The camera is slightly elevated above the pruned bounding box center to reduce occlusion by surrounding furniture. Depth maps are discarded.

The two images are saved under `examples/output/{scene_id}/local_views/`:
- `local_x.png` — view from the x-axis direction
- `local_y.png` — view from the y-axis direction

## 6. Description Reformulation (MCDR)

Conditioned on the two local side views, the VLM decomposes the original free-form utterance into structured *single target – single anchor* statements, and augments it with anchors that are visible in the pruned region but absent from the original description (see Appendix C, Fig. 7 of the paper for the prompt):

```bash
python MCDR.py \
    --scene_id scene0011_00 \
    --output_dir examples/output \
    --description "This is a single chair nearest the TV"
```

Requires `LGSP.py` to have been run first, since it consumes `local_views/local_x.png` and `local_views/local_y.png`.

The result is written to `examples/output/{scene_id}/reformulation.json`:

| Field | Content |
|---|---|
| `target` | Target object name |
| `anchors` | Anchors explicitly mentioned in the original description |
| `anchors_visual` | Anchors detected in the side views but absent from the description |
| `target_properties` / `augmented_target_properties` | Target attributes, stated and inferred |
| `utterances` / `augmented_utterances` | Single target – single anchor statements |
| `reformulated_description` | The flattened description passed to the grounder |
| `raw` | The model's unparsed output |

**Useful flags:**

| Flag | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace model ID |
| `--max_new_tokens` | `768` | Max tokens to generate (the structured output is longer than a bounding box) |
| `--sideview_max_pixels` | unset | Max pixels per side-view image. Set this if you hit CUDA OOM |
