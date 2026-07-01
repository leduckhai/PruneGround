# PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

Duc Cao Dinh*, Khai Le-Duc*, Florent Draye, Chris Ngo, Terry Jingchen Zhang, Bernhard Schölkopf, Zhijing Jin

## Status

- [x] Render guide (topview + oblique views)
- [x] Topview box prediction by VLM
- [ ] Prune scene
- [ ] Multiview render guide
- [ ] Reformulation
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
python render.py --scene_id scene0011_00 --data_dir examples/data --output_dir examples/output
```

This produces, under `examples/output/{scene_id}/`:
- `topview.png` — orthographic top-down rendering of the scene
- `sideviews/view_{1..4}_rgb.png` and `sideviews/view_{1..4}_depth.png` — oblique perspective RGB and depth images
- `metadata.json` — scene bounding box size (`m`, `n`, `k` in meters) and topview resolution (`w`, `h` in pixels)

## 4. Query with Qwen2.5-VL

Run the 2D bounding box prediction on a rendered scene (see Appendix C of the paper for the full prompt structure):

```bash id="zv0wtx"
python query_qwen.py \
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