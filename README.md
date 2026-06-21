# PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

Duc Cao Dinh*, Khai Le-Duc*, Florent Draye, Chris Ngo, Terry Jingchen Zhang, Bernhard Schölkopf, Zhijing Jin

# Updating ...

## 1. Environment Setup

Create a conda environment and install the dependencies:

```bash
conda create -n pruneground python=3.10 -y
conda activate pruneground
pip install -r requirements.txt
```

## 2. Data Preparation

All experiments are conducted on ScanRefer, Sr3D, and Nr3D, which are based on the ScanNet dataset. Make sure you have the ScanNet dataset downloaded on your device.

Refer to https://www.scan-net.org/ first to proceed with the download.

## 3. Render Topview and Oblique Views

For each scene, generate the topview image and the 4 oblique RGB-D side views used for VLM querying:

```bash
python render.py --scene_id scene0085_00 --data_dir examples/data --output_dir examples/output
```

This produces, under `examples/output/{scene_id}/`:
- `topview.png`: topview rendering of the scene
- `sideviews/view_{1..4}_rgb.png` and `sideviews/view_{1..4}_depth.png`: oblique RGB and depth images
- `metadata.json`: scene size (`m`, `n`, `k` in meters) and topview resolution (`w`, `h` in pixels)

## 4. Query with Qwen2.5-VL-3B

Run the 2D bounding box query on a rendered scene (see Appendix C of the paper for the prompt structure). If you need a specific CUDA build of `torch`/`torchvision` for your GPU (e.g. `--index-url https://download.pytorch.org/whl/cu128` for recent NVIDIA GPUs), reinstall those two packages after `pip install -r requirements.txt`.

```bash
python query_qwen.py \
    --scene_id scene0085_00 \
    --output_dir examples/output \
    --description "the chair next to the window"
```

By default this loads `Qwen/Qwen2.5-VL-3B-Instruct` from Hugging Face. Useful flags:
- `--model_name`: use a different checkpoint
- `--max_new_tokens`: control the generation length
- `--sideview_max_pixels`: max pixels per oblique side-view image fed to the vision encoder (lower this if you hit CUDA out-of-memory; default `256 * 28 * 28`). This does not affect `topview.png`, which is kept near its native resolution so the model's `bbox_2d` output stays aligned with the topview pixel grid.

This prints the model's raw `bbox_2d` prediction and saves an annotated copy at `examples/output/{scene_id}/topview_pred.png` with the predicted box drawn on the topview image.