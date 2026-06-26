# PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

Duc Cao Dinh*, Khai Le-Duc*, Florent Draye, Chris Ngo, Terry Jingchen Zhang, Bernhard Schölkopf, Zhijing Jin

> Updating ...

## 1. Environment Setup

Create a conda environment and install the required dependencies:

```bash id="2ymklr"
conda create -n pruneground python=3.10 -y
conda activate pruneground
pip install -r requirements.txt
```

## 2. Data Preparation

All experiments are conducted on **ScanRefer**, **Sr3D**, and **Nr3D**, which are based on the **ScanNet** dataset.

Please make sure the ScanNet dataset is downloaded on your device before running the code. Refer to the official ScanNet website for download instructions:

```text id="qe4rxc"
https://www.scan-net.org/
```

## 3. Render Topview and Oblique Views

For each scene, generate the topview image and the four oblique RGB-D side views used for VLM querying:

```bash id="zqv1qm"
python render.py \
    --scene_id scene0085_00 \
    --data_dir examples/data \
    --output_dir examples/output
```

This produces the following files under:

```text id="ltcmow"
examples/output/{scene_id}/
```

Generated outputs:

* `topview.png`: topview rendering of the scene
* `sideviews/view_{1..4}_rgb.png`: four oblique RGB side views
* `sideviews/view_{1..4}_depth.png`: four oblique depth side views
* `metadata.json`: scene size, given by `m`, `n`, and `k` in meters, and topview resolution, given by `w` and `h` in pixels

## 4. Query with Qwen2.5-VL-3B

Run the 2D bounding-box query on a rendered scene. See Appendix C of the paper for the prompt structure.

```bash id="zv0wtx"
python query_qwen.py \
    --scene_id scene0085_00 \
    --output_dir examples/output \
    --description "the chair next to the window"
```

By default, this loads the following Hugging Face checkpoint:

```text id="apenz6"
Qwen/Qwen2.5-VL-3B-Instruct
```

If you need a specific CUDA build of `torch` and `torchvision` for your GPU, reinstall those two packages after running `pip install -r requirements.txt`. For example, for recent NVIDIA GPUs:

```bash id="5658hr"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Useful flags:

* `--model_name`: use a different checkpoint
* `--max_new_tokens`: control the generation length
* `--sideview_max_pixels`: set the maximum number of pixels per oblique side-view image fed to the vision encoder

Lower `--sideview_max_pixels` if you encounter CUDA out-of-memory errors. The default value is:

```text id="5po4z6"
256 * 28 * 28
```

This setting does not affect `topview.png`, which is kept near its native resolution so that the model’s `bbox_2d` output remains aligned with the topview pixel grid.

The script prints the model’s raw `bbox_2d` prediction and saves an annotated image with the predicted box drawn on the topview image:

```text id="618thj"
examples/output/{scene_id}/topview_pred.png
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.
