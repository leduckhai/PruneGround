import argparse
import json
import os
import re

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROMPT_TEMPLATE = '''Task:
Given the description: "{description}", Determine the relevant region in the
topview image as a 2D bounding box.

Image order:
The images appear in the following order:
1. topview.png
2. view_1_rgb.png
3. view_1_depth.png
4. view_2_rgb.png
5. view_2_depth.png
6. view_3_rgb.png
7. view_3_depth.png
8. view_4_rgb.png
9. view_4_depth.png

Scene coordinate system:
- X axis: horizontal width = {m} meters
- Y axis: vertical length = {n} meters
- Z axis: upward height = {k} meters

Topview image:
- topview.png is a topview rendering of the scene
- width = {w} pixels corresponding to the X dimension
- height = {h} pixels corresponding to the Y dimension

Image coordinate system:
- (0, 0) is the top-left corner of the topview image
- x increases to the right
- y increases downward

Oblique views:
- Each oblique view consists of two consecutive images:
  first the RGB image, then the corresponding depth image.
- The cameras are placed at the midpoint of the four top
  edges of the axis-aligned 3D bounding box.

View mapping:
- view_1 (images 2-3: view_1_rgb.png, view_1_depth.png)
  camera looks from bottom to top in the topview
- view_2 (images 4-5: view_2_rgb.png, view_2_depth.png)
  camera looks from right to left in the topview
- view_3 (images 6-7: view_3_rgb.png, view_3_depth.png)
  camera looks from top to bottom in the topview
- view_4 (images 8-9: view_4_rgb.png, view_4_depth.png)
  camera looks from left to right in the topview

Reasoning procedure:
Step 1: Identify possible target objects and anchors mentioned in the description.
Step 2: Use the topview and oblique RGB-depth images to determine plausible
        spatial regions containing target candidates.
Step 3: Determine the minimal bounding region that covers those candidates.

Bounding box requirements:
- cover all candidate regions that could plausibly contain the target object
- exclude regions that are unlikely to contain the target
- include anchors mentioned in the description only when they are spatially
  close to the target candidates
- if the anchors are far from the target candidates, they should not be
  included

Output format:
After your reasoning, output the final answer as a JSON list on its own line,
using exactly this format (pixel coordinates in the topview image, x1 < x2,
y1 < y2):
```json
[{{"bbox_2d": [x1, y1, x2, y2], "label": "{description}"}}]
```
'''


def build_messages(scene_dir, description, metadata, sideview_max_pixels=None):
    # The topview image must keep (close to) its native resolution: its bbox_2d
    # output is interpreted in the pixel space of the image the model actually
    # sees, and the prompt tells the model that space is metadata["w"] x metadata["h"].
    # Only the oblique side views (context only, no coordinates read off them) are
    # safe to downscale to control vision-encoder memory use.
    image_files = [
        ("topview.png", None),
        ("sideviews/view_1_rgb.png", sideview_max_pixels),
        ("sideviews/view_1_depth.png", sideview_max_pixels),
        ("sideviews/view_2_rgb.png", sideview_max_pixels),
        ("sideviews/view_2_depth.png", sideview_max_pixels),
        ("sideviews/view_3_rgb.png", sideview_max_pixels),
        ("sideviews/view_3_depth.png", sideview_max_pixels),
        ("sideviews/view_4_rgb.png", sideview_max_pixels),
        ("sideviews/view_4_depth.png", sideview_max_pixels),
    ]

    content = []
    for name, max_pixels in image_files:
        image_entry = {"type": "image", "image": os.path.join(scene_dir, name)}
        if max_pixels is not None:
            image_entry["max_pixels"] = max_pixels
        content.append(image_entry)
    content.append({
        "type": "text",
        "text": PROMPT_TEMPLATE.format(description=description, **metadata),
    })

    return [{"role": "user", "content": content}]


def parse_bboxes(output_text):
    # Preferred: structured `[{"bbox_2d": [...], ...}]` JSON, as produced by
    # smaller/more instruction-strict checkpoints (e.g. the 3B model).
    json_match = re.search(r"\[\s*\{.*?\}\s*\]", output_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: larger models often narrate their reasoning and state the final
    # box as prose, e.g. "Top-left corner: (x=0, y=0)" / "Bottom-right corner:
    # (x=390, y=512)". Take the last two (x, y) pairs as the final answer.
    coords = re.findall(r"x\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*(-?\d+(?:\.\d+)?)", output_text)
    if len(coords) >= 2:
        (x1, y1), (x2, y2) = coords[-2], coords[-1]
        return [{"bbox_2d": [float(x1), float(y1), float(x2), float(y2)]}]

    raise ValueError(f"Could not parse bounding boxes from model output: {output_text}")


def draw_bboxes(topview_path, bboxes, output_path):
    image = Image.open(topview_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in bboxes:
        x1, y1, x2, y2 = box["bbox_2d"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, y1 + 2), box.get("label", ""), fill="red")
    image.save(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", default="scene0085_00")
    parser.add_argument("--output_dir", default="examples/output")
    parser.add_argument("--description", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--sideview_max_pixels", type=int, default=256 * 28 * 28,
        help="Max pixels per oblique side-view image fed to the vision encoder "
             "(lower to reduce GPU memory use). Does not affect topview.png, "
             "whose resolution must match metadata.json for bbox_2d to align."
    )
    args = parser.parse_args()

    scene_dir = os.path.join(args.output_dir, args.scene_id)
    with open(os.path.join(scene_dir, "metadata.json")) as f:
        metadata = json.load(f)

    messages = build_messages(scene_dir, args.description, metadata, args.sideview_max_pixels)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    trimmed_ids = output_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    print(output_text)

    bboxes = parse_bboxes(output_text)
    pred_path = os.path.join(scene_dir, "topview_pred.png")
    draw_bboxes(os.path.join(scene_dir, "topview.png"), bboxes, pred_path)
    print(f"Saved annotated topview to {pred_path}")
