import argparse
import json
import os
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MCDR_PROMPT_TEMPLATE = '''Task:

You are given two side-view images of a pruned 3D scene.

Images:
1. side_1.png
2. side_2.png

Description: "{description}"
Your goal is to analyze the description and the two side views to extract the
target object and anchors, and reformulate the description into structured
single target-single anchor utterances.

Reasoning procedure:
Step 1 - Analysis
Analyze the description and side views to identify:
- the target object
- the list of anchors explicitly mentioned in the description
Step 2 - Visual Anchor Detection
From the two side-view images, identify additional anchors that are visible
in the scene but NOT mentioned in the description.
Step 3 - Reformulate Description
Reformulate the description into a list of single target - single anchor
utterances.
Each utterance should describe the spatial relation between the target and one
anchor.

Output Structure:
- target name
- anchors name:
[list of anchors mentioned in the description]
- anchors not in description but visible:
[list of anchors visible in side views but not mentioned in the description]
- reformulated_description:
  + target properties
  (short and concise description of the target's properties)
  + augmented target properties
  (additional target properties that can be inferred
  from the side views but are not explicitly described)
  + single target single anchor utterances
  (list of utterances where each describes the relation between
  the target and one anchor)
  + augmented single target - single anchor utterances
  (list of utterances using anchors that are not mentioned in the
  description but detected in the side views)

Constraints:
- target properties and augmented target properties must be short and concise.
- the TOTAL number of single target - single anchor utterances must be at most 5.
- each utterance must follow this format:
target (+properties) + spatial relation + anchor (+properties)
- Use only clear spatial relations such as:
left of, right of, in front of, behind, next to, under, on, ...
- Do not produce explanations.
- Return only the structured output.

Example target properties:
- the chair is wooden and yellow
Example utterance:
- the chair is behind the black coffee table
'''


def build_mcdr_messages(local_views_dir, description, max_pixels=None):
    """Build the MCDR query: the two local side views + the reformulation prompt."""
    side_views = [
        os.path.join(local_views_dir, "local_x.png"),
        os.path.join(local_views_dir, "local_y.png"),
    ]
    for path in side_views:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing local side view: {path}. Run LGSP.py first to prune the "
                f"scene and render its local views."
            )

    content = []
    for path in side_views:
        entry = {"type": "image", "image": path}
        if max_pixels is not None:
            entry["max_pixels"] = max_pixels
        content.append(entry)
    content.append({
        "type": "text",
        "text": MCDR_PROMPT_TEMPLATE.format(description=description),
    })

    return [{"role": "user", "content": content}]


# Section headers of the structure requested in the prompt, most specific first so
# that e.g. "augmented target properties" is not swallowed by "target properties".
_SECTION_PATTERNS = [
    ("augmented_utterances", r"augmented\s+single\s+target"),
    ("utterances", r"single\s+target\s*[-–]?\s*single\s+anchor|single\s+target\s+single\s+anchor"),
    ("augmented_target_properties", r"augmented\s+target\s+propert"),
    ("target_properties", r"target\s+propert"),
    ("anchors_visual", r"anchors?\s+not\s+in\s+description|not\s+mentioned\s+in\s+the\s+description"),
    ("anchors", r"anchors?\s+name|^\s*[-+*]?\s*anchors?\s*:"),
    ("target", r"target\s+name|^\s*[-+*]?\s*target\s*:"),
]

_BULLET = re.compile(r"^\s*(?:[-+*•]|\d+[.)])\s*(.+)$")

# Container headers: they open a group of sub-sections rather than holding items
# themselves, so they must close the current section instead of being collected
# into it.
_CONTAINER_PATTERN = re.compile(
    r"reformulated[_\s]description|output\s+structure|reasoning\s+procedure", re.IGNORECASE
)


def parse_reformulation(output_text):
    """
    Parse the VLM's structured reformulation into a dict.

    The model does not always reproduce the requested layout exactly, so this is a
    best-effort section parser: it tracks the current section by header and collects
    the bullet items under it. The raw output is always kept so nothing is lost.
    """
    result = {
        "target": "",
        "anchors": [],
        "anchors_visual": [],
        "target_properties": [],
        "augmented_target_properties": [],
        "utterances": [],
        "augmented_utterances": [],
        "raw": output_text,
    }

    current = None
    for line in output_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Skip the placeholder/explanatory lines echoed from the prompt template.
        if stripped.startswith("(") and stripped.endswith(")"):
            continue

        if _CONTAINER_PATTERN.search(stripped):
            current = None
            continue

        header = None
        for name, pattern in _SECTION_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                header = name
                break

        if header is not None:
            current = header
            # A header may carry its value inline, e.g. "- target name: chair".
            inline = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            inline = inline.strip("[]").strip()
            if inline:
                _add(result, current, inline)
            continue

        if current is None:
            continue

        bullet = _BULLET.match(stripped)
        value = bullet.group(1).strip() if bullet else stripped
        value = value.strip("[]").strip()
        if value:
            _add(result, current, value)

    return result


def _add(result, section, value):
    if section == "target":
        if not result["target"]:
            result["target"] = value
    elif isinstance(result.get(section), list):
        # A single line may hold a comma-separated list, e.g. "chair, desk, lamp".
        if section in ("anchors", "anchors_visual") and "," in value:
            result[section].extend(v.strip() for v in value.split(",") if v.strip())
        else:
            result[section].append(value)


def format_reformulation(parsed):
    """Render the parsed reformulation as the description passed to the grounder."""
    lines = []
    if parsed["target"]:
        lines.append(f"Target: {parsed['target']}")
    for prop in parsed["target_properties"] + parsed["augmented_target_properties"]:
        lines.append(f"- {prop}")
    for utt in parsed["utterances"] + parsed["augmented_utterances"]:
        lines.append(f"- {utt}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", default="scene0011_00")
    parser.add_argument("--output_dir", default="examples/output")
    parser.add_argument("--description", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument(
        "--sideview_max_pixels", type=int, default=None,
        help="Max pixels per local side-view image fed to the vision encoder "
             "(lower to reduce GPU memory use). Unset keeps native resolution."
    )
    args = parser.parse_args()

    scene_dir = os.path.join(args.output_dir, args.scene_id)
    local_views_dir = os.path.join(scene_dir, "local_views")

    messages = build_mcdr_messages(
        local_views_dir, args.description, args.sideview_max_pixels
    )

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

    # See LGSP.py: Qwen2.5-VL defaults to do_sample=true, which makes the output
    # vary between runs. Force greedy decoding for reproducibility.
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    trimmed_ids = output_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    print(output_text)

    parsed = parse_reformulation(output_text)
    parsed["original_description"] = args.description
    parsed["reformulated_description"] = format_reformulation(parsed)

    out_path = os.path.join(scene_dir, "reformulation.json")
    with open(out_path, "w") as f:
        json.dump(parsed, f, indent=2)
    print(f"\nSaved reformulation to {out_path}")
