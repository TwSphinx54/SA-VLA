# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Optional, Union

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def to_tensor(
    array: Union[dict, torch.Tensor, np.ndarray, list, Any], device: str = "cpu"
) -> Union[dict, torch.Tensor]:
    """
    Copied from ManiSkill!
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    elif isinstance(array, torch.Tensor):
        ret = array.to(device)
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        ret = torch.tensor(array).to(device)
    else:
        if isinstance(array, list) and isinstance(array[0], np.ndarray):
            array = np.array(array)
        ret = torch.tensor(array, device=device)
    if ret.dtype == torch.float64:
        ret = ret.to(torch.float32)
    return ret


def list_of_dict_to_dict_of_list(
    list_of_dict: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """
    Convert a list of dictionaries to a dictionary of lists.

    Args:
        list_of_dict: List of dictionaries with same keys

    Returns:
        Dictionary where each key maps to a list of values
    """
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


def save_rollout_video(
    rollout_images: list[np.ndarray], output_dir: str, video_name: str, fps: int = 30
) -> None:
    """
    Saves an MP4 replay of an episode.

    Args:
        rollout_images: List of images from the episode
        output_dir: Directory to save the video
        video_name: Name of the output video file
        fps: Frames per second for the video
    """
    os.makedirs(output_dir, exist_ok=True)
    mp4_path = os.path.join(output_dir, f"{video_name}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()


def tile_images(
    images: list[Union[np.ndarray, torch.Tensor]], nrows: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Copied from maniskill https://github.com/haosulab/ManiSkill
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns to fit all the images.
    The images can also be batched (e.g. of shape (B, H, W, C)), but give images must all have the same batch size.

    if nrows is 1, images can be of different sizes. If nrows > 1, they must all be the same size.
    """
    # Sort images in descending order of vertical height
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns: list[list[Union[np.ndarray, torch.Tensor]]] = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched : 2 + batched]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1 + batched] for x in columns)

    is_torch = False
    if torch is not None:
        is_torch = isinstance(images[0], torch.Tensor)

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    if is_torch:
        output_image = torch.zeros(output_shape, dtype=images[0].dtype)
    else:
        output_image = np.zeros(output_shape, dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        if is_torch:
            column_image = torch.concatenate(column, dim=0 + batched)
        else:
            column_image = np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        output_image[..., :cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image


def put_text_on_image(
    image: np.ndarray, lines: list[str], max_width: int = 200
) -> np.ndarray:
    """
    Put text lines on an image with automatic line wrapping.

    Args:
        image: Input image as numpy array
        lines: List of text lines to add
        max_width: Maximum width for text wrapping
    """
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)

    new_lines = []
    for line in lines:
        words = line.split()
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            test_width = font.getlength(test_line)

            if test_width <= max_width:
                current_line.append(word)
            else:
                new_lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            new_lines.append(" ".join(current_line))

    y = -10
    for line in new_lines:
        bbox = draw.textbbox((0, 0), text=line)
        textheight = bbox[3] - bbox[1]
        y += textheight + 10
        x = 10
        draw.text((x, y), text=line, fill=(0, 0, 0))
    return np.array(image)


def put_info_on_image(
    image: np.ndarray,
    info: dict[str, float],
    extras: Optional[list[str]] = None,
    overlay: bool = True,
) -> np.ndarray:
    """
    Put information dictionary and extra lines on an image.

    Args:
        image: Input image
        info: Dictionary of key-value pairs to display
        extras: Additional text lines to display
        overlay: Whether to overlay text on image
    """
    lines = [
        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in info.items()
    ]
    if extras is not None:
        lines.extend(extras)
    return put_text_on_image(image, lines)


def list_of_dict_to_dict_of_batchified_tensor(
    list_of_dict: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert list[dict(str -> nested dict/tensor)] -> dict(str -> nested dict/tensor(N, ...)).

    Assumptions:
    - list_of_dict is non-empty.
    - All elements share the same nested dict structure.
    - Leaves are torch.Tensors with the same shape across samples.
    """
    if not list_of_dict:
        raise ValueError("list_of_dict cannot be empty")

    def merge_level(values: list[Any]) -> Any:
        """Merge a list of values (same structure) into a batch along dim=0."""
        first = values[0]

        # Nested dict: merge per key
        if isinstance(first, dict):
            merged: dict[str, Any] = {}
            for k in first.keys():
                # Collect k-th value from each sample
                sub_values = [v[k] for v in values]
                merged[k] = merge_level(sub_values)
            return merged

        # Tensor leaf: stack along batch dimension
        if isinstance(first, torch.Tensor):
            try:
                return torch.stack(values, dim=0)
            except Exception as e:
                raise ValueError(f"Failed to stack tensors at leaf level: {e}")

        # Unsupported leaf type
        raise TypeError(
            f"Unsupported leaf type {type(first)}; only dict and torch.Tensor are supported."
        )

    return merge_level(list_of_dict)

LOGICAL_GOAL_OPS = {
    "and",
    "or",
    "not",
    "forall",
    "exists",
    "imply",
    "when",
}

PLACEMENT_RELATIONS = {"on", "in"}
INTERACTION_RELATIONS = {"open", "close", "turnon", "turnoff"}
INTERACTION_DIRECTION = {
    "open": 1.0,
    "turnon": 1.0,
    "close": -1.0,
    "turnoff": -1.0,
}
RELATION_TO_MODE = {
    "on": "place",
    "in": "place",
    "open": "interact",
    "close": "interact",
    "turnon": "interact",
    "turnoff": "interact",
}

DEFAULT_DENSE_THRESHOLDS = {
    "th_gripper": 0.02,
    "th_place_distance": 0.04,
    "th_leave_distance": 0.04,
    "th_reach_distance": 0.10,
    "th_state_progress": 0.35,
}


def _tokenize_bddl(content: str) -> list[str]:
    content = content.replace("(", " ( ").replace(")", " ) ")
    return content.split()


def _parse_sexpr(tokens: list[str]):
    if len(tokens) == 0:
        return None, []
    token = tokens.pop(0)
    if token == "(":
        arr = []
        while len(tokens) > 0 and tokens[0] != ")":
            elem, tokens = _parse_sexpr(tokens)
            arr.append(elem)
        if len(tokens) == 0:
            raise ValueError("Invalid BDDL: missing ')' while parsing s-expression")
        tokens.pop(0)
        return arr, tokens
    if token == ")":
        raise ValueError("Invalid BDDL: unexpected ')' token")
    return token, tokens


def _find_bddl_section(root: Any, section_name: str):
    if not isinstance(root, list):
        return None
    section_name = section_name.lower()
    for item in root:
        if isinstance(item, list) and len(item) > 0:
            head = str(item[0]).lower()
            if head == section_name:
                return item
    return None


def _extract_goal_predicates(expr: Any, out: list[dict[str, Any]]):
    if not isinstance(expr, list) or len(expr) == 0:
        return

    op = str(expr[0])
    op_l = op.lower()

    if op_l in LOGICAL_GOAL_OPS:
        if op_l == "not":
            if len(expr) >= 2:
                _extract_goal_predicates(expr[1], out)
            return

        if op_l in {"forall", "exists"}:
            if len(expr) >= 3:
                _extract_goal_predicates(expr[2], out)
            return

        for child in expr[1:]:
            _extract_goal_predicates(child, out)
        return

    args = [x for x in expr[1:] if not isinstance(x, list)]
    out.append(
        {
            "relation": op,
            "args": args,
            "object": args[0] if len(args) > 0 else None,
            "destination": args[1] if len(args) > 1 else None,
        }
    )


def parse_bddl_problem(bddl_path: str) -> dict[str, Any]:
    """Parse a BDDL file and return native language and structured goal predicates."""
    with open(bddl_path, "r", encoding="utf-8") as f:
        content = f.read()

    tokens = _tokenize_bddl(content)
    root, _ = _parse_sexpr(tokens)

    language = None
    language_section = _find_bddl_section(root, ":language")
    if isinstance(language_section, list) and len(language_section) >= 2:
        language = " ".join(str(x) for x in language_section[1:]).strip()

    goals: list[dict[str, Any]] = []
    goal_section = _find_bddl_section(root, ":goal")
    if isinstance(goal_section, list) and len(goal_section) >= 2:
        _extract_goal_predicates(goal_section[1], goals)

    return {
        "language": language,
        "goals": goals,
    }


def parse_bddl_goals(bddl_path: str) -> list[dict[str, Any]]:
    """Parse BDDL file and return all goal predicates as a list of dicts."""
    return parse_bddl_problem(bddl_path).get("goals", [])


def parse_bddl_goal(bddl_path: str) -> dict[str, Any]:
    """Backward-compatible wrapper: return the first parsed goal predicate."""
    goals = parse_bddl_goals(bddl_path)
    return goals[0] if len(goals) > 0 else {}


def build_dense_goal_models(goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build explicit dense-reward models from parsed BDDL goals by relation."""
    models: list[dict[str, Any]] = []

    for goal_idx, g in enumerate(goals):
        if not isinstance(g, dict):
            continue

        relation = str(g.get("relation", "")).strip()
        relation_key = relation.lower()
        obj = g.get("object", None)
        dest = g.get("destination", None)
        mode = RELATION_TO_MODE.get(relation_key, "none")
        thresholds = dict(DEFAULT_DENSE_THRESHOLDS)

        if mode == "place":
            if obj is not None and dest is not None:
                models.append(
                    {
                        "goal_index": goal_idx,
                        "mode": "place",
                        "family": "place",
                        "relation": relation,
                        "relation_key": relation_key,
                        "object": obj,
                        "destination": dest,
                        "direction": 1.0,
                        "thresholds": thresholds,
                    }
                )
            else:
                models.append(
                    {
                        "goal_index": goal_idx,
                        "mode": "reach",
                        "family": "place",
                        "relation": relation,
                        "relation_key": relation_key,
                        "target": obj if obj is not None else dest,
                        "direction": 1.0,
                        "thresholds": thresholds,
                    }
                )
            continue

        if mode == "interact":
            target = obj if obj is not None else dest
            thresholds["th_state_progress"] = 0.30
            if target is None:
                models.append(
                    {
                        "goal_index": goal_idx,
                        "mode": "none",
                        "family": "interact",
                        "relation": relation,
                        "relation_key": relation_key,
                        "direction": INTERACTION_DIRECTION.get(relation_key, 1.0),
                        "thresholds": thresholds,
                    }
                )
            else:
                models.append(
                    {
                        "goal_index": goal_idx,
                        "mode": "interact",
                        "family": "interact",
                        "relation": relation,
                        "relation_key": relation_key,
                        "target": target,
                        "direction": INTERACTION_DIRECTION.get(relation_key, 1.0),
                        "thresholds": thresholds,
                    }
                )
            continue

        # unknown relation: keep a safe fallback that at least reaches related entity
        target = obj if obj is not None else dest
        if target is not None:
            models.append(
                {
                    "goal_index": goal_idx,
                    "mode": "reach",
                    "family": "reach",
                    "relation": relation,
                    "relation_key": relation_key,
                    "target": target,
                    "direction": 1.0,
                    "thresholds": thresholds,
                }
            )
        else:
            models.append(
                {
                    "goal_index": goal_idx,
                    "mode": "none",
                    "family": "none",
                    "relation": relation,
                    "relation_key": relation_key,
                    "direction": 0.0,
                    "thresholds": thresholds,
                }
            )

    return models


def select_primary_dense_model(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one primary dense model with explicit priority: place > interact > reach > none."""
    if not models:
        return {"mode": "none", "relation": None}

    priority = {"place": 0, "interact": 1, "reach": 2, "none": 3}
    return sorted(models, key=lambda m: priority.get(str(m.get("mode", "none")), 99))[0]

def process_plus_name(name: str) -> str:
    res = name
    if "_language_" in name:
        res = name.split("_language_")[0] + ".bddl"
    else:
        if "_view_" in name:
            res = name.split("_view_")[0] + ".bddl"
        else:
            if "_tb_" in name:
                res = name.split("_tb_")[0] + ".bddl"
            elif "_light_" in name:
                res = name.split("_light_")[0] + ".bddl"
            elif "_add_" in name:
                res = name.split("_add_")[0] + ".bddl"
            elif "_level" in name:
                res = name.split("_level")[0] + ".bddl"
            elif "_table_" in name:
                tables = name.split("_table_")
                if tables.__len__() > 2:
                    res = "_table_".join(tables[:-1]) + ".bddl"
                else:
                    res = tables[0] + ".bddl"
    return res