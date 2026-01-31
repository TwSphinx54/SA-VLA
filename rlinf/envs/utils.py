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

def parse_bddl_goal(bddl_path: str):
    """Parse BDDL file and return goals as list of dicts."""
    with open(bddl_path, 'r') as f:
        content = f.read()

    # tokenize: 空格分割 + 括号单独成 token
    content = content.replace('(', ' ( ').replace(')', ' ) ')
    tokens = content.split()

    # 简单的递归解析 S-expression
    def parse_sexpr(tokens):
        if len(tokens) == 0:
            return None, []
        token = tokens.pop(0)
        if token == '(':
            L = []
            while tokens[0] != ')':
                elem, tokens = parse_sexpr(tokens)
                L.append(elem)
            tokens.pop(0)  # pop ')'
            return L, tokens
        elif token == ')':
            raise ValueError("Unexpected )")
        else:
            return token, tokens

    sexpr, _ = parse_sexpr(tokens)

    # 找到 :goal
    goal_sexpr = None
    for item in sexpr:
        if isinstance(item, list) and len(item) > 0 and item[0] == ':goal':
            goal_sexpr = item[1]  # 跳过 :goal
            break

    if goal_sexpr is None:
        return []

    # 解析实际 predicate
    results = []

    def extract_predicates(expr):
        # expr 可以是 ['And', ['On', 'akita_black_bowl_1', 'plate_1'], ...]
        if isinstance(expr, list):
            if len(expr) == 0:
                return
            if expr[0] in ('And', 'Or'):
                for e in expr[1:]:
                    extract_predicates(e)
            else:
                # predicate
                results.append({
                    "relation": expr[0],
                    "object": expr[1] if len(expr) > 1 else None,
                    "destination": expr[2] if len(expr) > 2 else None
                })

    extract_predicates(goal_sexpr)
    return results[0]

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