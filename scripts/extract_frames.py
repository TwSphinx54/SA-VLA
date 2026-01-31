#!/usr/bin/env python3
import argparse
import os
import sys

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frames from an MP4 every k frames, optionally cropping to a grid cell."
    )
    p.add_argument("--input", "-i", default="outputs/eval_t/video/eval/seed_3/0.mp4", help="Path to an .mp4 file")
    p.add_argument("--output-dir", "-o", default="outputs/vids/eval_frames_1", help="Directory to write extracted frames")
    p.add_argument("--step", "-k", type=int, default=30, help="Extract every k frames (k>=1)")
    p.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Default grid size N for NxN when --grid-rows/--grid-cols are not set (N>=1)",
    )
    p.add_argument("--grid-rows", type=int, default=4, help="Grid rows R for RxC grid (R>=1); requires --grid-cols")
    p.add_argument("--grid-cols", type=int, default=5, help="Grid cols C for RxC grid (C>=1); requires --grid-rows")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--grid-index",
        type=int,
        default=5,
        help="Select a cell in RxC grid by index [0..R*C-1], row-major (0=top-left)",
    )
    g.add_argument("--grid-row", type=int, help="Grid row [0..R-1] (0=top)")
    p.add_argument("--grid-col", type=int, help="Grid col [0..C-1] (0=left); requires --grid-row")
    p.add_argument("--ext", default="png", choices=["jpg", "png"], help="Output image extension")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1-100) if ext=jpg")
    p.add_argument(
        "--concat-horizontal",
        action="store_true",
        help="Concatenate selected frames horizontally into one image (saved to parent of output-dir); disables per-frame saves",
    )
    p.add_argument(
        "--cell-video",
        action="store_true",
        help="Export a video containing only the selected grid cell; requires a grid selection",
    )
    return p.parse_args()


def resolve_grid(args: argparse.Namespace):
    if (args.grid_rows is None) != (args.grid_cols is None):
        raise ValueError("Must provide both --grid-rows and --grid-cols, or neither")
    rows = args.grid_rows if args.grid_rows is not None else args.grid_size
    cols = args.grid_cols if args.grid_cols is not None else args.grid_size
    if rows < 1 or cols < 1:
        raise ValueError("--grid-size/--grid-rows/--grid-cols must be >= 1")
    if args.grid_index is None and args.grid_row is None and args.grid_col is None:
        return None, rows, cols  # full frame
    max_idx = rows * cols - 1
    if args.grid_index is not None:
        if not (0 <= args.grid_index <= max_idx):
            raise ValueError(f"--grid-index must be in [0..{max_idx}]")
        return divmod(args.grid_index, cols), rows, cols
    if args.grid_row is None or args.grid_col is None:
        raise ValueError("Must provide both --grid-row and --grid-col")
    if not (0 <= args.grid_row < rows) or not (0 <= args.grid_col < cols):
        raise ValueError(f"--grid-row/--grid-col must be in [0..{rows-1}] and [0..{cols-1}]")
    return (args.grid_row, args.grid_col), rows, cols


def crop_to_grid(frame, row: int, col: int, grid_rows: int, grid_cols: int):
    h, w = frame.shape[:2]
    cell_w = w // grid_cols
    cell_h = h // grid_rows

    x0 = col * cell_w
    y0 = row * cell_h
    x1 = (col + 1) * cell_w if col < grid_cols - 1 else w
    y1 = (row + 1) * cell_h if row < grid_rows - 1 else h

    return frame[y0:y1, x0:x1]


def main() -> int:
    args = parse_args()
    if args.step < 1:
        print("Error: --step must be >= 1", file=sys.stderr)
        return 2

    try:
        grid, n_rows, n_cols = resolve_grid(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if args.cell_video and grid is None:
        print("Error: --cell-video requires specifying a grid cell (index or row/col).", file=sys.stderr)
        return 2

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"Error: input not found: {in_path}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"Error: failed to open video: {in_path}", file=sys.stderr)
        return 2

    frame_idx = 0
    saved = 0

    # Precompute imwrite params
    imwrite_params = []
    if args.ext == "jpg":
        q = int(args.jpeg_quality)
        q = 1 if q < 1 else 100 if q > 100 else q
        imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]

    concat_frames = [] if args.concat_horizontal else None
    cell_writer = None
    cell_video_path = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cell_frame = None
            if args.cell_video:
                r, c = grid  # grid is guaranteed not None here
                cell_frame = crop_to_grid(frame, r, c, n_rows, n_cols)
                if cell_writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    fps = fps if fps > 0 else 30
                    h, w = cell_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    cell_video_path = os.path.join(args.output_dir, f"cell_r{r}c{c}.mp4")
                    cell_writer = cv2.VideoWriter(cell_video_path, fourcc, fps, (w, h))
                    if not cell_writer.isOpened():
                        print(f"Error: failed to open cell video writer: {cell_video_path}", file=sys.stderr)
                        return 2
                cell_writer.write(cell_frame)

            if frame_idx % args.step == 0:
                out = frame
                suffix = ""
                if grid is not None:
                    out = cell_frame if cell_frame is not None else crop_to_grid(frame, *grid, n_rows, n_cols)
                    r, c = grid
                    suffix = f"_r{r}c{c}"

                if concat_frames is not None:
                    concat_frames.append(out)
                else:
                    out_name = f"frame_{frame_idx:08d}{suffix}.{args.ext}"
                    out_path = os.path.join(args.output_dir, out_name)
                    if not cv2.imwrite(out_path, out, imwrite_params):
                        print(f"Error: failed to write: {out_path}", file=sys.stderr)
                        return 2
                    saved += 1

            frame_idx += 1
    finally:
        cap.release()
        if cell_writer is not None:
            cell_writer.release()

    if concat_frames is not None:
        if concat_frames:
            stitched = cv2.hconcat(concat_frames)
            parent_dir = os.path.abspath(os.path.join(args.output_dir, os.pardir))
            os.makedirs(parent_dir, exist_ok=True)
            out_path = os.path.join(parent_dir, f"frames_concat.{args.ext}")
            if not cv2.imwrite(out_path, stitched, imwrite_params):
                print(f"Error: failed to write: {out_path}", file=sys.stderr)
                return 2
            saved = 1
        else:
            print("No frames selected; nothing to save.", file=sys.stderr)
            return 1

    target_dir = os.path.abspath(os.path.join(args.output_dir, os.pardir)) if args.concat_horizontal else args.output_dir
    if args.cell_video and cell_video_path:
        target_dir = f"{target_dir} (cell video: {cell_video_path})"
    print(f"Done. Read {frame_idx} frames, saved {saved} images to: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
