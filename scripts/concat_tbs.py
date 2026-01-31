import argparse
import os
import sys
import tempfile
import tensorflow as tf
from tensorboard.compat.proto import event_pb2
from tensorflow.python.lib.io import tf_record

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB

def check_file(path: str) -> None:
    if not os.path.isfile(path):
        print(f"Input file does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    if os.path.getsize(path) == 0:
        print(f"Input file is empty: {path}", file=sys.stderr)
        sys.exit(1)

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def _iter_events(path):
    for raw in tf_record.tf_record_iterator(path):
        ev = event_pb2.Event()
        ev.ParseFromString(raw)
        yield ev, raw

def get_step_bounds(path: str, min_valid_step: int = 1, require_summary: bool = False):
    sum_min = sum_max = None
    any_min = any_max = None
    for ev, _ in _iter_events(path):
        step = ev.step
        any_min = step if any_min is None else min(any_min, step)
        any_max = step if any_max is None else max(any_max, step)
        if step < min_valid_step:
            continue
        if require_summary and (ev.summary is None or len(ev.summary.value) == 0):
            continue
        sum_min = step if sum_min is None else min(sum_min, step)
        sum_max = step if sum_max is None else max(sum_max, step)
    if sum_min is not None:
        return sum_min, sum_max, True
    return any_min, any_max, False

def trim_events_before_step(src_path: str, cutoff_step: int):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tfevents.trimmed")
    writer = tf.io.TFRecordWriter(tmp.name)
    kept = 0
    for ev, raw in _iter_events(src_path):
        if ev.step < cutoff_step:
            writer.write(raw)
            kept += 1
    writer.close()
    return tmp.name, kept

def concat_files(input_paths, output_path, overwrite=False, cutoff_step: int | None = None) -> None:
    for p in input_paths:
        check_file(p)
    if os.path.exists(output_path) and not overwrite:
        print(f"Output file already exists and --overwrite not specified: {output_path}", file=sys.stderr)
        sys.exit(1)

    adjusted_paths = list(input_paths)
    trimmed_path = None
    try:
        if cutoff_step is not None:
            trimmed_path, kept = trim_events_before_step(input_paths[0], cutoff_step)
            adjusted_paths[0] = trimmed_path
            msg = f"Using manual cutoff_step={cutoff_step}, trimmed first log to step < {cutoff_step}"
            if kept == 0:
                msg += "; first log is empty after trimming"
            print(msg, file=sys.stderr)
        else:
            first_min, first_max, first_summary = get_step_bounds(input_paths[0], min_valid_step=1, require_summary=True)
            second_min, _, second_summary = get_step_bounds(input_paths[1], min_valid_step=1, require_summary=True)
            # Fallback to any-step bounds if no summary steps were found
            if not first_summary:
                first_min, first_max, _ = get_step_bounds(input_paths[0], min_valid_step=1, require_summary=False)
            if not second_summary:
                second_min, _, _ = get_step_bounds(input_paths[1], min_valid_step=1, require_summary=False)
            if first_max is not None and second_min is not None and second_min <= first_max:
                trimmed_path, kept = trim_events_before_step(input_paths[0], second_min)
                adjusted_paths[0] = trimmed_path
                msg = f"Detected overlapping steps (second log min step: {second_min}), trimmed first log to step < {second_min}"
                if kept == 0:
                    msg += "; first log is empty after trimming"
                print(msg, file=sys.stderr)

        ensure_parent_dir(output_path)
        with open(output_path, "wb") as out_f:
            for p in adjusted_paths:
                with open(p, "rb") as in_f:
                    while True:
                        buf = in_f.read(CHUNK_SIZE)
                        if not buf:
                            break
                        out_f.write(buf)

        total_size = sum(os.path.getsize(p) for p in adjusted_paths)
        out_size = os.path.getsize(output_path)
        print(f"Concatenation completed: {output_path}")
        print(f"Total input size: {total_size} bytes, output size: {out_size} bytes")
    finally:
        if trimmed_path and os.path.exists(trimmed_path):
            os.remove(trimmed_path)

def main():
    parser = argparse.ArgumentParser(description="Concatenate two TensorBoard event log files into one file")
    parser.add_argument(
        "--input1",
        # default="logs/raw_pt1_scan86/tensorboard/events.out.tfevents.1769082918.bbc-h800-2.4117868.0",
        default="logs/l1_scan_d_seed86/tensorboard/events.out.tfevents.concat.scan86",
        help="Path to the first input event log file"
    )
    parser.add_argument(
        "--input2",
        # default="logs/raw_pt2_scan86/tensorboard/events.out.tfevents.1769249806.bbc-h800-2.172274.0",
        # default="logs/raw_pt5_scan86/tensorboard/events.out.tfevents.1769499697.bbc-h800-2.526062.0",
        # default="logs/raw_pt5_2_scan86/tensorboard/events.out.tfevents.1769575368.bbc-h800-2.1297811.0",
        # default="logs/raw_pt5_3_scan86/tensorboard/events.out.tfevents.1769620778.bbc-h800-2.1816943.0",
        default="logs/raw_pt5_4_scan86/tensorboard/events.out.tfevents.1769621263.bbc-h800-2.1858723.0",
        help="Path to the second input event log file"
    )
    parser.add_argument(
        "--output",
        default="logs/scan_d_seed86/tensorboard/events.out.tfevents.concat.scan86",
        help="Path to the output event log file"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite if output file already exists"
    )
    parser.add_argument(
        "--cutoff-step",
        type=int,
        default=None,
        help="Manually specify cutoff step; use first file before this step, second file after"
    )
    args = parser.parse_args()

    concat_files([args.input1, args.input2], args.output, overwrite=args.overwrite, cutoff_step=args.cutoff_step)

if __name__ == "__main__":
    main()
