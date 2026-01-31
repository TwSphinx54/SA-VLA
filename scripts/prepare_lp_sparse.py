from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CATEGORIES = ("Robot Initial States", "Camera Viewpoints")


def _load_module_from_path(py_path: Path):
	spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Failed to import module from: {py_path}")
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _find_suite_map(mod: Any) -> Dict[str, List[str]]:
	"""
	Find a dict that looks like: {suite_name: [task_name, ...], ...}
	We don't assume the variable name.
	"""
	best: Optional[Dict[str, List[str]]] = None
	best_score = -1

	for _, v in mod.__dict__.items():
		if not isinstance(v, dict) or not v:
			continue
		if not all(isinstance(k, str) for k in v.keys()):
			continue
		score = 0
		ok = True
		for kk, vv in v.items():
			if not isinstance(vv, (list, tuple)):
				ok = False
				break
			if not all(isinstance(x, str) for x in vv):
				ok = False
				break
			# heuristic: task names tend to contain "_view_" and/or "_initstate_"
			score += sum(1 for x in vv if ("_view_" in x or "_initstate_" in x))
			# suites usually have non-trivial length
			score += min(len(vv), 50)
		if not ok:
			continue
		if score > best_score:
			best_score = score
			best = {k: list(vv) for k, vv in v.items()}

	if best is None:
		raise RuntimeError(
			"Could not find suite->tasks mapping dict in module. "
			"Expected something like {'libero_spatial': ['task1', ...], ...}."
		)
	return best


def _load_classification_json(path: Path) -> Dict[str, List[dict]]:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def _build_name_to_entry(classification: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, dict]:
	out: Dict[str, dict] = {}
	for suite_name, items in classification.items():
		if not isinstance(items, (list, tuple)):
			continue
		for it in items:
			if not isinstance(it, dict):
				continue
			name = it.get("name")
			if isinstance(name, str):
				# store original entry (keep category/id/etc.)
				out[name] = dict(it)
	return out


def _base_task_key(task_name: str) -> str:
	# Default: treat everything before "_view_" as the underlying instruction.
	# Example:
	#   pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_1
	# -> pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
	return task_name.split("_view_")[0]


def _sample(items: Sequence[str], n: int, rng: random.Random, with_replacement: bool) -> List[str]:
	if n <= 0:
		return []
	if not items:
		return []
	if with_replacement:
		return [items[rng.randrange(0, len(items))] for _ in range(n)]

	# Evenly spaced sampling WITHOUT replacement (deterministic).
	# Example: n=4 -> take first, last, and 2 evenly spaced in-between.
	m = len(items)
	if m <= n:
		return list(items)
	if n == 1:
		return [items[0]]

	# primary evenly-spaced indices (includes 0 and m-1)
	indices: List[int] = []
	seen = set()
	for i in range(n):
		pos = i * (m - 1) / (n - 1)
		idx = int(round(pos))
		if idx < 0:
			idx = 0
		elif idx > m - 1:
			idx = m - 1
		if idx not in seen:
			seen.add(idx)
			indices.append(idx)

	# if rounding caused collisions, fill deterministically with remaining indices
	if len(indices) < n:
		for idx in range(m):
			if idx in seen:
				continue
			seen.add(idx)
			indices.append(idx)
			if len(indices) == n:
				break

	indices.sort()
	return [items[i] for i in indices]


@dataclass(frozen=True)
class SparseResult:
	suite_to_tasks: Dict[str, List[str]]
	suite_to_classification_items: Dict[str, List[dict]]


def make_sparse(
	suite_to_tasks: Mapping[str, Sequence[str]],
	name_to_entry: Mapping[str, dict],
	n: int,
	seed: int,
	with_replacement: bool,
) -> SparseResult:
	rng = random.Random(seed)
	out_map: Dict[str, List[str]] = {}
	out_json: Dict[str, List[dict]] = {}

	for suite_name, tasks in suite_to_tasks.items():
		# group by base task key, then by category
		base_to_cat: Dict[str, Dict[str, List[str]]] = {}
		for t in tasks:
			entry = name_to_entry.get(t)
			if not entry:
				# Unknown in classification json; skip silently to keep script usable across different jsons.
				continue
			cat = entry.get("category")
			if cat not in CATEGORIES:
				continue
			base = _base_task_key(t)
			base_to_cat.setdefault(base, {}).setdefault(cat, []).append(t)

		selected: List[str] = []
		for base, cat_map in sorted(base_to_cat.items(), key=lambda kv: kv[0]):
			for cat in CATEGORIES:
				cands = sorted(cat_map.get(cat, []))
				selected.extend(_sample(cands, n, rng, with_replacement))

		# de-dup (only matters when with_replacement=True); keep order
		seen = set()
		selected_unique: List[str] = []
		for t in selected:
			if t in seen:
				continue
			seen.add(t)
			selected_unique.append(t)

		out_map[suite_name] = selected_unique

		# build json entries for selected tasks; reindex ids per suite
		items: List[dict] = []
		for new_id, t in enumerate(selected_unique):
			e = dict(name_to_entry[t])
			e["id"] = new_id
			e["name"] = t
			items.append(e)
		out_json[suite_name] = items

	return SparseResult(suite_to_tasks=out_map, suite_to_classification_items=out_json)


def _emit_py_suite_map(path: Path, var_name: str, suite_to_tasks: Mapping[str, Sequence[str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		f.write("# Auto-generated by /RLinf/scripts/make_sparse_suite.py; do not edit.\n")
		f.write(f"{var_name} = ")
		json.dump(suite_to_tasks, f, ensure_ascii=False, indent=2)
		f.write("\n")


def _emit_json(path: Path, suite_to_items: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(suite_to_items, f, ensure_ascii=False, indent=2)
		f.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
	ap = argparse.ArgumentParser(
		description=(
			"Sample n tasks per base-task per perturbation(category) from libero suite maps, "
			"and emit a new suite map + a matching classification json."
		)
	)
	ap.add_argument(
		"--suite-map-py",
		type=Path,
		default=Path("/opt/venv/openpi/libero/libero/libero/benchmark/libero_suite_task_map_spatial.py"),
		help="Path to libero_suite_task_map_spatial.py (or compatible).",
	)
	ap.add_argument(
		"--classification-json",
		type=Path,
		default=Path("/opt/venv/openpi/libero/libero/libero/benchmark/task_classification_spatial.json"),
		help="Path to task_classification_spatial.json (or compatible).",
	)
	ap.add_argument("--n", type=int, default=4, help="Sample count per (base task, category).")
	ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
	ap.add_argument(
		"--with-replacement",
		action="store_true",
		help="If set, sample with replacement when candidates < n.",
	)
	ap.add_argument(
		"--out-dir",
		type=Path,
		default=Path("/opt/venv/openpi/libero/libero/libero/benchmark"),
		help="Output directory (inside the repo).",
	)
	ap.add_argument(
		"--out-py-var",
		type=str,
		default="libero_task_map",
		help="Variable name to write in the generated .py suite map.",
	)
	args = ap.parse_args(argv)

	mod = _load_module_from_path(args.suite_map_py)
	suite_to_tasks = _find_suite_map(mod)

	classification = _load_classification_json(args.classification_json)
	name_to_entry = _build_name_to_entry(classification)

	res = make_sparse(
		suite_to_tasks=suite_to_tasks,
		name_to_entry=name_to_entry,
		n=args.n,
		seed=args.seed,
		with_replacement=args.with_replacement,
	)

	out_py = args.out_dir / f"{args.suite_map_py.stem}_sparse_n{args.n}.py"
	out_json = args.out_dir / f"{args.classification_json.stem}_sparse_n{args.n}.json"

	_emit_py_suite_map(out_py, args.out_py_var, res.suite_to_tasks)
	_emit_json(out_json, res.suite_to_classification_items)

	print(f"Wrote: {out_py}")
	print(f"Wrote: {out_json}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
