from __future__ import annotations

import argparse
import csv
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from libero.libero import get_libero_path

from rlinf.envs.libero.utils import get_benchmark_overridden
from rlinf.envs.utils import (
    build_dense_goal_models,
    parse_bddl_problem,
    process_plus_name,
    select_primary_dense_model,
)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass
class DecomposeCheck:
    suite: str
    task_idx: int
    task_name: str
    prompt_from_bddl: str
    bddl_file: str
    goal_count: int
    goal_family: str
    goal_signature: str
    dense_model_signature: str
    task_relations: str
    primary_mode: str
    primary_relation: str | None
    dense_reward_compatible: bool


def _normalize_relation(value: Any) -> str:
    return str(value).strip().lower()


def _build_task_name(task: Any, fallback: str) -> str:
    for attr in ("name", "task_name"):
        val = getattr(task, attr, None)
        if isinstance(val, str) and val:
            return val
    return fallback


def _goal_signature(goal: dict[str, Any]) -> str:
    relation = str(goal.get("relation", "")).strip()
    args = goal.get("args", None)
    if not isinstance(args, list) or not args:
        obj = goal.get("object", None)
        dest = goal.get("destination", None)
        args = [x for x in (obj, dest) if x is not None]
    args_text = ", ".join(str(x) for x in args)
    return f"{relation}({args_text})" if args_text else relation


def _model_signature(model: dict[str, Any]) -> str:
    mode = str(model.get("mode", "none"))
    relation = str(model.get("relation", model.get("relation_key", ""))).strip()
    if mode == "place":
        return f"place[{relation}]({model.get('object')} -> {model.get('destination')})"
    if mode == "reach":
        return f"reach[{relation}]({model.get('target')})"
    if mode == "interact":
        return f"interact[{relation}]({model.get('target')})"
    return f"none[{relation}]"


def _summarize_goals(goals: list[dict[str, Any]]) -> str:
    return " ; ".join(_goal_signature(g) for g in goals if isinstance(g, dict))


def _summarize_models(models: list[dict[str, Any]]) -> str:
    return " ; ".join(_model_signature(m) for m in models if isinstance(m, dict))


def _goal_family(primary_mode: str, primary_relation: str | None, goal_count: int) -> str:
    if goal_count > 1:
        return "multi-goal"
    relation = _normalize_relation(primary_relation)
    if primary_mode == "place":
        return "placement (On/In)"
    if primary_mode == "interact":
        if relation in {"open", "close"}:
            return "interact (Open/Close)"
        if relation in {"turnon", "turnoff"}:
            return "interact (TurnOn/TurnOff)"
        return "interact"
    if primary_mode == "reach":
        return "reach"
    return "none"


def _get_bddl_path(task: Any, is_libero_plus: bool) -> Path:
    bddl_file = getattr(task, "bddl_file")
    problem_folder = getattr(task, "problem_folder")
    bddl_for_goal = process_plus_name(bddl_file) if is_libero_plus else bddl_file
    bddl_path = Path(get_libero_path("bddl_files")) / problem_folder / bddl_for_goal
    if not bddl_path.exists():
        bddl_path = Path(get_libero_path("bddl_files")) / problem_folder / bddl_file
    return bddl_path


def _build_relation_example_row(
    *,
    suite: str,
    task_idx: int,
    task_name: str,
    prompt_from_bddl: str,
    bddl_file: str,
    goals: list[dict[str, Any]],
) -> DecomposeCheck:
    dense_models = build_dense_goal_models(goals)
    primary = select_primary_dense_model(dense_models)
    relations = sorted({str(g.get("relation", "")) for g in goals if isinstance(g, dict)})
    primary_mode = str(primary.get("mode", "none"))
    primary_relation = primary.get("relation", None)
    goal_count = len(goals)
    return DecomposeCheck(
        suite=suite,
        task_idx=task_idx,
        task_name=task_name,
        prompt_from_bddl=prompt_from_bddl,
        bddl_file=bddl_file,
        goal_count=goal_count,
        goal_family=_goal_family(primary_mode, primary_relation, goal_count),
        goal_signature=_summarize_goals(goals),
        dense_model_signature=_summarize_models(dense_models),
        task_relations="|".join(relations),
        primary_mode=primary_mode,
        primary_relation=primary_relation,
        dense_reward_compatible=(primary_mode != "none"),
    )


def _suite_sort_key(suite_name: str) -> int:
    try:
        return SUITES.index(suite_name)
    except ValueError:
        return len(SUITES)


def _task_sort_key(row: DecomposeCheck) -> tuple[int, int, str]:
    task_idx = row.task_idx if row.task_idx >= 0 else 10**9
    return (_suite_sort_key(row.suite), task_idx, row.task_name)


def _multi_task_sort_key(row: DecomposeCheck) -> tuple[int, int, str]:
    task_idx = row.task_idx if row.task_idx >= 0 else 10**9
    return (-row.goal_count, _suite_sort_key(row.suite), task_idx)


def _pick_first(rows: list[DecomposeCheck], predicate) -> DecomposeCheck | None:
    for row in rows:
        if predicate(row):
            return row
    return None


def _check_one_task(suite_name: str, suite: Any, task_idx: int, is_libero_plus: bool) -> DecomposeCheck:
    task = suite.get_task(task_idx)

    bddl_path = _get_bddl_path(task, is_libero_plus=is_libero_plus)
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL not found: {bddl_path}")

    bddl_problem = parse_bddl_problem(str(bddl_path))
    prompt_from_bddl = str(bddl_problem.get("language") or "")
    goals = bddl_problem.get("goals", [])
    task_name = _build_task_name(task, fallback=str(getattr(task, "bddl_file", "")))
    return _build_relation_example_row(
        suite=suite_name,
        task_idx=task_idx,
        task_name=task_name,
        prompt_from_bddl=prompt_from_bddl,
        bddl_file=str(bddl_path),
        goals=goals,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a curated flattened BDDL-native CSV with 3 single-goal families and 2 multi-goal tasks."
        )
    )
    parser.add_argument(
        "--is-libero-plus",
        action="store_true",
        default=True,
        help="Use libero-plus naming normalization for BDDL (default: enabled).",
    )
    parser.add_argument(
        "--no-libero-plus",
        dest="is_libero_plus",
        action="store_false",
        help="Disable libero-plus naming normalization.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/dense_reward_goal_decomposition_relation_examples.csv"),
        help="Path to the per-relation example CSV.",
    )
    args = parser.parse_args()

    all_rows: list[DecomposeCheck] = []

    for suite_name in SUITES:
        with redirect_stdout(io.StringIO()):
            suite = get_benchmark_overridden(suite_name)()
        n_tasks = suite.get_num_tasks()
        for task_idx in range(n_tasks):
            r = _check_one_task(
                suite_name=suite_name,
                suite=suite,
                task_idx=task_idx,
                is_libero_plus=args.is_libero_plus,
            )
            all_rows.append(r)

    single_rows = sorted([r for r in all_rows if r.goal_count == 1], key=_task_sort_key)
    multi_rows = sorted([r for r in all_rows if r.goal_count > 1], key=_multi_task_sort_key)

    curated_rows: list[tuple[str, DecomposeCheck]] = []
    chosen: set[tuple[str, int]] = set()

    for demo_group, predicate in (
        ("single-placement", lambda r: r.primary_mode == "place"),
        (
            "single-articulation-open-close",
            lambda r: r.primary_mode == "interact"
            and _normalize_relation(r.primary_relation) in {"open", "close"},
        ),
        (
            "single-switch-turnon-turnoff",
            lambda r: r.primary_mode == "interact"
            and _normalize_relation(r.primary_relation) in {"turnon", "turnoff"},
        ),
    ):
        row = _pick_first(single_rows, predicate)
        if row is not None and (row.suite, row.task_idx) not in chosen:
            curated_rows.append((demo_group, row))
            chosen.add((row.suite, row.task_idx))

    multi_examples: list[DecomposeCheck] = []
    seen_multi_signatures: set[str] = set()
    for row in multi_rows:
        signature = f"{row.task_relations}:{row.goal_signature}"
        if signature in seen_multi_signatures:
            continue
        multi_examples.append(row)
        seen_multi_signatures.add(signature)
        if len(multi_examples) == 2:
            break

    for idx, row in enumerate(multi_examples, start=1):
        curated_rows.append((f"multi-goal-{idx}", row))

    print("=" * 110)
    print("Dense Reward Curated Examples")
    print(f"is_libero_plus: {args.is_libero_plus}")
    print(f"rows_selected: {[group for group, _ in curated_rows]}")
    print("=" * 110)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "demo_group",
                "suite",
                "task_idx",
                "task_name",
                "goal_family",
                "goal_count",
                "prompt_from_bddl",
                "goal_signature",
                "dense_model_signature",
                "task_relations",
                "primary_mode",
                "primary_relation",
                "dense_reward_compatible",
                "bddl_file",
            ],
        )
        writer.writeheader()
        for demo_group, r in curated_rows:
            writer.writerow(
                {
                    "demo_group": demo_group,
                    "suite": r.suite,
                    "task_idx": r.task_idx,
                    "task_name": r.task_name,
                    "goal_family": r.goal_family,
                    "goal_count": r.goal_count,
                    "prompt_from_bddl": r.prompt_from_bddl,
                    "goal_signature": r.goal_signature,
                    "dense_model_signature": r.dense_model_signature,
                    "task_relations": r.task_relations,
                    "primary_mode": r.primary_mode,
                    "primary_relation": r.primary_relation,
                    "dense_reward_compatible": r.dense_reward_compatible,
                    "bddl_file": r.bddl_file,
                }
            )
    print(f"Saved CSV table to: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
