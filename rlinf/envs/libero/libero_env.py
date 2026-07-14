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

import copy
import os
from typing import Optional, Union

import gym
import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark
from libero.libero.envs import OffScreenRenderEnv
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.libero.utils import (
    get_benchmark_overridden,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import (
    build_dense_goal_models,
    list_of_dict_to_dict_of_list,
    parse_bddl_goal,
    parse_bddl_goals,
    process_plus_name,
    put_info_on_image,
    save_rollout_video,
    select_primary_dense_model,
    tile_images,
    to_tensor,
)


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        # Example structure: {task_id: {"relation": "On", "object": "akita_black_bowl_1", "destination": "plate_1"}}
        self.task_goal_meta = {}
        # Example structure: {task_id: {"models": [...], "primary": {...}}}
        self.task_dense_meta = {}

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []
        self.current_raw_obs = None

        self._dense_log_gripper = [[] for _ in range(self.num_envs)]
        self._dense_log_d_ro = [[] for _ in range(self.num_envs)]
        self._dense_log_d_od = [[] for _ in range(self.num_envs)]
        # dense reward trajectory per environment
        self._dense_log_reward = [[] for _ in range(self.num_envs)]
        # dense reward phase tracking: 0=Approach, 1=Relation-Actuation, 2=Stabilize
        self._dense_phase = np.zeros(self.num_envs, dtype=int)
        self._last_gripper = [None] * self.num_envs
        self._last_pos_ro = [None] * self.num_envs  # object-to-eef position
        self._last_pos_o = [None] * self.num_envs   # object world position
        self._last_d_od = [None] * self.num_envs    # object-destination distance
        # Initial distances (treated as max distance for normalization)
        self._init_d_ro = [None] * self.num_envs
        self._init_d_od = [None] * self.num_envs
        self._init_obj_state = [None] * self.num_envs

        # --- dense reward config (delta-only) ---
        self.use_dense_reward = bool(cfg.get("use_dense_reward", False))
        self.dense_reward_coef = float(cfg.get("dense_reward_coef", 1.0))
        self.dense_reward_clip = float(cfg.get("dense_reward_clip", 0.1))
        self._prev_d_ro_norm = [None] * self.num_envs
        self._prev_d_od_norm = [None] * self.num_envs
        self._prev_obj_state_score = [None] * self.num_envs
        self._dense_goal_idx = [0] * self.num_envs


    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []
        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param):
                seed = param.pop("seed")
                env = OffScreenRenderEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        task_descriptions = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(self.task_descriptions[env_id])
                continue
            task = self.task_suite.get_task(self.task_ids[env_id])

            bddl_file = process_plus_name(task.bddl_file) if self.cfg.is_libero_plus else task.bddl_file
            task_bddl_file_base = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, bddl_file
            )
            if not os.path.exists(task_bddl_file_base):
                task_bddl_file_base = os.path.join(
                    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
                )
            task_id = self.task_ids[env_id]
            if task_id not in self.task_goal_meta:
                goal_meta_first = parse_bddl_goal(task_bddl_file_base)
                goal_meta_all = parse_bddl_goals(task_bddl_file_base)
                dense_models = build_dense_goal_models(goal_meta_all)
                self.task_goal_meta[task_id] = goal_meta_first
                self.task_dense_meta[task_id] = {
                    "models": dense_models,
                    "primary": select_primary_dense_model(dense_models),
                }
            
            task_bddl_file = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": task_bddl_file,
                    "seed": self.seed,
                }
            )
            task_descriptions.append(task.language)
        self.task_descriptions = task_descriptions
        return env_fn_params

    def _resolve_obs_key(self, name: Optional[str], obs_env):
        """
        Resolve BDDL entity name to observation object key.
        Handles region-like suffixes and fallback prefix matches.
        """
        if name is None:
            return None

        name = str(name)
        if name in obs_env:
            return name

        suffixes = [
            "_contain_region",
            "_top_region",
            "_bottom_region",
            "_left_region",
            "_right_region",
            "_front_region",
            "_back_region",
            "_middle_region",
            "_center_region",
            "_region",
        ]

        candidates = [name]
        reduced = name
        changed = True
        while changed:
            changed = False
            for suf in suffixes:
                if reduced.endswith(suf):
                    reduced = reduced[: -len(suf)]
                    candidates.append(reduced)
                    changed = True

        for cand in candidates:
            if cand in obs_env:
                return cand

        for cand in candidates:
            for k in obs_env.keys():
                if k.startswith(cand + "_") or cand.startswith(k + "_"):
                    return k

        return None

    def _get_dense_goal_models(self, task_id: int):
        dense_meta = self.task_dense_meta.get(task_id, {})
        models = dense_meta.get("models", [])
        if not models:
            primary = dense_meta.get("primary", {"mode": "none"})
            models = [primary]
        return models

    def _reset_dense_goal_state(self, env_idx: int):
        self._dense_goal_idx[env_idx] = 0
        self._dense_phase[env_idx] = 0
        self._prev_d_ro_norm[env_idx] = None
        self._prev_d_od_norm[env_idx] = None
        self._prev_obj_state_score[env_idx] = None

    def _advance_dense_goal_state(self, env_idx: int):
        self._dense_goal_idx[env_idx] += 1
        self._dense_phase[env_idx] = 0
        self._prev_d_ro_norm[env_idx] = None
        self._prev_d_od_norm[env_idx] = None
        self._prev_obj_state_score[env_idx] = None

    def _set_dense_goal_state(self, env_idx: int, goal_idx: int):
        self._dense_goal_idx[env_idx] = max(0, int(goal_idx))
        self._dense_phase[env_idx] = 0
        self._prev_d_ro_norm[env_idx] = None
        self._prev_d_od_norm[env_idx] = None
        self._prev_obj_state_score[env_idx] = None

    def _compute_obj_state_score(self, env_idx: int, dense_meta: dict, raw_obs_env):
        obj_state = raw_obs_env.get("object-state", None)
        if obj_state is None:
            return None

        obj_state_vec = np.asarray(obj_state).ravel().astype(np.float32)
        if self._init_obj_state[env_idx] is None and obj_state_vec.size > 0:
            self._init_obj_state[env_idx] = obj_state_vec.copy()

        init_obj_state = self._init_obj_state[env_idx]
        if init_obj_state is None or init_obj_state.size != obj_state_vec.size:
            return None

        direction = float(dense_meta.get("direction", 1.0))
        state_delta = float(np.mean(obj_state_vec - init_obj_state))
        return float(np.tanh(2.0 * direction * state_delta))

    def _is_goal_satisfied(self, env_idx: int, dense_meta: dict, obs_env, raw_obs_env) -> bool:
        mode = str(dense_meta.get("mode", "none"))
        thresholds = dense_meta.get("thresholds", {})

        if mode == "place":
            obj_key = self._resolve_obs_key(dense_meta.get("object", None), obs_env)
            dest_key = self._resolve_obs_key(dense_meta.get("destination", None), obs_env)
            if (
                obj_key is None
                or dest_key is None
                or ("pos" not in obs_env[obj_key])
                or ("pos" not in obs_env[dest_key])
            ):
                return False
            pos_o = obs_env[obj_key]["pos"]
            pos_d = obs_env[dest_key]["pos"]
            d_od = (pos_o - pos_d).norm().item()
            th_place_distance = float(thresholds.get("th_place_distance", 0.04))
            return d_od < th_place_distance

        if mode == "interact":
            score = self._compute_obj_state_score(env_idx, dense_meta, raw_obs_env)
            th_state_progress = float(thresholds.get("th_state_progress", 0.35))
            return score is not None and score >= th_state_progress

        if mode == "reach":
            target_key = self._resolve_obs_key(dense_meta.get("target", None), obs_env)
            if target_key is None or ("to_robot0_eef_pos" not in obs_env[target_key]):
                return False
            d_ro = obs_env[target_key]["to_robot0_eef_pos"].norm().item()
            th_reach_distance = float(thresholds.get("th_reach_distance", 0.10))
            return d_ro <= th_reach_distance

        return True

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

    def update_reset_state_ids(self):
        if self.cfg.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        self._generator_ordered.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (self.num_group,), dtype=int
            )
        else:
            if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
                self.reset_state_ids_all = self.get_reset_state_ids_all()
                self.start_idx = 0
            reset_state_ids = self.reset_state_ids_all[self.seed_offset][
                self.start_idx : self.start_idx + num_reset_states
            ]
            self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.reward_dense = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.reward_dense[mask] = 0
            self._elapsed_steps[env_idx] = 0
            # reset dense-reward-related states for these envs
            idxs = np.atleast_1d(env_idx)
            for i in idxs:
                self._dense_log_gripper[int(i)] = []
                self._dense_log_d_ro[int(i)] = []
                self._dense_log_d_od[int(i)] = []
                self._dense_log_reward[int(i)] = []  # reset dense reward log
                self._reset_dense_goal_state(int(i))
                self._last_gripper[int(i)] = None
                self._last_pos_ro[int(i)] = None
                self._last_pos_o[int(i)] = None
                self._last_d_od[int(i)] = None
                self._init_d_ro[int(i)] = None
                self._init_d_od[int(i)] = None
                self._init_obj_state[int(i)] = None
                self._prev_d_ro_norm[int(i)] = None
                self._prev_d_od_norm[int(i)] = None
                self._prev_obj_state_score[int(i)] = None
                self._dense_goal_idx[int(i)] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.reward_dense[:] = 0.0
            self._elapsed_steps[:] = 0
            # reset all dense-reward-related states
            for i in range(self.num_envs):
                self._dense_log_gripper[i] = []
                self._dense_log_d_ro[i] = []
                self._dense_log_d_od[i] = []
                self._dense_log_reward[i] = []  # reset dense reward log
                self._reset_dense_goal_state(i)
                self._last_gripper[i] = None
                self._last_pos_ro[i] = None
                self._last_pos_o[i] = None
                self._last_d_od[i] = None
                self._init_d_ro[i] = None
                self._init_d_od[i] = None
                self._init_obj_state[i] = None
                self._prev_d_ro_norm[i] = None
                self._prev_d_od_norm[i] = None
                self._prev_obj_state_score[i] = None
                self._dense_goal_idx[i] = 0

    def _record_metrics(self, step_reward, dense_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.reward_dense += dense_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["task_id"] = self.task_ids.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward_sparse"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["reward_dense"] = self.reward_dense / episode_info["episode_len"]
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _extract_objects_states(self, obs):
        """
        Parse each non-robot object keys:
          <obj>_pos, <obj>_quat, <obj>_to_robot0_eef_pos, <obj>_to_robot0_eef_quat

        Returns:
        {
            'object1': {
                'pos': tensor,
                'quat': tensor,
                'to_robot0_eef_pos': tensor,
                'to_robot0_eef_rot': tensor (axis-angle),
            },
            'object2': { ... },
            ...
        }
        """
        # First parse with numpy, then convert to tensors at the end
        np_object_states = {}

        for key in obs.keys():
            # Exclude robot itself and images
            if key.startswith("robot0_"):
                continue
            if key in ("agentview_image", "robot0_eye_in_hand_image"):
                continue

            if key.endswith("_to_robot0_eef_pos"):
                base = key[: -len("_to_robot0_eef_pos")]
                np_object_states.setdefault(base, {})["to_robot0_eef_pos"] = np.asarray(
                    obs[key]
                ).ravel()
            elif key.endswith("_to_robot0_eef_quat"):
                base = key[: -len("_to_robot0_eef_quat")]
                np_object_states.setdefault(base, {})["to_robot0_eef_quat"] = np.asarray(
                    obs[key]
                ).ravel()
            elif key.endswith("_pos"):
                base = key[: -len("_pos")]
                np_object_states.setdefault(base, {})["pos"] = np.asarray(
                    obs[key]
                ).ravel()
            elif key.endswith("_quat"):
                # Exclude xxx_to_robot0_eef_quat, already handled above
                if key.endswith("_to_robot0_eef_quat"):
                    continue
                base = key[: -len("_quat")]
                np_object_states.setdefault(base, {})["quat"] = np.asarray(
                    obs[key]
                ).ravel()

        # Convert numpy arrays to tensors with expected keys
        object_states = {}
        for name, fields in np_object_states.items():
            obj_entry = {}
            if "pos" in fields:
                obj_entry["pos"] = to_tensor(fields["pos"])
            if "quat" in fields:
                obj_entry["quat"] = to_tensor(fields["quat"])
            if "to_robot0_eef_pos" in fields:
                obj_entry["to_robot0_eef_pos"] = to_tensor(fields["to_robot0_eef_pos"])
            if "to_robot0_eef_quat" in fields:
                obj_entry["to_robot0_eef_rot"] = to_tensor(quat2axisangle(fields["to_robot0_eef_quat"]))
            object_states[name] = obj_entry

        return object_states

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        obj_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)
            obj_states = self._extract_objects_states(obs)
            obj_states_list.append(obj_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        image_tensor = torch.stack(
            [
                value.clone().permute(2, 0, 1)
                for value in images_and_states["full_image"]
            ]
        )
        wrist_image_tensor = torch.stack(
            [
                value.clone().permute(2, 0, 1)
                for value in images_and_states["wrist_image"]
            ]
        )

        states = images_and_states["state"]

        obs = {
            "images": image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "obs_obj": obj_states_list,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            if self.task_ids[env_id] != task_ids[j]:
                reconfig_env_idx.append(env_id)
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]


        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        init_state = self._get_reset_states(env_idx=env_idx)
        self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        # Detect "full reset" (epoch-level reset in typical eval loops)
        env_idx_arr = np.asarray(env_idx)
        is_full_reset = (
            env_idx_arr.size == self.num_envs
            and np.array_equal(np.sort(env_idx_arr), np.arange(self.num_envs))
        )

        # In eval: when doing a full reset, always advance ordered reset ids (epoch -> new chunk of pool)
        if self.cfg.is_eval and self.use_fixed_reset_state_ids and is_full_reset:
            self.update_reset_state_ids()

        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            # IMPORTANT: in eval with fixed reset ids, default to ordered (not random),
            # so "epoch * num_envs covers the pool" holds.
            if self.cfg.is_eval and self.use_fixed_reset_state_ids:
                reset_state_ids = self.reset_state_ids[env_idx_arr]
            else:
                num_reset_states = len(env_idx)
                reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        sparse_reward = self._calc_step_reward(terminations)
        if not self.use_dense_reward:
            dense_reward = np.zeros(self.num_envs)
        else:
            gripper = [raw_obs[env_idx]["robot0_gripper_qpos"] for env_idx in range(self.num_envs)]
            dense_reward = self._calc_dense_reward(raw_obs, obs['obs_obj'], gripper)
        step_reward = list(np.asarray(sparse_reward) + np.asarray(dense_reward))

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "task": self.task_descriptions,
            }
            self.add_new_frames(raw_obs, plot_infos)

        infos = self._record_metrics(sparse_reward, dense_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # if past_dones.all():
        #     self.plot_dense_logs("/RLinf/output/")
        #     exit()

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones.cpu().numpy(), extracted_obs, infos
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if self.cfg.is_eval:
            self.update_reset_state_ids()
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _calc_dense_reward(self, raw_obs, obs, grippers, th_gripper=0.02, th_d_od=0.04):
        """
        Unified dense reward with 3 phases:
        - phase 0: Approach
        - phase 1: Relation-Actuation
        - phase 2: Stabilize

        Relation family only changes phase-1/phase-2 scoring behavior.
        """
        num_envs = len(obs)
        dense_rewards = []

        eps_grip = 1e-3
        eps_pos = 1e-3
        eps_den = 1e-6

        for env_idx in range(num_envs):
            gripper = grippers[env_idx]
            grip_scalar = float(np.max(gripper))

            obs_env = obs[env_idx]
            raw_obs_env = raw_obs[env_idx]
            task_id = self.task_ids[env_idx]
            goal_models = self._get_dense_goal_models(task_id)
            if len(goal_models) == 0:
                goal_models = [{"mode": "none"}]

            goal_idx = min(self._dense_goal_idx[env_idx], len(goal_models) - 1)

            # sequential consistency: if any previously completed sub-task is violated,
            # fallback to earliest violated goal and re-plan from there.
            if goal_idx > 0:
                fallback_idx = None
                for prev_idx in range(goal_idx):
                    if not self._is_goal_satisfied(
                        env_idx,
                        goal_models[prev_idx],
                        obs_env,
                        raw_obs_env,
                    ):
                        fallback_idx = prev_idx
                        break
                if fallback_idx is not None:
                    self._set_dense_goal_state(env_idx, fallback_idx)
                    goal_idx = fallback_idx

            dense_meta = goal_models[goal_idx]
            mode = dense_meta.get("mode", "none")
            thresholds = dense_meta.get("thresholds", {})
            th_gripper_local = float(thresholds.get("th_gripper", th_gripper))
            th_place_distance = float(thresholds.get("th_place_distance", th_d_od))
            th_state_progress = float(thresholds.get("th_state_progress", 0.35))
            th_reach_distance = float(thresholds.get("th_reach_distance", 0.10))

            d_ro = None
            d_od = None
            d_ro_norm = None
            d_od_norm = None
            pos_ro_np = None
            pos_o_np = None

            if mode == "place":
                obj_key = self._resolve_obs_key(dense_meta.get("object", None), obs_env)
                dest_key = self._resolve_obs_key(dense_meta.get("destination", None), obs_env)

                if (
                    obj_key is None
                    or dest_key is None
                    or ("to_robot0_eef_pos" not in obs_env[obj_key])
                    or ("pos" not in obs_env[obj_key])
                    or ("pos" not in obs_env[dest_key])
                ):
                    mode = "reach"
                    dense_meta = {
                        "mode": "reach",
                        "relation": dense_meta.get("relation", None),
                        "target": dense_meta.get("object", None),
                        "thresholds": thresholds,
                    }
                else:
                    pos_ro = obs_env[obj_key]["to_robot0_eef_pos"]
                    d_ro = pos_ro.norm().item()
                    pos_ro_np = pos_ro.detach().cpu().numpy()

                    pos_o = obs_env[obj_key]["pos"]
                    pos_d = obs_env[dest_key]["pos"]
                    pos_o_np = pos_o.detach().cpu().numpy()
                    pos_od = pos_o - pos_d
                    d_od = pos_od.norm().item()

            if mode == "reach":
                target_key = self._resolve_obs_key(dense_meta.get("target", None), obs_env)
                if target_key is None or ("to_robot0_eef_pos" not in obs_env[target_key]):
                    mode = "none"
                else:
                    pos_ro = obs_env[target_key]["to_robot0_eef_pos"]
                    d_ro = pos_ro.norm().item()
                    pos_ro_np = pos_ro.detach().cpu().numpy()

            obj_state_score = None
            if mode == "interact":
                target_key = self._resolve_obs_key(dense_meta.get("target", None), obs_env)
                if target_key is not None and ("to_robot0_eef_pos" in obs_env[target_key]):
                    pos_ro = obs_env[target_key]["to_robot0_eef_pos"]
                    d_ro = pos_ro.norm().item()
                    pos_ro_np = pos_ro.detach().cpu().numpy()
                obj_state_score = self._compute_obj_state_score(env_idx, dense_meta, raw_obs_env)

            # init normalization denominators
            if d_ro is not None and self._init_d_ro[env_idx] is None and d_ro > eps_den:
                self._init_d_ro[env_idx] = d_ro
            if d_od is not None and self._init_d_od[env_idx] is None and d_od > eps_den:
                self._init_d_od[env_idx] = d_od

            if d_ro is not None:
                d_ro_max = (
                    self._init_d_ro[env_idx]
                    if (self._init_d_ro[env_idx] is not None and self._init_d_ro[env_idx] > eps_den)
                    else d_ro + eps_den
                )
                d_ro_norm = float(np.clip(d_ro / d_ro_max, 0.0, 1.0))

            if d_od is not None:
                d_od_max = (
                    self._init_d_od[env_idx]
                    if (self._init_d_od[env_idx] is not None and self._init_d_od[env_idx] > eps_den)
                    else d_od + eps_den
                )
                d_od_norm = float(np.clip(d_od / d_od_max, 0.0, 1.0))

            if mode == "none" or d_ro_norm is None:
                reward = 0.0
                dense_rewards.append(reward)
                self._dense_log_gripper[env_idx].append(grip_scalar)
                self._dense_log_d_ro[env_idx].append(float("nan"))
                self._dense_log_d_od[env_idx].append(float("nan"))
                self._dense_log_reward[env_idx].append(reward)
                self._last_gripper[env_idx] = grip_scalar
                self._last_pos_ro[env_idx] = None
                self._last_pos_o[env_idx] = None
                self._last_d_od[env_idx] = None
                self._prev_obj_state_score[env_idx] = None
                continue

            prev_phase = int(self._dense_phase[env_idx])

            last_g = self._last_gripper[env_idx]
            last_pos_ro = self._last_pos_ro[env_idx]
            last_pos_o = self._last_pos_o[env_idx]

            pos_ro_stable = (pos_ro_np is not None) and (last_pos_ro is not None) and (
                np.linalg.norm(pos_ro_np - last_pos_ro) < eps_pos
            )
            pos_o_stable = (pos_o_np is not None) and (last_pos_o is not None) and (
                np.linalg.norm(pos_o_np - last_pos_o) < eps_pos
            )
            grip_stable = (last_g is not None) and (abs(grip_scalar - last_g) < eps_grip)

            cond_phase1 = False
            cond_phase2 = False

            if mode == "place":
                closed_now = grip_scalar < th_gripper_local
                opened_now = grip_scalar > th_gripper_local
                dest_reached = (d_od is not None) and (d_od < th_place_distance)
                cond_phase1 = closed_now and pos_ro_stable and grip_stable
                cond_phase2 = opened_now and dest_reached and pos_o_stable and grip_stable
            elif mode == "interact":
                near_target = d_ro_norm is not None and d_ro_norm <= th_reach_distance
                state_ready = obj_state_score is not None and obj_state_score >= th_state_progress
                cond_phase1 = near_target
                cond_phase2 = near_target and state_ready
            elif mode == "reach":
                near_target = d_ro is not None and d_ro <= th_reach_distance
                cond_phase1 = near_target
                cond_phase2 = near_target and pos_ro_stable and grip_stable

            if cond_phase2:
                phase = 2
            elif cond_phase1:
                phase = 1
            else:
                phase = 0

            self._dense_phase[env_idx] = phase

            # phase switch: init corresponding prev so switch-step delta ~= 0
            if phase != prev_phase:
                if phase == 1 and mode == "place":
                    self._prev_d_od_norm[env_idx] = d_od_norm
                else:
                    self._prev_d_ro_norm[env_idx] = d_ro_norm
                if mode == "interact":
                    self._prev_obj_state_score[env_idx] = obj_state_score

            goal_complete = phase == 2

            if phase == 0:
                prev = self._prev_d_ro_norm[env_idx]
                if prev is None:
                    reward = 0.0
                else:
                    reward = self.dense_reward_coef * (float(prev) - d_ro_norm)
                self._prev_d_ro_norm[env_idx] = d_ro_norm
            elif phase == 1:
                if mode == "place":
                    prev = self._prev_d_od_norm[env_idx]
                    if prev is None or d_od_norm is None:
                        reward = 0.0
                    else:
                        reward = self.dense_reward_coef * (float(prev) - d_od_norm)
                    self._prev_d_od_norm[env_idx] = d_od_norm
                elif mode == "interact":
                    reward_reach = 0.0
                    prev_ro = self._prev_d_ro_norm[env_idx]
                    if d_ro_norm is not None and prev_ro is not None:
                        reward_reach = float(prev_ro) - d_ro_norm

                    reward_state = 0.0
                    prev_state = self._prev_obj_state_score[env_idx]
                    if obj_state_score is not None and prev_state is not None:
                        reward_state = obj_state_score - float(prev_state)

                    reward = self.dense_reward_coef * (0.4 * reward_reach + 0.6 * reward_state)
                    self._prev_d_ro_norm[env_idx] = d_ro_norm
                    self._prev_obj_state_score[env_idx] = obj_state_score
                else:
                    reward = 0.0
                    self._prev_d_ro_norm[env_idx] = d_ro_norm
            else:  # phase == 2
                if mode == "place":
                    prev = self._prev_d_ro_norm[env_idx]
                    if prev is None:
                        reward = 0.0
                    else:
                        reward = self.dense_reward_coef * (d_ro_norm - float(prev))
                    self._prev_d_ro_norm[env_idx] = d_ro_norm
                elif mode == "interact":
                    retreat_delta = 0.0
                    prev_ro = self._prev_d_ro_norm[env_idx]
                    if d_ro_norm is not None and prev_ro is not None:
                        retreat_delta = d_ro_norm - float(prev_ro)

                    state_delta = 0.0
                    prev_state = self._prev_obj_state_score[env_idx]
                    if obj_state_score is not None and prev_state is not None:
                        state_delta = obj_state_score - float(prev_state)

                    reward = self.dense_reward_coef * (
                        0.3 * retreat_delta + 0.7 * max(state_delta, 0.0)
                    )
                    self._prev_d_ro_norm[env_idx] = d_ro_norm
                    self._prev_obj_state_score[env_idx] = obj_state_score
                else:
                    reward = 0.0
                    self._prev_d_ro_norm[env_idx] = d_ro_norm

            if self.dense_reward_clip is not None:
                reward = float(
                    np.clip(reward, -self.dense_reward_clip, self.dense_reward_clip)
                )

            dense_rewards.append(reward)

            self._dense_log_gripper[env_idx].append(grip_scalar)
            self._dense_log_d_ro[env_idx].append(d_ro)
            self._dense_log_d_od[env_idx].append(d_od if mode == "place" and d_od is not None else float("nan"))
            self._dense_log_reward[env_idx].append(reward)

            self._last_gripper[env_idx] = grip_scalar
            self._last_pos_ro[env_idx] = pos_ro_np
            self._last_pos_o[env_idx] = pos_o_np
            self._last_d_od[env_idx] = d_od if d_od is not None else None

            if goal_complete and self._dense_goal_idx[env_idx] < len(goal_models) - 1:
                self._advance_dense_goal_state(env_idx)

        return dense_rewards

    def plot_dense_logs(self, save_dir: str):
        """
        Plot per-env trajectories recorded in _calc_dense_reward:
        gripper, d_ro, d_od, dense_reward.

        Behavior:
        - Given a folder path save_dir, create it if it does not exist;
        - In that directory, generate num_env images:
          - env_{i}.png corresponds to environment i;
          - each image is a vertical concatenation of line plots
            for each metric in that environment.
        """
        os.makedirs(save_dir, exist_ok=True)

        import matplotlib.pyplot as plt
        import io

        for env_idx in range(self.num_envs):
            steps = np.arange(len(self._dense_log_gripper[env_idx]))

            # Ensure each env has at least one plot: if no records, use a single-step placeholder
            if len(steps) == 0:
                steps = np.array([0])
                metric_series = [
                    ("gripper", [0.0]),
                    ("d_ro", [0.0]),
                    ("d_od", [0.0]),
                    ("dense_reward", [0.0]),
                ]
            else:
                metric_series = [
                    ("gripper", self._dense_log_gripper[env_idx]),
                    ("d_ro", self._dense_log_d_ro[env_idx]),
                    ("d_od", self._dense_log_d_od[env_idx]),
                    ("dense_reward", self._dense_log_reward[env_idx]),
                ]

            images = []
            for name, values in metric_series:
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.plot(steps, values)
                ax.set_ylabel(name)
                ax.set_xlabel("step")
                ax.grid(True)
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plt.close(fig)

                buf.seek(0)
                img = plt.imread(buf)
                buf.close()
                images.append(img)

            # Vertically concatenate all metric images for this environment
            combined = np.concatenate(images, axis=0)

            out_path = os.path.join(save_dir, f"env_{env_idx}.png")
            plt.imsave(out_path, combined)

    def add_new_frames(self, raw_obs, plot_infos):
        images = []
        for env_id, raw_single_obs in enumerate(raw_obs):
            info_item = {
                k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()
            }
            img = raw_single_obs["agentview_image"][::-1, ::-1]
            img = put_info_on_image(img, info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []
