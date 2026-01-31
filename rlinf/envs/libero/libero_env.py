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
    list_of_dict_to_dict_of_list,
    put_info_on_image,
    save_rollout_video,
    tile_images,
    to_tensor,
    parse_bddl_goal,
    process_plus_name,
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
        # dense reward phase/state tracking: 0=Reach, 1=Place, 2=Leave
        self._dense_phase = np.zeros(self.num_envs, dtype=int)
        self._last_gripper = [None] * self.num_envs
        self._last_pos_ro = [None] * self.num_envs  # object-to-eef position
        self._last_pos_o = [None] * self.num_envs   # object world position
        self._last_d_od = [None] * self.num_envs    # object-destination distance
        # Initial distances (treated as max distance for normalization)
        self._init_d_ro = [None] * self.num_envs
        self._init_d_od = [None] * self.num_envs

        # --- dense reward config + delta caches (backward compatible defaults) ---
        self.dense_reward_mode = cfg.get("dense_reward_mode", "none")  # "state" | "delta" | "none"
        self.dense_reward_coef = float(cfg.get("dense_reward_coef", 1.0))
        self.dense_reward_clip = float(cfg.get("dense_reward_clip", 0.1))
        self.dense_reward_zero_on_first_step = bool(
            cfg.get("dense_reward_zero_on_first_step", True)
        )
        self._prev_d_ro_norm = [None] * self.num_envs
        self._prev_d_od_norm = [None] * self.num_envs

        # --- debug: print task id in real time (minimal, optional) ---
        self.print_task_id = bool(cfg.get("print_task_id", False))

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
            task_id = self.task_ids[env_id]
            if task_id not in self.task_goal_meta:
                goal_meta = parse_bddl_goal(task_bddl_file_base)
                self.task_goal_meta[task_id] = goal_meta
            
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
                self._dense_phase[int(i)] = 0
                self._last_gripper[int(i)] = None
                self._last_pos_ro[int(i)] = None
                self._last_pos_o[int(i)] = None
                self._last_d_od[int(i)] = None
                self._init_d_ro[int(i)] = None
                self._init_d_od[int(i)] = None
                self._prev_d_ro_norm[int(i)] = None
                self._prev_d_od_norm[int(i)] = None
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
                self._dense_phase[i] = 0
                self._last_gripper[i] = None
                self._last_pos_ro[i] = None
                self._last_pos_o[i] = None
                self._last_d_od[i] = None
                self._init_d_ro[i] = None
                self._init_d_od[i] = None
                self._prev_d_ro_norm[i] = None
                self._prev_d_od_norm[i] = None

    def _record_metrics(self, step_reward, dense_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.reward_dense += dense_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
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

        # --- debug print: task id used this time (per env reset/reconfigure) ---
        if self.print_task_id:
            for j, env_id in enumerate(env_idx):
                print(
                    f"[LiberoEnv pid? seed_offset={self.seed_offset}] env={int(env_id)} "
                    f"task_id={int(task_ids[j])} trial_id={int(trial_ids[j])}",
                    flush=True,
                )

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
        if self.dense_reward_mode == "none":
            dense_reward = np.zeros(self.num_envs)
        else:
            gripper = [raw_obs[env_idx]["robot0_gripper_qpos"] for env_idx in range(self.num_envs)]
            dense_reward = self._calc_dense_reward(obs['obs_obj'], gripper)
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

    def _calc_dense_reward(self, obs, grippers, th_gripper=0.02, th_d_od=0.04):
        """
        Dense reward with phase-dependent objective.

        Phase (dynamically inferred each step):
        - Reach  (phase 0): want d_ro_norm to go DOWN   (eef -> object)
        - Place  (phase 1): want d_od_norm to go DOWN   (object -> destination)
        - Leave  (phase 2): want d_ro_norm to go UP     (eef far away from object, after release & placed)
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
            task_id = self.task_ids[env_idx]
            goal = self.task_goal_meta[task_id]
            relation = goal.get("relation", None)
            obj = goal.get("object", None)
            dest = goal.get("destination", None)

            assert relation == "On", "Only 'On' relation is supported in dense reward."
            assert obj is not None, "Target object must be specified in dense reward."
            assert dest is not None, "Destination object must be specified in dense reward."

            # Reach: robot -> object
            pos_ro = obs_env[obj]["to_robot0_eef_pos"]  # tensor
            d_ro = pos_ro.norm().item()
            pos_ro_np = pos_ro.detach().cpu().numpy()

            # Place / Leave: object -> destination
            pos_o = obs_env[obj]["pos"]
            pos_d = obs_env[dest]["pos"]
            pos_o_np = pos_o.detach().cpu().numpy()
            pos_od = pos_o - pos_d
            d_od = pos_od.norm().item()

            # init normalization denominators
            if self._init_d_ro[env_idx] is None and d_ro > eps_den:
                self._init_d_ro[env_idx] = d_ro
            if self._init_d_od[env_idx] is None and d_od > eps_den:
                self._init_d_od[env_idx] = d_od

            d_ro_max = (
                self._init_d_ro[env_idx]
                if (self._init_d_ro[env_idx] is not None and self._init_d_ro[env_idx] > eps_den)
                else d_ro + eps_den
            )
            d_od_max = (
                self._init_d_od[env_idx]
                if (self._init_d_od[env_idx] is not None and self._init_d_od[env_idx] > eps_den)
                else d_od + eps_den
            )

            d_ro_norm = float(np.clip(d_ro / d_ro_max, 0.0, 1.0))
            d_od_norm = float(np.clip(d_od / d_od_max, 0.0, 1.0))

            # --- phase inference (stability-based; no d_ro threshold) ---
            prev_phase = int(self._dense_phase[env_idx])

            last_g = self._last_gripper[env_idx]
            last_pos_ro = self._last_pos_ro[env_idx]
            last_pos_o = self._last_pos_o[env_idx]

            # "target no longer changing" = stable
            pos_ro_stable = (last_pos_ro is not None) and (
                np.linalg.norm(pos_ro_np - last_pos_ro) < eps_pos
            )
            pos_o_stable = (last_pos_o is not None) and (
                np.linalg.norm(pos_o_np - last_pos_o) < eps_pos
            )

            # gripper stable
            grip_stable = (last_g is not None) and (abs(grip_scalar - last_g) < eps_grip)

            closed_now = grip_scalar < th_gripper
            opened_now = grip_scalar > th_gripper
            dest_reached = d_od < th_d_od

            # Place: grasped/holding => object-to-eef relative pose becomes stable + gripper stable
            cond_place = closed_now and pos_ro_stable and grip_stable
            # Leave: released + object is on destination (threshold) + object stops moving + gripper stable
            cond_leave = opened_now and dest_reached and pos_o_stable and grip_stable

            # Mutually exclusive by priority
            if cond_leave:
                phase = 2
            elif cond_place:
                phase = 1
            else:
                phase = 0

            self._dense_phase[env_idx] = phase

            # phase switch: init corresponding prev so switch-step delta ~= 0
            if phase != prev_phase:
                if phase in (0, 2):
                    self._prev_d_ro_norm[env_idx] = d_ro_norm
                else:  # phase == 1
                    self._prev_d_od_norm[env_idx] = d_od_norm

            # --- reward computation ---
            if self.dense_reward_mode == "delta":
                if phase == 0:
                    prev = self._prev_d_ro_norm[env_idx]
                    if prev is None and self.dense_reward_zero_on_first_step:
                        reward = 0.0
                    else:
                        prev_val = d_ro_norm if prev is None else float(prev)
                        reward = self.dense_reward_coef * (prev_val - d_ro_norm)
                    self._prev_d_ro_norm[env_idx] = d_ro_norm
                elif phase == 1:
                    prev = self._prev_d_od_norm[env_idx]
                    if prev is None and self.dense_reward_zero_on_first_step:
                        reward = 0.0
                    else:
                        prev_val = d_od_norm if prev is None else float(prev)
                        reward = self.dense_reward_coef * (prev_val - d_od_norm)
                    self._prev_d_od_norm[env_idx] = d_od_norm
                else:  # phase == 2 (Leave): reward eef moving away from object
                    prev = self._prev_d_ro_norm[env_idx]
                    if prev is None and self.dense_reward_zero_on_first_step:
                        reward = 0.0
                    else:
                        prev_val = d_ro_norm if prev is None else float(prev)
                        reward = self.dense_reward_coef * (d_ro_norm - prev_val)
                    self._prev_d_ro_norm[env_idx] = d_ro_norm

                if self.dense_reward_clip is not None:
                    reward = float(
                        np.clip(reward, -self.dense_reward_clip, self.dense_reward_clip)
                    )
            else:
                # legacy "state" dense reward (scaled down to avoid linear blow-up)
                if phase == 0:
                    reward = (1.0 - d_ro_norm) / self.cfg.max_episode_steps
                elif phase == 1:
                    reward = (1.0 - d_od_norm) / self.cfg.max_episode_steps
                else:  # phase == 2
                    reward = d_ro_norm / self.cfg.max_episode_steps

            dense_rewards.append(reward)

            # logs
            self._dense_log_gripper[env_idx].append(grip_scalar)
            self._dense_log_d_ro[env_idx].append(d_ro)
            self._dense_log_d_od[env_idx].append(d_od)
            self._dense_log_reward[env_idx].append(reward)

            # update last-step states
            self._last_gripper[env_idx] = grip_scalar
            self._last_pos_ro[env_idx] = pos_ro_np
            self._last_pos_o[env_idx] = pos_o_np
            self._last_d_od[env_idx] = d_od

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
