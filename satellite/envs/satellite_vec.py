# satellite_vec.py

import numpy as np
from typing import Any

from satellite.configs.satellite_config import SatelliteConfig
from satellite.utils.satellite_util import class_to_dict, sample_random_quaternion_batch, quat_diff, quat_diff_rad
from satellite.isaacgymenvs.vec_task import VecTask
from satellite.rewards.satellite_reward import (
    TestReward,
    WeightedSumReward,
    TwoPhaseReward,
    ExponentialStabilizationReward,
    ContinuousDiscreteEffortReward,
    ShapingReward,
    RewardFunction
)

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

class SatelliteVec(VecTask):
    def __init__(self, cfg: SatelliteConfig, rl_device: Any, sim_device: Any, graphics_device_id: int, headless: bool, 
                 virtual_screen_capture: bool = False, force_render: bool = False, reward_fn: RewardFunction = None):
        self.cfg = class_to_dict(cfg)
        self._cfg = cfg

        super().__init__(self.cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        self.max_episode_length = int(np.ceil(self._cfg.env.episode_length_s / self._cfg.sim.dt))

        ################# SETUP SIM #################
        self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.actor_root_state).view(self._cfg.env.num_envs, 13)
        #############################################

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.satellite_pos     = self.root_states[:, 0:3]
        self.satellite_quats   = self.root_states[:, 3:7]
        self.satellite_linvels = self.root_states[:, 7:10]
        self.satellite_angvels = self.root_states[:, 10:13]
        ########################################

        self.prev_angvel = self.satellite_angvels.clone()
        self.satellite_angacc = (self.satellite_angvels - self.prev_angvel) / self._cfg.sim.dt

        self.initial_root_states = self.root_states.clone()

        self.goal_quat = sample_random_quaternion_batch(self._cfg.env.device, self._cfg.env.num_envs)
        #self.goal_quat = torch.tensor( [0, 1, 0, 0], dtype=torch.float32, device=self._cfg.env.device).repeat((self._cfg.env.num_envs, 1))
        self.goal_ang_vel = torch.zeros((self._cfg.env.num_envs, 3), dtype=torch.float32, device=self._cfg.env.device)
        self.goal_ang_acc = torch.zeros((self._cfg.env.num_envs, 3), dtype=torch.float32, device=self._cfg.env.device)

        self.states_buf = torch.cat((self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc, self.satellite_angvels), dim=-1)
        self.obs_buf = torch.cat((self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc), dim=-1)

        if reward_fn is None:
            self.reward_fn: RewardFunction = TestReward()
        else:
            self.reward_fn = reward_fn
                    
    def create_sim(self) -> None:
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.create_envs(self._cfg.env.env_spacing, int(np.sqrt(self._cfg.env.num_envs)))

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.satellite_asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing[0], -spacing[1], -spacing[2])
        env_upper = gymapi.Vec3(spacing[0], spacing[1], spacing[2])

        for i in range(self._cfg.env.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.create_actor(i, env, self.satellite_asset, self._cfg.asset.init_pos_p, self._cfg.asset.init_pos_r, 1, self._cfg.asset.name)
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)

    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self._cfg.asset.root, self._cfg.asset.file)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        return asset

    ################################################################################################################################
    
    def termination(self) -> None:
        ids = torch.nonzero(self.reset_buf | self.timeout_buf, as_tuple=False).flatten()
        if ids.numel() != 0:
            self.reset_idx(ids)
        
    def reset_idx(self, ids: torch.Tensor) -> None:      
        print(f"[reset_idx] Reset envs: {ids.tolist()}")

        ################# SIM #################
        self.root_states[ids] = self.initial_root_states[ids].clone()

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(ids.to(dtype=torch.int32)), len(ids.to(dtype=torch.int32))
        )
        #######################################

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        ########################################

        self.prev_angvel[ids] = self.satellite_angvels[ids].clone()
        self.satellite_angacc[ids] = (self.satellite_angvels[ids] - self.prev_angvel[ids]) / self._cfg.sim.dt

        self.states_buf[ids] = torch.cat((self.satellite_quats, quat_diff(self.satellite_quats[ids], self.goal_quat[ids]), self.satellite_angacc[ids], self.satellite_angvels[ids]), dim=-1)
        self.obs_buf[ids] = torch.cat((self.satellite_quats, quat_diff(self.satellite_quats[ids], self.goal_quat[ids]), self.satellite_angacc[ids]), dim=-1)
        
        self.goal_quat[ids] = sample_random_quaternion_batch(self._cfg.env.device, len(ids))
        #self.goal_quat[ids] = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self._cfg.env.device).repeat((len(ids), 1))
        self.goal_ang_vel[ids] = torch.zeros((len(ids), 3), dtype=torch.float32, device=self._cfg.env.device)
        self.goal_ang_acc[ids] = torch.zeros((len(ids), 3), dtype=torch.float32, device=self._cfg.env.device)

        self.progress_buf[ids] = 0
        self.reset_buf[ids] = False
        self.timeout_buf[ids] = False

        self.reward_buf[ids] = 0.0

    def compute_observations(self) -> None:
        self.prev_angvel = self.satellite_angvels.clone()

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        ########################################

        self.satellite_angacc = (self.satellite_angvels - self.prev_angvel) / self._cfg.sim.dt

        self.states_buf = torch.cat((self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc, self.satellite_angvels), dim=-1)
        self.obs_buf = torch.cat((self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc), dim=-1)
        
        if self._cfg.env.sensor_noise_std > 0.0:
            self.obs_buf = self.obs_buf + torch.normal(mean=0.0, std=self._cfg.env.sensor_noise_std, 
                                                       size=self.obs_buf.shape, device=self._cfg.env.device)
        self.obs_buf = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)

        if self._cfg.env.sensor_noise_std > 0.0:
            self.states_buf = self.states_buf + torch.normal(mean=0.0, std=self._cfg.env.sensor_noise_std, 
                                                             size=self.states_buf.shape, device=self._cfg.env.device)
        self.states_buf = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs)

    def apply_torque(self, actions: torch.Tensor) -> None:
        if self._cfg.env.actuation_noise_std > 0.0:
            actions = actions + torch.normal(mean=0.0, std=self._cfg.env.actuation_noise_std, 
                                             size=actions.shape, device=self._cfg.env.device)
        self.actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        
        print(f"[apply_torque]: actions[0]=[{', '.join(f'{v:.2f}' for v in actions[0].tolist())}]")
        
        ################# SIM #################
        torque_tensor = torch.zeros((self.num_bodies * self._cfg.env.num_envs, 3), device=self._cfg.env.device)
        root_indices = torch.arange(self._cfg.env.num_envs, device=self._cfg.env.device, dtype=torch.long) * self.num_bodies
        torque_tensor[root_indices] = actions

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(torch.zeros_like(torque_tensor)),  
            gymtorch.unwrap_tensor(torque_tensor), 
            gymapi.ENV_SPACE
        )
        #######################################
    
    def compute_reward(self) -> None:
        self.reward_buf = self.reward_fn.compute(
            self.satellite_quats, self.satellite_angvels, self.satellite_angacc,
            self.goal_quat, self.goal_ang_vel, self.goal_ang_acc,
            self.actions
        )

    def check_termination(self) -> None:
        angle_diff = quat_diff_rad(self.satellite_quats, self.goal_quat)
        ang_vel_diff = torch.norm((self.satellite_angvels - self.goal_ang_vel), dim=1)
        
        timeout = self.progress_buf >= self.max_episode_length
        overspeed = torch.norm(self.satellite_angvels, dim=1) >= self._cfg.env.overspeed_ang_vel
        goal = ((angle_diff < self._cfg.env.threshold_ang_goal) & (ang_vel_diff < self._cfg.env.threshold_vel_goal)) 

        self.timeout_buf = (timeout | overspeed).to(torch.bool)
        self.reset_buf = (goal).to(torch.bool)
        
        timeout_ids = torch.nonzero(timeout, as_tuple=False).flatten()
        if len(timeout_ids) > 0:
            print(f"[check_termination] TIMEOUT or OVERSPEED in envs: {timeout_ids.tolist()}")
        
        reset_ids = torch.nonzero(self.reset_buf, as_tuple=False).flatten()
        if len(reset_ids) > 0:
            print(f"[check_termination] GOAL envs: {reset_ids.tolist()}")
        