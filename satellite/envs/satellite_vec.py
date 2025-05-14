# satellite_vec.py

from satellite.configs.satellite_config import SatelliteConfig
from satellite.utils.satellite_util import sample_random_quaternion_batch, quat_diff, quat_diff_rad
from satellite.isaacgymenvs.vec_task import VecTask
from satellite.rewards.satellite_reward import (
    TestReward,
    RewardFunction
)

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

import numpy as np

class SatelliteVec(VecTask):
    def __init__(self, cfg, headless: bool, reward_fn: RewardFunction = None):
        super().__init__(cfg, headless)

        ################# SETUP SIM #################
        self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.actor_root_state).view(self.num_envs, 13)
        self.satellite_pos     = self.root_states[:, 0:3]
        self.satellite_quats   = self.root_states[:, 3:7]
        self.satellite_linvels = self.root_states[:, 7:10]
        self.satellite_angvels = self.root_states[:, 10:13]
        #############################################

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        ########################################

        self.initial_root_states = self.root_states.clone()
        self.prev_angvel = self.satellite_angvels.clone()
        
        self.satellite_angacc = (self.satellite_angvels - self.prev_angvel) / self.dt

        self.goal_quat = sample_random_quaternion_batch(self.device, self.num_envs)
        #self.goal_quat = torch.tensor( [0, 1, 0, 0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
        self.goal_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.goal_ang_acc = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self.states_buf = torch.cat(
            (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc, self.satellite_angvels), dim=-1)
        self.obs_buf = torch.cat(
            (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc), dim=-1)

        if reward_fn is None:
            self.reward_fn: RewardFunction = TestReward()
        else:
            self.reward_fn = reward_fn
                    
    ################################################################################################################################
    
    def termination(self) -> None:
        ids = torch.nonzero(self.reset_buf | self.timeout_buf, as_tuple=False).flatten()
        if len(ids) > 0:
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

        self.satellite_angacc[ids] = (self.satellite_angvels[ids] - self.prev_angvel[ids]) / self.dt

        self.goal_quat[ids] = sample_random_quaternion_batch(self.device, len(ids))
        #self.goal_quat[ids] = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device).repeat((len(ids), 1))
        self.goal_ang_vel[ids] = torch.zeros((len(ids), 3), dtype=torch.float32, device=self.device)
        self.goal_ang_acc[ids] = torch.zeros((len(ids), 3), dtype=torch.float32, device=self.device)

        self.states_buf[ids] = torch.cat(
            (self.satellite_quats[ids], quat_diff(self.satellite_quats[ids], self.goal_quat[ids]), self.satellite_angacc[ids], self.satellite_angvels[ids]), dim=-1)
        self.obs_buf[ids] = torch.cat(
            (self.satellite_quats[ids], quat_diff(self.satellite_quats[ids], self.goal_quat[ids]), self.satellite_angacc[ids]), dim=-1)
        
        self.progress_buf[ids] = 0
        self.reset_buf[ids] = False
        self.timeout_buf[ids] = False

        self.reward_buf[ids] = 0.0

    def compute_observations(self) -> None:
        self.prev_angvel = self.satellite_angvels.clone()

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        ########################################

        self.satellite_angacc = (self.satellite_angvels - self.prev_angvel) / self.dt

        self.states_buf = torch.cat(
            (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc, self.satellite_angvels), dim=-1)
        self.obs_buf = torch.cat(
            (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc), dim=-1)
        
        print(f"[compute_observations]: satellite_quats[0]=[{', '.join(f'{v:.2f}' for v in self.satellite_quats[0].tolist())}]")
        print(f"[compute_observations]: satellite_quats[1]=[{', '.join(f'{v:.2f}' for v in self.satellite_quats[1].tolist())}]")
        print(f"[compute_observations]: satellite_quats[2]=[{', '.join(f'{v:.2f}' for v in self.satellite_quats[2].tolist())}]")

        if self.sensor_noise_std > 0.0:
            self.obs_buf = self.obs_buf + torch.normal(mean=0.0, std=self.sensor_noise_std, 
                                                       size=self.obs_buf.shape, device=self.device)
        self.obs_buf = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)

        if self.sensor_noise_std > 0.0:
            self.states_buf = self.states_buf + torch.normal(mean=0.0, std=self.sensor_noise_std, 
                                                             size=self.states_buf.shape, device=self.device)
        self.states_buf = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs)

    def apply_torque(self, actions: torch.Tensor) -> None:
        if self.actuation_noise_std > 0.0:
            actions = actions + torch.normal(mean=0.0, std=self.actuation_noise_std, 
                                             size=actions.shape, device=self.device)
        self.actions = torch.clamp(actions, -self.clip_actions, self.clip_actions) * self.torque_scale

        print(f"[apply_torque]: actions[0]=[{', '.join(f'{v:.2f}' for v in self.actions[0].tolist())}]")
        print(f"[apply_torque]: actions[1]=[{', '.join(f'{v:.2f}' for v in self.actions[1].tolist())}]")
        print(f"[apply_torque]: actions[2]=[{', '.join(f'{v:.2f}' for v in self.actions[2].tolist())}]")

        ################# SIM #################
        torque_tensor = torch.zeros((self.num_bodies * self.num_envs, 3), device=self.device)
        root_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long) * self.num_bodies
        torque_tensor[root_indices] = self.actions

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
        print(f"[compute_reward]: reward_buf[0]={self.reward_buf[0].item():.2f}")
        print(f"[compute_reward]: reward_buf[1]={self.reward_buf[1].item():.2f}")
        print(f"[compute_reward]: reward_buf[2]={self.reward_buf[2].item():.2f}")

    def check_termination(self) -> None:
        angle_diff = quat_diff_rad(self.satellite_quats, self.goal_quat)
        ang_vel_diff = torch.norm((self.satellite_angvels - self.goal_ang_vel), dim=1)
        
        timeout = self.progress_buf >= self.max_episode_length
        overspeed = torch.norm(self.satellite_angvels, dim=1) >= self.overspeed_ang_vel
        goal = ((angle_diff < self.threshold_ang_goal) & (ang_vel_diff < self.threshold_vel_goal)) 

        self.timeout_buf = (timeout | overspeed).to(torch.bool)
        self.reset_buf = (goal).to(torch.bool)
        
        timeout_ids = torch.nonzero(timeout, as_tuple=False).flatten()
        if len(timeout_ids) > 0:
            print(f"[check_termination] TIMEOUT or OVERSPEED in envs: {timeout_ids.tolist()}")
        
        reset_ids = torch.nonzero(self.reset_buf, as_tuple=False).flatten()
        if len(reset_ids) > 0:
            print(f"[check_termination] GOAL envs: {reset_ids.tolist()}")
        