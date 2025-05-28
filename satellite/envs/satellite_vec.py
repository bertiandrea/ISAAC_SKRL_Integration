# satellite_vec.py

from satellite.utils.satellite_util import sample_random_quaternion_batch, quat_diff, quat_diff_rad
from satellite.envs.vec_task import VecTask
from satellite.rewards.satellite_reward import (
    TestReward,
    RewardFunction
)

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

class SatelliteVec(VecTask):
    def __init__(self, cfg, reward_fn: RewardFunction = None):
        super().__init__(cfg)

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
        self.initial_root_states = self.root_states.clone()
        self.prev_angvel = self.satellite_angvels.clone()
        ########################################

        self.goal_quat = sample_random_quaternion_batch(self.device, self.num_envs)
        self.goal_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_ang_acc = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.torque_tensor = torch.zeros((self.num_bodies * self.num_envs, 3), device=self.device)
        self.root_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.int) * self.num_bodies
        self.force_tensor = torch.zeros_like(self.torque_tensor, device=self.device)

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
        self.root_states[ids] = self.initial_root_states[ids]
        actor_indices = ids.to(dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.actor_root_state, gymtorch.unwrap_tensor(actor_indices), len(actor_indices)
        )
        #######################################

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.prev_angvel = self.satellite_angvels.clone()
        ########################################

        self.goal_quat[ids] = sample_random_quaternion_batch(self.device, len(ids))
        self.goal_ang_vel[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)
        self.goal_ang_acc[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)

        self.progress_buf[ids] = 0
        self.reset_buf[ids] = False
        self.timeout_buf[ids] = False

        self.reward_buf[ids] = 0.0

    def compute_observations(self) -> None:
        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.satellite_angacc = (self.satellite_angvels - self.prev_angvel) / self.dt
        self.prev_angvel = self.satellite_angvels.clone()
        self.obs_buf = torch.cat(
            (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc, self.actions), dim=-1)
        self.states_buf = torch.cat(
            (self.obs_buf, self.satellite_angvels), dim=-1)
        ########################################

        print(f"[compute_observations]: satellite_quats[0]=[{', '.join(f'{v:.2f}' for v in self.satellite_quats[0].tolist())}]")
        print(f"[compute_observations]: satellite_quats[1]=[{', '.join(f'{v:.2f}' for v in self.satellite_quats[1].tolist())}]")
        print(f"[compute_observations]: satellite_quats[2]=[{', '.join(f'{v:.2f}' for v in self.satellite_quats[2].tolist())}]")

        if self.sensor_noise_std > 0.0:
            noise = torch.normal(mean=0.0, std=self.sensor_noise_std, size=self.state_space.shape, device=self.device)
            self.obs_buf = self.obs_buf + noise[:, :self.num_observations]
            self.states_buf = self.states_buf + noise[:, :self.num_states]
        self.obs_buf = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)
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
        self.torque_tensor[self.root_indices] = self.actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.force_tensor),  
            gymtorch.unwrap_tensor(self.torque_tensor), 
            gymapi.ENV_SPACE
        )
        #######################################
    
    def compute_reward(self) -> None:
        #self.reward_buf = self.reward_fn.compute(
        #    self.satellite_quats, self.satellite_angvels, self.satellite_angacc,
        #    self.goal_quat, self.goal_ang_vel, self.goal_ang_acc,
        #    self.actions
        #)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        print(f"[compute_reward]: reward_buf[0]={self.reward_buf[0].item():.2f}")
        print(f"[compute_reward]: reward_buf[1]={self.reward_buf[1].item():.2f}")
        print(f"[compute_reward]: reward_buf[2]={self.reward_buf[2].item():.2f}")

    def check_termination(self) -> None:
        angle_diff = quat_diff_rad(self.satellite_quats, self.goal_quat)
        ang_vel_diff = torch.norm((self.satellite_angvels - self.goal_ang_vel), dim=1)
        
        timeout = self.progress_buf >= self.max_episode_length
        overspeed = torch.norm(self.satellite_angvels, dim=1) >= self.overspeed_ang_vel
        goal = ((angle_diff < self.threshold_ang_goal) & (ang_vel_diff < self.threshold_vel_goal)) 

        self.timeout_buf = torch.where(timeout | overspeed, True, False)
        self.reset_buf = torch.where(goal, True, False)
        
        timeout_ids = torch.nonzero(timeout, as_tuple=False).flatten()
        if len(timeout_ids) > 0:
            print(f"[check_termination] TIMEOUT or OVERSPEED in envs: {timeout_ids.tolist()}")
        
        reset_ids = torch.nonzero(self.reset_buf, as_tuple=False).flatten()
        if len(reset_ids) > 0:
            print(f"[check_termination] GOAL envs: {reset_ids.tolist()}")
        
        
        