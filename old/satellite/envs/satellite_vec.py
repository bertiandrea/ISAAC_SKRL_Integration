# satellite_vec.py

from satellite.utils.satellite_util import sample_random_quaternion_batch, quat_diff, quat_diff_rad, quat_axis
from satellite.envs.vec_task import VecTask
from satellite.rewards.satellite_reward import (
    TestReward,
    TestRewardSmooth,
    RewardFunction
)
from satellite.pid.pid import PID
from satellite.controller.controller import SatelliteAttitudeController

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

from torch.profiler import record_function

import numpy as np

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

        self.controller_logic = getattr(cfg.controller, "controller_logic", False)
        if self.controller_logic:
            self.pid = PID(
                num_envs=self.num_envs,
                kp=getattr(cfg.pid.rate, "kp", 1.0),
                ki=getattr(cfg.pid.rate, "ki", 0.1),
                kd=getattr(cfg.pid.rate, "kd", 0.01),
                dt=self.dt,
                device=self.device
            )
            self.controller = SatelliteAttitudeController(
                num_envs=self.num_envs,
                device=self.device,
                pid=self.pid,
                torque_tau=getattr(cfg.controller, "torque_tau", 0.02)
            )
        
        self.heartbeat = getattr(cfg, 'heartbeat', False)
        self.profile = getattr(cfg, 'profile', False)
        self.debug_arrows = getattr(cfg.env, 'debug_arrows', False)

        if self.debug_arrows:
            self.draw_arrows()

    def draw_arrows(self):
        sat_pos = self.satellite_pos.cpu().numpy()
        local_dir = quat_axis(self.goal_quat, 2).cpu().numpy()
        self.gym.clear_lines(self.viewer)
        for i, env in enumerate(self.envs):
            start = sat_pos[i]
            end   = sat_pos[i] + local_dir[i] * 2.0
            self.gym.add_lines(
                self.viewer,
                env,
                1,
                np.array([start, end], dtype=np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
                
    ################################################################################################################################
    
    def reset_idx(self, ids: torch.Tensor) -> None:
        with record_function("$SatelliteVec__reset_idx__sim"):
            ################# SIM #################
            self.root_states[ids] = self.initial_root_states[ids]
            idx32 = ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, self.actor_root_state, gymtorch.unwrap_tensor(idx32), len(idx32)
            )
            #######################################

            ################# SIM #################
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.prev_angvel = self.satellite_angvels.clone()
            ########################################
                
        with record_function("$SatelliteVec__reset_idx__sample_goal"):
            self.goal_quat[ids] = sample_random_quaternion_batch(self.device, len(ids))

        with record_function("$SatelliteVec__reset_idx__reset_buffers"):
            self.goal_ang_vel[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)
            self.goal_ang_acc[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)

            self.progress_buf[ids] = 0
            self.reset_buf[ids] = False
            self.timeout_buf[ids] = False

            self.reward_buf[ids] = 0.0
        
        if self.controller_logic:
            with record_function("$SatelliteVec__reset_idx__reset_controller"):
                self.controller.reset(ids)

        if self.debug_arrows:
            with record_function("$SatelliteVec__reset_idx__draw_arrows"):
                self.draw_arrows()

    def apply_torque(self, actions: torch.Tensor) -> None:
        ############## CONTROLLER ###############
        if self.controller_logic:
            actions = self.controller.compute_control(
                actions=actions, 
                measured_angacc=self.satellite_angacc,
            )
        #########################################

        with record_function("$SatelliteVec__apply_torque__noise_and_clamp"):
            if self.actuation_noise_std > 0.0:
                actions = torch.add(
                    actions,
                    torch.normal(mean=0.0, std=self.actuation_noise_std, size=actions.shape, device=self.device)
                )
            
            self.actions = torch.mul(
                torch.clamp(actions, -self.clip_actions, self.clip_actions),
                self.torque_scale
            )

        ################# SIM #################
        with record_function("$SatelliteVec__apply_torque__sim"):
            self.torque_tensor[self.root_indices] = self.actions
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.force_tensor),  
                gymtorch.unwrap_tensor(self.torque_tensor), 
                gymapi.ENV_SPACE
            )
        #######################################
    
    def termination(self) -> None:
        with record_function("$SatelliteVec__termination"):      
            ids  = torch.nonzero(torch.logical_or(self.reset_buf, self.timeout_buf), as_tuple=False).flatten()
            if len(ids) > 0:
                self.reset_idx(ids)

    def compute_observations(self) -> None:
        ################# SIM #################
        with record_function("$SatelliteVec__compute_observations__sim"):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.satellite_angacc = torch.div(
                torch.sub(self.satellite_angvels, self.prev_angvel),
                self.dt
            )
            self.prev_angvel = self.satellite_angvels.clone()
            self.obs_buf = torch.cat(
                (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc, self.actions), dim=-1)
            self.states_buf = torch.cat(
                (self.obs_buf, self.satellite_angvels), dim=-1)
        ########################################

        with record_function("$SatelliteVec__compute_observations__noise_and_clamp"):
            if self.sensor_noise_std > 0.0:
                noise = torch.normal(mean=0.0, std=self.sensor_noise_std, size=self.state_space.shape, device=self.device)
                self.obs_buf = torch.add(self.obs_buf, noise[:, :self.num_observations])
                # No noise on states

            self.obs_buf = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)
            self.states_buf = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs)
   
    def compute_reward(self) -> None:
        with record_function("$SatelliteVec__compute_reward"):
            self.reward_buf = self.reward_fn.compute(
                self.satellite_quats, self.satellite_angvels, self.satellite_angacc,
                self.goal_quat, self.goal_ang_vel, self.goal_ang_acc,
                self.actions
            )

    def check_termination(self) -> None:
        with record_function("$SatelliteVec__check_termination"):
            angle_diff = quat_diff_rad(self.satellite_quats, self.goal_quat)
            ang_vel_diff = torch.norm(
                torch.sub(self.satellite_angvels, self.goal_ang_vel),
                dim=1
            )
            goal = torch.logical_and(
                torch.lt(angle_diff, self.threshold_ang_goal),
                torch.lt(ang_vel_diff, self.threshold_vel_goal)
            )
            
            timeout = torch.ge(self.progress_buf, self.max_episode_length)

            overspeed = torch.ge(
                torch.norm(self.satellite_angvels, dim=1),
                self.overspeed_ang_vel
            )

            self.reset_buf = torch.where(goal, True, False)

            self.timeout_buf = torch.where(
                torch.logical_and(
                    torch.logical_or(timeout, overspeed), torch.logical_not(self.reset_buf)
                    ), True, False)

        
        
        