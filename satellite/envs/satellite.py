# satellite.py

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
import torch
from isaacgym import gymutil, gymtorch, gymapi

from pathlib import Path
import numpy as np

from torch.profiler import record_function

from torch.utils.tensorboard import SummaryWriter

BASE_COLORS_SAT  = torch.tensor([[1,0,1], [0,1,1], [1,1,0]], dtype=torch.float32)
BASE_COLORS_GOAL = torch.tensor([[0,0,1], [0,1,0], [1,0,0]], dtype=torch.float32)

class Satellite(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, reward_fn: RewardFunction = None):
        self.cfg = cfg

        self.dt = cfg["sim"].get('dt', 1 / 60.0)  # seconds
        self.env_spacing = cfg["env"].get('envSpacing', 0.0)
        self.asset_name = cfg["env"]["asset"].get('assetName', 'satellite')
        self.asset_root = cfg["env"]["asset"].get('assetRoot', str(Path(__file__).resolve().parent.parent))
        self.asset_file = cfg["env"]["asset"].get('assetFileName', 'satellite.urdf')
        self.asset_init_pos_p = cfg["env"]["asset"].get('init_pos_p', [0.0, 0.0, 0.0])
        self.asset_init_pos_r = cfg["env"]["asset"].get('init_pos_r', [0.0, 0.0, 0.0, 1.0])
        self.actuation_noise_std = cfg["env"].get('actuation_noise_std', 0.0)
        self.sensor_noise_std = cfg["env"].get('sensor_noise_std', 0.0)
        self.torque_scale = cfg["env"].get('torque_scale', 1.0)
        self.threshold_ang_goal = cfg["env"].get('threshold_ang_goal', 0.01745)  # radians
        self.threshold_vel_goal = cfg["env"].get('threshold_vel_goal', 0.01745)  # radians/sec
        self.overspeed_ang_vel = cfg["env"].get('overspeed_ang_vel', 0.78540)  # radians/sec
        self.max_episode_length = cfg["env"].get('episode_length_s', 120) / self.dt  # seconds
        self.debug_arrows = cfg["env"].get('debug_arrows', False)
        self.heartbeat = cfg.get('heartbeat', False)
        
        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

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
        print(f"Initial root states: {self.initial_root_states[0]}")
        ########################################

        self.prev_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.actions_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.goal_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1) #sample_random_quaternion_batch(self.device, self.num_envs)
        self.goal_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_ang_acc = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.torque_tensor = torch.zeros((self.num_bodies * self.num_envs, 3), device=self.device)
        self.root_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.int) * self.num_bodies
        self.force_tensor = torch.zeros_like(self.torque_tensor, device=self.device)

        if reward_fn is None:
            self.reward_fn: RewardFunction = TestReward()
        else:
            self.reward_fn = reward_fn

        self.controller_logic = cfg["controller"].get("controller_logic", False)
        if self.controller_logic:
            self.pid = PID(
                num_envs=self.num_envs,
                device=self.device,
                dt=self.dt,
                kp=cfg["pid"]["rate"].get("kp", 1.0),
                ki=cfg["pid"]["rate"].get("ki", 0.1),
                kd=cfg["pid"]["rate"].get("kd", 0.01),
            )
            self.controller = SatelliteAttitudeController(
                torque_tau=cfg["controller"].get("torque_tau", 0.02),
                pid=self.pid,
                num_envs=self.num_envs,
                device=self.device,
            )

        if self.debug_arrows:
            self.draw_arrows()
        
        self.writer = SummaryWriter(comment="_satellite_reward")
        self.global_step = 0

    def create_sim(self) -> None:
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params) # Acquires the sim pointer
        self.create_envs(self.env_spacing, int(np.sqrt(self.num_envs)))

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.sat_glob_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            origin = self.gym.get_env_origin(env)
            self.sat_glob_pos[i] = torch.tensor([origin.x, origin.y, origin.z],
                                                dtype=torch.float32,
                                                device=self.device)
            self.create_actor(i, env, self.asset, self.asset_init_pos_p, self.asset_init_pos_r, 1, self.asset_name)
            self.envs.append(env)

    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        return asset
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)
        
                
    def draw_arrows(self):
        x_goal = quat_axis(self.goal_quat, 0)
        y_goal = quat_axis(self.goal_quat, 1)
        z_goal = quat_axis(self.goal_quat, 2)
        x_sat  = quat_axis(self.satellite_quats, 0)
        y_sat  = quat_axis(self.satellite_quats, 1)
        z_sat  = quat_axis(self.satellite_quats, 2)

        sat_lines = torch.cat([
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + x_sat * 1.5], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + y_sat * 1.5], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + z_sat * 1.5], dim=1),
        ], dim=0)  # → (3N,2,3)
        goal_lines = torch.cat([
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + x_goal * 2.0], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + y_goal * 2.0], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + z_goal * 2.0], dim=1),
        ], dim=0)  # → (3N,2,3)
        all_lines = torch.cat([sat_lines, goal_lines], dim=0)  # → (6N,2,3)

        colors_sat  = BASE_COLORS_SAT.repeat_interleave(self.num_envs, dim=0)   # (3N,3)
        colors_goal = BASE_COLORS_GOAL.repeat_interleave(self.num_envs, dim=0)  # (3N,3)
        all_colors = torch.cat([colors_sat, colors_goal], dim=0)  # (6N,3)

        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(
            self.viewer,
            None,
            6 * self.num_envs,
            all_lines.cpu().numpy(),
            all_colors.cpu().numpy()
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

        with record_function("$SatelliteVec__reset_idx__reset_buffers"):
            self.prev_angvel[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)
            self.actions_integral[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)

            self.goal_quat[ids] = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device).repeat(len(ids), 1) #sample_random_quaternion_batch(self.device, len(ids))
            self.goal_ang_vel[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)
            self.goal_ang_acc[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)

            self.progress_buf[ids] = 0
            self.reset_buf[ids] = False
            self.timeout_buf[ids] = False

            self.rew_buf[ids] = 0.0
        
        if self.controller_logic:
            with record_function("$SatelliteVec__reset_idx__reset_controller"):
                self.controller.reset(ids)
        
        if self.debug_arrows:
            with record_function("$SatelliteVec__reset_idx__draw_arrows"):
                self.draw_arrows()

    ################################################################################################################################
                
    def termination(self) -> None:
        with record_function("$SatelliteVec__termination"):      
            ids  = torch.nonzero(self.reset_buf, as_tuple=False).flatten()
            if len(ids) > 0:
                self.reset_idx(ids)
    
    def apply_torque(self, actions: torch.Tensor) -> None:
        ############## CONTROLLER ###############
        if self.controller_logic:
            actions = self.controller.compute_control(
                actions=actions, 
                measured_angacc=self.satellite_angacc
            )
        #########################################
        
        with record_function("$SatelliteVec__apply_torque__noise"):
            if self.actuation_noise_std > 0.0:
                actions = torch.add(
                    actions,
                    torch.normal(mean=0.0, std=self.actuation_noise_std, size=actions.shape, device=self.device)
                )

        with record_function("$SatelliteVec__apply_torque__scale_integrate_clamp"):
            self.actions = torch.mul(actions, self.torque_scale)
            self.actions_integral += torch.mul(self.actions, self.dt)

            self.actions = torch.clamp(self.actions, -self.clip_actions * self.torque_scale, self.clip_actions * self.torque_scale)
            self.actions_integral = torch.clamp(self.actions_integral, -self.clip_actions * self.torque_scale, self.clip_actions * self.torque_scale)

        #########################################
        self.writer.add_scalar('Actions/action_X', self.actions[0, 0].item(), global_step=self.global_step)
        self.writer.add_scalar('Actions/action_Y', self.actions[0, 1].item(), global_step=self.global_step)
        self.writer.add_scalar('Actions/action_Z', self.actions[0, 2].item(), global_step=self.global_step)
        self.writer.add_scalar('Actions/action_integral_X', self.actions_integral[0, 2].item(), global_step=self.global_step)
        self.writer.add_scalar('Actions/action_integral_Y', self.actions_integral[0, 2].item(), global_step=self.global_step)
        self.writer.add_scalar('Actions/action_integral_Z', self.actions_integral[0, 2].item(), global_step=self.global_step)

        self.global_step += 1

        assert not torch.isnan(self.actions).any(), f"actions has NaN: {self.actions, self.states_buf}"
        assert not torch.isinf(self.actions).any(), f"actions has Inf: {self.actions, self.states_buf}"
        assert not torch.isnan(self.actions_integral).any(), f"actions_integral has NaN: {self.actions_integral, self.states_buf}"
        assert not torch.isinf(self.actions_integral).any(), f"actions_integral has Inf: {self.actions_integral, self.states_buf}"
        #########################################

        ################# SIM #################
        with record_function("$SatelliteVec__apply_torque__sim"):
            self.torque_tensor[self.root_indices] = self.actions_integral
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.force_tensor),  
                gymtorch.unwrap_tensor(self.torque_tensor), 
                gymapi.LOCAL_SPACE,
            )
        #######################################
                
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
                (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), 
                 quat_diff_rad(self.satellite_quats, self.goal_quat).unsqueeze(-1), 
                 self.satellite_angacc, self.actions, self.actions_integral), dim=-1)
            self.states_buf = torch.cat(
                (self.obs_buf, self.satellite_angvels), dim=-1)
        ########################################

        with record_function("$SatelliteVec__compute_observations__noise"):
            if self.sensor_noise_std > 0.0:
                noise = torch.normal(mean=0.0, std=self.sensor_noise_std, size=self.state_space.shape, device=self.device)
                self.obs_buf = torch.add(self.obs_buf, noise[:, :self.num_observations])

        ########################################
        assert not torch.isnan(self.obs_buf).any(), f"self.obs_buf has NaN: {self.actions, self.obs_buf}"
        assert not torch.isinf(self.obs_buf).any(), f"self.obs_buf has Inf: {self.actions, self.obs_buf}"
        assert not torch.isnan(self.states_buf).any(), f"self.states_buf has NaN: {self.actions, self.states_buf}"
        assert not torch.isinf(self.states_buf).any(), f"self.states_buf has Inf: {self.actions, self.states_buf}"
        ########################################

    def compute_reward(self) -> None:
        with record_function("$SatelliteVec__compute_reward"):
            self.rew_buf = self.reward_fn.compute(
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

            self.timeout_buf = timeout
            self.reset_buf = torch.logical_or(timeout, overspeed)
    
    def pre_physics_step(self, actions):
        if self.heartbeat:
            return

        self.termination()

        self.apply_torque(actions)

    def post_physics_step(self):
        self.progress_buf += 1
        
        if self.heartbeat:
            return
        
        self.compute_observations()

        self.compute_reward()

        self.check_termination()

        if self.debug_arrows:
            with record_function("$SatelliteVec__post_physics_step__draw_arrows"):
                self.draw_arrows()