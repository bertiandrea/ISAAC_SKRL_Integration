# vec_task.py

from satellite.envs.params import Params

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

import sys
import numpy as np
from typing import Dict, Any, Tuple

class VecTask(Params):
    def __init__(self, config, headless: bool): 
        super().__init__(config, headless)

        self.create_sim()
        
        self.viewer = None
        if not self.headless:
            self.set_viewer()
        
        self.allocate_buffers()
        
        self.gym.prepare_sim(self.sim)

    def create_sim(self) -> None:
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(self.device_id, self.device_id, self.physics_engine, self.sim_params)
        self.create_envs(self.env_spacing, int(np.sqrt(self.num_envs)))

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing[0], -spacing[1], -spacing[2])
        env_upper = gymapi.Vec3(spacing[0], spacing[1], spacing[2])

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.create_actor(i, env, self.asset, self.asset_init_pos_p, self.asset_init_pos_r, 1, self.asset_name)
    
    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        return asset
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)
    
    def set_viewer(self):        
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.screen_width
        camera_props.height = self.screen_height
        self.viewer = self.gym.create_viewer(
            self.sim, camera_props)

        sim_params = self.gym.get_sim_params(self.sim)
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

        self.gym.viewer_camera_look_at(
            self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

    def render(self):
        if self.gym.query_viewer_has_closed(self.viewer):
            self.close()

        self.gym.step_graphics(self.sim)

        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.gym.sync_frame_time(self.sim)

    def pre_physics_step(self, actions: torch.Tensor) -> None:        
        self.termination()

        self.apply_torque(actions)

    def post_physics_step(self) -> None:      
        self.compute_observations()

        self.compute_reward()

        self.check_termination()

        self.progress_buf += 1

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        #self.pre_physics_step(actions)

        ######################################################################
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        ######################################################################
        
        #self.post_physics_step()
        
        return self.states_buf, \
            self.reward_buf.view(-1, 1), \
            self.reset_buf.view(-1, 1), \
            self.timeout_buf.view(-1, 1), \
            {}

    def reset(self):
        ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(ids)
        
        return self.states_buf, {}

    def close(self) -> None:
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        sys.exit()

    def destroy(self) -> None:
        self.close()