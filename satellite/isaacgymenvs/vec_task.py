# vec_task.py

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

import time
import sys
import numpy as np
from typing import Dict, Any, Tuple
from abc import ABC
from gymnasium import spaces

class Params(ABC):
    def __init__(self, config, headless: bool): 
        self.headless = headless

        self.screen_width = getattr(config, 'screen_width', 1920)
        self.screen_height = getattr(config, 'screen_height', 1080)

        ########################################################################
        
        self.device_type = getattr(config, 'device_type', 'cpu')
        self.device_id = getattr(config, 'device_id', -1)
        self.device = getattr(config, 'device', 'cpu')
        
        self.num_envs = getattr(config.env, 'num_envs', 1)
        self.num_agents = getattr(config.env, 'num_agents', 1)
        
        self.num_observations = getattr(config.env, 'num_observations', 0)
        self.num_states = getattr(config.env, 'num_states', 0)
        self.num_actions = getattr(config.env, 'num_actions', 0)

        self.clip_obs = getattr(config.env, 'clip_observations', np.Inf)
        self.clip_actions = getattr(config.env, 'clip_actions', np.Inf)

        ########################################################################

        self.dt: float = getattr(config.sim, "dt", 1.0 / 60.0)
        if config.sim.physics_engine == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            self.physics_engine = gymapi.SIM_PHYSX
        self.sim_params = self.parse_sim_params(config.sim)
        
        ########################################################################

        self.episode_length_s = getattr(config.env, 'episode_length_s', 600.0)
        self.max_episode_length = int(np.ceil(self.episode_length_s / self.dt))

        self.env_spacing = getattr(config.env, 'env_spacing', 0.0)
        
        self.asset_init_pos_p = getattr(config.asset, 'init_pos_p', [0.0, 0.0, 0.0])
        self.asset_init_pos_r = getattr(config.asset, 'init_pos_r', [0.0, 0.0, 0.0, 1.0])
        self.asset_name = getattr(config.asset, 'name', 'satellite')
        self.asset_root = getattr(config.asset, 'root', '.')
        self.asset_file = getattr(config.asset, 'file', 'satellite.urdf')

        self.sensor_noise_std = getattr(config.env, 'sensor_noise_std', 0.0)
        self.actuation_noise_std = getattr(config.env, 'actuation_noise_std', 0.0)
        self.torque_scale = getattr(config.env, 'torque_scale', 1.0)

        self.overspeed_ang_vel = getattr(config.env, 'overspeed_ang_vel', 1.57)
        self.threshold_ang_goal = getattr(config.env, 'threshold_ang_goal', 0.01745)
        self.threshold_vel_goal = getattr(config.env, 'threshold_vel_goal', 0.01745)

        ########################################################################

        self.obs_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -np.Inf, np.ones(self.num_actions) * np.Inf)



    def parse_sim_params(self, config_sim) -> gymapi.SimParams:
        sim_params = gymapi.SimParams()

        sim_params.dt = getattr(config_sim, "dt", 1.0 / 60.0)
        sim_params.num_client_threads = getattr(config_sim, "num_client_threads", 1)
        if self.device_type != 'cpu':
            sim_params.use_gpu_pipeline = getattr(config_sim, "use_gpu_pipeline", False)
        else:
            sim_params.use_gpu_pipeline = False
        sim_params.substeps = getattr(config_sim, "substeps", 2)
        sim_params.gravity = gymapi.Vec3(*getattr(config_sim, "gravity", [0.0, 0.0, -9.81]))
        if config_sim.up_axis == "y":
            sim_params.up_axis = gymapi.UP_AXIS_Y
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Z
        if config_sim.physics_engine == "flex":
            for opt, val in vars(config_sim.flex).items():
                setattr(sim_params.flex, opt, val)
        else:  
            for opt, val in vars(config_sim.physx).items():
                if opt == "contact_collection":
                    setattr(sim_params.physx, opt, gymapi.ContactCollection(val))
                else:
                    setattr(sim_params.physx, opt, val)

        return sim_params

class VecTask(Params):
    def __init__(self, config, headless: bool): 
        super().__init__(config, headless)

        self.last_frame_time: float = 0.0

        self.create_sim()
        
        self.set_viewer()
        
        self.allocate_buffers()
    
    def create_sim(self) -> None:
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(self.device_id, self.device_id, self.physics_engine, self.sim_params)
        self.create_envs(self.env_spacing, int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.satellite_asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing[0], -spacing[1], -spacing[2])
        env_upper = gymapi.Vec3(spacing[0], spacing[1], spacing[2])

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.create_actor(i, env, self.satellite_asset, self.asset_init_pos_p, self.asset_init_pos_r, 1, self.asset_name)
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)

    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        return asset
    
    def set_viewer(self):
        self.viewer = None
        if self.headless == True:
            return
        
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
        if self.viewer == None:
            return
        
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.gym.sync_frame_time(self.sim)

        now = time.time()

        delta = now - self.last_frame_time
        if delta < self.dt:
            time.sleep(self.dt - delta)

        self.last_frame_time = time.time()


    def pre_physics_step(self, actions: torch.Tensor) -> None:        
        self.termination()

        self.apply_torque(actions)

    def post_physics_step(self) -> None:      
        self.compute_observations()

        self.compute_reward()

        self.check_termination()

        self.progress_buf += 1
        
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self.pre_physics_step(actions)

        ######################################################################
        self.gym.simulate(self.sim)

        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        ######################################################################
        
        self.post_physics_step()
        
        return self.states_buf.to(self.device).clone(), \
            self.reward_buf.to(self.device).view(-1, 1).clone(), \
            self.reset_buf.to(self.device).view(-1, 1).clone(), \
            self.timeout_buf.to(self.device).view(-1, 1).clone(), \
            {}

    def reset(self):
        ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(ids)
        
        return self.states_buf.to(self.device).clone(), {}

    def close(self) -> None:
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def destroy(self) -> None:
        self.close()