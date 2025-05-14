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

EXISTING_SIM = None
SCREEN_RESOLUTION = {
    "width": 1920,
    "height": 1080
}

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM

class Env(ABC):
    def __init__(self, config, headless: bool): 
        self.headless = headless

        self.device_type = getattr(config, 'device', 'cpu')
        self.device_id = getattr(config, 'device', -1)
        if self.device_type == 'cuda' or self.device_type == 'gpu':
            self.device = getattr(config, 'device', 'cpu')
        elif self.device_type == 'cpu':
            self.device = 'cpu'
        else:
            raise ValueError(f"Invalid device type: {self.device_type}")
        
        self.num_envs = getattr(config.env, 'num_envs', 0)
        self.num_agents = getattr(config.env, 'num_agents', 1)
        
        self.num_observations = getattr(config.env, 'num_observations', 0)
        self.num_states = getattr(config.env, 'num_states', 0)
        self.num_actions = getattr(config.env, 'num_actions', 0)

        self.obs_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -np.Inf, np.ones(self.num_actions) * np.Inf)

        self.control_freq_inv = getattr(config.env, 'control_freq_inv', 1)

        self.clip_obs = getattr(config.env, 'clip_observations', np.Inf)
        self.clip_actions = getattr(config.env, 'clip_actions', np.Inf)

        self.render_fps: int = getattr(config.env, 'render_fps', -1)
        self.last_frame_time: float = 0.0

        self.dt: float = getattr(config.env.sim, "dt", 1.0 / 60.0)
        if config.env.sim.physics_engine == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif config.env.sim.physics_engine == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            raise ValueError(f"Invalid physics engine backend: {config.env.sim.physics_engine}")
        
        self.sim_params = self.parse_sim_params(config.env.sim)



    def parse_sim_params(self, config_sim) -> gymapi.SimParams:
        sim_params = gymapi.SimParams()

        sim_params.dt = getattr(config_sim, "dt", 1.0 / 60.0)
        sim_params.num_client_threads = getattr(config_sim, "num_client_threads", 1)
        if self.device_type != 'cpu' and self.device_id >= 0:
            sim_params.use_gpu_pipeline = getattr(config_sim, "use_gpu_pipeline", False)
        else:
            sim_params.use_gpu_pipeline = False
        sim_params.substeps = getattr(config_sim, "substeps", 2)
        sim_params.gravity = gymapi.Vec3(*getattr(config_sim, "gravity", [0.0, 0.0, -9.81]))
        if config_sim.up_axis == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        elif config_sim.up_axis == "y":
            sim_params.up_axis = gymapi.UP_AXIS_Y
        else:
            raise ValueError(f"Invalid physics up-axis: {config_sim.up_axis}")

        if config_sim.physics_engine == "physx":
            for opt, val in vars(config_sim.physx).keys():
                if opt == "contact_collection":
                    setattr(sim_params.physx, opt, gymapi.ContactCollection(val))
                else:
                    setattr(sim_params.physx, opt, val)
        elif config_sim.physics_engine == "flex":
            for opt, val in vars(config_sim.flex).keys():
                setattr(sim_params.flex, opt, val)
        else:
            raise ValueError(f"Invalid physics engine backend: {config_sim.physics_engine}")

        return sim_params

class VecTask(Env):
    def __init__(self, config, headless: bool, force_render: bool = False): 
        super().__init__(config, headless)
        self.force_render = force_render

        self.gym = gymapi.acquire_gym()

        self.create_sim()
        self.gym.prepare_sim(self.sim)

        self.set_viewer()
        
        self.allocate_buffers()

    def set_viewer(self):
        self.viewer = None

        if self.headless == False:
            camera_props = gymapi.CameraProperties()
            camera_props.width = SCREEN_RESOLUTION["width"]
            camera_props.height = SCREEN_RESOLUTION["height"]
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

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim
        
    def render(self):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            now = time.time()
            delta = now - self.last_frame_time
            if self.render_fps < 0:
                render_dt = self.dt * self.control_freq_inv
            else:
                render_dt = 1.0 / self.render_fps

            if delta < render_dt:
                time.sleep(render_dt - delta)

            self.last_frame_time = time.time()


    def pre_physics_step(self, actions: torch.Tensor) -> None:        
        self.termination()

        self.apply_torque(actions)

    def post_physics_step(self) -> None:  
        self.progress_buf += 1
    
        self.compute_observations()

        self.compute_reward()

        self.check_termination()
        
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self.pre_physics_step(actions)

        ######################################################################
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
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
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)