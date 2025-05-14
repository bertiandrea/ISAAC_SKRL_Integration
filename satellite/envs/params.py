# params.py

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

import numpy as np
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