# satellite_config.py

from satellite.configs.base_config import BaseConfig
from pathlib import Path

import isaacgym #BugFix
import torch

class SatelliteConfig(BaseConfig):
    seed = 42
    physics_engine = 'physx'

    class env:
        numEnvs = 1024
        num_envs = 1024

        numObservations = 7 # [x,y,z,w, ax,ay,az]
        num_observations = 7 # [x,y,z,w, ax,ay,az]

        numStates = 10 # [x,y,z,w, vx,vy,vz, ax,ay,az]
        num_states = 10 # [x,y,z,w, vx,vy,vz, ax,ay,az]

        numActions = 3
        num_actions = 3
        
        env_spacing = [4.0, 4.0, 4.0]

        sensor_noise_std = 0.0
        actuation_noise_std = 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        threshold_ang_goal = 0.01745        # soglia in radianti per orientamento
        threshold_vel_goal = 0.01745        # soglia in rad/sec per la differenza di velocità
        overspeed_ang_vel = 3.141           # soglia in rad/sec per l'overspeed
        episode_length_s = 30               # soglia in secondi per la terminazione di una singola simulazione
        
        #clipActions = 1
        #clipObservations = 1

    class asset:
        root = str(Path(__file__).resolve().parent.parent)
        file = "satellite.urdf"
        name = "satellite"

        init_pos_p = [0, 0, 0]    # posizione iniziale del satellite [x,y,z]
        init_pos_r = [0, 0, 0, 1] # attitude iniziale del satellite [x,y,z,w]

    class sim:
        dt = 1.0 / 60.0
        gravity = [0.0, 0.0, 0.0] # [m/s^2]
        up_axis = 'z'
        use_gpu_pipeline = torch.cuda.is_available()

        class physx:
            use_gpu = torch.cuda.is_available()
