# satellite_config.py

from satellite.configs.base_config import BaseConfig
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from pathlib import Path

import isaacgym #BugFix
import torch

CUDA = torch.cuda.is_available()

class SatelliteConfig(BaseConfig):
    set_seed = False
    seed = 42

    device_type = torch.device("cuda" if CUDA else "cpu")
    device_id = torch.cuda.current_device() if CUDA else -1
    device = device_type + ":" + str(device_id)

    class env:  
        num_envs = 4096
   
        num_observations = 11 # [x,y,z,w, dx,dy,dz,dw, ax,ay,az]

        num_states = 14 # [x,y,z,w, dx,dy,dz,dw, vx,vy,vz, ax,ay,az]

        num_actions = 3
        
        env_spacing = [4.0, 4.0, 4.0]

        sensor_noise_std = 0.0
        actuation_noise_std = 0.0
        
        threshold_ang_goal = 0.01745        # soglia in radianti per orientamento
        threshold_vel_goal = 0.01745        # soglia in rad/sec per la differenza di velocità
        overspeed_ang_vel = 1.57            # soglia in rad/sec per l'overspeed
        episode_length_s = 120              # soglia in secondi per la terminazione di una singola simulazione
        
        #clip_actions = 1
        #clip_observations = 1

        torque_scale = 10

        record_frames = False 
        
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
        use_gpu_pipeline = CUDA
        physics_engine = 'physx'

        class physx:
            use_gpu = CUDA

    class rl:
        class PPO:
            num_envs = 4096
            rollouts = ROLLOUTS
            learning_epochs = 8
            minibatch_size = 16384
            mini_batches = rollouts * num_envs // minibatch_size,
            discount_factor = 0.99
            lambda_ = 0.95
            learning_rate = 1e-3
            learning_rate_scheduler = KLAdaptiveRL
            learning_rate_scheduler_kwargs = {"kl_threshold": 0.016}
            grad_norm_clip = 1.0
            ratio_clip = 0.2
            value_clip = 0.2
            clip_predicted_values = True
            entropy_loss_scale = 0.00
            value_loss_scale = 1.0
            kl_threshold = 0
            rewards_shaper = lambda rewards, timestep, timesteps: rewards * 0.01
            state_preprocessor = RunningStandardScaler
            value_preprocessor = RunningStandardScaler
            random_timesteps = 0
            learning_starts = 0
            
            class experiment:
                    write_interval = 10
                    checkpoint_interval = 100
                    directory = "./runs"
                    wandb = False

        class trainer:
            rollouts = 16
            n_epochs = 8192
            timesteps = rollouts * n_epochs
            headless = False
            disable_progressbar = True   # whether to disable the progressbar. If None, disable on non-TTY
        

        
