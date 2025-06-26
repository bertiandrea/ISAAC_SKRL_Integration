# satellite_config.py

from pathlib import Path
import numpy as np

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

NUM_ENVS = 8
ROLLOUTS = 8

CONFIG = {
    # --- seed & devices ----------------------------------------------------
    "set_seed": False,
    "seed": 42,

    "physics_engine": "physx",

    "rl_device": "cuda:0",
    "sim_device": "cuda:0",
    "graphics_device_id": 0,
    "headless": True,
    "virtual_screen_capture": False,
    "force_render": False,

    "profile": False,

    "heartbeat": False,

    # --- env section -------------------------------------------------------
    "env": {
        "numEnvs": NUM_ENVS,

        "numObservations": 14,

        "numStates": 17,

        "numActions": 3,

        "envSpacing": 2.0,

        "sensor_noise_std": 0.0,
        "actuation_noise_std": 0.0,

        "threshold_ang_goal": 0.0872665,
        "threshold_vel_goal": 0.0174532,
        "overspeed_ang_vel": 0.78540,
        "episode_length_s": 30,

        "clipActions": np.inf,
        "clipObservations": np.inf,

        "torque_scale": 1,

        "debug_arrows": False,

        "asset": {

            "assetRoot": str(Path(__file__).resolve().parent.parent),
            "assetFileName": "satellite.urdf",
            "assetName": "satellite",

            "init_pos_p": [0, 0, 0],
            "init_pos_r": [0, 0, 0, 1],
            
            #"disable_gravity"
            #"collapse_fixed_joints"
            #"slices_per_cylinder"
            #"replace_cylinder_with_capsule"
            #"fix_base_link"
            #"default_dof_drive_mode"
            #"self_collisions"
            #"flip_visual_attachments"

            #"density"
            #"angular_damping"
            #"linear_damping"
            #"max_angular_velocity"
            #"max_linear_velocity"
            #"armature"
            #"thickness"
        },
    },

    # --- sim section -------------------------------------------------------
    "sim": {
        "dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, 0.0],
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "substeps": 2,

        #"num_client_threads"
        #"stress_visualization"
        #"stress_visualization_max"
        #"stress_visualization_min"

        "physx": {
            "use_gpu": True,
            #"solver_type" = 1
            #"num_threads" = 4
            #"num_position_iterations" = 4
            #"num_velocity_iterations" = 1
            #"contact_offset"
            #"rest_offset"
            #"bounce_threshold_velocity"
            #"contact_collection"
            #"default_buffer_size_multiplier"
            #"max_depenetration_velocity"
            #"max_gpu_contact_pairs"
            #"num_subscenes"
            #"always_use_articulations"
            #"friction_correlation_distance"
            #"friction_offset_threshold"
        },
        #"flex": {
            #"solver_type"
            #"num_outer_iterations"
            #"num_inner_iterations"
            #"relaxation"
            #"warm_start"
            #"contact_regularization"
            #"deterministic_mode"
            #"dynamic_friction"
            #"friction_mode"
            #"geometric_stiffness"
            #"max_rigid_contacts"
            #"max_soft_contacts"
            #"particle_friction"
            #"return_contacts"
            #"shape_collision_distance"
            #"shape_collision_margin"
            #"static_friction"
        #},
    },

    # --- RL / PPO hyper-params --------------------------------------------
    "rl": {
        "PPO": {
            "num_envs": NUM_ENVS,
            "rollouts": ROLLOUTS,
            "learning_epochs": 4,
            "mini_batches": 8,
            
            "experiment": {
                "write_interval": 1,
                "checkpoint_interval": 1,
                "directory": "./runs/satellite",
                "experiment_name": "satellite_test",
                "wandb": False,
            },
        },
        "trainer": {
            "rollouts": ROLLOUTS,
            "n_epochs": 2,
            "timesteps": ROLLOUTS * 2,
            "disable_progressbar": False,
            "headless": True,
        },
        "memory": {
            "rollouts": ROLLOUTS,
        },
    },

    # --- low-level controllers --------------------------------------------
    "controller": {
        "controller_logic": False
    },
    "pid": {
        "rate": {"kp": 0.5, "ki": 0.0, "kd": 0.1},
    },
    # --- logging -----------------------------------------------------------
    "log_reward": {
        "log_reward": False,
        "log_reward_interval": 100,
    },
    # --- CAPS --------------------------------------------------------------
    "CAPS": {
        "enabled": False,
        "lambda_temporal_smoothness": 0.0,  # λ_t
        "lambda_spatial_smoothness": 0.0,   # λ_s
        "noise_std": 0.00,                  # σ
    },
}