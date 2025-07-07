# train.py

from satellite.envs.quadcopter import Quadcopter
from satellite.models.custom_model import Policy, Value, Shared
from satellite.envs.wrappers.isaacgym_envs_wrapper import IsaacGymWrapper

import isaacgym
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

import argparse

HEADLESS = False
FORCE_RENDER = False
NUM_ENVS = 4096
ROLLOUTS = 16
N_EPOCHS = 100

CONFIG = {
    # --- seed & devices ----------------------------------------------------
    "set_seed": False,
    "seed": 42,

    "physics_engine": "physx",

    "rl_device": "cuda:0",
    "sim_device": "cuda:0",
    "graphics_device_id": 0,
    "headless": HEADLESS,
    "virtual_screen_capture": False,
    "force_render": FORCE_RENDER,

    # --- env section -------------------------------------------------------
    "env": {
        'numEnvs': NUM_ENVS,
        'envSpacing': 1.25,        
        'clipObservations': 5.0,  # no clipping
        'clipActions': 1.0,  # no clipping
        'enableCameraSensors': False
    },

    # --- sim section -------------------------------------------------------
    "sim": {
        'dt': 0.0166,
        'substeps': 2,
        'up_axis': 'z', 
        'use_gpu_pipeline': True, 
        'gravity': [0.0, 0.0, -9.81], 
        'physx': {
            'use_gpu': True, 
        },
        'num_position_iterations': 4,
        'num_velocity_iterations': 0,
        'contact_offset': 0.02,
        'rest_offset': 0.001,
        'bounce_threshold_velocity': 0.2,
        'max_depenetration_velocity': 1000.0,
        'default_buffer_size_multiplier': 5.0,
        'max_gpu_contact_pairs': 1048576, # 1024*1024
        'contact_collection': 0
    },

    # --- RL / PPO hyper-params --------------------------------------------
    "rl": {
        "PPO": {
            "num_envs": NUM_ENVS,
            "rollouts": ROLLOUTS,
            "learning_epochs": 8,
            "mini_batches": 8,
            
            "learning_rate_scheduler" : KLAdaptiveRL,
            "learning_rate_scheduler_kwargs" : {"kl_threshold": 0.016},
            "state_preprocessor" : RunningStandardScaler,
            "value_preprocessor" : RunningStandardScaler,
            "rewards_shaper" : lambda rewards, timestep, timesteps: rewards * 0.1,

            "discount_factor" : 0.99, #(γ) Future reward discount; balances immediate versus long-term return.
            "learning_rate" : 1e-3, #Step size for optimizer (e.g. Adam) when updating policy and value networks.
            "grad_norm_clip" : 1.0, #Maximum norm value to clip gradients, preventing exploding gradients.
            "ratio_clip" : 0.2, #(ϵ) PPO’s clipping threshold on the policy probability ratio to constrain updates.
            "value_clip" : 0.2, #Clipping range for value function targets to stabilize value updates.
            "clip_predicted_values" : True, #If enabled, clips the new value predictions to lie within the range defined by value_clip around the old predictions.
            "entropy_loss_scale" : 0.00, #Coefficient multiplying the entropy bonus; encourages exploration when > 0.
            "value_loss_scale" : 1.0, #Coefficient weighting the value function loss in the total loss.
            "kl_threshold" : 0, #Optional early-stop threshold on KL divergence between old and new policies (0 disables).
            "lambda" : 0.95, #(λ) GAE parameter for bias–variance trade-off in advantage estimation.

            "random_timesteps" : 0, #Number of initial timesteps with random actions before learning or policy-driven sampling.
            "learning_starts" : 0, #Number of environment steps to collect before beginning any gradient updates.
            
            "experiment": {
                "write_interval": "auto",
                "checkpoint_interval": "auto",
                "directory": "./runs/satellite",
                "wandb": False,
            },
        },
        "trainer": {
            "rollouts": ROLLOUTS,
            "n_epochs": N_EPOCHS,
            "timesteps": ROLLOUTS * N_EPOCHS,
            "disable_progressbar": False,
            "headless": HEADLESS,
        },
        "memory": {
            "rollouts": ROLLOUTS,
        },
    },
}

def main():

    if CONFIG["set_seed"]:
        set_seed(CONFIG["seed"])
    
    #################################################################################

    env = Quadcopter(
        cfg=CONFIG,
        rl_device=CONFIG["rl_device"],
        sim_device=CONFIG["sim_device"],
        graphics_device_id=CONFIG["graphics_device_id"],
        headless=CONFIG["headless"],
        virtual_screen_capture=CONFIG["virtual_screen_capture"],
        force_render= CONFIG["force_render"],
    )
        
    env = IsaacGymWrapper(env)

    #####################################################################################

    memory = RandomMemory(memory_size=CONFIG["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Shared(env.state_space, env.action_space, env.device)
    models["value"] = models["policy"]  # Shared model for policy and value
   
    CONFIG["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    CONFIG["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update(CONFIG["rl"]["PPO"])

    agent = PPO(models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.state_space,
            action_space=env.action_space,
            device=env.device)
    
    trainer = SequentialTrainer(cfg=CONFIG["rl"]["trainer"], env=env, agents=agent)

    trainer.train()
    
    #################################################################################

if __name__ == "__main__":
    main()