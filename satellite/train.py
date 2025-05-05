# train.py

import os
from satellite.configs.satellite_config import SatelliteConfig
from satellite.envs.satellite_vec import SatelliteVec
from satellite.models.custom_model import Policy, Value

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# fissiamo il seed per la riproducibilità
set_seed(42)

# 1) setup environment
env_cfg      = SatelliteConfig()
headless     = False
env          = SatelliteVec(cfg=env_cfg,
                            rl_device="cuda:0",
                            sim_device="cuda:0",
                            graphics_device_id=0,
                            headless=headless)

print(f"Envs: {env.num_envs}, \
        state_space: {env.state_space}, \
        obs_space: {env.obs_space}, \
        act_space: {env.act_space}, \
        device: {env.device}")

# 2) PPO config
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo.update({
    # raccolta rollouts e mini\u2010batch
    "rollouts":                   16,
    "learning_epochs":            8,
    "mini_batches":               4,
    # hyper\u2010parametri base
    "discount_factor":            0.99,
    "lambda":                     0.95,
    "learning_rate":              1e-3,
    "learning_rate_scheduler":    KLAdaptiveRL,
    "learning_rate_scheduler_kwargs": {"kl_threshold": 0.016},
    "grad_norm_clip":             1.0,
    "ratio_clip":                 0.2,
    "value_clip":                 0.2,
    "clip_predicted_values":      False,
    "entropy_loss_scale":         0.0,
    "value_loss_scale":           1.0,
    "kl_threshold":               0,
    "rewards_shaper":             lambda rewards, timestep, timesteps: rewards * 0.01,
    # normalizzazione degli input
    "state_preprocessor":         RunningStandardScaler,
    "state_preprocessor_kwargs":  {"size": env.state_space,   "device": env.device},
    "value_preprocessor":         RunningStandardScaler,
    "value_preprocessor_kwargs":  {"size": 1, "device": env.device},
    # rollout iniziali senza esplorazione randomizzata
    "random_timesteps":           0,
    "learning_starts":            0
})

log_dir = "./runs"
os.makedirs(log_dir, exist_ok=True)

cfg_ppo["experiment"] = {
    "write_interval": 10,
    "checkpoint_interval": 100,
    "directory": log_dir,
    "wandb": False  # Imposta a True se vuoi usare Weights & Biases
}

cfg_trainer = {
    "timesteps": 100000,
    "headless":  headless
}

# 3) memoria e modelli
memory = RandomMemory(memory_size=cfg_ppo["rollouts"],
                      num_envs=env.num_envs,
                      device=env.device)

policy = Policy(env.obs_space, env.act_space, env.device)
value  = Value(env.state_space, env.act_space, env.device)

models = {
    "policy": policy,
    "value":  value
}

# 4) istanzia agente e trainer
agent   = PPO(models=models,
              memory=memory,
              cfg=cfg_ppo,
              observation_space=env.state_space,
              action_space=env.act_space,
              device=env.device)

trainer = SequentialTrainer(cfg=cfg_trainer,
                            env=env,
                            agents=agent)

print("###################### DONE INIT ######################")
trainer.train()

env.destroy()
