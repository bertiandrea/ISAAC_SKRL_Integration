# train.py

import os
import argparse
from satellite.configs.satellite_config import SatelliteConfig
from satellite.envs.satellite_vec import SatelliteVec
from satellite.models.custom_model import Policy, Value
from satellite.rewards.satellite_reward import (
    TestReward,
    WeightedSumReward,
    TwoPhaseReward,
    ExponentialStabilizationReward,
    ContinuousDiscreteEffortReward,
    ShapingReward,
)

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Training con reward function selezionabile")
    parser.add_argument(
        "--reward-fn",
        choices=[
            "test",
            "weighted_sum",
            "two_phase",
            "exp_stabilization",
            "continuous_discrete_effort",
            "shaping"
        ],
        default="test",
        help="Which RewardFunction?"
    )
    return parser.parse_args()

# parsing degli argomenti
args = parse_args()

reward_map = {
    "test": TestReward,
    "weighted_sum": WeightedSumReward,
    "two_phase": TwoPhaseReward,
    "exp_stabilization": ExponentialStabilizationReward,
    "continuous_discrete_effort": ContinuousDiscreteEffortReward,
    "shaping": ShapingReward,
}
reward_fn = reward_map[args.reward_fn]()

# fissiamo il seed per la riproducibilità
set_seed(42)

# 1) setup environment
env_cfg      = SatelliteConfig()
headless     = False
env          = SatelliteVec(cfg=env_cfg,
                            rl_device="cuda:0",
                            sim_device="cuda:0",
                            graphics_device_id=0,
                            headless=headless,
                            reward_fn=reward_fn
)

print(f"Envs: {env.num_envs}, \
        state_space: {env.state_space}, \
        obs_space: {env.obs_space}, \
        act_space: {env.act_space}, \
        device: {env.device}")

num_actors     = env.num_envs           # numero di ambienti paralleli
horizon_length = 2048 // num_actors     # lunghezza della sequenza di rollouts
mini_epochs    = 4                      # numero di passate SGD per aggiornamento
minibatch_size = 256                    # dimensione del minibatch su cui calcolare un gradiente
max_epochs     = 2000                   # numero di cicli di aggiornamento (ogni ciclo usa horizon_length passi)

print(f"num_actors: {num_actors}, \
        horizon_length: {horizon_length}, \
        mini_epochs: {mini_epochs}, \
        minibatch_size: {minibatch_size}, \
        max_epochs: {max_epochs} \
        rollouts: {horizon_length}, \
        learning_epochs: {mini_epochs}, \
        mini_batches: {horizon_length * num_actors / minibatch_size}, \
        timesteps: {horizon_length * max_epochs}")

# 2) PPO config
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo.update({
    # environment
    "rollouts":                   horizon_length,
    "learning_epochs":            mini_epochs,
    "mini_batches":               horizon_length * num_actors / minibatch_size,
    # agent
    "discount_factor":            0.99,
    "lambda":                     0.95,
    "learning_rate":              3e-4,
    "learning_rate_scheduler":    KLAdaptiveRL,
    "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
    "grad_norm_clip":             1.0,
    "ratio_clip":                 0.2,
    "value_clip":                 0.2,
    "clip_predicted_values":      True,
    "entropy_loss_scale":         0.00,
    "value_loss_scale":           2.0,
    "kl_threshold":               0,
    "rewards_shaper":             lambda rewards, timestep, timesteps: rewards * 0.1,
    # preprocessing
    "state_preprocessor":         RunningStandardScaler,
    "state_preprocessor_kwargs":  {"size": env.state_space,   "device": env.device},
    "value_preprocessor":         RunningStandardScaler,
    "value_preprocessor_kwargs":  {"size": 1, "device": env.device},
    # training
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
    "timesteps": horizon_length * max_epochs,
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
