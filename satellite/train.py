# train.py

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

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import argparse

REWARD_MAP = {
    "test": TestReward,
    "weighted_sum": WeightedSumReward,
    "two_phase": TwoPhaseReward,
    "exp_stabilization": ExponentialStabilizationReward,
    "continuous_discrete_effort": ContinuousDiscreteEffortReward,
    "shaping": ShapingReward,
}

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training con reward function selezionabile")
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    return parser.parse_args()

def main():
    # 0) parsing degli argomenti
    args = parse_args()

    # 1) setup environment
    env_cfg = SatelliteConfig()

    if env_cfg.set_seed:
        set_seed(env_cfg.seed)

    env = SatelliteVec(cfg=env_cfg,
                    headless=args.headless,
                    reward_fn=REWARD_MAP[args.reward_fn](),
    )

    # 2) PPO and Trainer config
    env_cfg_dict = class_to_dict(env_cfg)
    print(env_cfg_dict)
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()

    env_cfg_dict["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space,   "device": env.device
    }
    env_cfg_dict["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }
    env_cfg_dict["rl"]["PPO"]["learning_rate_scheduler"] = KLAdaptiveRL
    env_cfg_dict["rl"]["PPO"]["learning_rate_scheduler_kwargs"] = { "kl_threshold": 0.016 }
    env_cfg_dict["rl"]["PPO"]["state_preprocessor"] = RunningStandardScaler
    env_cfg_dict["rl"]["PPO"]["value_preprocessor"] = RunningStandardScaler
    env_cfg_dict["rl"]["PPO"]["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01

    cfg_ppo.update(env_cfg_dict["rl"]["PPO"])

    cfg_trainer = env_cfg_dict["rl"]["trainer"]

    # 3) memoria
    memory = RandomMemory(memory_size=env_cfg_dict["rl"]["memory"]["rollouts"],
                        num_envs=env.num_envs,
                        device=env.device)

    # 4) modelli
    policy = Policy(env.obs_space, env.act_space, env.device)
    value  = Value(env.state_space, env.act_space, env.device)
    models = { "policy": policy, "value":  value}

    # 5) istanzia agente e trainer
    agent = PPO(models=models,
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

if __name__ == "__main__":
    main()
