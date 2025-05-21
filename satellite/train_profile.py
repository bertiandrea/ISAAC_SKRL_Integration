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

import isaacgym  # BugFix
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
import os

# ──────────────────────────────────────────────────────────────────────────────
# Profiler imports
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
)
# ──────────────────────────────────────────────────────────────────────────────

REWARD_MAP = {
    "test": TestReward,
    "weighted_sum": WeightedSumReward,
    "two_phase": TwoPhaseReward,
    "exp_stabilization": ExponentialStabilizationReward,
    "continuous_discrete_effort": ContinuousDiscreteEffortReward,
    "shaping": ShapingReward,
}

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if isinstance(val, list):
            result[key] = [class_to_dict(item) for item in val]
        else:
            result[key] = class_to_dict(val)
    return result

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training con reward function selezionabile")
    parser.add_argument(
        "--reward-fn",
        choices=list(REWARD_MAP.keys()),
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
    env = SatelliteVec(cfg=env_cfg, reward_fn=REWARD_MAP[args.reward_fn]())

    # 2) PPO and Trainer config
    env_cfg_dict = class_to_dict(env_cfg)
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    env_cfg_dict["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    env_cfg_dict["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }
    env_cfg_dict["rl"]["PPO"]["learning_rate_scheduler"] = KLAdaptiveRL
    env_cfg_dict["rl"]["PPO"]["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
    env_cfg_dict["rl"]["PPO"]["state_preprocessor"] = RunningStandardScaler
    env_cfg_dict["rl"]["PPO"]["value_preprocessor"] = RunningStandardScaler
    env_cfg_dict["rl"]["PPO"]["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
    cfg_ppo.update(env_cfg_dict["rl"]["PPO"])

    # 3) memoria
    memory = RandomMemory(
        memory_size=env_cfg_dict["rl"]["memory"]["rollouts"],
        num_envs=env.num_envs,
        device=env.device
    )

    # 4) modelli
    policy = Policy(env.obs_space, env.act_space, env.device)
    value  = Value(env.state_space, env.act_space, env.device)
    models = { "policy": policy, "value": value }

    # 5) istanzia agente e trainer
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg_ppo,
        observation_space=env.state_space,
        action_space=env.act_space,
        device=env.device
    )
    trainer = SequentialTrainer(cfg=env_cfg_dict["rl"]["trainer"],
                                env=env,
                                agents=agent)
    # ──────────────────────────────────────────────────────────────────────────
    # Setup PyTorch profiler
    log_dir = "/home/andreaberti/profiler_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    )
    # ──────────────────────────────────────────────────────────────────────────

    print("###################### DONE INIT ######################")
    prof.start()
    trainer.train()
    prof.stop()
    print(f"Profiler traces written to {log_dir}")

    env.destroy()

if __name__ == "__main__":
    main()
