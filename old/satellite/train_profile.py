# train.py

from satellite.configs.satellite_config import SatelliteConfig
from satellite.envs.satellite_vec import SatelliteVec
from satellite.models.custom_model import Policy, Value
from satellite.rewards.satellite_reward import (
    TestReward,
    TestRewardSmooth,
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
import pandas as pd

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
    "test_smooth": TestRewardSmooth,
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
    log_dir = "/home/andreaberti/profiler_logs/ISAAC_SKRL_Integration_old/satellite"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    # ──────────────────────────────────────────────────────────────────────────

    prof.start()
    trainer.train()
    prof.stop()

    output_path = "/home/andreaberti/profiler_text/ISAAC_SKRL_Integration_old/satellite/text_output.txt"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    events = prof.key_averages()

    with open(output_path, "w") as f:
        f.write(events.table(sort_by="self_cuda_time_total", row_limit=500))

        f.write("\n\n\n")

        f.write(events.table(sort_by="self_cpu_time_total", row_limit=500))

        f.write("\n\n\n")

        f.write(events.table(sort_by="self_cuda_memory_usage", row_limit=500))

        f.write("\n\n\n")

        f.write(events.table(sort_by="self_cpu_memory_usage", row_limit=500))

    rows = []
    for e in events:
        rows.append({
            "name":               e.key[:50],  # Truncate to 50 characters
            "self_cpu_time_ms":   e.self_cpu_time_total / 1e3,
            "cpu_time_ms":        e.cpu_time_total / 1e3,

            "self_cuda_time_ms":  e.self_device_time_total / 1e3,
            "cuda_time_ms":       e.device_time_total / 1e3,

            "self_cpu_memory_bytes":   e.self_cpu_memory_usage,
            "self_cuda_memory_bytes":  e.self_device_memory_usage,

            "cpu_memory_bytes":   e.cpu_memory_usage,
            "cuda_memory_bytes":  e.device_memory_usage,

            "count":              e.count,
            "flops":              e.flops,

            "device_type":        str(e.device_type),
        })
    df = pd.DataFrame(rows)
    
    df['order'] = df['name'].str[0].map({'#': 0, '$': 1}).fillna(2).astype(int)
    df = df.sort_values(['order', 'name'], ascending=[True, True])
    df = df.drop(columns='order')

    print(df.head(40))

    csv_path = "/home/andreaberti/profiler_text/ISAAC_SKRL_Integration_old/satellite/csv_output.csv"
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()