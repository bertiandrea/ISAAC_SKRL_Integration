# train.py

from satellite.configs.satellite_config_opt import SatelliteConfigOptimization
from satellite.envs.satellite import Satellite
from satellite.models.custom_model import Policy, Value
from satellite.envs.wrappers.isaacgym_envs import IsaacGymWrapper
from satellite.rewards.satellite_reward import (
    TestReward,
    TestRewardSmooth,
    WeightedSumReward,
    TwoPhaseReward,
    ExponentialStabilizationReward,
    ContinuousDiscreteEffortReward,
    ShapingReward,
)

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import argparse

import json
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tensorboard.backend.event_processing import event_accumulator

TENSORBOARD_TAG = "Reward / Instantaneous reward (mean)"
N_TRIALS = 25

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

REWARD_MAP = {
    "test": TestReward,
    "test_smooth": TestRewardSmooth,
    "weighted_sum": WeightedSumReward,
    "two_phase": TwoPhaseReward,
    "exp_stabilization": ExponentialStabilizationReward,
    "continuous_discrete_effort": ContinuousDiscreteEffortReward,
    "shaping": ShapingReward,
}

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

def sample_ppo_params(trial: optuna.Trial):
    return {
        "discount_factor": trial.suggest_float("discount_factor", 0.90, 0.999),
        "lambda":          trial.suggest_float("gamma", 0.90,   0.999),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-4, 1e-2),
        "grad_norm_clip": trial.suggest_float("grad_norm_clip", 0.1, 1.0),
        "ratio_clip":   trial.suggest_float("ratio_clip", 0.1, 0.3),
        "value_clip": trial.suggest_float("value_clip", 0.1, 0.3),
        "clip_predicted_values": trial.suggest_categorical("clip_predicted_values", [True, False]),
        "entropy_loss_scale": trial.suggest_float("entropy_loss_scale", 0.0, 0.05),
        "value_loss_scale": trial.suggest_float("value_loss_scale", 0.5, 2.0),
        "kl_threshold": trial.suggest_float("kl_threshold", 0.0, 0.1),
    }

def objective(trial: optuna.Trial) -> float:
    cfg_class = SatelliteConfigOptimization()
    cfg = class_to_dict(cfg_class)

    if cfg["set_seed"]:
        set_seed(cfg["seed"])
    
    env = Satellite(
        cfg=cfg,
        rl_device=cfg["rl_device"],
        sim_device=cfg["sim_device"],
        graphics_device_id=cfg["graphics_device_id"],
        headless=cfg["headless"],
        virtual_screen_capture=cfg["virtual_screen_capture"],
        force_render= cfg["force_render"],
        reward_fn=REWARD_MAP[args.reward_fn](cfg["log_reward"]["log_reward"], cfg["log_reward"]["log_reward_interval"])
    )
    
    env = IsaacGymWrapper(env)

    memory = RandomMemory(memory_size=cfg["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, env.device)
    models["value"] = Value(env.state_space, env.action_space, env.device)
   
    cfg["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    cfg["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }
    cfg["rl"]["PPO"]["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["rl"]["PPO"]["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
    cfg["rl"]["PPO"]["state_preprocessor"] = RunningStandardScaler
    cfg["rl"]["PPO"]["value_preprocessor"] = RunningStandardScaler
    cfg["rl"]["PPO"]["rewards_shaper"] = lambda rewards, t, T: rewards * 0.5
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    
    cfg_ppo.update(cfg["rl"]["PPO"])

    hp = sample_ppo_params(trial)
    cfg_ppo.update({
        "discount_factor":       hp["discount_factor"],
        "lambda":                hp["lambda"],
        "learning_rate":         hp["learning_rate"],
        "grad_norm_clip":        hp["grad_norm_clip"],
        "ratio_clip":            hp["ratio_clip"],
        "value_clip":            hp["value_clip"],
        "clip_predicted_values": hp["clip_predicted_values"],
        "entropy_loss_scale":    hp["entropy_loss_scale"],
        "value_loss_scale":      hp["value_loss_scale"],
        "kl_threshold":          hp["kl_threshold"],
    })

    agent = PPO(models=models,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.state_space,
                action_space=env.action_space,
                device=env.device)

    trainer = SequentialTrainer(cfg=cfg["rl"]["trainer"], env=env, agents=agent)

    trainer.train()
    
    env.close() # Force environment close to avoid memory leaks

    #############################################################################
    log_dir = cfg["rl"]["PPO"]["experiment"]["directory"] + "/" + cfg["rl"]["PPO"]["experiment"]["experiment_name"]

    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 10000})
    ea.Reload()

    print(f"Available Tags: {ea.Tags()['scalars']}")

    values = ea.Scalars(TENSORBOARD_TAG)
    if not values:
        raise RuntimeError(f"Tag '{TENSORBOARD_TAG}' non trovato in {log_dir}")
    mean_return = values[-1].value

    trial.report(mean_return, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mean_return

def main():
    global args
    args = parse_args()

    study = optuna.create_study(
        sampler=TPESampler(n_startup_trials=10, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        direction="maximize",
    )
    study.optimize(objective, n_trials=N_TRIALS)

    out_path = f"best_hyperparams.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"\n✅ Salvato in {out_path}")
    print(f"➤ mean_return migliore: {study.best_value:.3f}")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    
if __name__ == "__main__":
    main()