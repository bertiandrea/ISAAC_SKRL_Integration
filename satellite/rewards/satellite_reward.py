# satellite_reward.py

from abc import ABC, abstractmethod
import torch
import math

class RewardFunction(ABC):
    @abstractmethod
    def compute(self,
                quats, ang_vels, ang_accs,
                goal_quat, goal_ang_vel, goal_ang_acc,
                actions):
        """
        Compute reward given state and action tensors.
        Returns tensor of shape (num_envs,).
        """
        pass

class TestReward(RewardFunction):
    def __init__(self , alpha_q=10.0, alpha_omega=0.5, alpha_acc=0.2, k_dyn=0.5):
        self.alpha_q = alpha_q
        self.alpha_omega = alpha_omega
        self.alpha_acc = alpha_acc
        self.k_dyn       = k_dyn

    def compute(self,
                quats, ang_vels, ang_accs,
                goal_quat, goal_ang_vel, goal_ang_acc,
                actions):
        angle_diff = 2 * torch.acos(torch.sum(quats * goal_quat, dim=1).abs().clamp(0.0, 1.0))
        ang_vel_diff = torch.norm(ang_vels - goal_ang_vel, dim=1)
        ang_acc_diff = torch.norm(ang_accs - goal_ang_acc, dim=1)

        print(f"[compute_reward]: angle_diff[0]={math.degrees(angle_diff[0].item()):.2f}° ang_vel_diff[0]={math.degrees(ang_vel_diff[0].item()):.2f}°/s ang_acc_diff[0]={math.degrees(ang_acc_diff[0].item()):.2f}°/s²")

        # Angular accelerration and velocity only matter when close to the target
        dynamic_weight = torch.exp(-self.k_dyn * angle_diff)

        # Normalize the differences
        angle_diff = angle_diff / math.pi
        ang_vel_diff = ang_vel_diff / (2 * math.pi)
        ang_acc_diff = ang_acc_diff / (10 * (2 * math.pi))
       
        reward = - (self.alpha_q * angle_diff +  dynamic_weight * (self.alpha_omega * ang_vel_diff + self.alpha_acc * ang_acc_diff))
        
        print(f"[compute_reward]: reward[0]={(reward[0].item()):.2f}")
        
        return reward

class WeightedSumReward(RewardFunction):
    """
    r = - (α_q·φ + α_ω·||ω_err|| + α_acc·||acc_err||)
        + bonuses/penalties:
      + bonus_q when φ ≤ q_threshE
      + bonus_stable when φ ≤ q_threshE and ||ω_err|| ≤ ω_threshE
      + penalty_lvl1 when φ ≥ q_threshL or ||ω_err|| ≥ ω_threshL
      + penalty_lvl2 when φ ≥ 2·q_threshL or ||ω_err|| ≥ 2·ω_threshL
      + penalty_saturation if any action hits saturation
    """
    def __init__(self,
                 alpha_q=10.0, alpha_omega=3.0, alpha_acc=1.0,
                 q_threshE=1e-2, omega_threshE=1e-2,
                 q_threshL=1e-2, omega_threshL=1e-2,
                 bonus_q=200.0, bonus_stable=1000.0,
                 penalty_lvl1=-10.0, penalty_lvl2=-50.0,
                 action_saturation_thresh=None,
                 penalty_saturation=-10.0):
        self.alpha_q = alpha_q
        self.alpha_omega = alpha_omega
        self.alpha_acc = alpha_acc
        self.q_threshE = q_threshE
        self.omega_threshE = omega_threshE
        self.q_threshL = q_threshL
        self.omega_threshL = omega_threshL
        self.bonus_q = bonus_q
        self.bonus_stable = bonus_stable
        self.penalty_lvl1 = penalty_lvl1
        self.penalty_lvl2 = penalty_lvl2
        self.action_saturation_thresh = action_saturation_thresh
        self.penalty_saturation = penalty_saturation

    def compute(self,
                quats, ang_vels, ang_accs,
                goal_quat, goal_ang_vel, goal_ang_acc,
                actions):
        q_err = 2 * torch.acos(torch.sum(quats * goal_quat, dim=1).clamp(0.0, 1.0).abs())
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        acc_err = torch.norm(ang_accs - goal_ang_acc, dim=1)

        print(f"[compute_reward]: angle_diff[0]={math.degrees(q_err[0].item()):.2f}° ang_vel_diff[0]={math.degrees(omega_err[0].item()):.2f}°/s ang_acc_diff[0]={math.degrees(acc_err[0].item()):.2f}°/s²")
        
        base = - (self.alpha_q * q_err
                + self.alpha_omega * omega_err
                + self.alpha_acc * acc_err)
        bonus = torch.zeros_like(base)
        bonus = torch.where(q_err <= self.q_threshE, bonus + self.bonus_q, bonus)
        bonus = torch.where((q_err <= self.q_threshE) & (omega_err <= self.omega_threshE),
                            bonus + self.bonus_stable, bonus)
        bonus = torch.where((q_err >= self.q_threshL) | (omega_err >= self.omega_threshL),
                            bonus + self.penalty_lvl1, bonus)
        bonus = torch.where((q_err >= 2 * self.q_threshL) | (omega_err >= 2 * self.omega_threshL),
                            bonus + self.penalty_lvl2, bonus)
        if self.action_saturation_thresh is not None:
            sat = torch.any(actions.abs() >= self.action_saturation_thresh, dim=1)
            bonus = torch.where(sat, bonus + self.penalty_saturation, bonus)
        return base + bonus

class TwoPhaseReward(RewardFunction):
    """
    Phase 1: +r1_pos or r1_neg based on improvement until cos_val >= threshold.
    Phase 2: α·exp(-φ/β) afterwards.
    """
    def __init__(self,
                 threshold=0.999962,
                 r1_pos=0.1, r1_neg=-0.1,
                 alpha=10.0, beta=0.5):
        self.threshold = threshold
        self.r1_pos = r1_pos
        self.r1_neg = r1_neg
        self.alpha = alpha
        self.beta = beta
        self.prev_cos_val = None

    def compute(self,
                quats, ang_vels, ang_accs,
                goal_quat, goal_ang_vel, goal_ang_acc,
                actions):
        cos_val = torch.sum(quats * goal_quat, dim=1).clamp(0.0, 1.0).abs()
        phi = 2 * torch.acos(cos_val)
        if self.prev_cos_val is not None:
            r1 = torch.where(cos_val > self.prev_cos_val, self.r1_pos, self.r1_neg)
        else:
            r1 = torch.zeros_like(phi)
        r2 = self.alpha * torch.exp(-phi / self.beta)
        r = torch.where(cos_val < self.threshold, r1, r2)
        self.prev_cos_val = cos_val.clone()
        return r

class ExponentialStabilizationReward(RewardFunction):
    """
    r = exp(-φ/scale) - 1 when φ increasing, exp(-φ/scale) when φ improving; bonus if φ <= goal_rad.
    """
    def __init__(self, scale=0.14 * 2 * math.pi, bonus=9.0, goal_deg=0.25):
        self.scale = scale
        self.bonus = bonus
        self.goal_rad = math.radians(goal_deg)
        self.prev_cos_val = None

    def compute(self,
                quats, ang_vels, ang_accs,
                goal_quat, goal_ang_vel, goal_ang_acc,
                actions):
        cos_val = torch.sum(quats * goal_quat, dim=1).clamp(0.0, 1.0).abs()
        phi = 2 * torch.acos(cos_val)
        if self.prev_cos_val is not None:
            r = torch.where(cos_val > self.prev_cos_val,
                            torch.exp(-phi / self.scale),
                            torch.exp(-phi / self.scale) - 1)
        else:
            r = torch.zeros_like(phi)
        bonus = torch.where(phi <= self.goal_rad, self.bonus, torch.zeros_like(phi))
        self.prev_cos_val = cos_val.clone()
        return r + bonus

class ContinuousDiscreteEffortReward(RewardFunction):
    """
    r = r1 + r2 + r3;
    r1 = -(φ + ||ω_err|| + λ||u||);
    r2 = bonus if sup_norm ≤ error_thresh;
    r3 = penalty if sup_norm ≥ fail_thresh.
    """
    def __init__(self,
                 error_thresh=1e-2,
                 bonus=5.0,
                 effort_penalty=0.1,
                 fail_thresh=4.0,
                 fail_penalty=-100.0):
        self.error_thresh = error_thresh
        self.bonus = bonus
        self.effort_penalty = effort_penalty
        self.fail_thresh = fail_thresh
        self.fail_penalty = fail_penalty

    def compute(self,
                quats, ang_vels, ang_accs,
                goal_quat, goal_ang_vel, goal_ang_acc,
                actions):
        phi = 2 * torch.acos(torch.sum(quats * goal_quat, dim=1).clamp(0.0, 1.0).abs())
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        u_norm = torch.sum(actions.pow(2), dim=1)
        sup_norm = torch.max(phi, omega_err)
        r1 = -(phi + omega_err + self.effort_penalty * u_norm)
        r2 = torch.where(sup_norm <= self.error_thresh, self.bonus, torch.zeros_like(phi))
        r3 = torch.where(sup_norm >= self.fail_thresh, self.fail_penalty, torch.zeros_like(phi))
        return r1 + r2 + r3

class ShapingReward(RewardFunction):
    """
    Variants R1-R4: β and τ combinations for shaped reward.
    mode: 'R1','R2','R3','R4'.
    """
    def __init__(self, mode='R4'):
        assert mode in ['R1', 'R2', 'R3', 'R4'], "Unsupported mode"
        self.mode = mode
        self.prev_phi = None

    @staticmethod
    def beta1(delta_phi):
        return torch.where(delta_phi > 0, 0.5, 1.0)

    @staticmethod
    def beta2(delta_phi):
        return torch.exp(-0.5 * (math.pi + delta_phi))

    @staticmethod
    def tau1(phi):
        return torch.exp(2.0 - torch.abs(phi))

    @staticmethod
    def tau2(phi):
        return 14.0 / (1.0 + torch.exp(2.0 * torch.abs(phi)))

    def compute(self,
                quats, ang_vels, ang_accs,
                goal_quat, goal_ang_vel, goal_ang_acc,
                actions):
        phi = 2 * torch.acos(torch.sum(quats * goal_quat, dim=1).clamp(0.0, 1.0).abs())
        if self.prev_phi is not None:
            delta_phi = phi - self.prev_phi
        else:
            delta_phi = torch.zeros_like(phi)
        if self.mode in ['R1', 'R2']:
            b = ShapingReward.beta1(delta_phi)
        else:
            b = ShapingReward.beta2(delta_phi)
        if self.mode in ['R1', 'R3']:
            t = ShapingReward.tau1(phi)
        else:
            t = ShapingReward.tau2(phi)
        self.prev_phi = phi.clone()
        return b * t
