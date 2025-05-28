# controller.py

from satellite.utils.satellite_util import quat_diff

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

class SatelliteAttitudeController:
    def __init__(self, num_envs, device, dt, inertia_vals, pid_rate, pid_attitude):
        self.device = device
        self.num_envs = num_envs
        self.dt = dt
        self.inertia = torch.tensor(inertia_vals, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.pid_rate = pid_rate
        self.pid_attitude = pid_attitude

    def compute_control(self, ang_acc_des, ang_acc, ang_vel, quat, goal_quat, goal_ang_vel, goal_ang_acc):
        quat_error = quat_diff(quat, goal_quat)

        angvel_sp = self.pid_attitude.update(
            error=quat_error,
            setpoint=torch.zeros_like(quat_error),
            feedback=quat_error
        )

        angvel_error = angvel_sp - ang_vel
        
        torque_fb = self.pid_rate.update(
            error=angvel_error,
            setpoint=angvel_sp,
            feedback=ang_vel
        )

        torque = torque_fb + (self.inertia * ang_acc_des)

        return torque

    def reset(self, env_ids):
        self.pid_rate.reset(env_ids)
        self.pid_attitude.reset(env_ids)
