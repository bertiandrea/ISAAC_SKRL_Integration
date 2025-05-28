import torch
from satellite.utils.satellite_util import quat_diff

class SatelliteAttitudeController:
    def __init__(self, pid_att, pid_rate):
        self.pid_att = pid_att
        self.pid_rate = pid_rate

    def compute_control(self, current_quat, target_quat, current_angvel):
        quat_error = quat_diff(current_quat, target_quat)
        
        angvel_sp = self.pid_att.update(
            error=quat_error,
            setpoint=torch.zeros_like(quat_error),
            feedback=quat_error
        )

        angvel_error = angvel_sp - current_angvel
        torque_cmd = self.pid_rate.update(
            error=angvel_error,
            setpoint=angvel_sp,
            feedback=current_angvel
        )

        return torque_cmd

    def reset(self, env_ids):
        self.pid_att.reset(env_ids)
        self.pid_rate.reset(env_ids)
