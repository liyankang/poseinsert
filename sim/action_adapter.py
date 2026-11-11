import numpy as np


def eef_goal_from_source(source_goal, source_now, eef_now):
    """保持当前相对抓取位姿，求把 source 送到 source_goal 时的末端目标位姿。"""
    return source_goal @ np.linalg.inv(source_now) @ eef_now


def action_from_source_goal(source_goal, source_now, eef_now, max_pos_step=0.05, gripper=1.0):
    """把 source 目标位姿转成 OSC_POSE 的 7 维动作（位置伺服，朝向保持不变）。"""
    eef_goal = eef_goal_from_source(source_goal, source_now, eef_now)
    action = np.zeros(7)
    action[:3] = np.clip((eef_goal[:3, 3] - eef_now[:3, 3]) / max_pos_step, -1.0, 1.0)
    action[6] = gripper
    return action


def servo_source(env, source_goal, n_max=200, pos_tol=0.003, speed=0.02, gripper=1.0):
    """把 source 位置伺服到 source_goal，返回实际步数。"""
    for i in range(n_max):
        eef = env.eef_pose()
        eef_goal = eef_goal_from_source(source_goal, env.source_pose(), eef)
        if np.linalg.norm(eef_goal[:3, 3] - eef[:3, 3]) < pos_tol:
            return i + 1
        env.step(action_from_source_goal(source_goal, env.source_pose(), eef, speed, gripper))
    return n_max
