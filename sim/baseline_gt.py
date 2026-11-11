"""用真值位姿直接伺服插入，验证 sim 的动作链路与坐标系是否正确。"""

import numpy as np

from sim.action_adapter import eef_goal_from_source
from sim.env_wrapper import NutAssemblySim


def servo(env, source_goal, n_max=200, pos_tol=0.003, speed=0.02):
    """把 source 位置伺服到 source_goal，返回实际步数。"""
    for i in range(n_max):
        eef = env.eef_pose()
        eef_goal = eef_goal_from_source(source_goal, env.source_pose(), eef)
        dp = eef_goal[:3, 3] - eef[:3, 3]
        if np.linalg.norm(dp) < pos_tol:
            return i + 1
        action = np.zeros(7)
        action[:3] = np.clip(dp / speed, -1.0, 1.0)
        action[6] = 1.0
        env.step(action)
    return n_max


def main():
    env = NutAssemblySim()
    env.reset()
    peg = env.target_pose()[:3, 3]
    R_goal = env.source_pose()[:3, :3].copy()

    def goal(x, y, z):
        T_ = np.eye(4)
        T_[:3, :3] = R_goal
        T_[:3, 3] = (x, y, z)
        return T_

    print("目标柱位置", np.round(peg, 4))
    n = servo(env, goal(peg[0], peg[1], 1.02))
    print("移动到柱上方:", n, "步, source", np.round(env.source_pose()[:3, 3], 4))

    for z in np.linspace(1.01, 0.965, 10):
        servo(env, goal(peg[0], peg[1], z), n_max=30, pos_tol=0.002)
    print("下压结束, source", np.round(env.source_pose()[:3, 3], 4))

    env.release()
    release = np.zeros(7)
    release[6] = -1.0
    for _ in range(60):
        env.step(release)
    print("松开后 success:", env.success(), "on_pegs:", env.objects_on_pegs())

    retreat = np.zeros(7)
    retreat[2] = 1.0
    retreat[6] = -1.0
    for _ in range(10):
        env.step(retreat)
    print("后退后 success:", env.success(), "on_pegs:", env.objects_on_pegs(),
          "螺母位置:", np.round(env.source_pose()[:3, 3], 4))
    env.close()


if __name__ == "__main__":
    main()
