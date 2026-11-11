"""采集 NutAssembly 仿真示教数据，格式与真机一致：train/{idx}/{source_pose,target_pose}.npy"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as RR

from sim.action_adapter import eef_goal_from_source
from sim.env_wrapper import NutAssemblySim


def pose_to_7d(T_):
    return np.concatenate([T_[:3, 3], RR.from_matrix(T_[:3, :3]).as_quat()])  # xyzw


class Recorder:
    def __init__(self, env):
        self.env = env
        self.source = []
        self.target = []

    def add(self):
        self.source.append(pose_to_7d(self.env.source_pose()))
        self.target.append(pose_to_7d(self.env.target_pose()))

    def servo(self, source_goal, n_max=200, pos_tol=0.003, speed=0.02):
        """伺服并逐控制步记录，返回步数。"""
        for i in range(n_max):
            eef = self.env.eef_pose()
            eef_goal = eef_goal_from_source(source_goal, self.env.source_pose(), eef)
            dp = eef_goal[:3, 3] - eef[:3, 3]
            if np.linalg.norm(dp) < pos_tol:
                return i
            action = np.zeros(7)
            action[:3] = np.clip(dp / speed, -1.0, 1.0)
            action[6] = 1.0
            self.env.step(action)
            self.add()
        return n_max

    def hold(self, steps, up=False, gripper=-1.0):
        action = np.zeros(7)
        action[2] = 1.0 if up else 0.0
        action[6] = gripper
        for _ in range(steps):
            self.env.step(action)
            self.add()


def collect_one(env, rng, release_z=0.965):
    env.reset()
    rec = Recorder(env)
    peg = env.target_pose()[:3, 3]
    R_goal = env.source_pose()[:3, :3].copy()

    def goal(x, y, z):
        T_ = np.eye(4)
        T_[:3, :3] = R_goal
        T_[:3, 3] = (x, y, z)
        return T_

    # 随机初始横向偏差，制造多样的 source-in-target 起点
    dx, dy = rng.uniform(-0.03, 0.03, size=2)
    rec.add()
    rec.servo(goal(peg[0] + dx, peg[1] + dy, peg[2] + 0.25), speed=0.03)
    rec.servo(goal(peg[0], peg[1], rng.uniform(1.03, 1.08)))
    for z in np.linspace(1.03, release_z, 12)[1:]:
        rec.servo(goal(peg[0], peg[1], z), n_max=30, pos_tol=0.002)
    env.release()
    rec.hold(20, gripper=-1.0)
    rec.hold(8, up=True, gripper=-1.0)
    return np.array(rec.source), np.array(rec.target), env.success()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demos", type=int, default=100)
    parser.add_argument("--out", type=str, default="./data/sim")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = NutAssemblySim()
    ok = 0
    for i in range(args.start_idx, args.start_idx + args.num_demos):
        src, tgt, success = collect_one(env, rng)
        save_dir = os.path.join(args.out, "train", str(i))
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "source_pose.npy"), src)
        np.save(os.path.join(save_dir, "target_pose.npy"), tgt)
        ok += int(success)
        print(f"demo {i}: {len(src)} steps, success={success} ({ok}/{i - args.start_idx + 1})")
    env.close()


if __name__ == "__main__":
    main()
