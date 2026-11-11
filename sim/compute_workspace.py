"""用仿真示教数据统计 source-in-target 的平移范围，供训练归一化使用。"""

import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as RR


def poses_to_matrix(poses):
    mats = np.tile(np.eye(4), (len(poses), 1, 1))
    mats[:, :3, :3] = RR.from_quat(poses[:, 3:7]).as_matrix()
    mats[:, :3, 3] = poses[:, :3]
    return mats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/sim/train")
    parser.add_argument("--out", type=str, default="./data/sim/source_workspace.npy")
    args = parser.parse_args()

    rows = []
    for demo in sorted(glob.glob(os.path.join(args.data, "*"))):
        source = poses_to_matrix(np.load(os.path.join(demo, "source_pose.npy")))
        target = poses_to_matrix(np.load(os.path.join(demo, "target_pose.npy")))
        rel = np.einsum("nij,njk->nik", np.linalg.inv(target), source)
        pos = rel[:, :3, 3]
        rows.append([*pos.max(0), *pos.min(0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    workspace = np.array(rows)
    np.save(args.out, workspace)
    print("saved", args.out, workspace.shape)
    print("translation max", np.round(workspace[:, :3].max(0), 4),
          "min", np.round(workspace[:, 3:6].min(0), 4))


if __name__ == "__main__":
    main()
