import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils import transform_utils as T

from sim.config import SimConfig


class NutAssemblySim:
    """NutAssembly 单螺母插入环境。

    对外统一暴露世界系 4x4 位姿，替代真机里的 /source/pose、/target/pose、
    /right/pose_info 三个 ROS 话题；动作是 OSC_POSE 的 7 维控制量。

    抓取说明：Panda 夹爪无法稳定夹住螺母的圆环/handle，这里用“刚性跟随”
    模拟“已被夹爪抓取”的初始状态（与真机前提一致：策略启动时物体已在夹爪内）：
    每个控制步把螺母的 free joint 位姿设为 T_eef @ T_eef_source，释放后再交给物理。
    """

    def __init__(self, cfg=None, has_renderer=False):
        self.cfg = cfg or SimConfig()
        controller_cfg = load_controller_config(default_controller="OSC_POSE")
        self.env = suite.make(
            "NutAssembly",
            robots=self.cfg.robot,
            controller_configs=controller_cfg,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq=self.cfg.control_freq,
            horizon=self.cfg.horizon,
            ignore_done=True,
            reward_shaping=False,
            single_object_mode=2,
            nut_type=self.cfg.nut_type,
            initialization_noise=None,
            hard_reset=False,
        )
        self.robot = self.env.robots[0]
        self.sim = self.env.sim
        self.nut = self.env.nuts[self.env.nut_id]

        self._nut_body_id = self.env.obj_body_id[self.nut.name]
        self._nut_joint = self.nut.joints[0]
        self._nut_qadr = self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(self._nut_joint)]
        self._nut_dof = self.sim.model.jnt_dofadr[self.sim.model.joint_name2id(self._nut_joint)]
        self._carry_offset = None

    # ---------------- 位姿 ----------------
    def eef_pose(self):
        T_ = np.eye(4)
        T_[:3, 3] = self.sim.data.site_xpos[self.robot.eef_site_id]
        T_[:3, :3] = np.array(self.sim.data.site_xmat[self.robot.eef_site_id]).reshape(3, 3)
        return T_

    def source_pose(self):
        return self._body_pose(self._nut_body_id)

    def target_pose(self):
        body_id = self.env.peg2_body_id if self.cfg.nut_type == "round" else self.env.peg1_body_id
        return self._body_pose(body_id)

    def _body_pose(self, body_id):
        T_ = np.eye(4)
        T_[:3, 3] = self.sim.data.body_xpos[body_id]
        T_[:3, :3] = T.quat2mat(np.array(self.sim.data.body_xquat[body_id]))
        return T_

    # ---------------- 环境交互 ----------------
    def step(self, action):
        obs = self.env.step(np.asarray(action, dtype=float))
        if self._carry_offset is not None:
            self._set_source_pose(self.eef_pose() @ self._carry_offset)
        return obs

    def release(self):
        """松开螺母，之后交给物理接触。"""
        self._carry_offset = None

    def success(self):
        return bool(self.env._check_success())

    def objects_on_pegs(self):
        return np.array(self.env.objects_on_pegs)

    def close(self):
        self.env.close()

    # ---------------- reset ----------------
    def reset(self):
        """重置环境，并把螺母摆成“已被夹爪抓取”的初始状态。"""
        self.env.reset()
        self._grip(self.cfg.open_steps, close=False)

        T_eef = self.eef_pose()
        T_src = np.eye(4)
        T_src[:3, :3] = T.quat2mat(np.array(self.sim.data.body_xquat[self._nut_body_id]))  # reset 时螺母平放
        T_src[:3, 3] = T_eef[:3, 3] + np.array([0.0, 0.0, -self.cfg.carry_drop])
        self._set_source_pose(T_src)
        self._carry_offset = np.linalg.inv(T_eef) @ T_src

        self._grip(self.cfg.close_steps, close=True)
        return {"eef": self.eef_pose(), "source": self.source_pose(), "target": self.target_pose()}

    def _set_source_pose(self, T_):
        self.sim.data.qpos[self._nut_qadr:self._nut_qadr + 7] = np.concatenate(
            [T_[:3, 3], T.mat2quat(T_[:3, :3])]
        )
        self.sim.data.qvel[self._nut_dof:self._nut_dof + 6] = 0
        self.sim.forward()

    def _grip(self, steps, close):
        action = np.zeros(7)
        action[6] = 1.0 if close else -1.0
        for _ in range(steps):
            self.step(action)
