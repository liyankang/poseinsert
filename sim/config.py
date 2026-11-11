from dataclasses import dataclass


@dataclass
class SimConfig:
    """NutAssembly 仿真配置。"""

    robot: str = "Panda"
    nut_type: str = "round"          # round -> RoundNut，对应 peg2
    control_freq: int = 20
    horizon: int = 100000
    max_pos_step: float = 0.05       # OSC_POSE 单步最大平移 (m)
    max_rot_step: float = 0.5        # OSC_POSE 单步最大旋转 (rad)
    open_steps: int = 60             # reset 后张开夹爪的步数
    close_steps: int = 100           # 夹紧夹爪的步数
    carry_drop: float = 0.06         # 螺母在夹爪下方的刚性跟随偏移 (m)
