import numpy as np
from dataclasses import dataclass
from enum import Enum, auto   # [CHANGED] 新增：控制模式枚举
from discoverse.robots_env.airbot_play_base import AirbotPlayBase, AirbotPlayCfg
import time

# ------------------------------- 通用参数与模式 -------------------------------

class ControlMode(Enum):        # [CHANGED] 新增：通用控制模式枚举
    CSP = auto()   # Cyclic Synchronous Position
    CSV = auto()   # Cyclic Synchronous Velocity
    MIT = auto()   # MIT/Impedance-like
    PVT = auto()   # Position-Velocity-Time

@dataclass
class CSPParams:
    enabled: bool = True
    slew_rate: float | None = None   # 每步最大位置变化量（rad/step）
    zero_offset: float = 0.0         # set_zero 记录偏置

@dataclass
class CSVParams:                     # [CHANGED] 新增：CSV 占位参数结构
    enabled: bool = True
    v_ref: float = 0.0               # 速度参考（rad/s）
    v_limit: float | None = None     # 速度上限（可选）

@dataclass
class MITParams:                     # [CHANGED] 新增：MIT 占位参数结构
    enabled: bool = True
    # 典型 MIT 参数（Kp/Kd/Tau_ff），先占位
    kp: float = 0.0
    kd: float = 0.0
    tau_ff: float = 0.0
    q_ref: float = 0.0
    dq_ref: float = 0.0

@dataclass
class PVTParams:                     # [CHANGED] 新增：PVT 占位参数结构
    enabled: bool = True
    # 这里通常需要队列/轨迹点；增加 target 与当前插值状态
    target_q: float = 0.0
    current_q: float = 0.0
    dq_ref: float = 0.0

# ------------------------------- MotorShim 通用层 -------------------------------

class MotorShim:
    """
    通用关节适配层：维护 per-joint 的控制模式与对应参数。
    目前仅实现 CSP 的 apply_to_ctrl；其他模式留出接口与 NotImplementedError。
    """
    def __init__(self, i, mj_model, mj_data, dt_ctrl, lo=None, hi=None):
        self.i = i
        self.model = mj_model
        self.data  = mj_data
        self.dt    = float(dt_ctrl)

        # 如果提供了自定义限制，使用自定义限制；否则从模型读取
        if lo is not None and hi is not None:
            self.lo, self.hi = float(lo), float(hi)
        else:
            lo, hi = mj_model.actuator_ctrlrange[i]
            self.lo, self.hi = float(lo), float(hi)

        # [CHANGED] 默认模式 CSP；并行持有各模式参数对象
        self.mode = ControlMode.CSP
        self.csp = CSPParams()
        self.csv = CSVParams()
        self.mit = MITParams()
        self.pvt = PVTParams()

        # 当前实际关节位置
        q0 = float(mj_data.qpos[i])

        # CSP 初始目标设为当前位姿
        self.csp_q_ref = q0
        # CSV 模式下的内部位置参考：由速度积分得到
        # 这样在 v_ref = 0 时，保持上一次的 q_ref，实现“当前位置保持”
        self.csv_q_ref = q0
        # PVT 初始状态：保持当前位姿与零速度
        self.pvt.target_q = q0
        self.pvt.current_q = q0
        self.pvt.dq_ref = 0.0

    # ---------------- 模式切换与通用开关 ----------------

    # [CHANGED] 新增：设置模式
    def set_mode(self, mode: ControlMode):
        self.mode = ControlMode(mode)
        # 当切换到 CSV 模式时，将内部位置参考对齐到当前实际位置，
        # 避免刚切换时产生突变。
        if self.mode is ControlMode.CSV:
            self.csv_q_ref = float(self.data.qpos[self.i])

    def enable(self):   # 按当前模式开关（也可扩展为“全模式开关”）
        self._params_now().enabled = True

    def disable(self):
        self._params_now().enabled = False

    def _params_now(self):
        return {
            ControlMode.CSP: self.csp,
            ControlMode.CSV: self.csv,
            ControlMode.MIT: self.mit,
            ControlMode.PVT: self.pvt,
        }[self.mode]

    # ---------------- 各模式：上层 API（只写“参考/参数”，不碰 mj_data） ----------------

    # CSP
    def csp_cmd(self, q_target: float):
        self.csp_q_ref = float(np.clip(q_target, self.lo, self.hi))

    def set_zero(self):
        # 记录当前关节位置为零偏；实际应用在 apply_to_ctrl()
        self.csp.zero_offset = float(self.data.qpos[self.i])

    def set_csp_param(self, name: str, value):
        if name == "slew_rate":
            self.csp.slew_rate = float(value)  # rad/step
        elif name == "zero_offset":
            self.csp.zero_offset = float(value)
        elif name == "enabled":
            self.csp.enabled = bool(value)

    # CSV（占位）
    def csv_cmd(self, v_target: float):
        self.csv.v_ref = float(v_target)

    def set_csv_param(self, name: str, value):
        if name == "v_limit":
            self.csv.v_limit = None if value is None else float(value)
        elif name == "enabled":
            self.csv.enabled = bool(value)

    # MIT（占位）
    def mit_cmd(self, q_ref: float, dq_ref: float, kp: float, kd: float, tau_ff: float = 0.0):
        self.mit.q_ref = float(q_ref)
        self.mit.dq_ref = float(dq_ref)
        self.mit.kp = float(kp)
        self.mit.kd = float(kd)
        self.mit.tau_ff = float(tau_ff)

    # PVT（占位）
    def pvt_cmd(self, q_ref: float, dq_ref: float):
        q = float(np.clip(q_ref, self.lo, self.hi))
        self.pvt.target_q = q
        self.pvt.dq_ref = float(dq_ref)

    # ---------------- 仿真线程：真正写 ctrl ----------------

    def apply_to_ctrl(self):
        """
        只允许在“仿真线程”调用：将当前模式的参考值应用到 data.ctrl[i]
        目前**仅实现 CSP**。其他模式抛 NotImplementedError（留扩展位）。
        """
        i = self.i

        if self.mode is ControlMode.CSP:
            # ---- CSP 实现：与你原版一致 ----
            q = float(self.data.qpos[i])  # 当前实际位置
            if not self.csp.enabled:
                u = q  # 禁用时保持当前位置（等效软保持）
            else:
                # 目标减去零偏得到有效参考
                qref_eff = self.csp_q_ref - self.csp.zero_offset
                u_des = float(np.clip(qref_eff, self.lo, self.hi))

                # 斜率限制（可选）：控制“到位速度”
                if self.csp.slew_rate is not None:
                    prev = float(self.data.ctrl[i])
                    delta = np.clip(u_des - prev, -self.csp.slew_rate, self.csp.slew_rate)
                    u = prev + delta
                else:
                    u = u_des

            self.data.ctrl[i] = u
            return u

        elif self.mode is ControlMode.CSV:
            # ---- CSV 实现：速度控制 + 位置保持 ----
            q_now = float(self.data.qpos[i])  # 当前实际位置
            if not self.csv.enabled:
                u = q_now  # 禁用时保持当前位置（与 CSP 一致，软保持）
            else:
                v_ref = float(self.csv.v_ref)  # 目标速度 (rad/s)
                
                # 速度限制（如果设置了）
                if self.csv.v_limit is not None:
                    v_ref = np.clip(v_ref, -self.csv.v_limit, self.csv.v_limit)
                # 速度控制：通过积分速度更新“内部位置参考” csv_q_ref
                # 注意这里不再使用 q_now 作为基准，避免 v_ref 回到 0 时
                # 参考值跟随当前姿态漂移，导致无保持力矩。
                self.csv_q_ref += v_ref * self.dt
                # 限制 csv_q_ref 在关节范围内，防止累加超出限制
                self.csv_q_ref = float(np.clip(self.csv_q_ref, self.lo, self.hi))
                u_des = self.csv_q_ref

                # 再次限制（双重保险）
                u = float(np.clip(u_des, self.lo, self.hi))
            
            self.data.ctrl[i] = u
            return u

        elif self.mode is ControlMode.MIT:
            # 典型 MIT 需要关节动力学/力矩接口；此处先留空
            raise NotImplementedError("MIT control not implemented yet")

        elif self.mode is ControlMode.PVT:
            # ---- PVT 实现：位置-速度-时间控制 ----
            q_now = float(self.data.qpos[i])  # 当前实际位置
            if not self.pvt.enabled:
                u = q_now  # 禁用时保持当前位置（与 CSP 一致，软保持）
            else:
                q_target = self.pvt.target_q
                dq = abs(self.pvt.dq_ref)
                if dq > 0.0:
                    max_step = dq * self.dt
                    delta = np.clip(q_target - q_now, -max_step, max_step)
                    u = q_now + delta
                else:
                    u = q_target
                u = float(np.clip(u, self.lo, self.hi))
                self.pvt.current_q = u
            self.data.ctrl[i] = u
            return u

        else:
            raise ValueError(f"unknown mode: {self.mode}")

# ------------------------------- AirbotPlay Shim（默认 CSP） -------------------------------

class AirbotPlayShim(AirbotPlayBase):
    # 关节与夹爪的位置限制（6关节 + 1夹爪）
    MIN_Q = [-3.1416, -2.9671, -0.087266, -3.0107, -1.7628, -3.0107, 0.0]
    MAX_Q = [2.0944, 0.17453, 3.1416, 3.0107, 1.7628, 3.0107, 0.072]

    def __init__(self, config: AirbotPlayCfg):
        self._dt_ctrl = config.timestep * config.decimation
        super().__init__(config)

    def post_load_mjcf(self):
        super().post_load_mjcf()
        assert self.mj_model.nu >= self.nj, "actuator 数量不足以覆盖关节"
        # 使用自定义限制创建 MotorShim（如果关节数超出限制数组长度，则使用模型默认限制）
        self.motors = []
        for i in range(self.nj):
            lo = self.MIN_Q[i] if i < len(self.MIN_Q) else None
            hi = self.MAX_Q[i] if i < len(self.MAX_Q) else None
            self.motors.append(MotorShim(i, self.mj_model, self.mj_data, self._dt_ctrl, lo=lo, hi=hi))
        for m in self.motors:
            m.set_mode(ControlMode.CSP)

    # ---------------- 1-based 取电机（替代原 _id1to0） ----------------
    def get_motor(self, motor_id_1b: int) -> MotorShim:
        """
        对外使用 1-based ID（1..nj），内部自动做 1->0 的安全映射并返回 MotorShim 实例。
        """
        j = int(motor_id_1b)
        if j < 1 or j > self.nj:
            raise IndexError(f"motor_id out of range: {motor_id_1b} (valid 1..{self.nj})")
        return self.motors[j - 1]

    # ---------------- 对外 API：参数一律视为 1-based ----------------
    def set_mode_joint(self, motor_id: int, mode: ControlMode):
        self.get_motor(motor_id).set_mode(mode)

    def set_mode_all(self, mode: ControlMode):
        for m in self.motors:
            m.set_mode(mode)

    def enable_joint(self, motor_id: int, on: bool = True):
        m = self.get_motor(motor_id)
        (m.enable() if on else m.disable())

    def set_zero_joint(self, motor_id: int):
        self.get_motor(motor_id).set_zero()

    # ---- CSP
    def csp_joint(self, motor_id: int, q_target: float):
        self.get_motor(motor_id).csp_cmd(q_target)

    def set_slew_rate_joint(self, motor_id: int, vmax_rad_s: float):
        # 注意：你的 MotorShim 里把 slew_rate 定义为“rad/step”，这里继续按原有换算
        self.get_motor(motor_id).set_csp_param("slew_rate", vmax_rad_s * self._dt_ctrl)

    # ---- CSV（占位）
    def csv_joint(self, motor_id: int, v_target: float):
        self.get_motor(motor_id).csv_cmd(v_target)

    # ---- MIT（占位）
    def mit_joint(self, motor_id: int, q_ref: float, dq_ref: float, kp: float, kd: float, tau_ff: float = 0.0):
        self.get_motor(motor_id).mit_cmd(q_ref, dq_ref, kp, kd, tau_ff)

    # ---- PVT（占位）
    def pvt_joint(self, motor_id: int, q_ref: float, dq_ref: float):
        self.get_motor(motor_id).pvt_cmd(q_ref, dq_ref)

    # ---- 观测
    def get_joint_state(self, motor_id: int):
        m = self.get_motor(motor_id)
        i = m.i  # MotorShim 内部保存的（0-based）执行器索引；保持你原来的读法不变
        return {
            "position": float(self.mj_data.qpos[i]),
            "velocity": float(self.mj_data.qvel[i]),
        }

    # ---------------- 仿真线程：应用控制 ----------------
    def updateControl(self, action):
        for m in self.motors:
            try:
                m.apply_to_ctrl()
            except NotImplementedError:
                pass

    def post_physics_step(self): pass
    def getChangedObjectPose(self): return None
# ------------------------------- 小测例（保持与原 play 脚本一致） -------------------------------
def play_with_csp():
    cfg = AirbotPlayCfg()
    env = AirbotPlayShim(cfg)
    obs = env.reset()
    nj = env.nj

    vmax = 1.0
    for motor_id in range(1, nj+1):
        env.set_slew_rate_joint(motor_id, vmax)

    T = 5.0
    dt = cfg.timestep * cfg.decimation
    steps = int(T / dt)

    # [ADD] 采样与暂停控制
    t_probe = 2.0                           # 在 2.0 s 时做一次暂停与采样
    k_probe = max(1, int(t_probe / dt))     # 对应的步数索引
    hold_steps = max(1, int(0.20 / dt))     # 暂停 0.2 s 观察
    jq_prev = None                          # 用于有限差分估计速度

    try:
        for k in range(steps):
            t = k * dt

            # 正常给目标
            amp = np.array([0.3, 0.2, 0.25, 0.2, 0.15, 0.1, 0.02])[:nj]
            phs = np.linspace(0, np.pi, nj)
            q_target = amp * np.sin(2*np.pi*0.2*t + phs)

            for motor_id in range(1, nj+1):
                env.csp_joint(motor_id, float(q_target[motor_id-1]))

            # 推进一步
            obs, _, _, terminated, _ = env.step(None)

            # [ADD] 有限差分速度估计，和 obs["jv"] 做对比
            jq = np.array(obs["jq"])
            jv = np.array(obs["jv"])
            if jq_prev is None:
                jq_prev = jq.copy()
            fd_jv = (jq - jq_prev) / dt
            jq_prev = jq.copy()

            # 每 50 ms 打印一次
            if k % max(1, int(0.05/dt)) == 0:
                print(f"[run] t={t:5.2f}s  q[0:3]={jq[:3]}  jv[0:3]={jv[:3]}  fd_jv[0:3]={fd_jv[:3]}")

            # [CHANGED][ADD] 到达探测时刻：暂停一小段时间并打印 get_joint_state
            if k == k_probe:
                print("\n[probe] >>> entering hold window, freezing targets at current position ...")
                # 把目标固定为“当前实际位置”（相当于刹车），并保持一段时间
                for motor_id in range(1, nj+1):
                    env.csp_joint(motor_id, float(jq[motor_id-1]))

                for h in range(hold_steps):
                    obs_hold, _, _, _, _ = env.step(None)
                    jq_h = np.array(obs_hold["jq"])
                    jv_h = np.array(obs_hold["jv"])

                    # 直接读 shim 的关节状态（内部 0-based，外部 1-based 已在类里映射）
                    st_1 = env.get_joint_state(1)   # 看第1个关节
                    st_2 = env.get_joint_state(2)   # 看第2个关节

                    # 有限差分核对
                    if h == 0:
                        jq_prev_hold = jq_h.copy()
                    fd_jv_h = (jq_h - jq_prev_hold) / dt
                    jq_prev_hold = jq_h.copy()

                    print(f"[probe] hold {h+1}/{hold_steps}: "
                          f"q1={st_1['position']:.4f}, v1={st_1['velocity']:.4f} | "
                          f"q2={st_2['position']:.4f}, v2={st_2['velocity']:.4f} || "
                          f"obs_jv1={jv_h[0]:.4f}, fd_jv1={fd_jv_h[0]:.4f}")

                print("[probe] <<< leaving hold window, resume trajectory ...\n")

            if terminated:
                env.reset()

    except KeyboardInterrupt:
        pass


def play_with_gripper():
    """
    仅驱动“夹爪关节”（默认取第 nj 个关节）做小幅开合，
    并在 2.0 s 的保持窗口里采样，核对 position / velocity 的一致性。
    """
    cfg = AirbotPlayCfg()
    env = AirbotPlayShim(cfg)
    obs = env.reset()

    nj = env.nj
    grip_id = nj              # 外部 1-based ID，数组索引用 grip_id-1
    idx = grip_id - 1

    # 仅对夹爪设置斜率限制
    vmax = 1.0
    env.set_slew_rate_joint(grip_id, vmax)

    # 运行 5 秒
    T = 5.0
    dt = cfg.timestep * cfg.decimation
    steps = int(T / dt)

    # 夹爪开合幅度（弧度），可视情况调大一点（例如 0.03~0.05）
    amp = 0.06
    freq = 0.5   # 0.5 Hz

    # 探针与保持窗口
    t_probe   = 2.0
    k_probe   = max(1, int(t_probe / dt))
    hold_steps = max(1, int(0.25 / dt))   # 保持 0.25 s

    # 为有限差分做准备
    jq_prev = None
    moved_enough = False  # 用来判断夹爪是否真的在动

    try:
        for k in range(steps):
            t = k * dt

            # 目标：其他关节保持在当前观测位置，仅夹爪关节按正弦开合
            q_now = np.array(obs["jq"])
            q_target = q_now.copy()
            q_target[idx] = amp * np.sin(2*np.pi*freq*t)

            # 只下发夹爪目标（也可以对所有关节下发，其中非夹爪目标用 q_now）
            env.csp_joint(grip_id, float(q_target[idx]))

            # 推进一步
            obs, _, _, terminated, _ = env.step(None)

            # 观测
            jq = np.array(obs["jq"])
            jv = np.array(obs["jv"])

            # 有限差分速度
            if jq_prev is None:
                jq_prev = jq.copy()
            fd_jv = (jq - jq_prev) / dt
            jq_prev = jq.copy()

            # 简单判断夹爪是否真的在运动
            if abs(jq[idx]) > 1e-4 or abs(jv[idx]) > 1e-4:
                moved_enough = True

            # 每 50 ms 打印一次夹爪关节状态
            if k % max(1, int(0.05/dt)) == 0:
                print(f"[run] t={t:5.2f}s  grip.q={jq[idx]:+.5f}  "
                      f"jv={jv[idx]:+.5f}  fd_jv={fd_jv[idx]:+.5f}")

            # 到达探针时刻：冻结目标到当前位置，进入保持窗口
            if k == k_probe:
                print("\n[probe] >>> entering hold window, freeze gripper at current position ...")
                env.csp_joint(grip_id, float(jq[idx]))  # 把目标定住当前值

                jq_prev_hold = None
                for h in range(hold_steps):
                    obs_h, _, _, _, _ = env.step(None)
                    jq_h = np.array(obs_h["jq"])
                    jv_h = np.array(obs_h["jv"])

                    # 单关节读取（外部 1-based）：期望返回 {'position':..,'velocity':..}
                    st = env.get_joint_state(grip_id) if hasattr(env, "get_joint_state") else None
                    print(f"grid id: {grip_id}, idx :{idx}, st: {st}")
                    time.sleep(3)
                    # 有限差分
                    if jq_prev_hold is None:
                        jq_prev_hold = jq_h.copy()
                    fd_jv_h = (jq_h - jq_prev_hold) / dt
                    jq_prev_hold = jq_h.copy()

                    # 打印三方对比
                    if st is not None:
                        print(f"[probe] hold {h+1}/{hold_steps}: "
                              f"q={st['position']:+.5f}, v={st['velocity']:+.5f}  "
                              f"| obs_jv={jv_h[idx]:+.5f}, fd_jv={fd_jv_h[idx]:+.5f}")
                    else:
                        print(f"[probe] hold {h+1}/{hold_steps}: "
                              f"q={jq_h[idx]:+.5f} | obs_jv={jv_h[idx]:+.5f}, fd_jv={fd_jv_h[idx]:+.5f}")

                print("[probe] <<< leaving hold window, resume trajectory ...\n")

            if terminated:
                env.reset()

        # 结束后给出判定与建议
        if not moved_enough:
            print("\n[hint] 夹爪关节（假定为第 nj 个）几乎没有运动。"
                  "你的模型可能是腱驱或双指两关节：\n"
                  "  - 腱驱：应读取 data.ten_length/ten_velocity（或 shim 暴露的 gripper API）\n"
                  "  - 双指：用两个 finger 关节的 qpos 组合推导开度（例如左右之差或和）\n"
                  "  - 或确认第 nj 个是否真的是夹爪关节\n")

    except KeyboardInterrupt:
        pass


def play_with_csv():
    """
    测试 CSV 速度控制接口。
    驱动第一个关节以恒定速度运动，验证速度控制功能。
    """
    cfg = AirbotPlayCfg()
    env = AirbotPlayShim(cfg)
    obs = env.reset()
    nj = env.nj

    # 设置第一个关节为 CSV 模式
    motor_id = 1
    env.set_mode_joint(motor_id, ControlMode.CSV)
    env.enable_joint(motor_id, True)

    # 设置速度限制（可选）
    env.get_motor(motor_id).set_csv_param("v_limit", 2.0)  # 最大 2.0 rad/s

    # 运行 5 秒
    T = 5.0
    dt = cfg.timestep * cfg.decimation
    steps = int(T / dt)

    # 目标速度：第一个关节 0.1 rad/s，其他关节保持 0
    target_velocity = 0.1  # rad/s

    jq_prev = None
    print(f"[CSV Test] Starting velocity control test for joint {motor_id}")
    print(f"[CSV Test] Target velocity: {target_velocity} rad/s")
    print(f"[CSV Test] Duration: {T} s\n")

    try:
        for k in range(steps):
            t = k * dt

            # 持续发送速度命令（CSV 需要持续调用）
            env.csv_joint(motor_id, target_velocity)

            # 推进一步
            obs, _, _, terminated, _ = env.step(None)

            # 观测
            jq = np.array(obs["jq"])
            jv = np.array(obs["jv"])

            # 有限差分速度估计
            if jq_prev is None:
                jq_prev = jq.copy()
            fd_jv = (jq - jq_prev) / dt
            jq_prev = jq.copy()

            # 每 100 ms 打印一次
            if k % max(1, int(0.1/dt)) == 0:
                print(f"[CSV] t={t:5.2f}s  q[{motor_id-1}]={jq[motor_id-1]:+.5f}  "
                      f"jv={jv[motor_id-1]:+.5f}  fd_jv={fd_jv[motor_id-1]:+.5f}  "
                      f"target_v={target_velocity:.3f}")

            # 在 2.5 秒时改变速度方向
            if k == int(2.5 / dt):
                target_velocity = -0.1
                print(f"\n[CSV] Changing velocity direction to {target_velocity} rad/s\n")

            if terminated:
                env.reset()

        print(f"\n[CSV Test] Completed. Final position: {jq[motor_id-1]:+.5f} rad")

    except KeyboardInterrupt:
        pass


def play_with_csv_zero():
    """
    测试 CSV 在 v_ref = 0 时的行为：
    - 将第一个关节切到 CSV 模式并 enable
    - 在整个测试过程中持续发送 csv_joint(..., 0.0)
    - 观察关节在重力下是否会发生回落
    """
    cfg = AirbotPlayCfg()
    env = AirbotPlayShim(cfg)
    obs = env.reset()
    nj = env.nj

    # 将所有关节切换为 CSV 模式并使能
    for motor_id in range(1, nj + 1):
        env.set_mode_joint(motor_id, ControlMode.CSV)
        env.enable_joint(motor_id, True)

    T = 5.0
    dt = cfg.timestep * cfg.decimation
    steps = int(T / dt)

    jq_prev = None
    print(f"[CSV-0 Test] Start: all {nj} joints, v_ref=0.0, duration={T:.2f}s, dt={dt:.4f}s\n")

    try:
        for k in range(steps):
            t = k * dt

            # 持续对所有关节发送 v_ref = 0.0
            for motor_id in range(1, nj + 1):
                env.csv_joint(motor_id, 0.0)

            obs, _, _, terminated, _ = env.step(None)

            jq = np.array(obs["jq"])
            jv = np.array(obs["jv"])

            if jq_prev is None:
                jq_prev = jq.copy()
            fd_jv = (jq - jq_prev) / dt
            jq_prev = jq.copy()

            # 每 100 ms 打印一次部分关节的位置变化（前三个作为代表）
            if k % max(1, int(0.1 / dt)) == 0:
                rep = min(3, nj)
                qs = ", ".join(f"{jq[i]:+.5f}" for i in range(rep))
                vs = ", ".join(f"{jv[i]:+.5f}" for i in range(rep))
                fds = ", ".join(f"{fd_jv[i]:+.5f}" for i in range(rep))
                print(f"[CSV-0] t={t:5.2f}s  q[0:{rep}]={qs}  jv[0:{rep}]={vs}  fd_jv[0:{rep}]={fds}")

            if terminated:
                env.reset()

        rep = min(3, nj)
        qs = ", ".join(f"{jq[i]:+.5f}" for i in range(rep))
        print(f"\n[CSV-0 Test] Completed. Final q[0:{rep}]={qs}\n")

    except KeyboardInterrupt:
        pass


def play_with_pvt():
    """
    测试 PVT 位置-速度-时间控制接口。
    驱动第一个关节移动到目标位置，验证带速度限制的位置控制功能。
    """
    cfg = AirbotPlayCfg()
    env = AirbotPlayShim(cfg)
    obs = env.reset()
    nj = env.nj

    # 设置第一个关节为 PVT 模式
    motor_id = 1
    env.set_mode_joint(motor_id, ControlMode.PVT)
    env.enable_joint(motor_id, True)

    # 运行 5 秒
    T = 5.0
    dt = cfg.timestep * cfg.decimation
    steps = int(T / dt)

    # 初始位置和目标位置
    jq_init = np.array(obs["jq"])
    target_position = jq_init[motor_id-1] + 0.5  # 目标位置：当前位置 + 0.5 rad
    max_velocity = 0.5  # 最大速度限制：0.5 rad/s

    print(f"[PVT Test] Starting position-velocity-time control test for joint {motor_id}")
    print(f"[PVT Test] Initial position: {jq_init[motor_id-1]:+.5f} rad")
    print(f"[PVT Test] Target position: {target_position:+.5f} rad")
    print(f"[PVT Test] Max velocity: {max_velocity} rad/s")
    print(f"[PVT Test] Duration: {T} s\n")

    jq_prev = None
    reached_target = False

    try:
        for k in range(steps):
            t = k * dt

            # 发送 PVT 命令：目标位置和最大速度
            env.pvt_joint(motor_id, target_position, max_velocity)

            # 推进一步
            obs, _, _, terminated, _ = env.step(None)

            # 观测
            jq = np.array(obs["jq"])
            jv = np.array(obs["jv"])

            # 有限差分速度估计
            if jq_prev is None:
                jq_prev = jq.copy()
            fd_jv = (jq - jq_prev) / dt
            jq_prev = jq.copy()

            # 检查是否到达目标
            error = abs(jq[motor_id-1] - target_position)
            if error < 0.01 and not reached_target:
                reached_target = True
                print(f"\n[PVT] Target reached at t={t:.2f}s (error={error:.5f} rad)\n")

            # 每 100 ms 打印一次
            if k % max(1, int(0.1/dt)) == 0:
                print(f"[PVT] t={t:5.2f}s  q[{motor_id-1}]={jq[motor_id-1]:+.5f}  "
                      f"target={target_position:+.5f}  error={error:.5f}  "
                      f"jv={jv[motor_id-1]:+.5f}  fd_jv={fd_jv[motor_id-1]:+.5f}")

            # 在 2.5 秒时改变目标位置（如果还没到达）
            if k == int(2.5 / dt) and not reached_target:
                target_position = jq_init[motor_id-1] - 0.3  # 反向移动
                print(f"\n[PVT] Changing target to {target_position:+.5f} rad\n")

            if terminated:
                env.reset()

        if reached_target:
            print(f"\n[PVT Test] Completed. Successfully reached target position.")
        else:
            print(f"\n[PVT Test] Completed. Final position: {jq[motor_id-1]:+.5f} rad, "
                  f"error: {error:.5f} rad")

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    # play_with_csp()
    # play_with_gripper()
    # play_with_csv()
    # play_with_pvt()
    play_with_csv_zero()