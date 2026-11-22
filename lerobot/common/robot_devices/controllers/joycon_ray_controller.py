"""将 Joy-Con 的陀螺仪、摇杆和按键输入转换为单台 SO-101 从臂的关节目标。"""

import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.common.robot_devices.controllers.joycon_ray_input import (
    BUTTONS,
    REPORT_SECONDS,
    SAMPLE_SECONDS,
    JoyConRayInput,
)

DEFAULT_BUTTONS = {
    "right": {"motion": "R", "gripper": "ZR", "save": "A", "retry": "Y"},
    "left": {"motion": "L", "gripper": "ZL", "save": "Right", "retry": "Left"},
}


class RayKinematics:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("so101_ray.xml")))
        self.data = mujoco.MjData(self.model)
        # 取指尖参考点在腕部滚转轴上的投影作为工具点（TCP）。只拧腕时，该点的位置保持不变。
        self.tool_length = -self.model.site("gripper").pos[2]
        mujoco.mj_kinematics(self.model, self.data)
        self.yaw_sign = np.sign(self.data.xaxis[0, 2])

    def tool_pose(self, q):
        """输入五个模型关节角（弧度），返回基座坐标系中的工具点位置（米）和夹爪前向单位向量。"""
        self.data.qpos[:5] = q
        mujoco.mj_kinematics(self.model, self.data)
        forward = -self.data.xaxis[4]
        return self.data.xanchor[4] + self.tool_length * forward, forward

    def step(self, q, yaw, pitch, length, depth):
        """根据偏航、俯仰增量（弧度）和推进量（米）求下一组关节角。不可达时返回 None。"""
        goal = q.copy()
        goal[0] += self.yaw_sign * yaw
        tcp, forward = self.tool_pose(goal)
        swung = Rotation.from_rotvec(pitch * self.data.xaxis[1]).apply(forward)
        # 从当前工具点沿夹爪轴后退 length 得到俯仰支点，再沿新指向量出摇杆调整后的长度。
        target = tcp - length * forward + (length + depth) * swung

        # 肩、肘、腕俯仰共面。腕部补足总俯仰角，肩和肘用位置雅可比把工具点送到射线末端。
        goal[3] += pitch
        for _ in range(8):
            tcp, _ = self.tool_pose(goal)
            error = target - tcp
            if np.linalg.norm(error) < 1e-7:
                lower, upper = self.model.jnt_range[:5].T
                if np.all(goal >= lower - 2e-5) and np.all(goal <= upper + 2e-5):
                    return goal
                break
            jac = np.cross(self.data.xaxis[1:4], tcp - self.data.xanchor[1:4])
            # 保持夹爪俯仰角时，腕部角度增量等于肩、肘增量之和的负值。
            jac = (jac[:2] - jac[2]).T
            dq = np.linalg.solve(jac.T @ jac + 1e-7 * np.eye(2), jac.T @ error)
            dq = np.clip(dq, -0.12, 0.12)
            goal[1:3] += dq
            goal[3] -= dq.sum()
        return None


class JoyConRayController:
    # 总线按“水平伸直为零”标定。肩俯仰、腕滚转的正方向与模型相反；夹爪开度另用百分比。
    motor_sign = np.array([1, -1, 1, 1, -1])

    def __init__(
        self,
        name="right",
        ray_length_m=0.10,
        advance_m_s=0.06,
        gyro_gain=(1, 1, 1),
        stick_deadzone=0.15,
        gripper_open_pct=60,
        buttons=None,
        mouse_buttons=None,
    ):
        if name not in DEFAULT_BUTTONS:
            raise ValueError("name must be right or left")
        self.buttons = DEFAULT_BUTTONS[name] | dict(buttons or {})
        if (
            self.buttons.keys() != DEFAULT_BUTTONS[name].keys()
            or len(set(self.buttons.values())) != 4
            or not set(self.buttons.values()) <= set(BUTTONS[name])
        ):
            raise ValueError(f"Choose four distinct {name} Joy-Con buttons for motion, gripper, save, retry")
        self.gain = np.asarray(gyro_gain, dtype=float)
        if self.gain.shape != (3,) or not np.isfinite(self.gain).all():
            raise ValueError("gyro_gain must contain finite roll, pitch and yaw gains")
        if not (
            0.01 <= ray_length_m < np.inf
            and 0 < advance_m_s < np.inf
            and 0 <= stick_deadzone < 1
            and 0 < gripper_open_pct <= 100
        ):
            raise ValueError("Invalid ray length, advance speed, stick deadzone or gripper opening")
        self.input = JoyConRayInput(name, stick_deadzone)
        self.kinematics = RayKinematics()
        self.start_length = ray_length_m
        self.speed = advance_m_s
        self.open_pct = gripper_open_pct
        self.mouse_buttons = dict(mouse_buttons or {})
        if self.mouse_buttons and (
            not self.mouse_buttons.keys() <= {"left", "middle", "right"}
            or sorted(self.mouse_buttons.values()) != ["close", "open"]
        ):
            raise ValueError("Map two distinct mouse buttons (left, middle, right) to open and close")
        self.mouse_listener = None
        self.mouse_clicks = deque(maxlen=1)
        self.command = None
        self.previous_buttons = None
        self.active = False

    def _mouse_click(self, x, y, button, pressed, injected=False):
        action = self.mouse_buttons.get(getattr(button, "name", None))
        if pressed and action is not None and self.active:
            self.mouse_clicks.append((time.monotonic(), action))

    def _move(self, frame, measured):
        # 角速度相对于手柄自身坐标轴。按采样顺序累乘旋转，倾斜握持时也能单独拧腕。
        for omega in frame.gyro_rad_s:
            self.orientation *= Rotation.from_rotvec(self.gain * omega * SAMPLE_SECONDS)
        angles = self.orientation.as_euler("ZYX")
        # 同一姿态有两组等价欧拉角，取更接近上一帧的一组，避免手柄经过竖直方向时角度跳变。
        alternate = [angles[0] + np.pi, np.pi - angles[1], angles[2] + np.pi]
        changes = (np.array([angles, alternate]) - self.previous_angles + np.pi) % (2 * np.pi) - np.pi
        delta = changes[np.argmin(np.linalg.norm(changes, axis=1))]
        self.previous_angles += delta
        depth = max(-self.length + 0.01, self.speed * frame.stick * REPORT_SECONDS)
        q = np.deg2rad(self.command[:5]) * self.motor_sign
        # 先独立求腕部滚转。模型腕轴指向手腕，手柄滚转轴指向指尖，因此从模型角减去滚转增量。
        roll_limit = self.kinematics.model.jnt_range[4]
        roll = np.clip(q[4] - delta[2], *roll_limit)
        wrist_deg = np.clip(-np.rad2deg(roll), self.command[4] - 3, self.command[4] + 3)
        if abs(delta[2]) > 1e-12 and abs(wrist_deg - measured[4]) <= 5:
            self.command[4] = wrist_deg
        if not (np.any(delta[:2]) or depth):
            return
        # 优先尝试完整指令，靠近关节限位时缩小步幅。每步最多 3°，目标最多领先实测关节角 5°。
        for fraction in (1, 0.5, 0.25, 0.125):
            goal = self.kinematics.step(
                q, delta[0] * fraction, delta[1] * fraction, self.length, depth * fraction
            )
            if goal is None:
                continue
            motors = np.rad2deg(goal[:4]) * self.motor_sign[:4]
            if np.max(np.abs(motors - measured[:4])) <= 5 and np.max(np.abs(motors - self.command[:4])) <= 3:
                self.command[:4] = motors
                self.length += depth * fraction
                break

    def get_command(self, present_pose):
        """输入五轴关节角（度）和夹爪开度（%），输出六维目标、录制事件和夹爪状态。"""
        measured = np.asarray(present_pose, dtype=float)
        if measured.shape != (6,) or not np.isfinite(measured).all():
            raise ValueError("Expected five calibrated joint angles (deg) and gripper opening (%)")
        if self.command is None:
            self.command = measured.copy()
        recording = 0
        frames = self.input.poll()
        if self.mouse_buttons and self.mouse_listener is None:
            from pynput.mouse import Listener

            self.mouse_listener = Listener(on_click=self._mouse_click)
            self.mouse_listener.start()
        if frames is None:
            self.active = False
            self.previous_buttons = None
            self.mouse_clicks.clear()
        for frame in frames or ():
            pressed = frame.buttons
            if self.previous_buttons is None:
                # 启动或输入恢复后，先记下当前按键状态；松开再按才触发新的动作。
                self.previous_buttons = pressed
            edges = pressed - self.previous_buttons
            self.previous_buttons = pressed
            if self.buttons["gripper"] in edges:
                self.command[5] = 0 if self.command[5] >= self.open_pct / 2 else self.open_pct
            if self.buttons["retry"] in edges:
                recording = -1
            elif self.buttons["save"] in edges:
                recording = 1

            if self.buttons["motion"] not in pressed:
                self.active = False
                self.mouse_clicks.clear()
            elif self.buttons["motion"] in edges:
                # 每次重新接管都以实测姿态为起点，松开运动键期间换握姿不会让机械臂跳动。
                self.command[:5] = measured[:5]
                self.length = self.start_length
                self.orientation = Rotation.identity()
                self.previous_angles = np.zeros(3)
                self.active = True
            elif self.active:
                self._move(frame, measured)

        if frames and self.active and self.mouse_clicks:
            when, action = self.mouse_clicks.popleft()
            if time.monotonic() - when <= 0.20:
                self.command[5] = self.open_pct if action == "open" else 0
        # 录制循环使用 1 保存、-1 重录、0 继续。夹爪状态 0 供负载保护识别闭合指令。
        return self.command.astype(np.float32), recording, float(self.command[5] > 0)

    def close(self):
        if self.mouse_listener is not None:
            self.mouse_listener.stop()
            self.mouse_listener.join()
            self.mouse_listener = None
        self.mouse_clicks.clear()
        self.input.close()
        self.command = None
        self.previous_buttons = None
        self.active = False
