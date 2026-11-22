"""通过 hidapi 读取一代蓝牙 Joy-Con 的陀螺仪、摇杆和实体按键。"""

import logging
import struct
import time
from dataclasses import dataclass

import numpy as np

SAMPLE_SECONDS = 0.005
REPORT_SECONDS = 3 * SAMPLE_SECONDS
BUTTONS = {
    "right": ("Y", "X", "B", "A", "SR", "SL", "R", "ZR"),
    "left": ("Down", "Up", "Right", "Left", "SR", "SL", "L", "ZL"),
}


@dataclass
class RayInput:
    gyro_rad_s: np.ndarray  # 每行一次采样，三列依次为手柄前、左、上轴的角速度（弧度/秒）
    stick: float
    buttons: set[str]


class JoyConRayInput:
    def __init__(self, name, stick_deadzone):
        self.name = name
        self.deadzone = stick_deadzone
        self.device = None
        self.bias = None
        self.center = 2048
        self.rest_samples = []
        self.last_timer = None
        self.last_report_time = None

    def _open(self):
        import hid

        if not hasattr(hid, "device"):
            raise ImportError("Joy-Con ray control needs the hidapi package (hid.device).")
        device = hid.device()
        try:
            device.open(0x057E, {"left": 0x2006, "right": 0x2007}[self.name])
            # 依次启用 IMU、设置量程、切换到 0x30 完整报告，每步等待手柄确认。
            commands = ((0x40, b"\x01"), (0x41, b"\x03\x00\x01\x01"), (0x03, b"\x30"))
            for number, (command, payload) in enumerate(commands):
                device.write(
                    bytes([1, number]) + bytes.fromhex("00 01 40 40") * 2 + bytes([command]) + payload
                )
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    reply = device.read(64, 50)
                    if len(reply) >= 15 and reply[0] == 0x21 and reply[14] == command and reply[13] & 0x80:
                        break
                else:
                    raise OSError(f"No Joy-Con acknowledgement for subcommand {command:#04x}")
            device.set_nonblocking(1)
        except Exception:
            device.close()
            raise
        self.device = device
        self.last_report_time = time.monotonic()
        logging.info("Joy-Con connected. Leave buttons and stick released and hold still for one second.")

    def _decode(self, report):
        left = self.name == "left"
        button_byte = report[5 if left else 3]
        buttons = {key for bit, key in enumerate(BUTTONS[self.name]) if button_byte & (1 << bit)}
        offset = 6 if left else 9
        stick = (report[offset + 1] >> 4) | (report[offset + 2] << 4)
        raw = np.array([struct.unpack_from("<hhh", report, 19 + 12 * i) for i in range(3)])
        # ±2000°/s 量程按标称灵敏度换算。右手柄的 Y、Z 轴反向安装，翻转后左右手柄共用同一坐标系。
        rates = raw * np.deg2rad(0.070) * ([1, 1, 1] if left else [1, -1, -1])
        if self.bias is None:
            if buttons or abs(stick - 2048) > 500 or np.max(np.abs(rates)) > np.deg2rad(8):
                self.rest_samples.clear()
                return None
            self.rest_samples.append((rates, stick))
            # 连续 67 份报告约为一秒。用静置时的均值估计陀螺仪零偏，用中位数估计摇杆中心。
            if len(self.rest_samples) == 67:
                samples, centers = zip(*self.rest_samples, strict=True)
                samples = np.concatenate(samples)
                if np.max(samples.std(axis=0)) < np.deg2rad(1):
                    self.bias = samples.mean(axis=0)
                    self.center = float(np.median(centers))
                    logging.info("Joy-Con ready. Hold the motion button to steer.")
                self.rest_samples.clear()
            if self.bias is None:
                return None
        normalized = (stick - self.center) / (4095 - self.center if stick >= self.center else self.center)
        magnitude = max(0.0, abs(normalized) - self.deadzone) / (1 - self.deadzone)
        return RayInput(rates - self.bias, np.copysign(magnitude, normalized), buttons)

    def poll(self):
        """按顺序返回新报告。空列表表示暂无新输入，None 表示输入超时、需重新按键接管。"""
        if self.device is None:
            self._open()
        now = time.monotonic()
        stale = now - self.last_report_time > 0.20
        frames = []
        # 录制暂停后，排空超时期间积压的报告，避免恢复时补执行旧动作。
        while report := self.device.read(64):
            if len(report) < 49 or report[0] != 0x30 or report[1] == self.last_timer:
                continue
            self.last_timer = report[1]
            self.last_report_time = now
            if not stale:
                frame = self._decode(bytes(report))
                if frame is not None:
                    frames.append(frame)
        if stale:
            self.rest_samples.clear()
            return None
        return frames

    def close(self):
        if self.device is not None:
            self.device.close()
        self.device = self.bias = self.last_timer = self.last_report_time = None
        self.rest_samples.clear()
