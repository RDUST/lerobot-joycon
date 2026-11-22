"""Offline checks for the ray profile. No Bluetooth, serial port or camera is opened."""

import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot.common.robot_devices.controllers.joycon_ray_controller import JoyConRayController
from lerobot.common.robot_devices.controllers.joycon_ray_input import REPORT_SECONDS, JoyConRayInput, RayInput
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
from lerobot.common.utils.utils import init_hydra_config

START = np.array([15.0, 75.0, 85.0, 12.0, 18.0, 45.0], dtype=np.float32)
CONFIG = "lerobot/configs/robot/so101_joycon_ray.yaml"


def frame(buttons=(), gyro=(0, 0, 0), stick=0):
    return RayInput(np.tile(gyro, (3, 1)), stick, set(buttons))


@pytest.fixture
def control():
    controller = JoyConRayController()
    controller.input = Mock()
    return controller


def send(control, frames, measured=START):
    control.input.poll.return_value = frames
    return control.get_command(measured)


def take_control(control):
    command, _, _ = send(control, [frame(), frame("R")])
    np.testing.assert_array_equal(command, START)


def pose(control, motors):
    point, forward = control.kinematics.tool_pose(np.deg2rad(motors[:5].astype(float)) * control.motor_sign)
    return point, forward.copy()


@pytest.mark.parametrize("sign", [-1, 1])
def test_depth_along_tilted_tool(control, sign):
    take_control(control)
    start, forward = pose(control, START)
    command, _, _ = send(control, [frame("R", stick=sign)])
    end, direction = pose(control, command)
    np.testing.assert_allclose(end - start, forward * sign * 0.06 * REPORT_SECONDS, atol=2e-7)
    np.testing.assert_allclose(direction, forward, atol=3e-7)


def test_swing_and_twist(control):
    take_control(control)
    start, forward = pose(control, START)
    yawed, _, _ = send(control, [frame("R", gyro=(0, 0, 1))])
    swept, pointing = pose(control, yawed)
    assert np.dot(swept - start, np.cross([0, 0, 1], forward)) > 0.003
    pitched, _, _ = send(control, [frame("R", gyro=(0, -1, 0))], yawed)
    lifted, tilted = pose(control, pitched)
    assert lifted[2] > swept[2] and tilted[2] > pointing[2]
    twisted, _, _ = send(control, [frame("R", gyro=(1, 0, 0))], pitched)
    np.testing.assert_allclose(twisted[:4], pitched[:4], atol=1e-5)
    np.testing.assert_allclose(twisted[4] - pitched[4], np.rad2deg(REPORT_SECONDS), atol=1e-5)
    np.testing.assert_allclose(pose(control, twisted)[0], lifted, atol=1e-7)
    assert twisted[5] == pitched[5]


def test_release_stale_regrab_and_bounds(control):
    take_control(control)
    moved, _, _ = send(control, [frame("R", gyro=(1, 0, 0))])
    for frames in ([frame(stick=1)], None, [frame("R", stick=1)]):
        held, _, _ = send(control, frames)
        np.testing.assert_array_equal(held, moved)
    at_limit = START.copy()
    at_limit[0] = -np.rad2deg(control.kinematics.model.jnt_range[0, 1])
    command, _, _ = send(control, [frame(), frame("R"), frame("R", gyro=(0, 0, 1))], at_limit)
    np.testing.assert_allclose(command, at_limit, atol=1e-5)
    reversed_command, _, _ = send(control, [frame("R", gyro=(0, 0, -1))], command)
    assert reversed_command[0] > command[0]


def test_button_edges(control):
    send(control, [frame()])
    for buttons, opening, event in [
        (["ZR", "A"], 0, 1),
        (["ZR", "A"], 0, 0),
        ([], 0, 0),
        (["ZR", "Y"], 60, -1),
    ]:
        cmd, recording, state = send(control, [frame(buttons)])
        assert (cmd[5], recording, state) == (opening, event, int(opening > 0))


def test_optional_mouse_requires_motion_button(monkeypatch):
    control = JoyConRayController(mouse_buttons={"left": "close", "right": "open"})
    listener = Mock()
    monkeypatch.setitem(sys.modules, "pynput.mouse", SimpleNamespace(Listener=listener))
    control.input = Mock()
    send(control, [frame()])
    click = listener.call_args.kwargs["on_click"]
    click(0, 0, SimpleNamespace(name="left"), True)
    assert send(control, [frame("R")])[0][5] == START[5]
    for button, opening in [("left", 0), ("right", 60)]:
        click(0, 0, SimpleNamespace(name=button), True)
        assert send(control, [frame("R")])[0][5] == opening
    click(0, 0, SimpleNamespace(name="left"), True)
    assert send(control, None)[0][5] == 60
    control.close()
    listener.return_value.stop.assert_called_once()


def test_wrist_roll_when_position_ik_fails(control, monkeypatch):
    take_control(control)
    monkeypatch.setattr(control.kinematics, "step", lambda *args: None)
    cmd, _, _ = send(control, [frame("R", gyro=(1, 0, 0), stick=1)])
    np.testing.assert_array_equal(cmd[:4], START[:4])
    assert cmd[4] > START[4]


def report(timer, side, samples=((0, 0, 0),) * 3, stick=2048, keys=0):
    packet = bytearray(49)
    packet[:2] = (0x30, timer % 256)
    packet[5 if side == "left" else 3] = keys
    offset = 6 if side == "left" else 9
    packet[offset + 1] = (stick & 15) << 4
    packet[offset + 2] = stick >> 4
    for i, sample in enumerate(samples):
        struct.pack_into("<hhh", packet, 19 + 12 * i, *sample)
    return packet


@pytest.mark.parametrize(("side", "multiplier", "clutch"), [("left", 1, "L"), ("right", -1, "R")])
def test_full_hid_reports_calibration_and_freshness(side, multiplier, clutch, monkeypatch):
    reader = JoyConRayInput(side, 0.15)
    reader.device = Mock()
    reader.last_report_time = 0
    monkeypatch.setattr(
        "lerobot.common.robot_devices.controllers.joycon_ray_input.time.monotonic", lambda: 0.1
    )
    for i in range(67):
        reader.device.read.side_effect = [report(i, side, stick=2100), []]
        reader.poll()
    samples = ((10, 20, 30), (40, 50, 60), (70, 80, 90))
    packet = report(68, side, samples, 4095, 1 << 6)
    reader.device.read.side_effect = [packet, packet, []]
    (parsed,) = reader.poll()
    expected = np.deg2rad(np.array(samples) * 0.070 * [1, multiplier, multiplier])
    np.testing.assert_allclose(parsed.gyro_rad_s, expected)
    assert parsed.stick == 1 and parsed.buttons == {clutch}
    reader.device.read.side_effect = [report(69, side, stick=2150), []]
    assert reader.poll()[0].stick == 0
    reader.last_report_time = -1
    reader.device.read.side_effect = [report(70, side, samples, 4095, 1 << 6), []]
    assert reader.poll() is None


def test_hid_open_and_close(monkeypatch):
    device = Mock()
    hid = SimpleNamespace(device=lambda: device)
    monkeypatch.setitem(sys.modules, "hid", hid)
    ack = bytearray(15)
    ack[0], ack[13] = 0x21, 0x80
    device.read.side_effect = [ack[:14] + bytes([key]) for key in (0x40, 0x41, 0x03)] + [[]]
    reader = JoyConRayInput("right", 0.15)
    assert reader.poll() == []
    device.open.assert_called_once_with(0x057E, 0x2007)
    assert [call.args[0][10:] for call in device.write.call_args_list] == [
        b"\x40\x01",
        b"\x41\x03\x00\x01\x01",
        b"\x03\x30",
    ]
    reader.close()
    device.close.assert_called_once()


def test_profile_through_teleop_step(monkeypatch):
    cfg = init_hydra_config(CONFIG, ["+follower_arms.right.mock=true"])
    robot = make_robot(cfg)
    assert robot.robot_type == "so101" and robot.calibration_dir.name == "so101"
    camera = robot.cameras["right"]
    assert (camera.camera_index, camera.width, camera.height, camera.fps) == (2, 640, 480, 30)
    image = np.full((480, 640, 3), 73, dtype=np.uint8)
    monkeypatch.setattr(camera, "async_read", lambda: image)
    monkeypatch.setattr(camera, "disconnect", lambda: None)
    camera.logs["delta_timestamp_s"] = 0
    control = robot.controllers["right"]
    control.input = Mock()
    control.input.poll.return_value = [frame(), frame("R"), frame(["R", "A"], gyro=(2, 0, 0))]
    bus = Mock()
    bus.read.side_effect = lambda name: np.zeros(6) if name == "Present_Load" else START.copy()
    robot.follower_arms["right"] = bus
    robot.is_connected = True
    try:
        obs, action = robot.teleop_step(record_data=True)
        assert robot.camera_features["observation.images.right"]["shape"] == image.shape
        np.testing.assert_array_equal(obs["observation.images.right"], image)
        np.testing.assert_array_equal(obs["observation.state"], START)
        assert action["action"].shape == (6,) and action["action"][4] > START[4]
        assert robot.button_control == 1
        np.testing.assert_array_equal(bus.write.call_args.args[1], action["action"].numpy().astype(int))
    finally:
        robot.disconnect()
    control.input.close.assert_called_once()


def test_so101_horizontal_zero_and_quarter_turn_calibration(monkeypatch):
    cfg = init_hydra_config(CONFIG)
    bus = FeetechMotorsBus("mock", cfg.follower_arms.right.motors, mock=True)
    zero = np.array([2000, 2300, 2050, 2100, 1800, 2750], dtype=np.int32)
    rotated = zero + np.array([-1024, 1024, -1024, 1024, -1024, -800], dtype=np.int32)
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(bus, "read", Mock(side_effect=[np.zeros(6), zero.copy(), rotated.copy()]))
    bus.set_calibration(run_arm_manual_calibration(bus, "so101", "right", "follower"))
    np.testing.assert_allclose(bus.apply_calibration(zero, None), np.zeros(6))
    np.testing.assert_allclose(bus.apply_calibration(rotated, None), [90, 90, 90, 90, 90, 100])
    np.testing.assert_array_equal(bus.revert_calibration(np.array([90.] * 5 + [100.]), None), rotated)

    control = JoyConRayController()
    np.testing.assert_array_equal(control.motor_sign, [1, -1, 1, 1, -1])
    tip, forward = pose(control, np.array([0, 90, 90, 0, 0, 0]))
    np.testing.assert_allclose(tip, [0.020613, -0.375601, 0.266852], atol=1e-6)
    np.testing.assert_allclose(forward, [0, -1, 0], atol=1e-6)
