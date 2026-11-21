# LeRobot 学习进度

## 我的基础与目标
- 编程项目新手，刚学会 Git，Python 基础尚需确认和补充。
- 学习 SO-ARM100 + JoyCon 的控制、数据采集、训练与部署。
- 计划本地采集数据，使用 Colab GPU 训练，再下载模型到本地运行。

## 项目信息
- 仓库：https://github.com/RDUST/lerobot-joycon
- 当前学习分支：`learning`。
- 云开发环境已配置并发布，不需要重新配置。
- 发布前验证：103 项测试通过、9 项跳过；CPU 模型、仿真和视频处理验证通过。
- 真实机械臂操作和 Colab 训练尚未验证。

## 工作区约定

- 本页兼作学习导航和进度记录。本分支新增的学习说明放在 `learning/`。每次练习记录讲到哪里、核对了什么，以及下一步。
- 项目实现沿用 `lerobot/`，设备参数放在 `lerobot/configs/robot/`，命令行工具优先复用 `lerobot/scripts/`，教学实验沿用 `examples/`。历史改动由 Git 保存。
- 单臂教程将采集数据放在 `datasets/<任务名>/`，训练输出放在 `outputs/train/<实验名>/`。本地临时文件用 `tmp/`。

## 已介绍的内容
- 整体流程：遥操作采集 → 数据集 → 模型训练 → 真机推理。
- 代码架构：配置、程序入口、控制循环、机器人接口、硬件驱动、
  JoyCon 运动学、数据集、策略模型。
- 已完成 6 步源码阅读路线（配置、命令行入口、控制循环、单帧动作与过载保护、数据采集与保存、ACT 训练与真机推理），详细作答与纠偏记录见下方。

## 当前学习位置
6 步源码阅读路线已全部走通；下一步待确定（如：整理本地采集 / Colab 训练 / 本地推理的实战命令，或深入 Joy-Con 逆运动学 / ACT 模型源码）。

第一个练习（已完成 1–5 项）：
1. 找到机械臂串口。
2. 找到六个电机的名称和 ID。
3. 找到相机编号、分辨率、帧率。
4. 确认使用左还是右 JoyCon。
5. 理解 _target_ 对应哪个 Python 类。

### 2026-11-19 · 已核对相机配置，当前读第 4 项 Joy-Con 手柄选择

本次只读[单臂配置的 `follower_arms` 段](../lerobot/configs/robot/so100_joycon_single.yaml)，沿缩进找到右臂的串口（serial port；シリアルポート）。已在本机用 PowerShell 定位到第 27 行，并核对[电机驱动](../lerobot/common/robot_devices/motors/feetech.py)的 `connect()` 会用这个值尝试打开端口。本次验证范围为读取配置和源码。

本人已定位到 `if not self.port_handler.openPort():`。已核对第 27 行的值是单个 `/dev/lerobot_right`，并讲解了配置文件（configuration file；設定ファイル）的作用，以及 `openPort()` 返回 `False` 时如何进入报错分支。

第 2 项已用夹爪核对读法。本人正确读出 `gripper` 的编号为 6，型号为 `sts3215`。编号 1 对应示例中的 `shoulder_pan`。

第 3 项已完成。本人正确读出相机编号 2、图像宽 640 像素、高 480 像素、目标帧率每秒 30 帧。

第 4 项练习（源码推演）：本人选对方案 B，并找到 [Joy-Con 控制器](../lerobot/common/robot_devices/controllers/joycon_controller.py)的初始化参数 `name`。已讲解：B 将内层参数改为 `left`，传给 `JoyconRobotics(device=name)`；[机械臂控制循环](../lerobot/common/robot_devices/robots/manipulator.py)里的同名变量取自外层键 `right`，用来查找右臂。

### 2026-11-20 · 补齐 `self` 基础，完成配置练习第 4–5 项与第 2 步入口代码

本人反馈：能选出 B，但还无法解释 `self`、手柄的 `name` 参数和机械臂循环里的同名变量。先停下来彻底补齐 `self`。

- **`self` 概念与作答/纠偏**：
  - 本人用结构体和全局变量类比 `self`，准确判断 `self.name = name` 是把输入参数 `name` 赋给当前对象的 `name` 成员。
  - 纠偏 1：误以为 `def __init__(...)` 括号里的参数就是类的全部成员 → 已说明对象真正拥有的成员是在函数体内通过 `self.xxx = ...` 挂载的（如 `timeout` 被存成了 `self.max_retries`）。
  - 纠偏 2：误以为 `self.` 等价于全局变量 → 已说明 `self.xxx` 是“对象内共享、不同实例彼此独立”。
  - 纠偏 3：在 [Joy-Con 控制器](../lerobot/common/robot_devices/controllers/joycon_controller.py) 第 75 行 `get_command` 练习中，本人选对 `self.max_retries`，但误以为第 75 行仍在 `__init__` 内且 `timeout` 仍在 → 已按 Python 缩进纠正（`__init__` 在第 73 行已结束，`timeout` 已销毁）。
  - 纠偏 4：在 `pad_left.retries = 5` 是否影响 `pad_right.retries` 一题中答对“不会”，但误以为 `pad_left.retries` 下次调用会丢失、且将 `retries` 默认值记为 `timeout` 的 `60` → 已纠正：挂在 `self.` / 对象上的属性会持久保留，不会丢失；`self.retries` 初始值为 `0`（`self.max_retries` 才是 `60`）。
  - 巩固核对通过：本人准确判断普通局部变量 `x` 函数结束后消失、`self.target_gpos_last` 仍在，且 `pad_left` 与 `pad_right` 互不干扰（仅再次把 `pad_right.retries` 默认值写成了 60，已提醒区分 `retries=0` 与 `max_retries=60`）。

- **第 4 项（左右 Joy-Con 选择）完成**：
  - 预测仅改[单臂配置](../lerobot/configs/robot/so100_joycon_single.yaml)第 21 行为 `name: left` 时，本人正确答出蓝牙连接 `left`，但误判[机械臂控制循环](../lerobot/common/robot_devices/robots/manipulator.py)第 540 行循环会拿到 `left` 并报错找不到 `right`。
  - 纠偏后核对通过：用“储物柜外层门牌号（字典键）”与“柜子里的对象参数”讲解后，本人准确总结“去找机械臂只会读取外层键 `right:`，而不是内层的 `name="right"`”。第 4 项已完成。

- **第 5 项（`_target_` 与 Python 类）完成**：
  - 本人正确拆出 `ManipulatorRobot`（[manipulator.py](../lerobot/common/robot_devices/robots/manipulator.py)）和 `FeetechMotorsBus`（[feetech.py](../lerobot/common/robot_devices/motors/feetech.py)）的文件、类名与参数。
  - 疑问与补充：本人在仓库中未找到 Hydra → 已说明 Hydra 是外部第三方库，通过 [control_robot.py](../lerobot/scripts/control_robot.py) 第 568–569 行和 [factory.py](../lerobot/common/robot_devices/robots/factory.py) 的 `hydra.utils.instantiate(cfg)` 递归实例化对象（顶层 `ManipulatorRobot` 也会接收 `controllers`、`follower_arms`、`cameras` 等同级配置）。

- **路线图第 2 步（[control_robot.py](../lerobot/scripts/control_robot.py) → `teleoperate()`）完成**：
  - 作答与纠偏：本人正确找出 `--robot-path` 默认值为 `lerobot/configs/robot/koch.yaml`，并判断不传该参数会选错机械臂、`target` 和 `port`；但在预测 `teleoperate --fps 30` 的执行路径时，误判会进入 `calibrate` 且以为只调用了 `robot.connect` → 已对照第 571–576 行 `if / elif` 分流和第 176–185 行纠正，命令会进入 `teleoperate()` 并调用 `control_loop()`。
  - 巩固核对通过：本人准确答出录制数据集应使用子命令 `record`（走 `elif control_mode == "record": record(robot, **kwargs)`），并定位到 `control_loop` 来自 [control_utils.py](../lerobot/common/robot_devices/control_utils.py)。

### 2026-11-21 · 完成第 3–6 步：控制循环、单帧动作、数据保存、ACT 训练与真机推理

- **路线图第 3 步（[control_utils.py](../lerobot/common/robot_devices/control_utils.py) → `control_loop()`）完成**：
  - 作答与纠偏：纯遥操作场景下，本人正确答出走 `if teleoperate:`、不会执行第 276 行存数据分支，且算出 `fps=30`、耗时 `0.01s` 时 `busy_wait` 约等 `0.023s` 以对齐动作与画面；小纠偏是将第 276 行不执行的原因看成了第 236 行 `policy=None` → 已纠正对应第 234 行 `dataset=None`。
  - 延伸提问：询问机械臂时间如何读取 → 已说明舵机本身无时钟，统一由主机通过 `time.perf_counter()` 计时。

- **路线图第 4 步（[manipulator.py](../lerobot/common/robot_devices/robots/manipulator.py) → `teleop_step()`）完成**：
  - 作答通过：本人准确识别 `Present_Load[-1]` 与 `goal_pos[-1]` 对应 6 号夹爪舵机 `gripper`，并结合 `feedback = self.follower_arms[name].read("Present_Position")` 理解过载保护（超载先停住卸力，下一帧再尝试往目标移动）。
  - 延伸追问：本人敏锐指出该机制无法精准控力且可能引起波动和抖动 → 已确认判断正确，STS3215 在位置控制模式（`Mode=0`）下仅用负载阈值作低成本防烧保护，并靠调低 `P_Coefficient=16` 缓解。

- **路线图第 5 步（数据采集与保存：`observation`、`action`、`add_frame`、`save_episode`）完成**：
  - 作答通过：对照 [control_robot.py](../lerobot/scripts/control_robot.py) 第 318–326 行和 [lerobot_dataset.py](../lerobot/common/datasets/lerobot_dataset.py)，本人准确判断重录分支里的 `continue` 会跳过 `save_episode` 与 `recorded_episodes += 1`、重新开始循环；准确答出真机推理时采集 `observation` 喂给 AI、AI 输出 `action`。

- **路线图第 6 步（[train.py](../lerobot/scripts/train.py)、ACT 策略与真机推理）完成**：
  - ACT 与训练配置作答通过：对照 [act_so100_real_single.yaml](../lerobot/configs/policy/act_so100_real_single.yaml)，本人准确读出默认训练 `2000` 步、每 `1000` 步存一次（在第 1000 和 2000 步存档），并正确指出 `[3, 480, 640]` 中 `480`/`640` 为高和宽、相机名 `right` 需与单臂配置键匹配，否则会找不到对应数据报错。
  - 真机推理作答与纠偏：本人正确定位到 [control_robot.py](../lerobot/scripts/control_robot.py) 第 226–227 行 `init_policy(...)` 加载模型，并正确答出 `/ 255` 将像素归一到 `0~1` 强度、`.permute(2, 0, 1)` 后与 `[3, 480, 640]` 对齐；两处小纠偏：① 实际把 `control_loop` 从人控切到 AI 控的开关是下游 [control_utils.py](../lerobot/common/robot_devices/control_utils.py) 第 224 行 `teleoperate=policy is None`；② 形状最前面的 `3` 表示 RGB 三个颜色通道数，而不是像素值大小。

## 后续阅读顺序
1. [已完成] so100_joycon_single.yaml：设备如何配置。
2. [已完成] control_robot.py → teleoperate()：命令从哪里进入。
3. [已完成] control_utils.py → control_loop()：循环执行什么。
4. [已完成] manipulator.py → teleop_step()：一次动作如何完成。
5. [已完成] 数据采集与保存：observation、action、add_frame、save_episode。
6. [已完成] 最后学习 train.py、ACT 策略和真机推理。

## 希望的指导方式
- 一次只讲一个小目标，先解释概念，再结合实际代码。
- 不默认我已经熟悉 Python、深度学习或机器人术语。
- 每次给一个小练习，根据我的反馈继续。
- 基础读值已熟悉时，练习提高到追踪参数、引用源码并预测行为。
- 区分“已经讲过”和“我已经完成”，不要提前标记掌握。
- 未提出修改要求时，以阅读和讲解为主。