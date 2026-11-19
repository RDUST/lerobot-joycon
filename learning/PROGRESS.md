# LeRobot 学习进度

## 我的基础与目标
- 编程项目新手，刚学会 Git，Python 基础尚需确认和补充。
- 学习 SO-ARM100 + JoyCon 的控制、数据采集、训练与部署。
- 计划本地采集数据，使用 Colab GPU 训练，再下载模型到本地运行。

## 项目信息
- 仓库：https://github.com/RDUST/lerobot-joycon
- 分支：main
- 云开发环境已配置并发布，不需要重新配置。
- 发布前验证：103 项测试通过、9 项跳过；CPU 模型、仿真和视频处理验证通过。
- 真实机械臂操作和 Colab 训练尚未验证。

## 已介绍的内容
- 整体流程：遥操作采集 → 数据集 → 模型训练 → 真机推理。
- 代码架构：配置、程序入口、控制循环、机器人接口、硬件驱动、
  JoyCon 运动学、数据集、策略模型。
- 尚未完成逐行代码学习，也未反馈完成练习。

## 当前学习位置
从配置文件开始：
lerobot/configs/robot/so100_joycon_single.yaml

第一个练习：
1. 找到机械臂串口。
2. 找到六个电机的名称和 ID。
3. 找到相机编号、分辨率、帧率。
4. 确认使用左还是右 JoyCon。
5. 理解 _target_ 对应哪个 Python 类。

## 后续阅读顺序
1. so100_joycon_single.yaml：设备如何配置。
2. control_robot.py → teleoperate()：命令从哪里进入。
3. control_utils.py → control_loop()：循环执行什么。
4. manipulator.py → teleop_step()：一次动作如何完成。
5. 数据采集与保存：observation、action、add_frame、save_episode。
6. 最后学习 train.py、ACT 策略和真机推理。

## 希望的指导方式
- 一次只讲一个小目标，先解释概念，再结合实际代码。
- 不默认我已经熟悉 Python、深度学习或机器人术语。
- 每次给一个小练习，根据我的反馈继续。
- 区分“已经讲过”和“我已经完成”，不要提前标记掌握。
- 未提出修改要求时，以阅读和讲解为主。