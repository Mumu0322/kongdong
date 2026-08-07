"""aubo_workbench: AUBO + Gemini 435Le 手眼标定 / 机械臂信息 / TCP 示教工具集。

模块地图
--------
config.py           所有可调参数（dataclass 单例）
geometry.py          4x4 变换、旋转、位姿格式转换等纯数学函数
io_utils.py          文件系统小工具（建目录、时间戳、矩阵<->list）
camera.py            Gemini 435Le 相机封装（依赖 pyorbbecsdk）
robot.py             AUBO 只读位姿会话（手眼标定专用，依赖 pyaubo_sdk）
charuco_detect.py    ChArUco 检测 + 点云 3D 板位姿拟合
drawing.py           OpenCV 中文绘制、面板/进度条等 UI 基础组件
visualization.py     采集主界面合成（RGB + 深度 + 质量看板）
quality.py           画面质量评分
samples.py           标定样本的持久化（JSON/CSV/归档）
solve.py             手眼求解、非线性精修、冲突诊断、自动剔除
capture.py           单帧/五帧批量采集逻辑
gui_common.py        GUI 通用小工具（日志重定向、手动位姿解析）
gui_handeye.py       手眼标定 GUI 面板 + OpenCV 窗口入口
robot_info.py        AUBO 机械臂信息只读查询（CLI + 库函数）
motion_control.py    AUBO 上下电、点动、点位和复位控制 GUI
tcp_teach.py         TCP 示教工具（4点法/3点法）
charuco_height_error.py ChArUco RGB/深度多高度误差实验与统计报告
workbench.py         机械臂信息/运动控制/TCP示教/手眼标定/伞架孔验证统一工作台窗口（默认入口）

设计要点
--------
1. 所有配置都是模块级单例（见 config.py），GUI 表单直接改这些字段，
   不再有任何地方 `exec(compile(...))` 把整段脚本塞进运行时命名空间。
2. `robot.AuboPoseSession`（只读，用于手眼标定）和
   `tcp_teach.TcpTeachSession`（会写 TCP 偏移）故意保持两个独立的类，
   避免只读采集流程被误用去修改机械臂参数。
3. 采集画面的核心循环（`capture._run_burst_loop`）只写一份，
   OpenCV 窗口版和 Tk GUI 版都调用它，只是展示回调不同。
"""

__all__ = [
    "config", "geometry", "io_utils", "camera", "robot", "charuco_detect",
    "drawing", "visualization", "quality", "samples", "solve", "capture",
    "gui_common", "gui_handeye", "robot_info", "motion_control", "tcp_teach",
    "charuco_height_error", "workbench",
]

__version__ = "2.0.0"
