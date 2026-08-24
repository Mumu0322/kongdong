# AUBO Workbench：两阶段孔洞定位工具

当前项目使用 AUBO 机械臂、Orbbec Gemini 435Le、YOLO 和 RGB-D 完成孔洞定位、在手标定、TCP 示教以及夹爪控制。

## 孔洞定位流程

当前保留的定位流程为 `run_yolo_eye_in_hand_optimized.py` 中的两阶段流程：

1. 在初始 RGB-D 画面中由 YOLO 检测孔并手动选择目标孔。
2. 机器人到约 340 mm 位置，使用孔口外侧深度环带拟合局部平面、中心和法向。
3. 可选地复用经过现场验证的粗定位缓存；缓存不兼容或验证失败时自动回退现场粗定位。
4. 机器人到约 260 mm，仅使用 RGB/YOLO 多帧精定位，并锁定在粗定位点云锚点附近。
5. 将去畸变像素中心与粗平面求交，进行倾斜圆心修正和最终 TCP 目标规划。

默认运行只生成预览报告；真实运动必须显式启用运动开关并通过手眼质量门。

## 入口

- `run_workbench.py`：主 GUI。
- `run_yolo_eye_in_hand_optimized.py`：两阶段孔洞定位核心。
- `run_hole_localization_pycharm.py`：IDE 直接运行配置入口。
- `run_coarse_to_fine_offset_test.py`：单孔偏移诊断入口。

GUI 中的孔洞定位页面提供模型、手眼文件、粗/精定位参数、缓存开关、批量粗定位、结果查看和偏移测试。

## 主要模块

- `aubo_workbench/optics.py`：去畸变、相机光线、平面求交和倾斜圆心修正。
- `aubo_workbench/fitting.py`：平面/球面拟合。
- `aubo_workbench/coarse_cache.py`：粗定位缓存建立、兼容性检查和现场验证。
- `aubo_workbench/camera.py`：RGB、RGB-D 和点云帧采集。
- `aubo_workbench/geometry.py`：坐标变换和位姿计算。
- `aubo_workbench/motion_control.py`、`robot.py`：机器人运动和只读位姿会话。

## 数据与安全

运行报告位于 `C:\MM\aubo_tools\data\hole_localization_runs`，粗定位持久化缓存位于 `C:\MM\aubo_tools\data\hole_localization_coarse_cache`。这些目录是运行数据，不应作为源码批量提交。

当前手眼候选和 ChArUco 补偿模型仍需结合独立真值完成生产验收。离线测试不能替代真实相机、机器人和工艺安全验证。

## 测试

在项目目录运行：

```powershell
python -m pytest tests -q
```

真实硬件测试前，先使用预览模式核对报告、目标位姿、缓存验证结果和运动安全门。
