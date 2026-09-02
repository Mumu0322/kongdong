# AUBO Workbench：两阶段孔洞定位工具

当前项目使用 AUBO 机械臂、Orbbec Gemini 435Le、YOLO 和 RGB-D 完成孔洞定位、在手标定、TCP 示教以及夹爪控制。

## 孔洞定位流程

当前保留的定位流程为 `run_yolo_eye_in_hand_optimized.py` 中的两阶段流程：

1. 在初始 RGB-D 画面中由 YOLO 检测孔并手动选择目标孔。
2. 机器人到约 340 mm 位置，使用孔口外侧深度环带拟合局部平面、中心和法向。
3. 可选地复用经过现场验证的粗定位缓存；缓存不兼容或验证失败时自动回退现场粗定位。
4. 机器人到约 260 mm，仅使用 RGB/YOLO 多帧精定位，并锁定在粗定位点云锚点附近。
5. 将去畸变像素中心与粗平面求交，进行倾斜圆心修正和最终 TCP 目标规划。

### 多孔一次建图与按孔号调用

在共享粗定位和共享精定位都开启时，可以把本轮所有合格孔保存为独立孔位地图。建图阶段只完成定位，不执行最终安放：

多孔不会被强制塞进同一个相机视野：340 mm粗定位和260 mm精定位都会按孔位投影范围自动拆组，每组在本组综合位置上方进行稳定连拍；某组失败不会取消其他组。

```powershell
python run_yolo_eye_in_hand_optimized.py `
  --batch-coarse-localization `
  --batch-fine-localization `
  --hole-map-mode build `
  --execute `
  --allow-experimental-handeye
```

不指定 `--hole-map-path` 时，地图会自动保存到 `C:\MM\aubo_tools\data\hole_localization_maps\hole-map-时间\hole_map.json`，并且只有全部选定孔通过质量门时才更新 `C:\MM\aubo_tools\data\hole_localization_maps\current.json`。调用时不指定地图路径即可自动调用当前完整地图；也可以按孔号调用历史版本，调用过程不启动相机和 YOLO：

```powershell
python run_yolo_eye_in_hand_optimized.py `
  --hole-map-mode execute `
  --hole-ids 3 1 2 `
  --execute
```

地图是当前工件/机器人循环内的结果；工件移动、重新装夹、TCP或标定改变后应重新建图。地图保存每孔的精定位 XY、粗定位 Z/法向/平面和执行参考姿态，ChArUco 补偿在调用时统一应用一次。

建图版本目录还会保存共享粗定位点云的 `pointcloud_raw.npz`、基坐标系 `pointcloud_base.ply`、孔中心标记 `hole_centers_base.ply` 和 `pointcloud_preview.jpg`。GUI 中的“查看点云”可打开二维投影预览，“打开三维PLY”可交给 Open3D 或 CloudCompare 检查点云、粗定位中心和最终孔位。

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
- `aubo_workbench/hole_map.py`：多孔最终孔位地图的生成、版本指针、读取和结构校验。
- `aubo_workbench/hole_map_visualization.py`：地图点云归档、PLY和JPG诊断产物。
- `aubo_workbench/camera.py`：RGB、RGB-D 和点云帧采集。
- `aubo_workbench/geometry.py`：坐标变换和位姿计算。
- `aubo_workbench/motion_control.py`、`robot.py`：机器人运动和只读位姿会话。

## 数据与安全

运行报告位于 `C:\MM\aubo_tools\data\hole_localization_runs`，孔位地图位于 `C:\MM\aubo_tools\data\hole_localization_maps`，粗定位持久化缓存位于 `C:\MM\aubo_tools\data\hole_localization_coarse_cache`。这些目录是运行数据，不应作为源码批量提交。

当前手眼候选和 ChArUco 补偿模型仍需结合独立真值完成生产验收。离线测试不能替代真实相机、机器人和工艺安全验证。

## 测试

在项目目录运行：

```powershell
python -m pytest tests -q
```

真实硬件测试前，先使用预览模式核对报告、目标位姿、缓存验证结果和运动安全门。
