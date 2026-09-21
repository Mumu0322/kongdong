# AUBO Workbench：两阶段孔洞定位工具

当前项目使用 AUBO 机械臂、Orbbec Gemini 435Le、YOLO 和 RGB-D 完成孔洞定位、在手标定、TCP 示教以及夹爪控制。

自动流程的“原点”优先读取 AUBO 控制器中保存的 `getHomePosition()` 关节原点；
`data/aubo_home_point.json` 只作为控制器读取失败时的软件备用点。两者不是同一份数据，
修改示教器原点后无需把旧 JSON 当作新的控制器原点，但应在启动定位前确认控制器原点和当前 TCP 配置一致。

## 手眼标定

工作台手眼页保留采集、求解、独立验证和手动归档样本。旧 `--opencv-ui` 界面、
自动按残差删点、贪心试删子集和空深度看板已移除。历史采集文件与归档保留。

位姿自动读取控制器当前生效的 TCP（`getTcpPose`，相对机器人基坐标系），
同时读取实际 TCP 偏置 `getActualTcpOffset`，并校验法兰位姿乘偏置是否与 TCP 一致。
界面显示实时 XYZ / RxRyRz 和实际偏置，无需选择 `tcp/tool` 或填写工具名称。
SDK 未提供示教器工具坐标名称的读取接口；请用显示的数值核对当前生效配置。
读取失败时拒绝采集，不回退到手动位姿或法兰。采集中切换 TCP 会被拒绝；
更换 TCP 后需要使用新的采集会话。

“求解标定”使用固定拟合集计算矩阵，留出组不参与拟合。结果区直接显示结论、
两组各自的平移 RMS / 最大误差和下一步。数值目标统一为 RMS ≤ 0.10 mm、
最大误差 ≤ 0.20 mm；样本不足时显示“待验证”，不会把低拟合残差当作验证通过。
结果是标定板位姿一致性，不代表机械臂绝对定位精度。数值达标后仍需完整独立验证。

旋转均值使用矩阵 SVD 投影，避免 ±180° 处旋转向量平均错误。固定性预检检查
相对均值的平移/旋转 RMS 和最大值；不再使用误差长度的标准差。

诊断输出仍为 `aubo_tools/data/handeye_diagnostic_current.json`，保留矩阵、
样本分组和各算法数值供追溯，但不安装为生产手眼文件。

## 孔洞定位流程

当前保留的定位流程为 `run_yolo_eye_in_hand_optimized.py` 中的两阶段流程。初始选孔既可手动点击，也可在固定观察位显式启用静态自动分区实验：

1. 在初始 RGB-D 画面中由 YOLO 检测孔并手动选择目标孔。
2. 机器人到约 340 mm 位置，使用孔口外侧深度环带拟合局部平面、中心和法向。
3. 可选地复用经过现场验证的粗定位缓存；缓存不兼容或验证失败时自动回退现场粗定位。
4. 机器人到约 260 mm，仅使用 RGB/YOLO 多帧精定位，并锁定在粗定位点云锚点附近。
5. 将去畸变像素中心与粗平面求交，进行倾斜圆心修正和最终 TCP 目标规划。

### 第四策略：340 mm纯点云中心直达

选择第四策略后，程序把独立点云观察高度默认设为340 mm，并先按“当前选中区域”的局部边界
识别最外围孔洞。外围孔洞先分组，每组最多3个；外围完成后才处理内部孔洞，内部每组最多5个，
两阶段都优先形成3～5孔组；余数或共同视野/几何质量不允许时，允许1～2孔，不为了凑数放宽视野和质量门。
外围组同样执行长宽比和紧凑度门限；近似直线的三孔会拆成更小的组，避免共同点云覆盖不完整。
边界识别按选中孔中心的局部几何计算，不使用托盘外框或图像边缘；断开的选区和凹形边界不会跨区凑组。
机械臂到位后先确认停稳，再默认额外等待1秒，随后清理旧帧并采集点云。
每组默认采集15帧、每孔至少10帧有效数据；点云质量不足的孔直接记录失败，不转入260 mm精定位或历史缓存回退。
正常模式以点云中心规划最终点，最终XY只记录现有 ChArUco 补偿；勾选“仅采集评估”时只移动到340 mm观察位并保存测量，
不执行最终 XY/Z或夹爪动作。分组搜索、停稳等待、清理旧帧和采集均有超时，进度会写入 `progress.json`。

命令行等价参数如下：

```powershell
python run_yolo_eye_in_hand_optimized.py `
  --coarse-direct-final `
  --coarse-direct-final-height-mm 340 `
  --coarse-direct-final-max-group-size 5 `
  --coarse-direct-final-early-stop-extra-frames 5 `
  --coarse-direct-final-settle-delay-s 1 `
  --coarse-direct-final-capture-only `
  --execute
```

完成多轮 A/B/C 试拍后，可用独立工具汇总报告；该工具只读 `report.json`，不会改写孔位地图：

```powershell
python tools/summarize_coarse_direct_experiment.py `
  --label A=C:\runs\A_340mm_5holes `
  --label B=C:\runs\B_340mm_3holes `
  --label C=C:\runs\C_340mm_1hole `
  --output-dir C:\runs\coarse_experiment_summary
```

### 单臂静止伞架自动分区实验

当前推荐使用鼠标画工作区域，无需提供伞架中心。在 GUI“检测与地图”点击“鼠标画区 / 修改区域”，启动持续 RGB 视频流。选择区域编号，把机械臂调整到该观察位后点击“确认当前位姿并取图”，程序会锁定最新视频帧并只读保存 TCP 位姿和关节角；随后左键逐点画轮廓，右键撤销一点，点击“闭合并保存本区域”；换编号可画多个区域，最后点击“保存配置并使用”。程序自动保存 `partition_mode=polygons` 配置并启用自动选孔。按钮“打开已有图像（离线）”只用于无相机时的离线画区验证，不能产生本次机械臂位姿记录。旧的中心角度模式仍兼容，但下面的 `origin_px` 要求仅适用于旧模式。

画区模式以孔中心判定归属：边线和顶点算区域内；重叠或公共边固定归编号最小的区域，再过滤活动扇区，不产生待确认状态。所有画区的并集就是工作范围，范围外检测记为 `outside_drawn_regions`。这种规则只能覆盖已检测到的孔，不能证明视觉没有漏检。相机与伞架位置、图像尺寸必须与画区时一致；程序会拒绝分辨率不一致的配置，观察位改变需要重新画区。

画区 JSON 与 `.reference.png` 原图保存在扇区信息目录；各 `Sxx` 归档记录实际多边形及归属规则，不再为手画工作区域记录60°角度范围。确认时的帧图保存在对应 `Sxx/captures`，位姿记录保存在 `auto_sector_selection_pose_records.json`，同时写入主配置的 `sector_pose_records`。从工作台打开的编辑器要求每个已画区域都有对应的确认记录后才能保存；离线按钮仅用于无硬件的逻辑验证。

当前位姿记录用于区域复现、绑定和审计；自动选孔仍按当前输入图像的像素多边形执行，运行某个扇区前要把机械臂置于该扇区对应的观察位。代码目前没有把不同观察位的像素多边形自动变换到统一工件坐标。

自动分支只在初始 RGB-D 画面做候选孔筛选、扇区归属、同帧去重和审计输出；初始画面不生成粗定位地图。普通策略仍单独执行340 mm粗定位和260 mm精定位；选择第四策略时改用上述340 mm纯点云流程。先在 GUI 完成视频流、位姿确认和画区，生成 `data/hole_localization_sector_info/auto_sector_selection.json`，再运行：

```powershell
python run_yolo_eye_in_hand_optimized.py `
  --auto-select-holes `
  --auto-sector-config data/hole_localization_sector_info/auto_sector_selection.json `
  --no-execute
```

只想用一张RGB图验证扇区线和选孔结果时，使用 `run_auto_sector_selection.py`；它不连接机器人、不读取深度，会在输出目录写入叠加图和 `auto_sector_selection.json`。自动两阶段每轮也会在运行目录写入同名报告和 `01_home_auto_sector_selection.png`。未配置参考孔位表时，报告中的孔号为临时 `Sxx-Pxxx`，不能当作跨次稳定孔号。扇区边界候选会保留在报告和叠加图中，但必须先经粗定位或补拍确认，不会直接进入机器人执行列表；去重还会检查检测框重叠和尺度，避免相邻孔仅因中心距离较近而被合并。

扇区信息另外归档到 `data/hole_localization_sector_info`（可用环境变量 `AUBO_WORKBENCH_SECTOR_INFO_DIR` 修改）。目录下固定建立 `S01` 到 `S06`，每个扇区保存 `sector_definition.json`、`latest.json` 和按运行/预览编号保存的快照；根目录的 `index.json` 记录最近一次更新和各扇区路径。

GUI 首次打开时会把 `configs/auto_sector_selection_static_template.json` 复制为 `data/hole_localization_sector_info/auto_sector_selection.json`，并自动填入配置路径。推荐直接点击“鼠标画区 / 修改区域”完成视频取帧、位姿记录和多边形绘制；模板中的 `origin_px` 只在旧的 `radial` 角度模式中使用。

如果同时使用六扇区地图建图，自动配置和 `--sector-id` 必须只指向同一个扇区；普通两阶段实验可以选择多个活动扇区。

### 340 mm粗定位建图与实时精定位调用

建图阶段只把本轮340 mm粗定位的孔中心、平面、法向和质量写入独立地图，不执行260 mm精定位，也不执行最终安放：

多孔不会被强制塞进同一个相机视野：340 mm粗定位和260 mm精定位都会按孔位投影范围自动拆组，每组在本组综合位置上方进行稳定连拍；某组失败不会取消其他组。

```powershell
python run_yolo_eye_in_hand_optimized.py `
  --batch-coarse-localization `
  --no-batch-fine-localization `
  --hole-map-mode build `
  --execute `
  --allow-experimental-handeye
```

不指定 `--hole-map-path` 时，地图会自动保存到 `aubo_tools/data/hole_localization_maps/hole-map-时间/hole_map.json`；普通地图只有全部选定孔通过质量门时才更新 `current.json`，六扇区地图会保留已完成扇区并按版本更新入口。调用时不指定地图路径即可自动调用当前地图；也可以按孔号调用历史版本。调用过程会重新启动相机和YOLO，在当前伞架状态下执行260 mm精定位；需要真实下发最终目标时再加 `--move-final-xy`：

```powershell
python run_yolo_eye_in_hand_optimized.py `
  --hole-map-mode execute `
  --hole-ids 3 1 2 `
  --execute
```

地图是当前工件/机器人循环内的粗定位导航结果；工件移动、重新装夹、TCP或标定改变后应重新建图。地图不保存精定位 XY、最终孔心、补偿或TCP目标；这些数据只在每次调用的当前运行报告中产生。

建图版本目录还会保存共享粗定位点云的 `pointcloud_raw.npz`、基坐标系 `pointcloud_base.ply`、孔中心标记 `hole_centers_base.ply` 和 `pointcloud_preview.jpg`。GUI 中的“查看点云”可打开二维投影预览，“打开三维PLY”可交给 Open3D 或 CloudCompare 检查点云、粗定位中心和孔位覆盖情况。

默认运行只生成预览报告；真实运动必须显式启用运动开关并通过手眼质量门。

定位流程连续3次取不到有效相机帧（包含预热丢帧）会终止本轮，并在报告中记录
`failure_type=camera_stream_unavailable`，不会再按孔失败进入后续分组或补拍。
在SDK每次取帧3秒超时的情况下，持续无帧通常约9秒后报错；这不包含设备关闭耗时。
机器人未上电、运动下发失败或到位等待超时也会终止本轮，记录
`failure_type=robot_motion_failed`。GUI显示具体故障；恢复相机/机器人状态后重新启动任务。
软件的及时终止不能修复网口相机本身的断流，仍需结合设备连接和SDK日志排查。

## 入口

- `run_workbench.py`：主 GUI。
- `run_yolo_eye_in_hand_optimized.py`：两阶段孔洞定位核心。
- `run_auto_sector_selection.py`：静止伞架单帧自动分区/选孔离线预览。
- `tools/summarize_coarse_direct_experiment.py`：汇总第四策略多轮点云报告，不写回地图。
- `run_hole_localization_pycharm.py`：IDE 直接运行配置入口。
- `run_coarse_to_fine_offset_test.py`：单孔偏移诊断入口。

GUI 中的孔洞定位页面提供模型、手眼文件、粗/精定位参数、缓存开关、批量粗定位、结果查看和偏移测试。

## 主要模块

- `aubo_workbench/optics.py`：去畸变、相机光线、平面求交和倾斜圆心修正。
- `aubo_workbench/fitting.py`：平面/球面拟合。
- `aubo_workbench/coarse_cache.py`：粗定位缓存建立、兼容性检查和现场验证。
- `aubo_workbench/hole_map.py`：340 mm粗定位地图的生成、版本指针、读取和结构校验。
- `aubo_workbench/hole_map_visualization.py`：地图点云归档、PLY和JPG诊断产物。
- `aubo_workbench/auto_sector_selection.py`：静态图像坐标系中的扇区归属、候选筛选、去重和完整性审计。
- `aubo_workbench/camera.py`：RGB、RGB-D 和点云帧采集。
- `aubo_workbench/geometry.py`：坐标变换和位姿计算。
- `aubo_workbench/motion_control.py`、`robot.py`：机器人运动和只读位姿会话。

## 数据与安全

运行报告位于 `aubo_tools/data/hole_localization_runs`，孔位地图位于 `aubo_tools/data/hole_localization_maps`，粗定位持久化缓存位于 `aubo_tools/data/hole_localization_coarse_cache`。这些目录是运行数据，不应作为源码批量提交。

当前手眼候选和 ChArUco 补偿模型仍需结合独立真值完成生产验收。离线测试不能替代真实相机、机器人和工艺安全验证。

## 测试

在项目目录运行：

```powershell
python -m pytest tests -q
```

真实硬件测试前，先使用预览模式核对报告、目标位姿、缓存验证结果和运动安全门。
