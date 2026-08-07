# AUBO + Gemini 435Le 工具集（重构版）

对原来 3000+ 行单文件脚本做的整体重构。机械臂信息读取、机械臂运动控制、
TCP 示教和手眼标定模块均保留。当前正式手眼链路使用“RGB ChArUco二维角点 +
RGB内参 + solvePnP”，求得 `T_tcp_rgb_camera`，与YOLO/椭圆2D孔中心使用同一RGB
光学坐标系；旧点云手眼只保留为归档诊断。手眼流程仍拆成“≥8组诊断求解”和
“≥11组E7独立验证”，诊断结果不能直接用于运动。

## 目录结构

```
aubo_workbench_project/
├── run_workbench.py      # 默认入口：工作台（机械臂信息/运动控制/TCP示教/手眼标定）
├── run_cad_registration_preview.py  # CAD 配准 Stage 0：只预览，不连接机器人
├── run_yolo_eye_in_hand_optimized.py # CAD运动/旧点云定位主脚本
├── run_hole_localization_pycharm.py  # 直接Run的CAD运动路径配置入口
├── run_handeye.py         # 只启动手眼标定（Tk GUI 或 --opencv-ui 旧版窗口）
├── run_robot_info.py      # 命令行只读查询机械臂信息
└── aubo_workbench/        # 核心包
    ├── config.py           # 所有可调参数（dataclass 单例，集中管理）
    ├── geometry.py         # 4x4 变换、旋转、位姿格式转换（纯数学，无副作用）
    ├── cad_model.py        # STEP 顶面孔提取与 cad_hole_model.json 校验
    ├── cad_registration.py # CAD↔RGB 匹配、平面PnP、质量门和叠加报告
    ├── io_utils.py         # 文件系统小工具
    ├── camera.py           # Gemini 435Le 相机封装
    ├── robot.py            # AUBO 只读位姿会话（手眼标定专用）
    ├── charuco_detect.py   # ChArUco检测 + 正式RGB-PnP板位姿 + 旧点云诊断
    ├── drawing.py          # OpenCV 中文绘制、面板等 UI 基础组件
    ├── visualization.py    # 采集主界面合成（RGB+深度+质量看板）
    ├── quality.py          # 画面质量评分
    ├── samples.py          # 标定样本持久化（JSON/CSV/归档）
    ├── solve.py            # RGB/旧点云坐标源隔离的诊断求解与冲突分析
    ├── e7_handeye.py       # RGB手眼E7预先留出验证、原始数据清单和候选输出
    ├── capture.py          # 单帧/五帧批量采集
    ├── gui_common.py       # GUI 通用小工具
    ├── gui_handeye.py      # 手眼标定 GUI + OpenCV 窗口入口
    ├── robot_info.py       # 机械臂信息只读查询（CLI + 库函数）
    ├── motion_control.py   # 机械臂上下电、点动、点位和复位控制
    ├── tcp_teach.py        # TCP 示教工具（4点法/3点法）
    └── workbench.py        # 工作台主窗口
```

TCP 示教页会分别显示机器人基坐标系下的当前法兰和 TCP `XYZ (mm)`，并保留
SDK 原始的 `m/rad` 六维位姿用于核查。AUBO `tcpOffsetIdentify()` 四点法返回
TCP 相对法兰的平移 `XYZ` 三项；程序会保留当前 TCP 的姿态三项，组成可继续
做姿态标定的完整六维偏移。计算后还会单独显示多姿态共同接触点在机器人
基坐标系下的 `XYZ (mm)`，避免把法兰系 TCP 偏移误当成基坐标位置。

## 这次做了什么

### 1. 结构 / 可维护性
- **去掉了 `exec(compile(embedded_source, ...))` 的内嵌脚本反模式。**
  原文件把"机械臂信息读取 CLI"和"TCP 示教 GUI"整段整段地写成字符串，
  在运行时 `exec` 成两个隐藏模块（`_embedded_aubo_robot_info` /
  `_embedded_tcp_teach_gui`），只是为了塞进同一个 .py 文件里还不冲突命名。
  现在这两块就是普通模块（`robot_info.py` / `tcp_teach.py`），可以正常
  `import`、正常被 IDE 跳转、正常被单独测试，不再需要这种 hack。
- **按职责拆成 17 个模块**（数学 / 相机 / 机械臂 / 检测 / 求解 / 采集 / GUI
  各自独立），原来 3000 行找一个函数要靠搜索，现在文件名基本就能定位。
- **采集循环去重**：原来 OpenCV 窗口版和 GUI 按钮版各写了一遍几乎一样的
  "连续抓 5 帧再选 1 帧"逻辑（`capture_burst_samples` /
  `capture_burst_samples_gui`），现在两者共享同一个 `capture._run_burst_loop`，
  只是把"怎么把画面显示出去"换成了不同的回调，以后改采集逻辑只用改一处。
- **配置依旧是模块级单例**（`config.py` 里的 `BOARD_CFG` / `CAMERA_CFG` /
  `ROBOT_CFG` / ...），这是刻意保留的：GUI 表单改 IP、改保存路径都是直接
  写这些字段，全局单例是最省事的方式；但现在它们集中定义在一个文件里，
  不再散落。

### 2. 健壮性 / 安全边界
- **两个"AUBO 会话"类刻意不合并**：`robot.AuboPoseSession` 只读 TCP/Tool
  位姿，绝不下发指令，用于手眼标定；`tcp_teach.TcpTeachSession` 会调用
  `config.setTcpOffset()` 真正修改机械臂参数。原文件里两者名字都叫
  `AuboSession`，靠内嵌 exec 的隔离命名空间才没有互相覆盖——这是很容易在
  后续维护中踩坑的地方，现在是两个类型不同、导入路径不同的类，不可能混用。
- CSV 字段列表、JSON 序列化等原来在多处重复定义的常量（如
  `_CSV_FIELDS`）现在只定义一次。
- **诊断与正式验证分离**：8组样本只允许生成
  `handeye_diagnostic_current.json`；E7要求至少11组、至少20%且不少于3组预先留出，
  留出集编号不参与拟合，也不按残差挑选。
- **采集记录可追溯**：五帧采集会记录相机序列号、profile、RGB/Depth时间戳、
  标定板视野区域，以及每张相机帧前后的TCP读数。缺少这些字段的旧样本仍可诊断，
  但不能通过E7。
- **RGB坐标链统一**：正式样本保存 `T_rgb_board`、PnP内点数和像素重投影误差；
  E7只接受 `calibration_frame=rgb_camera`，并输出 `T_tcp_rgb_camera`。深度图仍可显示
  和保存用于现场核查，但不参与正式手眼求解。
- **旧样本隔离**：工作台启动RGB手眼页时，旧 `pointcloud` 样本会移动到
  `archived_legacy_pointcloud_*`，不会与新RGB样本混合拟合或覆盖当前结论。
- **文件状态同步**：“归档最后样本”会把JSON和图像移出活动目录并重写CSV，程序重启后
  不会把已移除样本重新加载。诊断结果和E7候选均采用唯一current文件原子替换，不生成重复副本。

### 3. 已验证正确性
用合成数据做了端到端回归（见下方“如何验证”），确认几何变换、
`cv2.calibrateHandEye` 五种方法、非线性精修、样本冲突诊断、
E7预先留出和JSON输出正常：
- 无噪声合成数据 → 求解出的 `T_tcp_rgb_camera` 与真值误差在浮点精度量级
  （浮点精度极限），验证了 Kabsch/RANSAC/hand-eye 数学没有在重构中被改动。
- 加噪声合成数据端到端跑通 `solve_and_save`，JSON 正常写出，冲突诊断和
  目标达标判断都按预期工作。
- 30组无噪声合成数据按固定规则拆成24组标定、6组验证；验证集不参与拟合，
  求解矩阵恢复真值，候选仍保持 `validated=false` 和 `do_not_use_for_motion=true`。
- 合成RGB ChArUco角点在完全不提供深度图/点云的情况下通过PnP恢复板位姿，
  验证正式算法没有暗中依赖435Le深度。

### 4. 现场仍需验证的点
- 当前RGB内参来自435Le SDK当前profile。正式E7前必须确认分辨率、畸变参数和安装后
  对焦状态保持一致；如改用离线高精度内参，必须重新采集全部RGB手眼样本。
- PnP重投影门当前为RMSE `<=0.35 px`、最大误差 `<=1.00 px`。这些是保守软件门，
  不能替代真实装机后的E7独立验证和最终E9/E10精度试验。
- `CAMERA_CFG.save_dir` / `SOLVE_CFG.output_json` 默认值仍是写死的 Windows
  路径 `C:\MM\...`，跟原脚本一致；如果你想换成相对路径或者从环境变量读，
  告诉我可以再改。

## 如何运行

### CAD 配准预览（Stage 0）

启动文件是：

`C:\MM\aubo_tools\aubo_workbench_project\run_cad_registration_preview.py`

它只做 RGB/CAD 预览，不创建 `AuboMotionSession`；现场采集模式会通过现有的
`AuboPoseSession` 只读获取 TCP 位姿，不读取深度或点云。即使质量门通过，报告中的
`motion_allowed` 也固定为 `false`。运行前可以提供近似 `T_camera_cad`（CAD 坐标系→RGB
相机坐标系），也可以在现场第一帧人工确认至少 4 组 `detection_id → CAD-xx` 映射；
不能让程序在对称孔位之间任意选择。

离线回放时，把 1 张或 5 张 RGB 图像作为 `--image` 传入；结果写到
`C:\MM\aubo_tools\data\cad_registration_runs\<时间戳>`，包括原图、叠加图、CSV/JSON
映射表和 `cad_registration_report.json`。没有 `--image` 时，脚本连接 AUBO 只读 SDK 和
Gemini Color 流，默认采集 5 帧；窗口中按 `C/空格` 拍摄、`S/Enter` 保存、`R` 重拍、
`Q` 退出。采集期间请保持机械臂静止，程序不会发送运动指令。

### CAD 运动路径（Stage 1，当前新增）

直接运行：

`C:\MM\aubo_tools\aubo_workbench_project\run_hole_localization_pycharm.py`

这个入口默认打开 CAD 运动路径的预览模式，不发送运动指令。每次 CAD 流程启动时，程序会在
当前位置连接 AUBO 只读位姿和 Gemini RGB 流，自动采集 5 帧并重新计算 CAD 配准；旧报告中的
位姿不会直接复用，自动配准失败或出现歧义时禁止继续。通过质量门后画面中会把 11 个 CAD 孔投影出来。
鼠标点击需要处理的 CAD 孔，按 `Enter` 或空格确认，
然后报告中写入每个目标的 CAD 基坐标中心、法向、固定 RZ 和 260 mm TCP 位姿。

真实运动路径已经实现，但默认不会开启。生产运动必须使用已验证手眼；如果只是现场
实验，必须在该入口明确把 `ALLOW_EXPERIMENTAL_HANDEYE`、`EXECUTE_MOTION` 和
`I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE` 都改为 `True`。实验运行会明确标记为
`completed_experimental_handeye`，不能作为生产精度结论。当前
`e7_handeye_candidate_current.json` 尚未通过生产手眼质量门，不能按生产模式运行。

Stage 1 的运动顺序是：当前位置自动 CAD 配准 → 回原点选择孔组 → 340 mm 只采集共享高度深度
→ CAD 孔中心/法向规划并安全移动到 260 mm → RGB/YOLO 多帧中心只做最终 XY 修正
→ 保留最终 Z、机械爪 X/Z 偏置和基坐标 `+Y 0.3 mm`。340 mm 深度只提供高度偏差，
不再执行点云中心/法向粗定位。结果写入
`C:\MM\aubo_tools\data\cad_motion_runs\cad-motion-<时间戳>`。

STEP 解析依赖当前 `lip_env310` 环境中的 `cadquery-ocp`。脚本会从
`孔位板_JXDZ26-KWB-001.STEP` 生成并校验
`C:\MM\aubo_tools\data\cad_model\cad_hole_model.json`；不会在解析失败时生成假数据。

```bash
# CAD 配准预览：STEP + 5 张 RGB 图像（不运动）
python C:\MM\aubo_tools\aubo_workbench_project\run_cad_registration_preview.py \
  --step "C:\MM\aubo_tools\aubo_workbench_project\孔位板_JXDZ26-KWB-001.STEP" \
  --image frame_01.png --image frame_02.png --image frame_03.png --image frame_04.png --image frame_05.png \
  --prior-camera-cad approximate_T_camera_cad.json \
  --base-camera approximate_T_base_camera.json

# 现场采集：不传 --image；启动后按窗口提示用键盘拍摄 5 帧
python C:\MM\aubo_tools\aubo_workbench_project\run_cad_registration_preview.py

# 默认：工作台
python run_workbench.py

# 工作台“孔洞定位”页面还提供“偏移容忍度测试（单孔）”按钮，
# 可直接设置半径/方向并运行下面的偏移测试，不需要单独打开命令行。

# 只要手眼标定
python run_handeye.py
python run_handeye.py --opencv-ui   # OpenCV窗口：h诊断、v E7验证、d归档最后样本

# 与手眼界面同时运行：逐点移动到40个候选位姿，拍照仍由人工完成
python run_handeye_pose_sequence.py             # 只预览，不连接机器人
python run_handeye_pose_sequence.py --execute   # 实机交互模式，默认P01-P40
python run_handeye_pose_sequence.py --execute --start-index 18  # 从P18恢复

# ChArUco高度×3x3视野实验（默认只预览，不运动）
python run_charuco_height_error_experiment.py

# 现场确认路径安全后：自动控制Z，XY由人工移动，每个网格点采20帧
python run_charuco_height_error_experiment.py --execute-motion

# 单高度3x3网格冒烟测试，每位置10帧
python run_charuco_height_error_experiment.py --heights-mm 340 --frames-per-position 10 --execute-motion

# 如需恢复旧的纯高度实验
python run_charuco_height_error_experiment.py --height-only --execute-motion

# 命令行查看机械臂信息
python run_robot_info.py --ip 192.168.50.200 --port 30004

# 粗定位后精定位的视野偏移容忍度测试（默认只生成计划，不运动）
python run_coarse_to_fine_offset_test.py --no-execute

# 实机测试：只选择一个孔；不执行最终XY、最终Z和基坐标Y+0.3 mm
python run_coarse_to_fine_offset_test.py --execute --allow-experimental-handeye

# 实机测试：额外执行正式流程的最终XY、降Z、基坐标Y+0.3 mm，到达每个目标点后等待确认
python run_coarse_to_fine_offset_test.py --execute --allow-experimental-handeye --include-final-motion
```

`run_coarse_to_fine_offset_test.py` 按默认的 0/5/10/15/20 mm 五个半径，
每个非零半径按 45° 间隔采 8 个方向，共 33 个位置。每个位置都从精定位中心
重新出发，固定姿态和RZ，只做相机横向偏移，然后运行当前RGB精定位质量门。测试
结果写入 `C:\MM\aubo_tools\data\hole_localization_runs\coarse-to-fine-offset-*`，
包括 `report.json`、`offset_samples.csv` 和极坐标图。只有同一半径的所有方向都通过，
该半径才会被汇总为支持半径。默认不执行最终插入动作；勾选/指定
`--include-final-motion` 后，会按正式顺序执行最终XY、降Z、基坐标Y+0.3 mm，
到达最终目标点后等待人工确认；确认或取消后都会先回升，再返回精定位中心。
GUI 目标点暂停时提供“确认并继续”和“标记当前点有误差并继续”两个选项；
标记结果会写入报告并在极坐标图中以紫色显示，不会再次询问该点。

视野实验固定使用板中心在RGB相机坐标系中的 `T_rgb_board[2,3]` 作为高度：
300/320/340/360 mm，步进20 mm。程序只自动修改TCP的Z，不发送XY运动；每个高度由
操作者依次把板中心移动到3x3目标位置，确认后每个位置采20帧。
程序会先预热并锁定彩色曝光/增益，逐高度人工确认运动，再按20个非重叠10帧批次采集。
输出位于 `C:\MM\aubo_tools\data\charuco_height_error\<run_id>`，包括逐帧CSV、批次/高度
汇总、JSON报告、趋势图、箱线图、通过率图和每批代表图。由于没有外部长度真值，该报告
只评价RGB-PnP、深度和二者交叉差异的内部一致性与重复性，不声称绝对测量精度。

依赖：`opencv-contrib-python`（需要 `cv2.aruco`）、`numpy`、`Pillow`（可选，
没装的话中文会自动退化成 ASCII）、`scipy`（可选，没装则跳过非线性精修）、
`cadquery-ocp`（仅 CAD STEP 解析预览使用）、
`pyaubo_sdk`（当前项目会自动查找 `C:\MM\third_party\aubo_sdk`，
也兼容项目目录下的 `third_party/aubo_sdk`，或直接安装到当前 Python 环境）、
`pyorbbecsdk`。

## 如何验证（不需要真实硬件）

```bash
python -m py_compile aubo_workbench/*.py run_workbench.py run_handeye.py run_robot_info.py
python -m unittest discover -s tests -v
```

以及仓库里 `config` / `geometry` / `io_utils` / `drawing` / `charuco_detect` /
`quality` / `camera` / `samples` / `solve` / `visualization` 这些不依赖
AUBO/Orbbec 硬件 SDK 的模块，都可以直接 `import` 跑单测——这也是这次重构
特意让它们不依赖硬件 SDK 的原因之一（原文件里所有函数都挤在一个 import
了 `pyaubo_sdk` 的文件里，哪怕只想测一下坐标变换，也得先能导入机械臂 SDK）。
