# AUBO Workbench：CAD 配准孔位定位工具

这是当前 `C:\MM` 项目的主说明。项目使用 AUBO 机械臂 SDK、Orbbec Gemini 435Le 和 YOLO 完成孔位定位、在手标定、CAD 配准运动、TCP 示教以及夹爪控制。

当前项目同时保留两条孔位定位路径。CAD 路径是正在联调的新主线，原来的多段位移/点云两阶段路径仍然保留，可用于兼容、对照和回退：

- STEP/CAD 模型提供孔中心、孔号、孔顶面高度和法向；
- RGB + YOLO 负责建立 CAD 坐标系到相机坐标系的配准；
- 340 mm 只用于选定孔组的一次共享深度采集，深度只修正共同高度；
- 260 mm 使用 RGB/YOLO 做每个孔的最终 XY 精修；
- ChArUco TCP-XY 补偿、最终 Z、工具/夹爪偏置以及基坐标 `+Y 0.3 mm` 保持在最终运动链路中；
- 原两阶段路径仍使用“初始选孔 → 340 mm 粗定位 → 点云中心/法向闭环 → 260 mm RGB 精定位 → 最终目标点”的多段位移流程；
- 当前实验手眼候选尚未通过生产质量门，实验结果不能直接作为生产精度结论。

当前完整阶段评审见：[CAD 配准运动阶段 DR V2.0](docs/reviews/DR_镀膜伞具孔位板_CAD配准运动_阶段评审_V2.0.md)。旧版三孔/点云 DR 已移入 `docs/archive`，仅用于历史追溯。

项目目录约定见工作区根目录的 [`WORKSPACE_MAP.md`](../../WORKSPACE_MAP.md)。源码、测试和当前文档在本目录；运行数据仍在 `C:\MM\aubo_tools\data`；厂商 SDK 和驱动目录不在本次整理范围内。

## 1. 推荐使用方式：主 GUI

推荐从下面的文件启动工作台：

`C:\MM\aubo_tools\aubo_workbench_project\run_workbench.py`

启动后按功能进入：

| GUI 页面 | 用途 |
|---|---|
| 孔洞定位 | CAD 配准预览、CAD 目标预览、340 mm 共享深度、260 mm 精定位和结果查看 |
| 机器人控制 | 上下电、复位、点动和点位运动 |
| 夹爪控制 | 打开独立夹爪窗口；主工作台仍可继续操作 |
| 标定中心 | TCP 示教、RGB ChArUco 手眼标定和验证 |
| 系统信息 | 只读查看机器人和连接状态 |

主 GUI 是日常操作入口。只有需要离线回放、自动化测试或排查底层数据时，才直接运行下面的 Python 文件。

### 1.1 CAD 配准预览

入口：

`C:\MM\aubo_tools\aubo_workbench_project\run_cad_registration_preview.py`

这是阶段 0 预览流程，**不会发送机器人运动指令**。实时模式只读取 AUBO 当前 TCP 位姿和 Gemini RGB 流，不读取深度、不获取点云。主要输出：

- RGB 原图；
- CAD 11 孔投影、绿色 CAD 圆、孔号和 YOLO 框叠加图；
- CAD 与 YOLO 中心残差矢量；
- 映射 CSV/JSON；
- `cad_registration_report.json`。

预览窗口常用操作：

- `C` 或空格：采集一帧；
- `S` 或 Enter：保存当前配准结果；
- `R`：重拍；
- `Q`：退出；
- `M`：进入鼠标映射；按照提示点击检测框和右侧 CAD 圆确认对应关系；
- `A`：尝试自动生成候选映射；
- `P`：复用已经确认且仍然有效的映射。

自动匹配出现对称多解时，必须人工确认或加载已确认映射，程序不允许任意选择一个对称解。

### 1.2 CAD 运动流程

当前 CAD 运动核心在：

`C:\MM\aubo_tools\aubo_workbench_project\run_yolo_eye_in_hand_optimized.py`

直接配置入口在：

`C:\MM\aubo_tools\aubo_workbench_project\run_hole_localization_pycharm.py`

正式使用优先从 GUI 点击 CAD 相关按钮。每次开始一次真实运动前，流程会先连接运动会话并回到保存的原点/观察位，等待机器人稳定后再重新做实时 CAD 配准，策略为 `fresh_live_auto_each_run`，不直接复用上一次报告中的旧位姿。预览模式不自动移动机器人。

安全默认值：主脚本、PyCharm 配置入口和 GUI 默认只预览，不执行真实运动、不启用实验手眼、不执行最终 XY。只有明确传入 `--execute` 或在配置/GUI 中逐项打开对应开关，且通过独立安全门后，才允许下发运动。

运动流程为：

1. 连接机器人并回到保存的原点/观察位；
2. 等待机器人稳定，在原点处采集 5 帧 RGB，完成 CAD 与当前相机画面的自动配准；
3. 根据原点处的 CAD 投影选择本次要处理的一个或多个孔；
4. 根据所选孔的 CAD 中心和法向规划共同视野位置，使所选孔尽可能处于相机视野中心；
5. 在共同视野位置约 340 mm 采集一次 RGB-D，用于得到孔组共享高度偏移；
6. 逐孔移动到 CAD 规划的约 260 mm 精定位位姿；
7. 丢弃稳定等待帧后采集 RGB/YOLO，使用 CAD 投影附近的检测计算最终 XY 修正；
8. 继续沿用 ChArUco TCP-XY 补偿；
9. 执行最终 Z、工具/夹爪偏置以及基坐标 `+Y 0.3 mm`；
10. 到达每个目标后保存结果，并由操作员确认是否继续下一个孔。

注意：340 mm 不是某一个孔单独的参考高度。单孔时使用单孔深度；选定 2～3 个孔时使用选定孔共享高度；选定不少于 4 个孔时使用至少 4 个有效孔进行共享高度估计。这样不需要对每个孔重复做 340 mm 深度采集。

### 1.3 原有非 CAD 两阶段路径（保留）

原来的“多段位移确定孔位”没有删除。它由 `run_yolo_eye_in_hand_optimized.py` 的 `run_two_stage_hole_localization()` 实现，在 GUI 中对应“旧流程/开始两阶段定位”。这条路径不使用 CAD 孔号来确定目标，而是使用相机初始画面中的 YOLO 选孔和 RGB-D 局部点云建立每个孔的三维中心、局部平面和法向。

它的实际步骤是：

1. 回原点后在初始 RGB-D 画面中点击一个或多个孔，按 Enter/空格确认；
2. 按初始孔号逐孔处理，先安全抬升/横移，再移动到该孔约 340 mm 粗定位位；
3. 在 340 mm 采集点云孔口外侧环带，只保留朝相机最近的前表面深度簇，拟合局部平面，求 YOLO 中心射线与平面的交点、孔中心和法向；
4. 根据中心偏差和法向误差做最多 2 次粗定位闭环修正，仍未通过质量门则停止；
5. 依据该孔点云平面逐次只修正基坐标 Z，到约 260 mm；
6. 在 260 mm 只采 RGB/YOLO 多帧，精定位中心仍锁定在该孔的点云锚点附近，并用 RGB 中心与粗平面求精定位点；
7. 保留 ChArUco TCP-XY 补偿、最终 Z、工具/夹爪偏置和基坐标 `+Y 0.3 mm`；
8. 当前孔完成后，输入 `m` 才开始下一个已选孔；每个孔的粗中心、法向、精中心和运动结果都会写入两阶段报告；
9. 本轮所有孔完成后，真实运动模式先让机器人保持当前位置；随后等待操作员点击 GUI 的“开始下一轮检测”（命令行输入 `m`），确认后才回到原点，确认完全停止后再采集下一轮初始画面。相机和机器人会话保持运行；在选孔窗口按 `Esc` 或取消确认可结束会话。预览模式只执行一轮。

两条路径的职责区别如下：

| 项目 | CAD 路径 | 原两阶段路径 |
|---|---|---|
| 孔位/孔号来源 | CAD 11 孔模型与 RGB 配准 | 初始 YOLO 选孔 + RGB-D 点云几何 |
| 340 mm 作用 | 选定孔组一次共享高度采集 | 每个孔分别到 340 mm 做点云粗定位和闭环 |
| 260 mm 作用 | CAD 规划位姿，RGB/YOLO 只修最终 XY | 点云平面确定 Z，RGB/YOLO 求精定位中心 |
| 法向来源 | CAD 法向 | 340 mm 点云局部平面法向 |
| 结果目录 | `data\cad_motion_runs\cad-motion-*` | `data\hole_localization_runs\two-stage-*` |
| 入口开关 | `CAD_MOTION_MODE=True` 或 `--cad-motion` | `CAD_MOTION_MODE=False` 且 `TWO_STAGE_MODE=True`，或 `--two-stage-hole-localization` |

旧两阶段在真实运动模式下默认启用粗定位缓存复用：初始选孔后在当前位置追加 3 帧 RGB-D，单孔至少 2 帧有效才写入当前运行目录的 `coarse_cache/manifest.json` 和 `hole_XX.npz`；同时把成功粗定位的几何摘要和局部点云保存到固定的 `data\hole_localization_coarse_cache`。缓存点云使用“孔口外侧环带 + 最近前表面”策略；算法策略变化时旧缓存自动拒绝并重建。后续运行按机器人 base 坐标匹配历史孔位，不依赖上一次孔号；匹配成功后先安全到达 340 mm，再直接调用缓存的中心、平面和法向，不再对目标孔采现场复核帧，也不再用现场深度改写缓存 Z，随后进入260 mm高度修正和RGB精定位。相机序列号、手眼矩阵/路径、内参和畸变不兼容时拒绝历史缓存；缓存缺失或损坏时自动执行原粗定位并刷新当前运行及持久化缓存。预览模式不建立或使用缓存，CAD 路径不读取该开关。

命令行可显式控制：`--reuse-coarse-cache`（默认）或 `--no-reuse-coarse-cache`，以及跨运行的 `--reuse-persistent-coarse-cache`（默认）或 `--no-reuse-persistent-coarse-cache`。0.3秒停稳缓冲只用于缓存缺失/损坏后的完整粗定位回退，直接命中缓存时不等待；可用 `--coarse-settle-delay-s` 调整回退缓冲。base 坐标匹配、缓存直接调用/回退状态及估算节省时间会写入两阶段 `report.json`。

`run_hole_localization_pycharm.py` 当前默认将 `CAD_MOTION_MODE=True`；主脚本参数的默认值仍是 `CAD_MOTION=False`、`two_stage_hole_localization=True`。因此直接运行主脚本时必须明确确认自己要进入哪条路径，不能根据文件名猜测。

## 2. CAD 模型和坐标约定

### 2.1 CAD 文件

STEP 源文件：

`C:\MM\aubo_tools\aubo_workbench_project\孔位板_JXDZ26-KWB-001.STEP`

已生成的 CAD 孔模型：

`C:\MM\aubo_tools\data\cad_model\cad_hole_model.json`

当前模型基线：

| 项目 | 当前值 |
|---|---|
| 模型版本 | `cad_hole_model_v1` |
| 顶面孔数量 | 11 |
| 单位 | mm |
| 顶面高度 | `Z=6 mm` |
| 顶面法向 | `+Z` |

### 2.2 位姿计算

CAD 注册得到：

```text
T_camera_cad = CAD 坐标系 → RGB 相机坐标系
```

初始位置的机器人 TCP 和手眼变换合成：

```text
T_base_cad = T_base_camera_initial @ T_camera_cad
```

之后从 CAD 孔中心和法向计算每个孔的基坐标目标。一次运行中工件不移动时，不需要逐孔重新示教；每次新的运动运行会重新匹配当前画面并立即应用。

当前项目坐标约定为：

- 工具 `X` 正方向向前；
- 工具 `Y` 正方向向左；
- 最终基坐标 `+Y 0.3 mm` 在最终 Z 动作之后执行。

## 3. 质量门

| 环节 | 当前规则 |
|---|---|
| CAD 单帧匹配 | 至少 4 个有效对应孔；正深度；法向方向正确 |
| CAD 重投影 | RMSE ≤ 2 px，最大误差 ≤ 4 px |
| CAD 多帧稳定性 | 5 帧中至少 3 帧有效；中心 P95 ≤ 1.5 mm |
| 340 mm 共享深度 | 默认 8 帧，至少 5 帧有效；平面 RMSE ≤ 3.5 mm；高度散布 P95 ≤ 2.0 mm |
| 260 mm 精定位 | 默认 12 帧，至少 6 帧有效；丢弃 10 帧稳定帧；中心散布 P95 ≤ 1.5 px；CAD-YOLO 匹配距离 ≤ 70 px |
| 手眼 | 生产必须使用已验证手眼；实验手眼只能显式覆盖 |

任一质量门失败都不应进入下一阶段。阶段 0 预览即使质量通过，报告中的 `motion_allowed` 也固定为 `False`。

## 4. 直接配置入口的安全注意事项

`run_hole_localization_pycharm.py` 是方便在 IDE 中直接运行的配置入口，不应把它的开关值理解为生产安全策略。它默认只预览；真实运动、实验手眼和最终 XY 开关需要在每次现场运行前由授权人员显式复核：

| 配置项 | 作用 | 生产要求 |
|---|---|---|
| `CAD_MOTION_MODE` | `True` 选择 CAD 路径；`False` 保留原两阶段点云路径 | 生产路径由阶段验收结论决定；切换前必须确认报告模式 |
| `TWO_STAGE_MODE` | 非 CAD 路径的“初始选孔→340→260→逐孔目标点”开关 | 需要保留原多段位移时保持开启 |
| `EXECUTE_MOTION` | 是否真的发送机器人运动指令 | 预览/调试必须关闭；生产由授权人员打开 |
| `ALLOW_EXPERIMENTAL_HANDEYE` | 是否允许未验证手眼运行 | 生产必须关闭 |
| `I_HAVE_CHECKED_ROBOT_PATH_AND_WORKSPACE` | 路径和工作空间确认 | 真实运动前必须人工确认 |
| `CAD_MOVE_FINAL_XY` | 是否执行 260 mm 后的最终 XY | 先用预览/单孔验证，再按现场权限开启 |
| `CAD_BASE_Y_TRIM_MM` | 最终基坐标 Y 修正 | 当前为 `0.3 mm`，变更需重新评审 |

最新运行报告的 `status` / `handeye_validated` / `experimental_motion_override` 三个字段以实际报告文件为准。当前链路已跑通，但这不表示手眼结果可以用于生产。

## 5. 最近一次 CAD 运动结果

具体数值以 [DR V2.0](docs/reviews/DR_镀膜伞具孔位板_CAD配准运动_阶段评审_V2.0.md) 和 `data\cad_motion_runs\` 下的运行报告为准，这里不再复制一份，避免两处数字各自漂移。

需要注意的判读规则：CAD-YOLO 平均距离是诊断字段，不是独立的生产放行门；最终验收还要看孔边缘覆盖、实际 TCP 记录和机械末端误差。

## 6. 重要代码目录

```text
aubo_workbench_project/
├── run_workbench.py                       # 主 GUI，推荐入口
├── run_cad_registration_preview.py        # CAD Stage 0 预览，不运动
├── run_yolo_eye_in_hand_optimized.py      # CAD 运动核心，含旧两阶段点云路径
├── run_hole_localization_pycharm.py        # IDE 直接运行配置入口
├── run_handeye_pose_sequence.py            # 手眼候选位姿序列实验
├── run_charuco_height_error_experiment.py  # ChArUco 高度/视野实验
├── run_coarse_to_fine_offset_test.py       # 粗到精视野偏移实验
├── aubo_workbench/                         # 共 33 个模块，下面只列主要的
│   ├── cad_model.py                        # CAD JSON 加载和校验
│   ├── cad_registration.py                 # CAD↔RGB 匹配、PnP/IPPE、质量门
│   ├── coarse_cache.py                     # 旧两阶段当前运行/基坐标持久化缓存
│   ├── geometry.py                         # 4x4 变换、旋转、位姿格式转换（纯数学）
│   ├── optics.py                           # 去畸变、像素光线、光线求平面、倾斜圆心修正
│   ├── fitting.py                          # 平面/球面最小二乘拟合，含外点剔除
│   ├── motion_guards.py                    # 运动前状态校验和到位等待（不依赖 tkinter）
│   ├── io_utils.py                         # JSON 可序列化转换和 CSV 落盘
│   ├── paths.py                            # 项目/数据/模型路径和连接默认值
│   ├── config.py                           # 机器人连接参数覆盖
│   ├── gui_hole_localization.py            # 孔洞定位页面
│   ├── gripper_control.py                  # 独立夹爪控制窗口
│   ├── workbench.py                        # 工作台主窗口
│   ├── robot.py                            # AUBO 只读位姿会话
│   ├── motion_control.py                   # 机器人运动控制
│   ├── tcp_teach.py                        # TCP 示教
│   ├── gui_handeye.py                      # 手眼标定 GUI
│   └── ...
├── pyproject.toml                          # 依赖与打包配置
├── conftest.py                             # 测试 sys.path 处理
├── tests/                                  # 离线单元测试
└── docs/
    ├── reviews/
    │   └── DR_镀膜伞具孔位板_CAD配准运动_阶段评审_V2.0.md
    ├── planning/
    │   └── CAD配准替代点云粗定位_技术方案_V1.0.md
    └── archive/
        └── DR_镀膜伞具三孔眼在手定位_阶段评审_V1.0.md  # 历史版本
```

## 7. 结果目录

| 目录 | 内容 |
|---|---|
| `C:\MM\aubo_tools\data\cad_model` | STEP 解析后的 CAD 孔模型 |
| `C:\MM\aubo_tools\data\cad_registration_runs` | CAD 预览原图、叠加图、映射和配准报告 |
| `C:\MM\aubo_tools\data\cad_motion_runs` | CAD 真实/预览运动报告、共享深度和逐孔精定位结果 |
| `C:\MM\aubo_tools\data\cad_motion_runs\...\fine_visualizations` | 260 mm 每孔逐帧可视化、CAD 投影、YOLO 框和残差 |
| `C:\MM\aubo_tools\data\hole_localization_runs` | 旧两阶段报告；启用缓存时每次运行的 `two-stage-*\coarse_cache` 保存 manifest 和局部点云 NPZ |
| `C:\MM\aubo_tools\data\hole_localization_coarse_cache` | 旧两阶段跨运行 base 坐标缓存；按当前初始选孔的 base 坐标匹配后，在安全到达 340 mm 后直接调用 |
| `C:\MM\aubo_tools\data\e7_candidates` | 手眼候选和验证状态 |
| `C:\MM\aubo_tools\data\tcp_absolute_xy_model` | ChArUco TCP-XY 模型及报告 |
| `C:\MM\aubo_tools\data\charuco_height_error` | ChArUco 高度/视野实验数据 |

### NPZ 局部点云可视化

粗定位缓存的 NPZ 已经包含逐帧环带点云、帧索引、相机内参、缓存采集位姿、
基坐标变换和质量统计，不需要转换成 PLY。使用工具查看单孔：

~~~
python tools\visualize_coarse_cache.py --cache-dir C:\MM\aubo_tools\data\hole_localization_coarse_cache --hole-id 3 --show
~~~

需要和 RGB 图叠加时，传入与缓存采集时相同相机位姿下的 RGB 图片：

~~~
python tools\visualize_coarse_cache.py --cache-dir C:\MM\aubo_tools\data\hole_localization_coarse_cache --hole-id 3 --rgb C:\path\to\rgb.png --output C:\MM\hole_03_cache.png
~~~

真实运行建立初始缓存时，三帧共享RGB快照会保存到本次运行的
coarse_cache\initial_cache_rgb_frame_00.png 等文件，并记录在报告的
coarse_cache.rgb_snapshot_images 中。
工具左侧显示三维点云和缓存法向，右侧把相机坐标点投影到 RGB 图；
参数 --coordinate-frame base 可切换到基坐标显示，参数 --frame 1 可只查看某一帧，
参数 --list-holes 可列出缓存孔号。RGB 图片必须和缓存点云来自同一采集位姿，
否则投影叠加只具有参考意义。

需要使用前端窗口时：

~~~
python tools\visualize_coarse_cache.py --gui --cache-dir C:\MM\aubo_tools\data\hole_localization_coarse_cache
~~~

前端支持孔号和帧号切换、相机/基坐标切换、RGB选择或自动快照、点数抽样、
缓存中心/法向/RGB帧中心开关、三维旋转缩放，以及当前视图PNG导出。

## 8. 夹爪控制

在主 GUI 点击“夹爪控制”后会打开独立窗口，主工作台不被锁住。夹爪驱动文件为：

`C:\MM\aubo_tools\JiaZhua\z_erg_20c.py`

控制参数包括串口、从站地址、波特率、通信超时、重试次数、夹持速度/电流/位置，以及旋转速度/电流/绝对角度/相对角度。每个动作在后台线程执行，窗口会显示连接、动作和错误状态。夹爪动作不自动作为 CAD 配准质量门，正式接入取放流程前需要单独完成机械安全和限位验证。

## 9. 手眼和 ChArUco 约束

- 正式 RGB 手眼使用 RGB ChArUco 角点、RGB 内参和 `solvePnP`；旧点云手眼仅保留用于历史诊断。
- 旧两阶段定位的点云不是死代码：它仍负责非 CAD 模式的孔中心、平面和法向确定，并保留多段安全位移与粗定位闭环。
- 手眼流程需要诊断数据和独立验证数据分离；诊断结果不能直接作为运动依据。
- 当前 `e7_handeye_candidate_current.json` 尚未通过生产质量门。
- ChArUco TCP-XY 模型只修正最终 XY，不改变 CAD 提供的孔中心 Z 和法向。
- 生产运行必须能明确显示手眼文件、验证状态和实验覆盖状态；未验证时应阻止运动。

## 10. 离线验证

代码或配置变更后，在 `lip_env310` 环境中重新运行下面的命令；真实硬件测试前先完成不运动 CAD 预览。这里不写具体测试条数，避免每次加测试都要改文档。

连接密码不再写入业务源码。可以在启动前设置 `AUBO_PASSWORD` 环境变量，或直接在 GUI/命令行输入；未提供密码时不会自动尝试登录。

```text
python -m py_compile aubo_workbench/*.py run_workbench.py
python -m pytest -q
python -m unittest discover -s tests -v   # 等价的备选
```

两条测试命令都必须在本目录（`aubo_workbench_project`）下执行。在别的目录跑 `unittest discover -s tests` 会因为找不到 `tests` 而输出 `Ran 0 tests ... OK`，看起来像通过，实际什么都没测。

离线测试覆盖 CAD 模型/注册、几何变换、RGB/ChArUco、运动计划、质量门、报告字段和夹爪控制接口；它不替代真实相机、机器人、手眼和夹爪的现场验收。

## 11. 生产前检查清单

- [ ] STEP、`cad_hole_model.json` 和当前实物是同一孔位板；
- [ ] 相机分辨率、RGB 内参、畸变参数和安装姿态与采集手眼时一致；
- [ ] CAD 11 孔投影在连续多帧中稳定覆盖真实孔边缘，孔号没有错映射；
- [ ] 自动匹配对称解会拒绝或要求确认；
- [ ] 340 mm 共享深度在单孔、2～3 孔和多孔场景分别通过；
- [ ] 260 mm 每孔精定位结果已保存并复核最终 XY；
- [ ] ChArUco TCP-XY 模型通过独立验证；
- [ ] E7 手眼候选标记为已验证并允许生产运动；
- [ ] `EXECUTE_MOTION`、实验手眼覆盖和工作空间确认开关已经由授权人员复核；
- [ ] 最终 Z、工具/夹爪偏置和基坐标 `+Y 0.3 mm` 与工艺要求一致；
- [ ] 夹爪限位、动作顺序和异常停止已经单独验证；
- [ ] 所有实验结果目录和报告已归档。

**当前状态：CAD 配准和 CAD 运动链路已完成实验联调；手眼质量门、重复性和生产机械精度验收未完成。**
