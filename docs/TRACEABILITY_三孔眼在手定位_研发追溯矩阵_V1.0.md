# 镀膜伞具三孔眼在手定位项目——研发追溯矩阵

| 文档项 | 内容 |
|---|---|
| 文档编号 | TR-EIH-20260730-001 |
| 版本 / 日期 | V1.0 / 2026-07-30 |
| 适用范围 | 当前眼在手三孔定位、ChArUco XY 修正、逐孔安放及其研发证据。 |
| 需求来源 | 三孔定位与逐孔安放流程；[阶段技术评审 DR](DR_镀膜伞具三孔眼在手定位_阶段评审_V1.0.md)。 |
| 状态定义 | **已验证**：实现、自动化测试和已保存运行记录均可追溯。**实验状态**：实现和记录存在，但原始记录明确未达到独立验证/生产门槛。**待补充**：实施存在但缺少指定独立证据或正式评审记录。**缺证据**：没有可定位的实现或验证路径。 |
| 评审边界 | 本矩阵只追溯机器人相对视觉定位实验；不声明外部绝对精度，也不声明生产验收。 |

## 1. 本次确认的验证入口

| 类别 | 入口 / 记录 | 本次确认结果 |
|---|---|---|
| 当前核心实现 | `run_yolo_eye_in_hand_optimized.py` | 两阶段三孔流程、ChArUco XY 修正、逐孔安放均有可定位实现。 |
| 离线自动化测试 | `C:\Users\j1005\.conda\envs\lip_env310\python.exe -m unittest discover -s tests -v` | 2026-07-30 重跑：`Ran 70 tests in 17.047s`，`OK`。 |
| RGB 相对定位 | `data\charuco_height_error\charuco-rgb-precision-20260724_113843_546\report.json` | 200–300 mm、11 个高度、1100/1100 帧完成；11/11 高度通过既定质量门，最佳高度 300 mm。 |
| TCP-XY 模型 | `data\tcp_absolute_xy_model\charuco-tcp-xy-20260727_174559\report.json` | 9 点 ChArUco 仿射模型；状态 `needs_more_or_better_touch_data`。 |
| 三孔实验 | `data\hole_localization_runs\two-stage-20260729_170949\report.json` | 状态 `completed_experimental_handeye`；3 孔均记录为 `placement_confirmed`。 |
| 手眼候选 | `data\e7_candidates\e7_handeye_candidate_current.json` | `validated=false`、`production_eligible=false`、`do_not_use_for_motion=true`。 |

## 2. 研发追溯矩阵

| Req ID | 需求（来源中的明确要求） | 设计引用 | 实现引用 | 自动化测试 | 运行 / 数据证据 | 状态 |
|---:|---|---|---|---|---|---|
| REQ-001 | 在同一粗定位和精定位视角下，推算 3 个孔的中心、法向和目标 TCP 位姿。 | DR §3“当前方案与实现状态”；DR §5“最新三孔实验结果”。 | `run_yolo_eye_in_hand_optimized.py`：`_select_coarse_holes_with_surface` 在粗定位确认位生成三孔粗位姿；三孔结果组织保留 `coarse_hole_pose_m_rad`、`hole_pose_m_rad` 和 `tcp_target_pose_m_rad`。 | `tests/test_optimized_final_xy.py` 三孔参考孔/相对几何匹配、球面射线求交、粗阶段表面法向回归。 | `two-stage-20260729_170949/report.json`：`hole_count=3`，3 个 `final_result.holes`；每孔有孔中心、法向、目标 TCP。 | 已验证 |
| REQ-002 | 粗定位应在约 340 mm 使用 RGB-D 建立孔周局部几何，闭环居中/垂直后冻结表面，并在粗阶段确定三孔身份与粗位姿。 | DR §3 第 2–3 阶段；粗阶段三孔位姿说明。 | `run_yolo_eye_in_hand_optimized.py`：初始选择、粗帧采集、粗融合、`coarse_multi_pose` 阶段和粗阶段表面模型字段。 | `tests/test_two_stage_hole_localization.py` 孔面法向姿态；`test_optimized_final_xy.py` 粗定位法向质量门与粗表面每孔法向。 | 三孔报告：粗定位 15/15 有效帧、中心散布 P95 0.186 px；闭环复采 15/15、中心散布 P95 0.124 px、冻结平面 RMSE 2.543 mm。 | 已验证 |
| REQ-003 | 精定位应在约 260 mm 仅使用 RGB，对每孔进行圆/椭圆中心多帧鲁棒融合；孔中心由 RGB 射线与粗阶段表面模型相交得到，姿态沿用粗阶段每孔位姿。 | DR §3 第 4–5 阶段；“粗深度几何 + 精 RGB 圆心”说明。 | `run_yolo_eye_in_hand_optimized.py`：`_prepare_fine_holes_from_coarse` 刷新精拍锚点；`_capture_multi_fine_burst` 多孔精拍；`_coarse_surface_point_and_normal_from_pixel` 使用粗阶段表面模型。 | `tests/test_two_stage_hole_localization.py` 像素去畸变后反投影、精拍离群融合；`test_optimized_final_xy.py` 融合后 P95 不劣化。 | 三孔报告：孔 1/2/3 圆心 P95 分别为 0.117/0.362/0.258 px；有效帧分别为 37/43、42/43、38/43。 | 已验证 |
| REQ-004 | 三孔最终 TCP 的 XY 应使用 ChArUco 仿射模型修正；Z 与姿态不使用该模型修正。 | DR §3 第 6 阶段；DR §4.2 ChArUco TCP-XY 模型。 | `run_yolo_eye_in_hand_optimized.py:648` 最终 TCP XY 规划；`:1802–1827` 默认 ChArUco 模型配置；`:2030–2047` 每孔 XY 修正和模式记录。 | `tests/test_two_stage_hole_localization.py:40` XY 修正保持 Z/姿态；`test_optimized_final_xy.py:81` 默认模型保持 Z/姿态；`:95` 固定偏置覆盖。 | 模型报告：9 点；训练 P95 0.294 mm；留一角点 P95 0.535 mm（阈值 0.500 mm）；三孔运行 XY 修正量分别为 `[0.607, 2.410]`、`[0.710, 2.406]`、`[0.539, 2.353]` mm。 | 实验状态 |
| REQ-005 | 三孔须逐孔安放；每孔路径为先抬 Z、再移动 XY/姿态、后降 Z；到最终点后基坐标 Y 正方向 +0.2 mm，并保留实际到位 TCP。 | DR §3 第 7 阶段；DR §5.3 三孔顺序安放。 | `run_yolo_eye_in_hand_optimized.py:1061` 顺序安放；`:1124` `final_base_y_trim_mm`；`:1133` `placement_confirmed`。 | `tests/test_optimized_final_xy.py:100` 最终 Z 规划；`:113` Y +0.2 mm 保持 X/Z/姿态；`:67` 最终 XY 运动默认开启。 | 三孔报告：安放顺序 1→2→3；各孔均 `placement_confirmed`；各孔记录 `final_base_y_trim_mm=0.2` 和 `placement_tcp_pose_m_rad`。 | 已验证 |
| REQ-006 | RGB 相对定位范围应以机器人 TCP 相对运动为参考，明确可用高度范围和最佳高度，不作为外部绝对精度。 | DR §4.1 RGB 相对定位实验。 | `run_charuco_height_error_experiment.py` RGB 精度范围模式与高度汇总；主项目将精拍高度配置为约 260 mm。 | `tests/test_charuco_height_error.py:146` 高度序列；`:153` 五点目标；`:196` 连续通过范围选择；`:249` RGB 模式不启动深度。 | RGB 报告：200–300 mm、11 高度、5 点×20 帧、1100/1100 帧；所有高度通过门槛；最佳高度 300 mm，距离误差 P95 0.095 mm。 | 已验证 |
| REQ-007 | 手眼候选状态必须被如实记录；未通过独立验证时不得在 DR 中描述为“已验证手眼”或生产结论。 | DR §4.3 手眼候选；DR §6 当前事实与边界。 | `run_yolo_eye_in_hand_optimized.py` 读取手眼候选并在报告中保留 `experimental_handeye_override`、`handeye_path`。 | `tests/test_handeye_e7.py:264` 未验证候选写入；`tests/test_charuco_point_experiment.py:66` 未验证候选显式标记为实验状态。 | `e7_handeye_candidate_current.json`：`validated=false`、`production_eligible=false`；三孔报告状态 `completed_experimental_handeye`。 | 实验状态 |
| NFR-001 | DR 需要通过 Word 页面渲染及负责人/评审人签署，形成正式阶段评审记录。 | DR 封面填写栏、验证要求。 | [DR Word 文件](DR_镀膜伞具三孔眼在手定位_阶段评审_V1.0.docx)。 | 无自动化测试。 | Word 与 Markdown 已做结构和标题一致性检查；当前记录中未包含页面渲染截图、负责人签署或评审人签署。 | 待补充 |

## 3. 追溯结论

| 汇总项 | 数量 | 说明 |
|---|---:|---|
| 已验证 | 5 | REQ-001、REQ-002、REQ-003、REQ-005、REQ-006：均可关联实现、自动化测试和保存的实验报告。 |
| 实验状态 | 2 | REQ-004 受 ChArUco 模型当前状态限制；REQ-007 的手眼候选明确未验证。 |
| 待补充 | 1 | NFR-001 缺少 Word 页面渲染留档和评审签署。 |
| 缺证据 | 0 | 本次明确纳入的需求均有至少一条实现或记录路径；不代表外部绝对精度已验证。 |

## 4. 缺口与最小补充动作

| 级别 | 项目 | 证据依据 | 最小补充动作 |
|---|---|---|---|
| ⚠️ 跟进 | ChArUco TCP-XY 模型 | 留一角点交叉验证 P95 为 0.535 mm，高于记录内 0.500 mm 验收阈值；模型状态为 `needs_more_or_better_touch_data`。 | 以新的 TCP 触碰记录扩充/复采模型数据；重新生成模型报告并更新 REQ-004 状态。 |
| ⚠️ 跟进 | 手眼候选 | 候选文件明确为 `validated=false`、`production_eligible=false`。 | 依据独立验证流程补齐标定板固定、验证位姿和人工确认信息；仅在候选文件字段变化且有相应证据时更新状态。 |
| ⚠️ 跟进 | DR 形式审阅 | 当前 Word 未保存页面渲染检查产物，负责人/评审人仍为空。 | 在具备 Word 或 LibreOffice 的环境渲染 PDF/页面图，逐页检查后签署封面，并将签署日期和审阅记录加入本矩阵。 |

## 5. 非本矩阵结论

以下事项没有被写为“缺证据”，因为它们不属于本次明确需求，也未被 DR 作为阶段结论：

- 外部独立基准下的孔中心绝对 XY/Z 精度；
- 生产节拍、长期稳定性和量产验收；
- 未经独立验证的手眼候选用于生产运动。

## 6. 数据追溯路径

| 证据 ID | 文件 / 目录 |
|---|---|
| EVID-RGB-001 | `C:\MM\aubo_tools\data\charuco_height_error\charuco-rgb-precision-20260724_113843_546\report.json` |
| EVID-XY-001 | `C:\MM\aubo_tools\data\tcp_absolute_xy_model\charuco-tcp-xy-20260727_174559\report.json` |
| EVID-RUN-001 | `C:\MM\aubo_tools\data\hole_localization_runs\two-stage-20260729_170949\report.json` |
| EVID-HAND-001 | `C:\MM\aubo_tools\data\e7_candidates\e7_handeye_candidate_current.json` |
| EVID-TEST-001 | 2026-07-30 在 `C:\MM\aubo_tools\aubo_workbench_project` 执行 `python -m unittest discover -s tests -v`；70 项通过。 |
