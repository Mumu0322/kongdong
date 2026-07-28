#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中管理所有可调参数。

设计说明：
- 原脚本把配置分散成模块级 dataclass 实例（BOARD_CFG / CAMERA_CFG / ...），
  被其余所有模块 import 后直接读写字段。这里保留同样的“全局单例配置”
  设计（GUI 需要在运行时改 IP / 保存目录等字段），但集中到一个文件里，
  避免配置定义散落在业务代码之间。
- 所有其余模块都应该 `from aubo_workbench.config import BOARD_CFG, ...`
  而不是各自 new 一份，否则运行时修改（比如 GUI 表单里改 IP）不会生效。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class BoardConfig:
    # 外形尺寸 400 x 300 mm 只是实体板外框；真正用于 ChArUco 几何的是图案尺寸。
    outer_width_mm: float = 400.0
    outer_height_mm: float = 300.0

    # 图案尺寸 360 x 270 mm，大格边长 30 mm，因此 12 x 9 个大格。
    pattern_width_mm: float = 360.0
    pattern_height_mm: float = 270.0
    square_length_mm: float = 30.0
    marker_length_mm: float = 22.5
    squares_x: int = 12
    squares_y: int = 9

    # 当前 CC400-30-22.5 检测验证使用 DICT_5X5_1000。
    aruco_dict_name: str = "DICT_5X5_1000"

    # ChArUco 检测与 3D 拟合质量门槛。
    min_charuco_corners: int = 18
    min_valid_3d_corners: int = 14
    max_corner_3d_rmse_mm: float = 2.5
    max_plane_rmse_mm: float = 2.0

    # 正式RGB手眼：用ChArUco二维角点、已知板尺寸和RGB内参做PnP。
    rgb_pnp_ransac_reprojection_px: float = 0.80
    rgb_pnp_ransac_iterations: int = 200
    rgb_pnp_ransac_confidence: float = 0.999
    max_rgb_reprojection_rmse_px: float = 0.35
    max_rgb_reprojection_error_px: float = 1.00


@dataclass
class CameraConfig:
    save_dir: str = r"C:\MM\aubo_tools\charuco_pointcloud_calib"
    min_valid_z_mm: float = 250.0
    max_valid_z_mm: float = 1200.0
    depth_vis_min_mm: float = 250.0
    depth_vis_max_mm: float = 1200.0
    align_to_color: bool = True
    xyz_window_radius_px: int = 3      # 3 => 7x7 中值取 3D 点
    board_mask_sample_step_px: int = 2  # 板面点云采样间隔，越小越准但慢
    board_plane_ransac_iter: int = 160
    board_plane_ransac_tol_mm: float = 1.8
    overlay_alpha: float = 0.32

    # 孔定位使用的彩色流必须显式选择，不能依赖 SDK 返回的 profile index=0。
    # 0 表示不强制该维度；在满足指定项的 profile 中优先选分辨率最高者。
    preferred_color_width: int = 0
    preferred_color_height: int = 0
    preferred_color_fps: int = 30
    preferred_color_formats: tuple[str, ...] = ("RGB", "BGR", "YUYV", "UYVY", "MJPG")

    # 生产测量前应先让自动曝光稳定，再锁定曝光/增益。None 表示只读取、不修改。
    lock_color_auto_exposure: bool = False
    color_exposure: int | None = None
    color_gain: int | None = None

    # SDK原生时域滤波只在显式诊断参数下启用；默认none保证标定/旧工作台行为不被暗改。
    depth_temporal_weight: float = 0.40
    depth_temporal_diff_scale: float = 0.10


@dataclass
class HoleVisionConfig:
    """镀膜伞孔 RGB 亚像素检测与局部平面拟合门槛。"""

    # v5 方案明确为三种固定型号，而不是 65–75 mm 连续变化的单一型号。
    # min/max 只保留为“自动诊断模式”的粗筛包络；生产流程必须显式指定型号。
    supported_hole_diameters_mm: tuple[float, ...] = (65.0, 70.0, 75.0)
    hole_diameter_min_mm: float = 65.0
    hole_diameter_max_mm: float = 75.0
    hole_model_crosscheck_tolerance_mm: float = 2.0

    gaussian_kernel: int = 5
    canny_low: int = 45
    canny_high: int = 135
    min_edge_points: int = 18
    min_axis_px: float = 10.0
    max_axis_px: float = 1000.0
    max_axis_ratio: float = 3.0
    ransac_iterations: int = 240
    ransac_residual_px: float = 0.80
    max_fit_rmse_px: float = 0.65
    min_inlier_ratio: float = 0.68
    min_angular_coverage_deg: float = 240.0
    expected_diameter_tolerance_ratio: float = 0.18

    # 沿 YOLO 先验半径采样灰度剖面，以暗区/亮区稳健中位数的 50% 强度交点定位孔沿。
    # 该方法避免 Otsu 阈值在相邻像素间跳变，也避免只取最大梯度时切到内/外双边缘。
    photometric_angle_count: int = 360
    photometric_radial_step_px: float = 0.125
    photometric_search_half_width_ratio: float = 0.16
    photometric_level_gap_inner_ratio: float = 0.08
    photometric_level_gap_outer_ratio: float = 0.22
    photometric_min_contrast: float = 8.0
    photometric_min_crossing_gradient: float = 0.8
    photometric_recenter_iterations: int = 3
    photometric_recenter_convergence_px: float = 0.01
    # 至少有这些相反方向孔沿对时，才允许用椭圆二次型反推采样中心偏移。
    photometric_opposed_pair_min_count: int = 36
    photometric_opposed_center_max_correction_px: float = 3.0

    # 深度只用于孔周围环带平面，不使用孔中心单像素深度。
    # 孔外窄环带：旧1.10–1.55在伞架外缘孔会同时覆盖相邻孔、外轮廓和较大曲率范围。
    # 新范围保留约4 px边缘隔离，同时把有效带宽控制在约8–12 px（当前90–125 px孔径）。
    plane_ring_inner_scale: float = 1.08
    plane_ring_outer_scale: float = 1.28
    plane_other_hole_exclusion_scale: float = 1.05
    plane_exclusion_min_confidence: float = 0.70
    plane_min_points: int = 80
    plane_ransac_iterations: int = 240
    plane_ransac_tol_mm: float = 1.0
    plane_max_rmse_mm: float = 0.80
    plane_min_inlier_ratio: float = 0.40
    # 仅用于发现“最近表面”的候选，不是放行门；最终仍必须通过上面的0.40。
    plane_front_candidate_min_ratio: float = 0.35
    local_surface_model: str = "quadratic_inverse_depth_tangent"
    surface_max_abs_gradient_mm_per_px: float = 1.0
    surface_max_abs_curvature_mm_per_px2: float = 0.02
    surface_layer_separation_mm: float = 3.0
    surface_layer_min_hypotheses: int = 3
    surface_layer_relative_consensus: float = 0.90
    physical_scale_min_frames: int = 20
    physical_scale_min_pass_ratio: float = 0.90

    # v5 E0.2 静态快测要求 200 帧；E1 正式重复性要求 500–1000 帧。
    # 物理指标只有在提供 mm/px 或局部平面 Jacobian 后才判定。
    repeatability_min_frames: int = 200
    repeatability_min_pass_ratio: float = 0.90
    repeatability_max_p95_mm: float = 0.05
    e1_min_frames: int = 500
    e1_min_pass_ratio: float = 0.995

    # 生产定位输出采用非重叠稳健批次；200 帧 E0 至少形成 20 个独立 10 帧批次。
    fusion_burst_frames: int = 10
    fusion_min_valid_frames_per_burst: int = 7
    fusion_min_bursts: int = 20
    fusion_min_pass_ratio: float = 0.90
    fusion_max_p95_mm: float = 0.05
    fusion_outlier_floor_px: float = 0.50

    # 深度噪声明显高于RGB孔心：500帧选型集上5/10/15/20/25帧窗口的Z P95分别为
    # 0.524/0.386/0.288/0.231/0.240 mm。生产默认取20帧以保留余量，2D仍用上面的10帧。
    # 这里只能验证固定场景重复性；E3/E4绝对精度仍需量块/位移计和精密倾角台真值。
    surface_pose_fusion_burst_frames: int = 20
    surface_pose_fusion_min_valid_frames_per_burst: int = 14
    surface_pose_fusion_min_bursts: int = 20
    surface_pose_fusion_min_pass_ratio: float = 0.90
    surface_pose_fusion_outlier_floor_mm: float = 1.0
    surface_pose_fusion_outlier_mad_scale: float = 3.5
    surface_pose_min_valid_frame_ratio: float = 0.90
    surface_pose_fused_max_z_p95_mm: float = 0.30
    surface_pose_fused_max_normal_p95_deg: float = 0.30

    # v5 正式端到端指标；0.20 mm 作为内部放行裕量，不冒充最终验收口径。
    final_xy_radial_p95_mm: float = 0.25
    engineering_release_xy_radial_p95_mm: float = 0.20

    # E0 画面/深度可用性门槛，用于先区分成像问题与检测算法问题。
    diagnostic_min_brightness: float = 35.0
    diagnostic_max_brightness: float = 225.0
    diagnostic_min_contrast: float = 12.0
    diagnostic_min_sharpness: float = 20.0
    diagnostic_max_underexposed_ratio: float = 0.40
    diagnostic_max_overexposed_ratio: float = 0.40
    diagnostic_min_valid_depth_ratio: float = 0.05


@dataclass
class RobotConfig:
    # 默认自动读取 AUBO 当前 TCP 位姿。只读，不写 TCP，不控制运动。
    robot_pose_read_enable: bool = True
    ip: str = os.getenv("AUBO_ROBOT_IP", "192.168.1.100")
    rpc_port: int = 30004
    user: str = os.getenv("AUBO_ROBOT_USER", "AUBO")
    password: str = os.getenv("AUBO_ROBOT_PASSWORD", "")
    request_timeout_ms: int = 1500
    network_precheck_timeout_s: float = 1.0

    # tcp：读取当前 TCP 在基坐标系下的位姿，推荐用于眼在手。
    # tool：读取当前工具/法兰位姿；只有明确要以法兰为手眼末端时才使用。
    pose_source: str = "tcp"

    require_power_on: bool = True
    require_steady: bool = True
    reject_collision: bool = True

    # 自动读取失败时，可按 m 手动输入 AUBO SDK 位姿，单位 m / rad。
    manual_pose_sdk_m_rad: tuple[float, float, float, float, float, float] | None = None


@dataclass
class RobotCameraIntegrationConfig:
    """自动入孔所需权威证据文件位置；不使用可手工翻转的解锁布尔值。"""

    production_camera_serial: str = "CP4B85P001L"
    dome_model_path: str = r"C:\MM\aubo_tools\data\dome_hole_model.json"
    handeye_validation_evidence_path: str = (
        r"C:\MM\aubo_tools\data\e7_handeye_validation_current.json"
    )
    tcp_payload_validation_evidence_path: str = (
        r"C:\MM\aubo_tools\data\e8_tcp_payload_validation_current.json"
    )
    part_clearance_evidence_path: str = (
        r"C:\MM\aubo_tools\data\part_clearance_validation_current.json"
    )
    v5_validation_bundle_path: str = (
        r"C:\MM\aubo_tools\data\v5_validation_evidence_current.json"
    )
    physical_integration_evidence_path: str = (
        r"C:\MM\aubo_tools\data\physical_integration_validation_current.json"
    )
    motion_authorization_evidence_path: str = (
        r"C:\MM\aubo_tools\data\automatic_motion_authorization_current.json"
    )
    # 可选：camera_depth / cad_fixture / external_sensor / coarse_depth_with_compliance。
    # 当前跨孔位证据已否决camera_depth作为全局精密高度源，因此默认保持未配置。
    insertion_height_source: str = "unconfigured"
    insertion_height_validation_evidence: str = (
        r"C:\MM\aubo_tools\data\insertion_height_validation_current.json"
    )
    camera_depth_cross_position_gate_path: str = (
        r"C:\MM\aubo_tools\data\hole_diagnostics\surface_pose_cross_position_gate_20260716.json"
    )


@dataclass
class SolveConfig:
    # 8组只允许做诊断求解；正式E7由 E7HandEyeConfig 单独控制。
    min_samples_for_solve: int = 8
    output_json: str = r"C:\MM\aubo_tools\data\handeye_diagnostic_current.json"
    also_write_compatible_key_t_tooltcp_cam: bool = False
    enable_nonlinear_refine: bool = True
    nonlinear_rotation_weight_mm: float = 80.0
    nonlinear_max_nfev: int = 300
    load_existing_samples_on_start: bool = True


@dataclass
class E7HandEyeConfig:
    """正式RGB-PnP E7交叉验证；与8组诊断及旧点云样本严格分开。"""

    # 超过 10 组即可进入正式流程：11 组中至少 8 组用于拟合、3 组独立留出验证。
    minimum_total_poses: int = 11
    minimum_validation_fraction: float = 0.20
    minimum_validation_poses: int = 3
    maximum_validation_center_scatter_rms_mm: float = 0.10
    require_tcp_pose_source: bool = True
    allow_manual_pose: bool = False
    candidate_dir: str = r"C:\MM\aubo_tools\data\e7_candidates"

    # 视野覆盖必须能由原始样本自动计算，不能由人工布尔值直接声称通过。
    center_region_half_width_ratio: float = 0.22
    center_region_half_height_ratio: float = 0.22
    minimum_distinct_edge_regions: int = 2

    # 这些门只用于判定单次采集期间机器人是否真正静止。
    maximum_pose_bracket_xyz_mm: float = 0.05
    maximum_pose_bracket_abc_deg: float = 0.01


@dataclass
class AutoPruneConfig:
    # 按 a 自动剔除时使用；普通 h 求解不受影响。
    max_remove_per_run: int = 8
    # 与当前E7最低总样本数保持一致；18张数据允许剔除，最低保留11张。
    min_remaining_samples: int = 11

    # 仅用于发现毫米级冲突样本；不能作为本项目生产手眼验收：
    # mean 控制整体稳定性，max 控制最差样本，max_translation_error 控制明显坏样本。
    target_translation_mean_mm: float = 0.70
    target_translation_max_mm: float = 1.00
    max_translation_error_mm: float = 1.20

    # ChArUco / 点云质量硬门槛。超过这些值的样本会优先作为候选。
    min_valid_3d_count: int = 80
    max_corner_rmse_mm: float = 0.65
    max_plane_rmse_mm: float = 0.70

    # 删除一个样本后至少要让整体指标有可见改善，避免误删正常覆盖点。
    min_mean_improvement_mm: float = 0.02
    min_max_improvement_mm: float = 0.12


@dataclass
class ConflictDiagnosisConfig:
    # 每次按 h 求解时，自动分析哪些样本和主一致集合冲突；只写报告，不移动/删除原始样本。
    enable_on_solve: bool = True

    # 诊断时最多尝试剔除多少个样本；这是为了找出冲突来源，不代表建议最终只保留这么少。
    max_remove_for_report: int = 12
    min_report_samples: int = 9

    # 只有剩余样本数足够多且指标达标时，才认为稳健子集可作为“可靠标定结果”。
    # 少量点即使内残差很好，也可能只是过拟合，不应当作为通用 0.5 mm 方案。
    min_accept_samples: int = 18
    target_translation_mean_mm: float = 0.70
    target_translation_rmse_mm: float = 0.80
    target_translation_max_mm: float = 1.00

    # 初始全样本残差超过这些阈值的样本，会在报告里标为疑似冲突/严重冲突。
    suspect_residual_mm: float = 2.00
    bad_residual_mm: float = 3.00


@dataclass
class AutoCaptureConfig:
    # 当前版本已经删除自动抓拍；本配置只保留毫米级诊断门槛和手动批量采集参数。
    enable: bool = False

    # 手动批量采集：按一次 c，连续取 5 帧合格观测，再从同一稳定 TCP 姿态簇中选 1 帧保存。
    manual_burst_frames: int = 5

    # 为了保证最终确实取得满 5 帧，允许最多尝试的帧数。
    manual_burst_max_attempts: int = 12

    # 连续取帧间隔，避免 5 帧完全挤在同一瞬间，也给相机缓存一点刷新时间。
    burst_interval_s: float = 0.06

    # 五帧采集前检查机器人 TCP 波动。
    burst_pose_stability_xyz_mm: float = 0.05
    burst_pose_stability_abc_deg: float = 0.01
    reject_unstable_burst_pose: bool = True

    # 选择同一 TCP 姿态簇时使用的近邻阈值。
    burst_pose_cluster_xyz_mm: float = 0.05
    burst_pose_cluster_abc_deg: float = 0.01
    burst_min_pose_cluster_frames: int = 3

    # True：只有满足毫米级单帧诊断门槛才保存；这不是正式E7手眼验收。
    require_quality_ok_for_burst: bool = True

    # 毫米级一致性诊断，只用于剔除冲突样本。生产必须另做E7独立交叉验证，RMS<=0.10mm。
    diagnostic_result_mean_mm: float = 0.70
    diagnostic_result_rmse_mm: float = 0.80
    diagnostic_result_max_mm: float = 1.00

    # 综合评分与图像质量门槛。
    min_score: float = 90.0
    min_charuco_corners: int = 78
    min_valid_3d_corners: int = 74
    max_corner_rmse_mm: float = 0.65
    max_corner_max_error_mm: float = 2.0
    max_plane_rmse_mm: float = 0.70

    # RGB手眼正式采样门；点云毫米指标只保留给旧点云诊断。
    min_rgb_pnp_inliers: int = 78
    max_rgb_reprojection_rmse_px: float = 0.35
    max_rgb_reprojection_error_px: float = 1.00
    max_rgb_center_offset_ratio: float = 0.85

    # 标定板在画面中的位置/大小要求。
    min_board_area_ratio: float = 0.055
    max_board_area_ratio: float = 0.55
    max_center_offset_ratio: float = 0.25

    # 灰度图像清晰度/亮度/对比度要求。
    min_brightness: float = 55.0
    max_brightness: float = 200.0
    min_contrast: float = 28.0
    min_sharpness: float = 90.0


# 全局单例：整个应用共享同一份配置对象，GUI 表单/命令行都是直接改这些字段。
BOARD_CFG = BoardConfig()
CAMERA_CFG = CameraConfig()
HOLE_VISION_CFG = HoleVisionConfig()
ROBOT_CFG = RobotConfig()
ROBOT_CAMERA_INTEGRATION_CFG = RobotCameraIntegrationConfig()
SOLVE_CFG = SolveConfig()
E7_HAND_EYE_CFG = E7HandEyeConfig()
AUTO_PRUNE_CFG = AutoPruneConfig()
CONFLICT_DIAG_CFG = ConflictDiagnosisConfig()
AUTO_CAPTURE_CFG = AutoCaptureConfig()

ESC_KEY = 27
MODE_NAMES = {1: "joint", 2: "base", 3: "world", 4: "tool", 5: "user"}
JOINT_LABELS = ("J1", "J2", "J3", "J4", "J5", "J6")
CARTESIAN_LABELS = ("X", "Y", "Z", "A", "B", "C")
