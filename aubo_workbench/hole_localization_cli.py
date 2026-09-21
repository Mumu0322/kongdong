"""Command-line interface for the hole-localization runner."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from aubo_workbench.hole_localization_models import TwoStageConfig


def build_parser(
    *,
    default_model: Path,
    default_handeye: Path,
    default_execute_motion: bool,
    default_allow_experimental_handeye: bool,
    default_two_stage_hole_localization: bool,
    default_move_final_xy: bool,
) -> argparse.ArgumentParser:
    DEFAULT_MODEL = default_model
    DEFAULT_HANDEYE = default_handeye
    DEFAULT_EXECUTE_MOTION = default_execute_motion
    DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE = default_allow_experimental_handeye
    DEFAULT_TWO_STAGE_HOLE_LOCALIZATION = default_two_stage_hole_localization
    DEFAULT_MOVE_FINAL_XY = default_move_final_xy
    p = argparse.ArgumentParser(description="YOLO 选孔并计算眼在手目标位姿")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--image", type=Path, help="离线 RGB 图；不指定则使用 Gemini RGB-D")
    p.add_argument("--intrinsics", type=Path, help="离线图像使用的 RGB 内参 JSON")
    p.add_argument("--target-depth-mm", type=float, help="离线无深度图时使用的平面深度（仅粗略预览）")
    p.add_argument("--execute", dest="execute", action="store_true", default=DEFAULT_EXECUTE_MOTION,
                   help="显式启用真实运动；不传入时只预览，不连接运动控制")
    p.add_argument("--no-execute", dest="execute", action="store_false",
                   help="仅预览：不连接运动控制或下发机器人运动")
    p.add_argument("--allow-experimental-handeye", dest="allow_experimental_handeye", action="store_true",
                   default=DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE,
                   help="显式允许当前实验手眼结果；默认只接受已验证手眼")
    p.add_argument("--require-validated-handeye", dest="allow_experimental_handeye", action="store_false",
                   help="只允许已获生产授权的手眼结果")
    p.add_argument("--speed-m-s", type=float, default=0.08,
                   help="精确靠近/闭环修正的 moveLine 速度(m/s)，默认0.08")
    p.add_argument("--acc-m-s2", type=float, default=0.25,
                   help="精确靠近/闭环修正的 moveLine 加速度(m/s²)，默认0.25")
    p.add_argument("--transit-speed-m-s", type=float, default=0.15,
                   help="孔间安全过渡及粗定位导航速度(m/s)，默认0.15")
    p.add_argument("--transit-acc-m-s2", type=float, default=0.45,
                   help="孔间安全过渡及粗定位导航加速度(m/s²)，默认0.45")
    p.add_argument("--approach-speed-m-s", type=float, default=0.12,
                   help="粗定位校正及非接触下降速度(m/s)，默认0.12")
    p.add_argument("--approach-acc-m-s2", type=float, default=0.35,
                   help="粗定位校正及非接触下降加速度(m/s²)，默认0.35")
    p.add_argument("--offset-mm", type=float, default=0.0, help="沿机器人基坐标 Z 方向的安全偏置")
    p.add_argument("--two-stage-hole-localization", dest="two_stage_hole_localization", action="store_true",
                   default=DEFAULT_TWO_STAGE_HOLE_LOCALIZATION,
                   help="兼容参数：默认执行 原点选孔->340mm粗定位->260mm RGB精定位")
    p.add_argument("--single-stage", dest="two_stage_hole_localization", action="store_false",
                   help="仅诊断使用：关闭默认两阶段流程")
    p.add_argument(
        "--auto-select-holes",
        dest="auto_select_holes",
        action="store_true",
        default=False,
        help=(
            "初始固定观察位启用自动扇区划分和候选孔筛选；"
            "必须同时提供 --auto-sector-config"
        ),
    )
    p.add_argument(
        "--auto-sector-config",
        type=Path,
        default=None,
        help="静止伞架自动分区JSON配置（鼠标多边形模式不需要伞架中心）",
    )
    p.add_argument(
        "--auto-sector-ids",
        type=int,
        nargs="+",
        default=None,
        metavar="SECTOR",
        help="只执行指定扇区；默认使用配置中的全部扇区",
    )
    p.add_argument(
        "--auto-exclude-boundary-candidates",
        dest="auto_exclude_boundary_candidates",
        action="store_true",
        default=False,
        help="将边界候选标记为配置排除；边界候选始终不会直接执行",
    )
    p.add_argument(
        "--map-hole-selection-mode",
        choices=("auto", "manual"),
        default="manual",
        help=(
            "建立地图时的最终选孔方式：auto直接使用自动扇区候选，"
            "manual在自动扇区候选范围内由操作者重新选择；默认manual"
        ),
    )
    p.add_argument("--coarse-height-mm", type=float, default=340.0,
                   help="两阶段模式的孔面RGB-Z粗定位高度，默认340")
    p.add_argument("--fine-height-mm", type=float, default=260.0,
                   help="两阶段模式的孔面RGB-Z精定位高度，默认260")
    p.add_argument(
        "--per-hole-fine-safe-z-margin-mm",
        type=float,
        default=20.0,
        help=(
            "逐孔精拍回退时，安全横移高度相对精拍目标Z的余量(mm)，"
            "默认20；最终低位落位仍使用60 mm"
        ),
    )
    p.add_argument("--coarse-frames", type=int, default=10, help="两阶段最终粗定位最大有效RGB-D帧数")
    p.add_argument(
        "--coarse-settle-delay-s", type=float, default=0.6,
        help="到达配置的粗定位观察位后等待的停稳缓冲时间(秒)，默认0.6",
    )
    p.add_argument(
        "--coarse-recapture-settle-discard-frames",
        type=int,
        default=10,
        help="每次粗定位重拍前重新停稳后丢弃的RGB-D预热帧数，默认10",
    )
    p.add_argument(
        "--coarse-max-corrections",
        type=int,
        default=2,
        help="单孔粗定位最多姿态纠偏次数，默认2；最后一次采集只做验证",
    )
    p.add_argument(
        "--batch-coarse-localization",
        dest="batch_coarse_localization",
        action="store_true",
        default=False,
        help="批量粗定位模式：在配置的共同观察高度一次性检测当前组孔的位姿和深度",
    )
    p.add_argument(
        "--no-batch-coarse-localization",
        dest="batch_coarse_localization",
        action="store_false",
        help="关闭批量粗定位，恢复逐孔340 mm粗定位",
    )
    p.add_argument(
        "--save-all-capture-overlays",
        dest="save_all_capture_overlays",
        action="store_true",
        default=False,
        help="保存共享粗/精定位每一帧叠加图；默认关闭，只保存每次采集最后一帧",
    )
    p.add_argument(
        "--no-save-all-capture-overlays",
        dest="save_all_capture_overlays",
        action="store_false",
        help="关闭共享粗/精定位逐帧叠加图保存",
    )
    p.add_argument(
        "--batch-coarse-frames",
        type=int,
        default=15,
        help="普通/第四策略共同位姿稳定连拍帧数，默认15；同一批帧覆盖当前组孔",
    )
    p.add_argument(
        "--batch-coarse-min-valid",
        type=int,
        default=10,
        help="普通/第四策略共同粗定位每孔最少有效帧数，默认10",
    )
    p.add_argument(
        "--batch-coarse-early-stop-extra-frames",
        type=int,
        default=2,
        help="普通粗定位达到最少有效帧后额外确认的帧数，默认2；第四策略有独立默认值5",
    )
    p.add_argument(
        "--batch-coarse-settle-discard-frames",
        type=int,
        default=10,
        help="批量粗定位正式采集前丢弃的停稳预热RGB-D帧数，默认10",
    )
    p.add_argument(
        "--batch-coarse-min-holes-per-frame",
        type=int,
        default=None,
        help="批量粗定位模式每帧最少有效孔数，默认为None（表示全部选中孔）",
    )
    p.add_argument(
        "--batch-coarse-group-pose-refinement",
        dest="batch_coarse_group_pose_refinement",
        action="store_true",
        default=True,
        help="共享粗定位到达每组名义位姿后进行稳定多帧现场复核并按实测几何纠偏，默认开启",
    )
    p.add_argument(
        "--no-batch-coarse-group-pose-refinement",
        dest="batch_coarse_group_pose_refinement",
        action="store_false",
        help="关闭共享粗定位专用的到位现场复核；其它模式不受影响",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-frames", type=int, default=5,
        help="共享粗定位现场复核的稳定RGB-D帧数，默认5",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-min-valid-frames", type=int, default=3,
        help="共享粗定位现场复核每孔最少有效帧数，默认3",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-max-iterations", type=int, default=8,
        help="共享粗定位现场复核最多规划/复测轮数，默认8；支持大纠偏分步收敛",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-settle-discard-frames", type=int, default=5,
        help="共享粗定位现场复核每轮停稳后丢弃的RGB-D帧数，默认5",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-tracking-tolerance-px", type=float, default=45.0,
        help="共享粗定位现场复核检测框到投影锚点的最大匹配距离(px)，默认45",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-max-center-scatter-p95-px", type=float, default=1.5,
        help="共享粗定位现场复核每孔中心跨帧散布P95门限(px)，默认1.5",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-max-tracking-distance-p95-px", type=float, default=20.0,
        help="共享粗定位现场复核检测框跟踪距离P95门限(px)，默认20",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-max-reprojection-error-px", type=float, default=8.0,
        help="共享粗定位现场复核实测点重投影误差门限(px)，默认8",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-pose-tolerance-mm", type=float, default=2.0,
        help="共享粗定位现场复核最终位姿平移门限(mm)，默认2",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-rotation-tolerance-deg", type=float, default=1.0,
        help="共享粗定位现场复核最终姿态门限(度)，默认1",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-min-move-mm", type=float, default=0.5,
        help="共享粗定位现场复核触发机器人纠偏的最小平移量(mm)，默认0.5",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-max-correction-mm", type=float, default=5.0,
        help="共享粗定位现场复核允许的最大单步纠偏平移(mm)，默认5",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-max-correction-rotation-deg",
        type=float, default=2.0,
        help="共享粗定位现场复核允许的最大单步纠偏旋转(度)，默认2；仍同时受5mm平移门限约束",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-direct-motion",
        dest="batch_coarse_pose_refine_direct_motion",
        action="store_true",
        default=True,
        help="同一孔组340mm观察位的小步闭环纠偏直接原位MoveL，默认开启",
    )
    p.add_argument(
        "--no-batch-coarse-pose-refine-direct-motion",
        dest="batch_coarse_pose_refine_direct_motion",
        action="store_false",
        help="关闭同组原位小步纠偏，恢复抬升、横移、下降的保守路径",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-max-total-correction-mm",
        type=float,
        default=25.0,
        help="共享粗定位现场复核允许的累计纠偏平移上限(mm)，默认25",
    )
    p.add_argument(
        "--batch-coarse-pose-refine-max-total-correction-rotation-deg",
        type=float,
        default=7.0,
        help="共享粗定位现场复核允许的累计纠偏旋转上限(度)，默认7",
    )
    p.add_argument(
        "--map-build-safe-z-margin-mm", type=float, default=100.0,
        help="建图安全横移相对目标高度的Z余量(mm)，默认100",
    )
    p.add_argument(
        "--batch-coarse-view-margin-px",
        type=float,
        default=50.0,
        help="批量粗定位共同位姿的视野边缘安全余量(px)，默认50",
    )
    p.add_argument(
        "--batch-coarse-max-view-span-ratio", type=float, default=0.35,
        help="共享粗定位单组孔包围盒最大图像跨度比例，超过后自动拆组，默认0.35",
    )
    p.add_argument(
        "--batch-coarse-max-group-size", type=int, default=5,
        help="共享粗定位单组最多孔数，默认5",
    )
    p.add_argument(
        "--batch-coarse-group-max-aspect-ratio", type=float, default=2.0,
        help="共享粗定位单组3D点集最大长宽比，超过后拆组，默认2.0",
    )
    p.add_argument(
        "--batch-coarse-group-adjacency-factor", type=float, default=1.8,
        help="共享粗定位相邻孔距离相对最近邻中位数的最大倍数，默认1.8",
    )
    p.add_argument(
        "--batch-coarse-group-max-xy-diameter-mm", type=float, default=150.0,
        help="共享粗定位单组孔位在基坐标XY平面的最大直径(mm)，超过后拆组，默认150",
    )
    p.add_argument(
        "--batch-coarse-group-max-normal-spread-deg", type=float, default=3.0,
        help="共享粗定位单组最大法向离散角度(度)，默认3",
    )
    p.add_argument(
        "--batch-fine-localization",
        dest="batch_fine_localization",
        action="store_true",
        default=True,
        help="260mm批量精定位：每个视野一次采集并精定位全部选中孔，默认开启",
    )
    p.add_argument(
        "--no-batch-fine-localization",
        dest="batch_fine_localization",
        action="store_false",
        help="关闭260mm批量精定位，恢复逐孔精定位",
    )
    p.add_argument(
        "--coarse-direct-final",
        dest="coarse_direct_final",
        action="store_true",
        default=False,
        help="第四策略：独立高度点云中心直达，只使用ChArUco XY纠偏；点云不可用时记录失败",
    )
    p.add_argument(
        "--coarse-direct-final-height-mm",
        type=float,
        default=340.0,
        help="第四策略点云拍摄高度(mm)，指相机光轴到孔面距离，默认340",
    )
    p.add_argument(
        "--coarse-direct-final-max-group-size",
        type=int,
        default=5,
        help="第四策略内部共同观察位最多拍摄孔数，必须为1到5，默认5；外围孔自动先分组且最多3孔，所有分组仍执行紧凑度门限",
    )
    p.add_argument(
        "--coarse-direct-final-early-stop-extra-frames",
        type=int,
        default=5,
        help="第四策略达到最少有效帧后额外确认的帧数，默认5",
    )
    p.add_argument(
        "--coarse-direct-final-settle-delay-s",
        type=float,
        default=1.0,
        help="第四策略确认机器人停稳后额外等待的时间(秒)，默认1",
    )
    p.add_argument(
        "--coarse-direct-final-group-planning-timeout-s",
        type=float,
        default=10.0,
        help="第四策略分组搜索最长时间(秒)，超时保留已找到的完整分组或进入有界回退，默认10",
    )
    p.add_argument(
        "--coarse-direct-final-settle-discard-timeout-s",
        type=float,
        default=10.0,
        help="第四策略停稳后清理旧RGB-D帧的最长时间(秒)，默认10",
    )
    p.add_argument(
        "--coarse-direct-final-capture-timeout-s",
        type=float,
        default=60.0,
        help="第四策略一组点云采集的最长时间(秒)，默认60",
    )
    p.add_argument(
        "--coarse-direct-final-steady-timeout-s",
        type=float,
        default=45.0,
        help="第四策略等待机器人到位/停稳的最长时间(秒)，默认45",
    )
    p.add_argument(
        "--coarse-direct-final-max-consecutive-frame-failures",
        type=int,
        default=3,
        help="第四策略连续取帧失败次数上限，达到后中止当前组，默认3",
    )
    p.add_argument(
        "--coarse-direct-final-capture-only",
        dest="coarse_direct_final_capture_only",
        action="store_true",
        default=False,
        help="第四策略仅采集评估：仍移动到观察位并拍点云，不执行最终XY/Z动作",
    )
    p.add_argument(
        "--batch-fine-joint-localization",
        dest="batch_fine_joint_localization",
        action="store_true",
        default=True,
        help="260mm共享精定位启用多孔平面刚体联合XY；仅影响共享精定位路径，默认开启",
    )
    p.add_argument(
        "--no-batch-fine-joint-localization",
        dest="batch_fine_joint_localization",
        action="store_false",
        help="关闭260mm多孔联合XY，回退为现有共享精拍逐孔XY结果",
    )
    p.add_argument(
        "--batch-fine-joint-min-holes", type=int, default=2,
        help="共享精定位联合求解每帧所需的最少孔数，默认2",
    )
    p.add_argument(
        "--batch-fine-joint-min-valid-frames", type=int, default=5,
        help="共享精定位联合变换稳定验收所需的有效帧数，默认5",
    )
    p.add_argument(
        "--batch-fine-joint-max-residual-mm", type=float, default=1.0,
        help="共享精定位联合刚体拟合最大孔级残差(mm)，默认1.0",
    )
    p.add_argument(
        "--batch-fine-joint-max-translation-mm", type=float, default=5.0,
        help="共享精定位联合解在各内点孔实际位置的最大修正量门限(mm)，默认5",
    )
    p.add_argument(
        "--batch-fine-joint-max-yaw-deg", type=float, default=3.0,
        help="共享精定位联合平面旋转幅度门限(deg)，默认3",
    )
    p.add_argument(
        "--batch-fine-joint-stable-translation-mm", type=float, default=0.25,
        help="共享精定位联合跨帧平移稳定门限(mm)，默认0.25",
    )
    p.add_argument(
        "--batch-fine-joint-stable-yaw-deg", type=float, default=0.15,
        help="共享精定位联合跨帧旋转稳定门限(deg)，默认0.15",
    )
    p.add_argument(
        "--batch-fine-joint-local-residual-weight", type=float, default=0.25,
        help="联合XY结果中保留逐孔局部残差的权重，默认0.25",
    )
    p.add_argument(
        "--batch-fine-joint-local-residual-limit-mm", type=float, default=0.5,
        help="逐孔局部残差参与联合XY的最大限幅(mm)，默认0.5",
    )
    p.add_argument(
        "--batch-fine-pointcloud-xy-fusion",
        dest="batch_fine_pointcloud_xy_fusion",
        action="store_true",
        default=True,
        help="共享精定位最终XY启用粗拍点云支撑中心的门控融合，默认开启",
    )
    p.add_argument(
        "--no-batch-fine-pointcloud-xy-fusion",
        dest="batch_fine_pointcloud_xy_fusion",
        action="store_false",
        help="关闭共享精定位最终XY的点云中心门控融合",
    )
    p.add_argument(
        "--batch-fine-pointcloud-xy-weight", type=float, default=0.35,
        help="共享精定位最终XY中点云中心权重，默认0.35",
    )
    p.add_argument(
        "--batch-fine-pointcloud-xy-max-correction-mm", type=float, default=0.6,
        help="点云中心对共享精定位最终XY的最大修正量(mm)，默认0.6",
    )
    p.add_argument(
        "--batch-fine-pointcloud-xy-agreement-gate-mm", type=float, default=2.5,
        help="粗精XY超过该差异时拒绝点云融合(mm)，默认2.5",
    )
    p.add_argument(
        "--batch-fine-view-margin-px",
        type=float,
        default=50.0,
        help="260mm批量精定位共同位姿的视野边缘安全余量(px)，默认50",
    )
    p.add_argument(
        "--batch-fine-max-view-span-ratio", type=float, default=0.55,
        help="共享精定位单组孔包围盒最大图像跨度比例，超过后自动拆组，默认0.55",
    )
    p.add_argument(
        "--batch-fine-max-group-size", type=int, default=4,
        help="共享精定位单组最多孔数，默认4",
    )
    p.add_argument(
        "--batch-fine-group-max-aspect-ratio", type=float, default=1.8,
        help="共享精定位单组3D点集最大长宽比，超过后拆组，默认1.8",
    )
    p.add_argument(
        "--batch-fine-group-adjacency-factor", type=float, default=1.8,
        help="共享精定位相邻孔距离相对最近邻中位数的最大倍数，默认1.8",
    )
    p.add_argument(
        "--batch-fine-group-max-normal-spread-deg", type=float, default=5.0,
        help="共享精定位单组最大法向离散角度(度)，默认5",
    )
    p.add_argument(
        "--batch-fine-frames", type=int, default=8,
        help="260mm共同位姿严格几何圆心连拍最大帧数，默认8",
    )
    p.add_argument(
        "--batch-fine-min-valid", type=int, default=5,
        help="260mm共同精定位每孔最少严格几何圆心帧数，默认5",
    )
    p.add_argument(
        "--batch-fine-stable-min-frames", type=int, default=5,
        help="260mm共同精定位稳定验收所需严格几何圆心帧数，默认5",
    )
    p.add_argument(
        "--batch-fine-settle-discard-frames", type=int, default=10,
        help="260mm批量精定位正式采集前最少丢弃的停稳RGB帧数；随后自动确认已追上实时帧，默认10",
    )
    p.add_argument(
        "--batch-fine-inplace-recovery-frames", type=int, default=4,
        help="共享精拍质量不足时在当前位置追加采集的RGB帧数，默认4；仍失败才移动补拍",
    )
    p.add_argument(
        "--batch-fine-supplement-rounds", type=int, default=2,
        help="260mm共享精拍质量不足时移动到失败孔共同观察位的补拍轮数上限，默认2；第二步仅在首步被限幅时执行",
    )
    p.add_argument(
        "--batch-fine-max-geometric-anchor-distance-px", type=float, default=5.0,
        help="共享精拍几何圆心到粗定位投影锚点的最大距离(px)，默认5",
    )
    p.add_argument(
        "--batch-fine-fallback-max-coarse-to-fine-xy-mm", type=float, default=3.0,
        help="联合定位失败时允许逐孔回退的最大粗精XY位移(mm)，默认3",
    )
    p.add_argument(
        "--batch-fine-supplement-max-view-span-ratio", type=float, default=0.55,
        help="共享补拍单组孔投影包围盒最大图像跨度比例，默认0.55",
    )
    p.add_argument(
        "--batch-fine-in-group-pose-adjustment",
        dest="batch_fine_in_group_pose_adjustment",
        action="store_true",
        default=True,
        help="共享精拍失败孔启用同高度附近的组内受限位姿调整，默认开启",
    )
    p.add_argument(
        "--no-batch-fine-in-group-pose-adjustment",
        dest="batch_fine_in_group_pose_adjustment",
        action="store_false",
        help="关闭组内受限位姿调整，恢复安全高度共享补拍路径",
    )
    p.add_argument(
        "--batch-fine-in-group-max-adjustments", type=int, default=2,
        help="每个共享精定位组最多组内位姿调整次数，默认2；未发生有效限幅运动时不会追加第二步",
    )
    p.add_argument(
        "--batch-fine-in-group-max-xy-mm", type=float, default=8.0,
        help="组内单次直接调整最大XY距离(mm)，默认8",
    )
    p.add_argument(
        "--batch-fine-in-group-max-z-mm", type=float, default=3.0,
        help="组内单次直接调整最大Z距离(mm)，默认3",
    )
    p.add_argument(
        "--batch-fine-in-group-max-rotation-deg", type=float, default=2.0,
        help="组内单次RX/RY姿态调整最大旋转角(度)，默认2",
    )
    p.add_argument(
        "--batch-fine-in-group-min-normal-holes", type=int, default=2,
        help="允许组内姿态调整所需的最少可靠法向孔数，默认2",
    )
    p.add_argument(
        "--batch-fine-in-group-max-normal-spread-deg", type=float, default=3.0,
        help="允许组内姿态调整的最大法向离散角(度)，默认3",
    )
    p.add_argument(
        "--batch-fine-per-hole-fallback",
        dest="batch_fine_per_hole_fallback",
        action="store_true",
        default=True,
        help="共享精拍失败孔逐孔补拍并继续流程，默认开启",
    )
    p.add_argument(
        "--no-batch-fine-per-hole-fallback",
        dest="batch_fine_per_hole_fallback",
        action="store_false",
        help="关闭共享精拍失败孔的逐孔补拍；仅用于兼容旧行为",
    )
    p.add_argument(
        "--hole-map-mode",
        choices=("none", "build", "execute", "repair"),
        default="none",
        help=(
            "孔位地图模式：build建立孔位地图；"
            "execute调用粗地图并每次实时执行260 mm精定位；"
            "repair重新粗/精定位指定单孔并生成新地图版本，默认none"
        ),
    )
    p.add_argument(
        "--map-build-localization-mode",
        choices=("coarse_only", "per_hole"),
        default="coarse_only",
        help=(
            "建图定位方式：coarse_only只保存340 mm粗定位；"
            "per_hole逐孔执行340 mm粗定位和260 mm精定位，并保存每孔精定位参考；"
            "两种方式都不把最终TCP动作写入地图"
        ),
    )
    p.add_argument(
        "--hole-map-path",
        type=Path,
        default=None,
        help="孔位地图JSON路径；build不指定时自动版本化生成，execute/repair不指定时调用当前地图",
    )
    p.add_argument(
        "--hole-map-seed-correction-path",
        type=Path,
        default=None,
        help=(
            "地图种子XY纠正模型；execute模式默认自动读取地图目录下的"
            "hole_seed_correction.json"
        ),
    )
    p.add_argument(
        "--hole-map-seed-correction",
        dest="hole_map_seed_correction",
        action="store_true",
        default=True,
        help="地图执行时启用逐孔基准生成的XY种子纠正，默认开启",
    )
    p.add_argument(
        "--no-hole-map-seed-correction",
        dest="hole_map_seed_correction",
        action="store_false",
        help="本次地图执行不加载种子纠正模型",
    )
    p.add_argument(
        "--sector-id",
        type=int,
        default=None,
        help="六扇区地图的扇区编号 1..6；旋转扇区 build/execute/repair 时必须指定",
    )
    p.add_argument(
        "--hole-ids",
        type=int,
        nargs="+",
        default=None,
        help="execute模式要调用的孔号，例如 --hole-ids 1 3 2；不指定则调用全部有效孔",
    )
    p.add_argument(
        "--repair-hole-id",
        type=int,
        default=None,
        help="repair模式要重新定位的单个孔号；必须与 --sector-id 一起使用",
    )
    p.add_argument(
        "--shared-cache-validation", action="store_true", default=False,
        help="固定点云复用：按共同视野分组，在每组340mm少帧验证后跳过通过孔的逐孔粗定位",
    )
    p.add_argument(
        "--shared-cache-validation-frames", type=int, default=3,
        help="共享缓存快速验证每组RGB-D帧数，默认3",
    )
    p.add_argument(
        "--shared-cache-validation-min-valid", type=int, default=2,
        help="共享缓存快速验证每孔最少有效帧数，默认2",
    )
    p.add_argument(
        "--shared-cache-validation-view-margin-px", type=float, default=50.0,
        help="共享缓存验证共同340mm位姿的视野边缘余量(px)，默认50",
    )
    p.add_argument(
        "--optimize-hole-order", action="store_true", default=False,
        help="按当前批量精定位目标XY的最近邻顺序处理孔；默认保持初始选择顺序",
    )
    p.add_argument("--fine-frames", type=int, default=20, help="两阶段精定位最大有效RGB帧数")
    p.add_argument("--fine-settle-discard-frames", type=int, default=10,
                   help="每次精定位采集前丢弃的机器人/相机预热RGB帧数，默认10")
    p.add_argument("--fine-retries", type=int, default=2,
                   help="单孔精定位质量门失败后的自动重拍次数，默认2")
    p.add_argument(
        "--reuse-coarse-cache", dest="reuse_coarse_cache", action="store_true", default=True,
        help="旧两阶段流程复用本次运行初始多孔局部点云缓存；验证失败自动回退完整粗定位",
    )
    p.add_argument(
        "--no-reuse-coarse-cache", dest="reuse_coarse_cache", action="store_false",
        help="关闭旧两阶段局部点云缓存复用，保持原始粗定位流程",
    )
    p.add_argument(
        "--reuse-persistent-coarse-cache",
        dest="reuse_persistent_coarse_cache",
        action="store_true",
        default=True,
        help="旧两阶段按机器人基坐标复用跨运行粗定位缓存；验证失败自动回退",
    )
    p.add_argument(
        "--no-reuse-persistent-coarse-cache",
        dest="reuse_persistent_coarse_cache",
        action="store_false",
        help="关闭跨运行基坐标粗定位缓存，仅使用当前运行缓存",
    )
    p.add_argument("--move-final-xy", dest="move_final_xy", action="store_true", default=DEFAULT_MOVE_FINAL_XY,
                   help="显式启用精定位后的最终 TCP XY 微调")
    p.add_argument("--no-move-final-xy", dest="move_final_xy", action="store_false",
                   help="仅排障使用：关闭精定位后的最终 TCP XY 微调")
    p.add_argument("--tcp-xy-offset-mm", type=float, nargs=2, metavar=("DX", "DY"), default=None,
                   help="临时固定TCP XY补偿(mm)；默认不施加任何XY偏置")
    # 工作台 GUI 可用这些参数覆盖本机默认连接配置；命令行既有用法保持兼容。
    p.add_argument("--robot-ip", type=str, help="AUBO RPC IP（工作台传入）")
    p.add_argument("--robot-port", type=int, help="AUBO RPC 端口（工作台传入）")
    p.add_argument("--robot-user", type=str, help="AUBO 用户名（工作台传入）")
    p.add_argument("--robot-password", type=str, help="AUBO 密码（工作台传入）")
    p.add_argument("--robot-timeout-ms", type=int, help="AUBO 请求超时毫秒（工作台传入）")

    # TwoStageConfig 是两阶段参数默认值的唯一来源。CLI 中 add_argument
    # 仍保留逐项帮助文本，但不再独立维护另一套默认值。
    config_defaults = TwoStageConfig()
    parser_destinations = {action.dest for action in p._actions}
    synchronized_defaults = {
        field.name: getattr(config_defaults, field.name)
        for field in fields(config_defaults)
        if field.name in parser_destinations
    }
    synchronized_defaults["fine_retries"] = config_defaults.fine_retry_count
    p.set_defaults(**synchronized_defaults)
    return p
