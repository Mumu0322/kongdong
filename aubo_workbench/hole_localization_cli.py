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
    default_final_target_mode: str,
    final_target_mode_gripper: str,
    final_target_mode_normal: str,
) -> argparse.ArgumentParser:
    DEFAULT_MODEL = default_model
    DEFAULT_HANDEYE = default_handeye
    DEFAULT_EXECUTE_MOTION = default_execute_motion
    DEFAULT_ALLOW_EXPERIMENTAL_HANDEYE = default_allow_experimental_handeye
    DEFAULT_TWO_STAGE_HOLE_LOCALIZATION = default_two_stage_hole_localization
    DEFAULT_MOVE_FINAL_XY = default_move_final_xy
    DEFAULT_FINAL_TARGET_MODE = default_final_target_mode
    FINAL_TARGET_MODE_GRIPPER = final_target_mode_gripper
    FINAL_TARGET_MODE_NORMAL = final_target_mode_normal
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
    p.add_argument("--coarse-height-mm", type=float, default=340.0,
                   help="两阶段模式的孔面RGB-Z粗定位高度，默认340")
    p.add_argument("--fine-height-mm", type=float, default=260.0,
                   help="两阶段模式的孔面RGB-Z精定位高度，默认260")
    p.add_argument("--coarse-frames", type=int, default=10, help="两阶段最终粗定位最大有效RGB-D帧数")
    p.add_argument(
        "--coarse-settle-delay-s", type=float, default=0.6,
        help="到达340 mm粗定位位后等待的停稳缓冲时间(秒)，默认0.6",
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
        help="批量粗定位模式：在340mm一次性检测所有选中孔的位姿和深度，然后按视野批量精定位",
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
        help="340mm共同位姿稳定连拍帧数，默认15；同一批帧覆盖全部选中孔",
    )
    p.add_argument(
        "--batch-coarse-min-valid",
        type=int,
        default=10,
        help="340mm共同粗定位每孔最少有效帧数，默认10",
    )
    p.add_argument(
        "--batch-coarse-early-stop-extra-frames",
        type=int,
        default=2,
        help="340mm粗定位达到最少有效帧后额外确认的帧数，默认2；满足稳定门后提前结束",
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
        "--batch-coarse-pose-refine-max-iterations", type=int, default=2,
        help="共享粗定位现场复核最多规划/复测轮数，默认2",
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
        "--batch-coarse-view-margin-px",
        type=float,
        default=50.0,
        help="批量粗定位共同位姿的视野边缘安全余量(px)，默认50",
    )
    p.add_argument(
        "--batch-coarse-max-view-span-ratio", type=float, default=0.55,
        help="共享粗定位单组孔包围盒最大图像跨度比例，超过后自动拆组，默认0.55",
    )
    p.add_argument(
        "--batch-coarse-max-group-size", type=int, default=9,
        help="共享粗定位单组最多孔数，默认9",
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
        "--batch-coarse-group-max-xy-diameter-mm", type=float, default=180.0,
        help="共享粗定位单组孔位在基坐标XY平面的最大直径(mm)，超过后拆组，默认180",
    )
    p.add_argument(
        "--batch-coarse-group-max-normal-spread-deg", type=float, default=8.0,
        help="共享粗定位单组最大法向离散角度(度)，默认8",
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
        "--batch-fine-joint-max-residual-mm", type=float, default=1.5,
        help="共享精定位联合刚体拟合最大孔级残差(mm)，默认1.5",
    )
    p.add_argument(
        "--batch-fine-joint-max-translation-mm", type=float, default=5.0,
        help="共享精定位联合平移幅度门限(mm)，默认5",
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
        "--batch-fine-view-margin-px",
        type=float,
        default=50.0,
        help="260mm批量精定位共同位姿的视野边缘安全余量(px)，默认50",
    )
    p.add_argument(
        "--batch-fine-max-view-span-ratio", type=float, default=0.60,
        help="共享精定位单组孔包围盒最大图像跨度比例，超过后自动拆组，默认0.60",
    )
    p.add_argument(
        "--batch-fine-max-group-size", type=int, default=3,
        help="共享精定位单组最多孔数，默认3",
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
        "--batch-fine-supplement-rounds", type=int, default=1,
        help="260mm共享精拍质量不足时移动到失败孔共同观察位的补拍轮数，默认1",
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
        "--batch-fine-supplement-max-view-span-ratio", type=float, default=0.60,
        help="共享补拍单组孔投影包围盒最大图像跨度比例，默认0.60",
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
        choices=("none", "build", "execute"),
        default="none",
        help=(
            "多孔孔位地图模式：build完成共享粗/精定位并保存地图；"
            "execute按地图孔号直接运动，不启动相机，默认none"
        ),
    )
    p.add_argument(
        "--hole-map-path",
        type=Path,
        default=None,
        help="孔位地图JSON路径；build不指定时自动版本化生成，execute不指定时调用当前地图",
    )
    p.add_argument(
        "--hole-ids",
        type=int,
        nargs="+",
        default=None,
        help="execute模式要调用的孔号，例如 --hole-ids 1 3 2；不指定则调用全部有效孔",
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
        "--final-target-mode",
        choices=(FINAL_TARGET_MODE_GRIPPER, FINAL_TARGET_MODE_NORMAL),
        default=DEFAULT_FINAL_TARGET_MODE,
        help="最终点模式：gripper=机械爪模式(X+64,Z+50)，normal=平常模式(无X/Z偏置)",
    )
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
