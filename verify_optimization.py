#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证共享粗定位和组内精定位优化参数是否正确应用

运行此脚本来检查优化配置是否生效
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aubo_workbench.hole_localization_models import TwoStageConfig


try:
    # Windows 传统控制台常用 GBK，避免验证脚本因为状态 emoji 不能输出而
    # 把“配置验证通过”误报成失败。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def test_optimized_config():
    """测试优化后的配置参数"""

    # 模拟命令行参数
    class Args:
        # 基础参数
        coarse_height_mm = 340.0
        fine_height_mm = 260.0

        # 优化参数
        batch_coarse_frames = 18
        batch_coarse_min_valid = 12
        batch_coarse_group_pose_refinement = True
        batch_coarse_pose_refine_max_iterations = 8
        batch_coarse_pose_refine_frames = 7
        batch_coarse_pose_refine_min_valid_frames = 5
        batch_coarse_pose_refine_max_correction_mm = 5.0
        batch_coarse_pose_refine_max_correction_rotation_deg = 2.0
        batch_coarse_pose_refine_max_total_correction_mm = 25.0
        batch_coarse_pose_refine_max_total_correction_rotation_deg = 7.0
        map_build_safe_z_margin_mm = 100.0
        coarse_settle_delay_s = 0.8
        batch_fine_max_view_span_ratio = 0.55
        batch_fine_max_group_size = 4
        batch_fine_supplement_rounds = 1
        batch_fine_in_group_pose_adjustment = True
        batch_fine_in_group_max_adjustments = 1
        batch_fine_in_group_max_xy_mm = 8.0
        batch_fine_in_group_max_z_mm = 3.0
        batch_fine_in_group_max_rotation_deg = 2.0
        batch_fine_in_group_min_normal_holes = 2
        batch_fine_in_group_max_normal_spread_deg = 3.0
        batch_fine_max_geometric_anchor_distance_px = 5.0
        batch_fine_fallback_max_coarse_to_fine_xy_mm = 3.0

    args = Args()

    # 创建配置
    try:
        cfg = TwoStageConfig.from_namespace(args)

        print("=" * 80)
        print("✅ 优化参数验证成功")
        print("=" * 80)
        print()

        # 显示关键优化参数
        print("📊 关键优化参数对比：")
        print()
        print("参数名称                                   | 默认值   | 优化值   | 改善")
        print("-" * 80)

        default = TwoStageConfig()

        checks = [
            ("map_build_safe_z_margin_mm", "建图安全余量(mm)", 100.0, 100.0, "保持安全值"),
            ("batch_coarse_frames", "粗定位采集帧数", 15, 18, "⬆️ +3帧"),
            ("batch_coarse_min_valid", "粗定位最少有效帧", 10, 12, "⬆️ +2帧"),
            ("batch_coarse_group_pose_refinement", "启用位姿纠偏", True, True, "✓ 启用"),
            ("batch_coarse_pose_refine_max_iterations", "纠偏最大迭代次数", 8, 8, "支持分步收敛"),
            ("batch_coarse_pose_refine_frames", "纠偏验证帧数", 5, 7, "⬆️ +2帧"),
            ("batch_coarse_pose_refine_min_valid_frames", "纠偏最少有效帧", 3, 5, "⬆️ +2帧"),
            ("batch_coarse_pose_refine_max_correction_mm", "纠偏单步平移(mm)", 5.0, 5.0, "小步闭环"),
            ("batch_coarse_pose_refine_max_correction_rotation_deg", "纠偏单步旋转(°)", 2.0, 2.0, "大姿态误差加快收敛"),
            ("batch_coarse_pose_refine_max_total_correction_mm", "纠偏累计平移(mm)", 25.0, 25.0, "累计保护"),
            ("batch_coarse_pose_refine_max_total_correction_rotation_deg", "纠偏累计旋转(°)", 7.0, 7.0, "累计保护"),
            ("coarse_settle_delay_s", "稳定等待时间(s)", 0.6, 0.8, "⬆️ +0.2s"),
            ("batch_fine_max_view_span_ratio", "精定位组视野跨度", 0.55, 0.55, "大组受限"),
            ("batch_fine_max_group_size", "精定位单组最多孔数", 4, 4, "3～4孔优先"),
            ("batch_fine_supplement_rounds", "精定位补拍轮数", 1, 1, "最多一次"),
            ("batch_fine_in_group_pose_adjustment", "启用组内位姿微调", True, True, "✓ 启用"),
            ("batch_fine_in_group_max_adjustments", "组内最多调整次数", 1, 1, "最多一次"),
            ("batch_fine_in_group_max_xy_mm", "组内最大XY(mm)", 8.0, 8.0, "平移限幅"),
            ("batch_fine_in_group_max_z_mm", "组内最大Z(mm)", 3.0, 3.0, "高度限幅"),
            ("batch_fine_in_group_max_rotation_deg", "组内最大姿态角(°)", 2.0, 2.0, "RX/RY限幅"),
            ("batch_fine_in_group_min_normal_holes", "姿态调整最少法向孔数", 2, 2, "禁止单孔调姿"),
            ("batch_fine_in_group_max_normal_spread_deg", "姿态调整法向离散(°)", 3.0, 3.0, "法向一致性门"),
            ("batch_fine_max_geometric_anchor_distance_px", "精定位几何锚点门(px)", 5.0, 5.0, "保持严格"),
            ("batch_fine_fallback_max_coarse_to_fine_xy_mm", "粗精安全回退门(mm)", 3.0, 3.0, "保持严格"),
        ]

        all_pass = True
        for attr_name, display_name, expected_default, expected_optimized, improvement in checks:
            actual_value = getattr(cfg, attr_name)
            default_value = getattr(default, attr_name)

            # 检查默认值是否正确
            if default_value != expected_default:
                print(f"⚠️  {display_name:40} | {default_value:8} | {actual_value:8} | 默认值异常")
                all_pass = False
                continue

            # 检查优化值是否应用
            if actual_value == expected_optimized:
                status = "✅"
            else:
                status = "❌"
                all_pass = False

            print(f"{status} {display_name:40} | {expected_default:8} | {actual_value:8} | {improvement}")

        print()
        print("=" * 80)

        if all_pass:
            print("✅ 所有优化参数已正确应用！")
            print()
            print("📈 需要现场复测的效果：")
            print("  • 组内视野跨度、XY直径和法向离散会被收紧")
            print("  • 大位姿偏差会拆成小步闭环纠偏")
            print("  • 精定位失败孔每组最多执行一次受限XY/Z/RX/RY微调")
            print("  • 真实精度和成功率不能仅由参数修改保证，需用新建图报告确认")
            print()
            print("💡 验证建议：")
            print("  1. 运行建图流程")
            print("  2. 查看叠加图：*_batch_coarse_*_last_frame.png")
            print("  3. 查看纠偏图：*_pose_refine_*_accepted.png")
            print("  4. 检查日志中的 plane_rmse_mm、center_scatter_p95_px 等指标")
            return 0
        else:
            print("❌ 部分参数未正确应用，请检查配置")
            return 1

    except Exception as e:
        print("=" * 80)
        print("❌ 配置验证失败")
        print("=" * 80)
        print(f"错误信息：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_optimized_config())
