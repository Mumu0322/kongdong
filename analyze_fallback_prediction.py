#!/usr/bin/env python3
"""
分析共享精定位回退模式，建立预测模型

思路：
1. 从逐孔精定位数据中提取每个孔的"真实"特征（基准）
2. 从共享精定位数据中提取回退的孔
3. 分析回退孔与成功孔的特征差异
4. 建立预测模型：哪些孔在共享模式下容易失败
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# 数据路径
PER_HOLE_DIR = Path(r"C:\MM_two\aubo_tools\data\hole_localization_runs\two-stage-20260916_153939")
SHARED_DIR = Path(r"C:\MM_two\aubo_tools\data\hole_localization_runs\two-stage-20260916_164346")

def extract_per_hole_features(frames_csv):
    """从逐孔精定位数据中提取每个孔的特征"""
    df = pd.read_csv(frames_csv)

    # 按孔分组
    hole_features = {}

    # 从stage字段提取孔号
    df['hole_id'] = df['stage'].str.extract(r'hole_(\d+)')[0].astype(float)

    for hole_id, group in df.groupby('hole_id'):
        if pd.isna(hole_id):
            continue

        hole_id = int(hole_id)

        # 分离粗定位和精定位数据
        coarse_frames = group[group['stage'].str.contains('coarse', na=False)]
        fine_frames = group[group['stage'].str.contains('fine', na=False)]

        features = {
            'hole_id': hole_id,

            # === 粗定位特征 ===
            'coarse_attempts': len(coarse_frames['stage'].unique()),  # 粗定位尝试次数
            'coarse_total_frames': len(coarse_frames),
            'coarse_plane_rmse_mean': coarse_frames['plane_rmse_mm'].mean() if len(coarse_frames) > 0 else None,
            'coarse_plane_rmse_std': coarse_frames['plane_rmse_mm'].std() if len(coarse_frames) > 0 else None,
            'coarse_tracking_distance_mean': coarse_frames['tracking_distance_px'].mean() if len(coarse_frames) > 0 else None,
            'coarse_tracking_distance_max': coarse_frames['tracking_distance_px'].max() if len(coarse_frames) > 0 else None,
            'coarse_center_u_std': coarse_frames['center_u_px'].std() if len(coarse_frames) > 0 else None,
            'coarse_center_v_std': coarse_frames['center_v_px'].std() if len(coarse_frames) > 0 else None,

            # === 精定位特征 ===
            'fine_total_frames': len(fine_frames),
            'fine_valid_frames': len(fine_frames[fine_frames['error'].isna()]),
            'fine_rejected_frames': len(fine_frames[fine_frames['error'].notna()]),
            'fine_ellipse_residual_mean': fine_frames['ellipse_residual_px'].mean() if len(fine_frames) > 0 else None,
            'fine_ellipse_residual_std': fine_frames['ellipse_residual_px'].std() if len(fine_frames) > 0 else None,
            'fine_geometric_anchor_distance_mean': fine_frames['geometric_anchor_distance_px'].mean() if len(fine_frames) > 0 else None,
            'fine_geometric_anchor_distance_max': fine_frames['geometric_anchor_distance_px'].max() if len(fine_frames) > 0 else None,
            'fine_tracking_distance_mean': fine_frames['tracking_distance_px'].mean() if len(fine_frames) > 0 else None,
            'fine_center_u_mean': fine_frames['center_u_px'].mean() if len(fine_frames) > 0 else None,
            'fine_center_v_mean': fine_frames['center_v_px'].mean() if len(fine_frames) > 0 else None,
            'fine_center_u_std': fine_frames['center_u_px'].std() if len(fine_frames) > 0 else None,
            'fine_center_v_std': fine_frames['center_v_px'].std() if len(fine_frames) > 0 else None,
        }

        hole_features[hole_id] = features

    return hole_features

def extract_shared_results(result_summary_path):
    """从共享精定位结果中提取哪些孔回退了"""
    with open(result_summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析result_summary.txt
    fallback_holes = []
    shared_success_holes = []

    for line in lines:
        if line.startswith('H'):
            parts = line.split()
            if len(parts) >= 3:
                hole_id = int(parts[0][1:])  # H01 -> 1
                source = parts[2]

                if source == 'per_hole_fallback':
                    fallback_holes.append(hole_id)
                elif source.startswith('shared'):
                    shared_success_holes.append(hole_id)

    return {
        'fallback_holes': fallback_holes,
        'shared_success_holes': shared_success_holes
    }

def analyze_features():
    """分析特征差异"""
    print("=== 提取逐孔精定位特征（基准真值） ===")
    per_hole_features = extract_per_hole_features(PER_HOLE_DIR / "frames.csv")
    print(f"提取了 {len(per_hole_features)} 个孔的特征")

    print("\n=== 提取共享精定位结果 ===")
    shared_results = extract_shared_results(SHARED_DIR / "result_summary.txt")
    print(f"回退孔: {shared_results['fallback_holes']}")
    print(f"共享成功孔: {shared_results['shared_success_holes'][:10]}...")

    # 构建训练数据
    print("\n=== 构建特征对比 ===")

    fallback_features = []
    success_features = []

    for hole_id in shared_results['fallback_holes']:
        if hole_id in per_hole_features:
            feat = per_hole_features[hole_id].copy()
            feat['label'] = 'fallback'
            fallback_features.append(feat)

    for hole_id in shared_results['shared_success_holes']:
        if hole_id in per_hole_features:
            feat = per_hole_features[hole_id].copy()
            feat['label'] = 'success'
            success_features.append(feat)

    # 转换为DataFrame
    df_fallback = pd.DataFrame(fallback_features)
    df_success = pd.DataFrame(success_features)

    print(f"\n回退孔数量: {len(df_fallback)}")
    print(f"成功孔数量: {len(df_success)}")

    # 特征对比
    print("\n" + "="*80)
    print("特征对比：回退孔 vs 成功孔")
    print("="*80)

    feature_cols = [col for col in df_fallback.columns if col not in ['hole_id', 'label']]

    comparison = []
    for col in feature_cols:
        fallback_mean = df_fallback[col].mean()
        success_mean = df_success[col].mean()
        fallback_std = df_fallback[col].std()
        success_std = df_success[col].std()

        if pd.notna(fallback_mean) and pd.notna(success_mean):
            diff_pct = ((fallback_mean - success_mean) / success_mean * 100) if success_mean != 0 else 0

            comparison.append({
                'feature': col,
                'fallback_mean': fallback_mean,
                'success_mean': success_mean,
                'diff_pct': diff_pct,
                'fallback_std': fallback_std,
                'success_std': success_std
            })

    df_comparison = pd.DataFrame(comparison)
    df_comparison['abs_diff_pct'] = df_comparison['diff_pct'].abs()
    df_comparison = df_comparison.sort_values('abs_diff_pct', ascending=False)

    # 显示最显著的差异
    print("\n最显著的特征差异（按差异百分比排序）：")
    print("-" * 120)
    print(f"{'特征':<45} {'回退孔均值':<15} {'成功孔均值':<15} {'差异%':<12} {'区分度':<10}")
    print("-" * 120)

    for _, row in df_comparison.head(20).iterrows():
        feature = row['feature']
        fallback_val = row['fallback_mean']
        success_val = row['success_mean']
        diff_pct = row['diff_pct']

        # 计算区分度（简单版：标准差比）
        discriminative = "高" if abs(diff_pct) > 20 else "中" if abs(diff_pct) > 10 else "低"

        print(f"{feature:<45} {fallback_val:<15.3f} {success_val:<15.3f} {diff_pct:>+11.1f}% {discriminative:<10}")

    # 保存详细对比
    df_comparison.to_csv('fallback_feature_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细特征对比已保存到: fallback_feature_comparison.csv")

    # 保存所有特征数据
    df_all = pd.concat([df_fallback, df_success], ignore_index=True)
    df_all.to_csv('all_hole_features.csv', index=False, encoding='utf-8-sig')
    print(f"所有孔特征数据已保存到: all_hole_features.csv")

    return df_comparison, df_all

def generate_prediction_rules(df_comparison):
    """根据特征差异生成预测规则"""
    print("\n" + "="*80)
    print("预测规则建议")
    print("="*80)

    # 选择区分度高的特征（差异>20%）
    high_discriminative = df_comparison[df_comparison['abs_diff_pct'] > 20]

    print("\n基于以下特征的预测规则：")
    for _, row in high_discriminative.head(10).iterrows():
        feature = row['feature']
        fallback_val = row['fallback_mean']
        success_val = row['success_mean']
        diff_pct = row['diff_pct']

        if diff_pct > 0:
            threshold = (fallback_val + success_val) / 2
            print(f"\n  IF {feature} > {threshold:.2f}:")
            print(f"     → 高风险（回退孔均值={fallback_val:.2f}, 成功孔均值={success_val:.2f}）")
        else:
            threshold = (fallback_val + success_val) / 2
            print(f"\n  IF {feature} < {threshold:.2f}:")
            print(f"     → 高风险（回退孔均值={fallback_val:.2f}, 成功孔均值={success_val:.2f}）")

    print("\n" + "="*80)
    print("实施建议")
    print("="*80)
    print("""
1. 在共享精定位分组时，可以使用以上特征来评估每个孔的"共享友好度"
2. 将高风险孔单独分组，或者直接跳过共享，走逐孔流程
3. 可以在batch_fine_workflow中添加预筛选逻辑：
   - 读取粗定位缓存中的特征
   - 计算每个孔的风险评分
   - 高风险孔直接标记为per_hole_fallback
4. 这样可以避免共享尝试 → 失败 → 补拍 → 再失败 → 回退的浪费时间
    """)

if __name__ == '__main__':
    df_comparison, df_all = analyze_features()
    generate_prediction_rules(df_comparison)
