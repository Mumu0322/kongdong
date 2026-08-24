# 批量粗定位功能设计方案

## 需求
为旧两阶段孔洞检测增加批量粗定位功能：
- **粗定位阶段（340mm）**：一次性检测多个孔的位姿和深度
- **精定位阶段（260mm）**：逐个移动到每个孔，分别拍摄确定最终位置

## 当前实现状态

### 两阶段流程（当前逐孔处理）
1. 在初始位置选择多个孔
2. **逐个孔**：移动到340mm → RGB-D粗定位 → 移动到260mm → RGB精定位

## 实现方案

### 方案A：为两阶段流程增加批量粗定位模式（推荐）

#### 新增命令行参数
```python
--batch-coarse-localization    # 启用批量粗定位模式（默认关闭，保持兼容）
--batch-coarse-frames N        # 批量粗定位采集帧数，默认15
--batch-coarse-min-valid N     # 批量粗定位最少有效帧数，默认10
--batch-coarse-min-holes N     # 每帧最少有效孔数，默认等于选中孔数
```

#### 工作流程
```
1. 初始位置：用户点击选择N个孔（保持现有逻辑）
2. 计算整组孔的共同观察位姿（340mm）
   - 使所有孔都在视野中心区域
   - 固定RZ，调整XY使孔组居中
3. 移动到共同340mm位姿
4. **批量采集所有孔的深度**（15帧RGB-D）
   - 每帧检测所有YOLO孔
   - 为每个孔计算：中心、法向、深度
   - 跨帧融合：每个孔的中心中位数、法向融合、深度中位数
5. 逐孔精定位（保持现有260mm逻辑）
   - 移动到孔1的260mm位 → RGB精定位
   - 移动到孔2的260mm位 → RGB精定位
   - ...
```

#### 核心函数设计

```python
def _batch_coarse_localization_at_340mm(
    selected_holes: list[dict],  # 初始选中的孔
    current_tcp: np.ndarray,
    handeye: Any,
    pipeline: Any,
    align: Any,
    chain: Any,
    model: Any,
    cfg: BatchCoarseConfig,
    intrinsics: Any,
    run_dir: Path,
) -> dict[int, dict[str, Any]]:
    """
    在340mm共同位姿一次性检测所有孔的位姿和深度
    
    返回: {hole_id: {
        "center_base_mm": [x, y, z],
        "normal_base": [nx, ny, nz],
        "depth_mm": float,
        "valid_frames": int,
        "center_scatter_p95_mm": float,
    }}
    """
    pass

def _plan_batch_coarse_group_pose(
    selected_holes: list[dict],
    current_tcp: np.ndarray,
    handeye: Any,
    fixed_rz_rad: float,
    intrinsics: Any,
    target_height_mm: float,
    view_margin_px: float,
) -> tuple[np.ndarray, dict]:
    """
    规划能看到所有孔的340mm共同位姿
    用于两阶段流程的共同观察位姿规划
    """
    pass
```

#### 修改点
1. **`run_yolo_eye_in_hand_optimized.py`**：
   - 在 `build_parser()` 增加 `--batch-coarse-localization` 等参数
   - 在 `_run_sequential_hole_workflow()` 增加批量粗定位分支
   - 新增 `_batch_coarse_localization_at_340mm()` 函数
   - 新增 `_plan_batch_coarse_group_pose()` 函数

2. **GUI `gui_hole_localization.py`**：
   - 在"旧定位"页面增加"批量粗定位"选项
   - 增加批量粗定位参数输入框

## 推荐实现

采用方案A的原因：
1. 保持向后兼容：默认关闭，不影响现有用户
2. 灵活性高：不依赖外部模型，纯视觉驱动
3. 性能提升：多孔场景下，粗定位只需一次340mm采集

## 实现优先级
1. **核心逻辑**：批量粗定位函数（~300行代码）
2. **参数接口**：命令行和GUI参数（~50行代码）
3. **工作流集成**：修改 `_run_sequential_hole_workflow`（~100行代码）
4. **测试验证**：单元测试和现场测试

## 预期效果
- **3孔场景**：粗定位从 3×30秒 → 1×30秒（节省约1分钟）
- **5孔场景**：粗定位从 5×30秒 → 1×30秒（节省约2分钟）
- **质量保持**：融合多帧数据，精度不降低
