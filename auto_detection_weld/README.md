# Auto Detection Weld

这个目录提供一套独立于现有 `FoundationStereo` 主流程的原型，用于完成以下任务：

1. 接收自然语言指令，例如“检测两块钢板并提取中间 V 字坡口”。
2. 结合 `color + depth + camera_intrinsics.json` 识别两块钢板区域。
3. 从双平面几何关系和图像语义中提取焊接坡口/拼接区域。
4. 输出 2D 可视化、3D 坐标、坡口中心线和机械臂可用的候选引导位姿。

## 需求拆解

### 输入

- `color/*.png` 或单张彩色图像
- `depth/*.npy` 或 `depth/*.png`
- `camera_intrinsics.json`
- 自然语言提示词，例如：
  - `两块钢板`
  - `两块钢板之间的焊接 V 字坡口`
  - `定位拼接区域并给出三维引导`

### 目标输出

- 两块钢板的 2D 区域掩膜与边界框
- 候选坡口/拼接区域掩膜
- 坡口 2D 中心线
- 坡口 3D 中心线
- 焊接目标点 `target_point_m`
- 焊枪接近方向 `approach_direction_camera`
- 焊缝方向 `seam_direction_camera`
- 简单轨迹点：预接近点、接近点、焊接起点、焊接终点

## 方案设计

### 1. 自然语言引导检测

推荐正式方案是：

- 文本检测：`GroundingDINO` 或 OWL-ViT 类 open-vocabulary detector
- 精细分割：`SAM2/SAM` 类 segmentor
- 语言到几何：把“steel plate / weld joint / groove / seam”转为候选 ROI

当前这个原型目录里同时提供两条路径：

- `OpenVocabularyDetector`：如果环境里装了 `transformers` 并提供兼容模型，可以走开放词汇检测。
- `HeuristicPlateDetector`：不依赖大模型，只基于颜色、深度和平面几何做钢板候选分割，方便先打通流程。

### 2. 钢板提取

在 depth 点云中提取两个主平面：

- 将深度图反投影为 3D 点云
- 用 RANSAC/平面拟合提取两个最大平面
- 将平面重新投影成图像掩膜
- 结合颜色边缘和面积、连通性筛掉噪声

这一步解决“哪两块板是要焊接的板”。

### 3. 坡口/拼接区域提取

从两块板的几何关系中构建坡口带：

- 计算两块板的边界
- 生成两边界之间的中间带状区域
- 在该带中融合以下线索：
  - 深度凹槽响应
  - 法向变化
  - 图像暗线/高梯度
  - 两平面交界关系
- 得到坡口 mask 与中心线

### 4. 三维定位与机械臂引导

- 对坡口中心线反投影得到 3D 点集
- PCA 拟合焊缝方向 `seam_direction_camera`
- 利用两板法向计算坡口角平分方向 `groove_bisector`
- 生成：
  - `pre_approach_point_m`
  - `approach_point_m`
  - `weld_start_point_m`
  - `weld_end_point_m`

注意：这里输出的是**相机坐标系**下的目标。如果要给机械臂，需要再乘以 `T_base_camera`。

## 目录结构

- `cli.py`：命令行入口
- `pipeline.py`：主流程
- `detector.py`：自然语言检测和钢板候选提取
- `geometry.py`：平面拟合、坡口提取、3D 轨迹生成
- `io_utils.py`：数据读取与保存
- `config.py`：参数定义

## 运行示例

```bash
python auto_detection_weld/cli.py \
  --data_dir /path/to/orbbec_recording \
  --output_dir /path/to/orbbec_recording/weld_auto \
  --prompt "检测两块钢板，并提取中间焊接V字坡口，输出三维引导点" \
  --max_frames 10
```

如果只想跑单帧：

```bash
python auto_detection_weld/cli.py \
  --data_dir /path/to/orbbec_recording \
  --output_dir /path/to/orbbec_recording/weld_auto \
  --prompt "定位拼接区域并输出焊接方向" \
  --frame_id 000025
```

## 混合版推荐流程

当前更推荐使用：

- `foundation_stereo` 环境执行主流程和深度几何
- `SAM2` 通过本地 `FastSAM-Demo` 的 Python 环境调用
- `Qwen2.5-VL` 通过本地 `python39` 环境调用

推荐命令：

```bash
conda run -n foundation_stereo python auto_detection_weld/cli.py \
  --data_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260405_161057 \
  --output_dir /tmp/weld_hybrid_run \
  --start_frame_id 000030 \
  --max_frames 10 \
  --depth_source foundation_stereo \
  --runtime_mode save_only \
  --max_color_frame_delta 2 \
  --enable_sam 1 \
  --enable_qwen 1 \
  --qwen_max_new_tokens 128
```

说明：

- `--start_frame_id 000030`：从彩色帧开始连续存在的区段启动，适合这组数据。
- `--max_color_frame_delta 2`：允许深度和彩色帧有很小的编号偏差，避免早期数据完全配不到彩色图。
- `--enable_sam 1`：先用 SAM 产生钢板候选区域。
- `--enable_qwen 1`：再用 Qwen-VL 对候选区域做语义推理和筛选。

如果只想验证单帧：

```bash
conda run -n foundation_stereo python auto_detection_weld/cli.py \
  --data_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260405_161057 \
  --output_dir /tmp/weld_hybrid_single \
  --frame_id 000060 \
  --depth_source foundation_stereo \
  --runtime_mode save_only \
  --max_color_frame_delta 0 \
  --enable_sam 1 \
  --enable_qwen 1
```

输出 JSON 里会额外保存：

- `detector_backend`：`sam_qwen`、`sam_geometry` 或 `heuristic`
- `sam_candidates`：候选钢板区域的几何摘要
- `reasoning`：Qwen-VL 选择的候选 ID 和理由

## 当前原型的边界

- 这是第一版工程原型，重点是把“语言 -> ROI -> 几何 -> 3D 目标”链条打通。
- 如果现场反光强、钢板颜色接近背景、深度噪声大，建议正式版接入 `GroundingDINO + SAM2 + 时序滤波`。
- 若后续要真正控制机械臂，还需要：
  - 手眼标定 `T_base_camera`
  - 工具坐标系 `T_tool_tcp`
  - 安全高度与碰撞约束
  - 轨迹平滑与速度规划
