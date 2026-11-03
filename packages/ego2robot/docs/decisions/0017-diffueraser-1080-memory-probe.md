# ADR-0017：DiffuEraser 1080p 与显存压缩实验

- 状态：实验完成，视觉结果待用户审阅；不替换正式 M5 路线
- 日期：2026-09-28
- 类型：[ENGINEERING]；论文未指定 DiffuEraser、裁块或量化参数
- 范围：官方 EgoDex episode 000，RTX 5070 12 GB，阴影扩展 mask

## 背景与决策

用户要求尝试 1080p，并评估权重量化和推理降显存。沿用 [ADR-0016](0016-diffueraser-comparison.md) 固定的官方 DiffuEraser 代码和权重、`2-Step` PCM、`nframes=22`、`seed=0`。先用 22 帧做整帧 1920×1080 探针，再用原分辨率的 656×368 裁块处理 52 帧，最后在同一裁块上测试 TorchAO INT8 weight-only。所有运行均与正式 M5 隔离。

整帧探针的先验是现有 ProPainter 的 `restored_frames`：画布是 1920×1080，但先验模型实际在 640×360 推理后放大。不能将其视为高分辨率先验。原视频、shadow mask 与先验以 FFV1 封装，逐帧输入哈希、PTS、参数见各候选 manifest。VAE tiling/slicing 启用；PyTorch 显存分配上限设 0.85。相关实验参数在 [`m5_diffueraser_1080.py`](../../tools/m5_diffueraser_1080.py)、[`m5_diffueraser_tiled_1080.py`](../../tools/m5_diffueraser_tiled_1080.py)、[`m5_diffueraser_quant_probe.py`](../../tools/m5_diffueraser_quant_probe.py) 固定。

## 整帧 1080p 结果

- [attention slicing 运行 manifest](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-1080-direct/manifest.json)、[失败状态图](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-1080-direct/review-frame-000000.png)：注意力处 CUDA OOM，单次申请约 1.96 GiB，未产生模型帧。
- [默认 SDPA 运行 manifest](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-1080-sdpa/manifest.json)、[失败状态图](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-1080-sdpa/review-frame-000000.png)：UNet 特征计算处 CUDA OOM，单次申请约 3.40 GiB，未产生模型帧。

第二次仍 OOM 表明瓶颈不限于注意力实现；此配置与这张 12 GB 卡上不能完成整帧 1080p 推理。官方 [DiffuEraser 仓库](https://github.com/lixiaowen-xw/DiffuEraser)报告 640×360 约 12 GB、960×540 约 20 GB、720p 约 33 GB（L20 上的估计），并未提供 1080p 保证。

## 原像素裁块 1080p 画布

52 帧 shadow mask 并集位于 `x=639..1919, y=374..1079`。四个 656×368 裁块覆盖该区域，横纵交叠各 16 px；使用原视频原像素裁切、相同位置的 shadow mask 和先验，分别运行 DiffuEraser，再羽化重叠并仅在 mask 附近合成到 1920×1080 原画布。见 [裁块 grid](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-1080-tiled/tile-grid-frame-000029.png)、[manifest](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-1080-tiled/manifest.json)、[52 帧并排视频](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-1080-tiled/comparison.mp4)、[第 29 帧](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-1080-tiled/review-frame-000029.png)。四块均解码 52 帧，单块 CUDA peak reserved 为 `8,694,792,192` bytes。合成画布为 1080p，**模型每次实际只看 656×368**。

第 0 和 29 帧可见残手、硬边及桌面纹理断裂，尤其在物体/手臂穿过裁块边界的位置。16 px 重叠不足以给扩散模型连续上下文。这个实验不能当作质量合格的 1080p 擦除输出；合成成功不代表视觉 gate 通过。

## INT8 权重量化结果

固定裁块 0 的 52 帧，用 TorchAO 0.16.0 `Int8WeightOnlyConfig` 分别量化 `UNetMotionModel` 与 `BrushNetModel` 的 `torch.nn.Linear`，其余模型与输入不变。量化的是权重，非中间激活；以下为 PyTorch CUDA allocated/reserved，非整机显存峰值。

| 裁块 0 推理 | 模型加载后 allocated | 量化后 allocated | 推理 peak reserved | 代表帧质量 |
|---|---:|---:|---:|---|
| FP16，无量化 | 未单独记录 | 未单独记录 | 8,694,792,192 B | 基准，有裁块固有缺陷 |
| UNet + BrushNet INT8 | 5,554,902,528 B | 4,545,887,232 B | 7,730,102,272 B | 出现大块黑色无效区域 |
| 仅 UNet INT8 | 5,554,902,528 B | 4,813,555,712 B | 8,009,023,488 B | 同样出现黑色无效区域 |
| 仅 BrushNet INT8 | 5,554,902,528 B | 5,287,234,048 B | 8,462,008,320 B | 抽看的第 0、29、44 帧接近 FP16，无明显黑块；完整视频尚待审阅 |

见 [全部量化 manifest](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-quant-int8/manifest.json)与[第 29 帧](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-quant-int8/review/frame-000029.png)、[UNet 单独量化 manifest](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-quant-int8-unet/manifest.json)、[BrushNet 单独量化 manifest](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-quant-int8-brushnet/manifest.json)及[第 29 帧](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-quant-int8-brushnet/review/frame-000029.png)。UNet 路线伴有输出转 uint8 时 NaN warning；黑块成因还需层级定位，不能断言是权重量化方法的一般缺陷。BrushNet 的约 267 MB 常驻节省和约 233 MB peak reserved 节省，远不足以填补整帧 1080p 的多 GiB OOM 缺口。

从两份主权重的 safetensors 形状粗分：总约 21.99 亿参数，其中二维矩阵约 9.93 亿、卷积四维约 12.05 亿。即使全部二维矩阵从 FP16 变 INT8，理论原始权重最多省约 0.93 GiB（实际实现另有 scale/metadata）；4-bit 约 1.39 GiB。实际可量化的 `nn.Linear` 子集小于或等于二维矩阵总量，且推理临时张量不随之同比下降。这个上界不保证整帧 1080p 可运行。

## 后续显存选项与限制

1. **保留默认 PyTorch SDPA，不与 attention slicing 同启**：这次 slicing 没解决 OOM；[Diffusers 文档](https://huggingface.co/docs/diffusers/api/pipelines/overview)明确警告 SDPA 或 xFormers 已启用时不应再叠加 attention slicing。
2. **VAE tiling/slicing**：已在探针中启用。tiling 分块处理 VAE 图像，slicing 主要改善多图 batch；它们不压缩 UNet 的全部时空中间张量。[官方内存指南](https://huggingface.co/docs/diffusers/main/optimization/memory)。
3. **CPU offload**：可减少同时驻留 GPU 的组件权重，但牺牲速度；本项目上游构造函数立即将 pipeline 移到 CUDA，不能未经适配直接调用 offload 并声称有效。即使卸载 VAE/text encoder，UNet 的整帧激活 OOM 仍可能存在。[官方内存指南](https://huggingface.co/docs/diffusers/main/optimization/memory)。
4. **缩短时间窗口或空间裁剪**：可降低激活规模；本次 `nframes=22` 是上游默认值，上游还要求有效视频至少 22 帧。更小的 `nframes` 是否可用尚未验证。裁块已证明显存可控但产生可见边界；更大的交叠、跨块一致性或局部背景底板需要独立视觉验证。
5. **INT4/更多模块量化**：权重还可进一步压缩，但当前 UNet INT8 已产生无效像素，降低位宽更不能直接视为可用。当前环境 Diffusers 0.29.2 的自定义模型/LoRA 也不能直接套用新版 Diffusers 的 [BitsAndBytes 加载示例](https://huggingface.co/docs/diffusers/quantization/bitsandbytes)。

当前可复现的结论是：12 GB 卡上完成了 52 帧原像素裁块推理和 1080p 画布合成，但视觉质量较差；整帧 1080p 两次 OOM；BrushNet INT8 仅能小幅省显存，UNet INT8 结果不可用。用户仍需查看图像和视频并决定视觉 gate。
