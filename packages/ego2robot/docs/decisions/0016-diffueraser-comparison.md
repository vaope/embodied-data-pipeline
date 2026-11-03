# ADR-0016：DiffuEraser 的 episode 000 人臂擦除对照

- 状态：实验完成，视觉结果待用户审阅；不替换正式 M5 路线
- 日期：2026-09-28
- 类型：[ENGINEERING]，论文没有给出 DiffuEraser 配置
- 范围：官方 EgoDex episode 000 的 52 帧、640×360 预览；原 refined mask 与阴影扩展 mask 各跑一次

## 背景与决策

用户要求实跑 DiffuEraser，与已有 ProPainter 结果比较。使用官方 [DiffuEraser 代码](https://github.com/lixiaowen-xw/DiffuEraser)（commit `8e6f279ac7531e27ad1849c6f8dab5372a8597e7`）和 [官方权重](https://huggingface.co/lixiaowen/diffuEraser)，隔离于正式 M5 stage。分别对原 refined mask 和现有阴影扩展 mask 运行一次；每组内两种模型使用同一组现存 640×360 输入 PNG 和 mask PNG。各组已有的 ProPainter 处理帧作为 DiffuEraser 的 prior，以避免重复运行 ProPainter。输入以 FFV1 无损视频封装，52 帧及原 mask 像素经解码核对；固定 30 fps 只用于模型输入与预览，原 PTS 由候选 manifest 逐帧保存。

运行参数由 [`tools/m5_diffueraser_compare.py`](../../tools/m5_diffueraser_compare.py) 固定：`max_img_size=640`、`nframes=22`、`mask_dilation_iter=4`、`seed=0`、`blended=true`、PCM `2-Step`。取 `mask_dilation_iter=4` 是沿用当前 ProPainter 预览的名义膨胀次数，**两种模型内部的形态学算子不同**，因此只称“相同输入 mask”，不称“相同有效修复区域”。DiffuEraser 输出为其官方 MP4 编码路径；ProPainter 对照为已有 PNG 原处理帧。跨 mask 的比较仅用于解释覆盖率对阴影的影响。

## 运行证据

- [候选 manifest](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser/manifest.json)记录逐帧 PTS、输入哈希、模型权重 SHA-256、官方代码 revision、配置、运行时与结果视频哈希。
- [四路并排视频](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser/comparison.mp4)：源画面、原 mask ProPainter、扩展阴影 mask ProPainter、原 mask DiffuEraser。
- [第 29 帧四路图](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser/comparison-frames/four-route-000029.png)及同目录其他 7 张代表帧；[第 29 帧输入 mask 与同 mask 结果图](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser/comparison-frames/frame-000029.png)。
- [DiffuEraser 独立视频](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser/diffueraser-result.mp4)可用于时序审阅。实际解码 52 帧。PyTorch 报告 peak allocated `8,210,681,344` bytes、peak reserved `9,298,771,968` bytes；这不等于整个系统的 GPU 峰值。
- [阴影扩展 mask 的候选 manifest](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-shadow/manifest.json)、[四路并排视频](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-shadow/comparison.mp4)、[第 29 帧](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-shadow/comparison-frames/four-route-000029.png)与[独立 DiffuEraser 视频](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/diffueraser-shadow/diffueraser-result.mp4)。实际解码 52 帧，peak allocated `8,207,721,984` bytes、peak reserved `9,449,766,912` bytes。

## 观察与限制

人工抽看帧 0、29、44：两组 DiffuEraser 都去掉了手臂主体，透明盒和纸盒仍可见。**原 mask 组**的桌面手形阴影继续保留，与同输入 mask 的 ProPainter 相比没有解决当前阴影问题。**扩展阴影 mask 组**的左侧手形残影在这些帧上明显变淡，某些区域看起来比同 mask ProPainter 更干净；但右侧仍有暗影，局部桌面和透明盒周边的细节也更柔和。改善主要来自扩展修复区域，模型带来的进一步收益和时序稳定性仍需用户看完整并排视频判断；没有无人的真实场景作为客观保真基准。

本实验只证明 12GB RTX 5070 在这个 52 帧、640×360、复用先验的配置下可运行。它不证明 1080p 可运行，也不代表 M5 视觉 gate 已获用户验收。任何后续正式替换均需单独审查物体保护、原 PTS 对齐、结果边界与完整视频。
