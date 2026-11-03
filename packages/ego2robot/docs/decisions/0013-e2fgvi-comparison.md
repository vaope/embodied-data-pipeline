# ADR-0013：E2FGVI-HQ 作为 M5 替代路线的 episode 000 对照

- 状态：候选推理受环境依赖阻塞；无模型质量结论；不改变正式 M5 路线
- 日期：2026-09-27
- 范围：仅限 episode 000、640×360 预览，供与已运行的 ProPainter 预览对照
- 先决条件：episode 000 的 M3 投影视频已由用户认可；M5 结果仍需独立视觉验收

## 决策

保留 E2FGVI-HQ 作为可复查的 M5 候选，不以它替换正式 ProPainter stage。先固定官方代码与权重，并用相同的 52 张 640×360 ProPainter 处理输入帧、同一组 refined masks 和相同 PTS sidecar 建立对照。只有官方依赖可导入且真实推理完成后，才比较两种模型的修复质量。

当前环境未安装 E2FGVI 所需的 MMCV 自定义算子，且 WSL 缺少 `nvcc`，因此没有运行 E2FGVI-HQ。候选目录中的 E2FGVI-HQ 面板明确标为“未推理”；面板同时显示真实 source、refined mask 和真实 ProPainter 结果。它们是障碍状态图，不是 E2FGVI 结果，不能据此评价模型优劣。

## 官方来源与固定资产

- 官方代码：[MCG-NKU/E2FGVI](https://github.com/MCG-NKU/E2FGVI)，固定 commit `709cbe319edc21b8a365a28e14cba595a93d62cf`。
- 官方模型说明、依赖、`test.py` 参数及输出行为：[README（固定 commit）](https://github.com/MCG-NKU/E2FGVI/blob/709cbe319edc21b8a365a28e14cba595a93d62cf/README.md)、[test.py（固定 commit）](https://github.com/MCG-NKU/E2FGVI/blob/709cbe319edc21b8a365a28e14cba595a93d62cf/test.py)、[environment.yml（固定 commit）](https://github.com/MCG-NKU/E2FGVI/blob/709cbe319edc21b8a365a28e14cba595a93d62cf/environment.yml)。
- 官方 HQ 权重：[Google Drive 文件](https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view)，文件 `E2FGVI-HQ-CVPR22.pth`，下载大小 `164,535,938` bytes，SHA-256 `afff989d41205598a79ce24630b9c83af4b0a06f45b137979a25937d94c121a5`；本机保存于 `assets/third_party/E2FGVI/release_model/`。Drive 匿名请求需经官方 virus-scan 确认页再下载，未遇到登录墙。
- 官方仓库声明代码采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)；仅限非商业用途，商业使用须联系作者。

## 运行兼容性证据

官方 `test.py` 的模型实现导入 `mmcv.ops.ModulatedDeformConv2d` / `modulated_deform_conv2d`、`mmcv.cnn` 和 `mmcv.runner`。上游 `environment.yml` 固定的是 Python 3.7.13、PyTorch 1.5.1 + CUDA 10.1 和 `mmcv-full==1.4.8`。本机用于 M5 的隔离运行时为 Python 3.12.13、PyTorch 2.10.0+cu128、CUDA 12.8，能访问 RTX 5070（12 GB）；NumPy、OpenCV、Pillow、Matplotlib 和 tqdm 均可导入，但 `mmcv` / `mmcv.ops` 缺失，且没有 `nvcc`。在没有可验证的兼容 MMCV CUDA 扩展前不尝试模型推理，也不替换成未经官方适配的算子。

这次阻塞发生在模型加载之前。12 GB GPU 对 640×360 HQ 推理的实际显存占用尚未测得，不能仅凭分辨率声称模型可运行或显存足够。没有修改 M5 stage、安装/编译 MMCV、载入 checkpoint 或产生任何 E2FGVI 模型输出。

## 可复现对照设置

如果后续提供经验证的 MMCV 扩展，工具 `tools/m5_e2fgvi_compare.py` 使用官方 `test.py`，参数为 `--model e2fgvi_hq --set_size --width 640 --height 360 --step 10 --num_ref -1 --neighbor_stride 5 --savefps 30`。输入来自现有 ProPainter stage 的 `processing_input/frames` 与 `processing_input/masks`，逐文件复制到该候选目录并记录哈希；避免重新缩放后两条路线输入发生差异。E2FGVI 官方 `read_mask()` 会额外应用 `cv2.MORPH_CROSS` 3×3 kernel、4 次膨胀；ProPainter 预览使用同一 refined-mask 输入及其对应的官方四次 cross dilation。ProPainter 对照取同一 stage 的 640×360 `processing_frames`，不是 restore 到 1920×1080 的正式画布结果。

源视频的 `frame_index`、timestamp_s、source/index/mask hashes 保存在候选 manifest；固定 30 fps 的 MP4 仅供预览，PTS sidecar 才是逐帧时间依据。选取至少 8 张代表帧制作 2×2 对照图：共享输入、共享 mask overlay、ProPainter 官方输出、E2FGVI-HQ 输出或清楚标明的未推理面板。状态视频、图像、输入副本、阻碍报告和 manifest 只写入 `episode000/previews/m5-alternatives/e2fgvi-hq/`，按输入、mask、权重、代码 revision 与配置指纹支持 resume。

## 当前运行记录

- 状态：`blocked`；52 帧未运行 E2FGVI-HQ。
- 唯一运行阻塞：隔离运行时无法导入 `mmcv.ops`（`ModuleNotFoundError: No module named 'mmcv'`）；此外该 WSL 环境没有 `nvcc`，所以未尝试构建 CUDA 扩展。
- 环境诊断：`episode000/previews/m5-alternatives/e2fgvi-hq/manifest.json`。
- 说明及运行障碍：[blockers.md](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/e2fgvi-hq/blockers.md)。
- 8 张 source/mask/ProPainter 与 E2FGVI 状态对照图：[comparison-frames](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/e2fgvi-hq/comparison-frames/)。
- 52 帧状态视频：[comparison-preview.mp4](../../data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/e2fgvi-hq/comparison-preview.mp4)。

本记录不是 M5 gate 通过、模型效果胜负或用户视觉验收的声明。E2FGVI-HQ 权重/模型许可为非商业限制；当前 ProPainter 对照已实际推理，E2FGVI-HQ 仍无输出。
