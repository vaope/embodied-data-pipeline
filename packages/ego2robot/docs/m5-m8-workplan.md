# M5–M8 实施与交付计划

更新时间：2026-09-27。负责人：主 Agent 负责架构、接口、集成审查和进度；Luna worker（gpt-6-luna / max）负责范围明确的实现。最终验收人：用户。

## 本轮交付状态（2026-09-27）

- **已交付代码**：Luna 完成 `vision/mask_refine.py`、`vision/m5_contracts.py`，在 `PipelineConfig` 加入兼容旧配置的 M5 字段，并将 `vision` 渲染导出改为惰性加载。主 Agent 已静态审查这些文件及实际配置导入。
- **已记录**：两份规范的开发补充、ADR-0011、官方接口调研及 episode 000 用户验收记录 `quality/reviews/m3-video.json`。
- **已准备运行环境**：WSL Ubuntu 22.04、Python 3.12.13、torch 2.10.0+cu128 / torchvision 0.25.0+cu128。GPU 可用，SAM3 builder 与 ProPainter core 实际导入成功。依赖快照为 `docs/runtime/m5-wsl-requirements.txt`。
- **历史权重记录**：原研发环境已有 SAM3 与 ProPainter 权重；发布副本不附这些文件。使用者显式配置本机路径，ProPainter 模型目录示例为 `assets/third_party/ProPainter/weights/`。原 episode 000 的哈希记录不随代码分发。
- **官方代码固定**：SAM3 `2345a4ad109ac29c569da749c91d84f10dc08c40`；ProPainter `e870e79321c31b733e2031af5aa2fb1fe3ac7eec`。本机共享 M5 模型环境是工程候选环境，实际推理兼容性仍待检查。
- **Luna 已恢复执行**：额度中断后按用户要求重试，已交付模型 bridge/worker、两个 stage、预览模块、M5 profile、注册入口及环境诊断工具。主 Agent 已完成集成审查与真实样本运行。
- **实际环境检查已完成**：`tools/m5_runtime_report.py` 在现有 episode 000 上输出 READY，确认 SAM3 / ProPainter 导入、CUDA、1+3 个 checkpoint 的存在及 SHA-256；图像证据为 `previews/m5-readiness/readiness.png`。该检查不执行模型推理，不代表 M5 视觉验收。
- **真实运行已完成**：SAM3 52/52 帧（87.53 秒，run `1821e4b8037c`）；ProPainter 640×360 52/52 帧（72.97 秒，run `7d03aa82cbcd`）。重跑 `4c0807e3d49d` 两个 stage 都命中缓存。52 个非空 mask 仅表示产物存在，不等于视觉覆盖合格。
- **数据核对**：源帧、遮罩、背景的 frame_index / PTS 一致；208 个图像校验和全部一致；背景尺寸一致；实际膨胀修复区域外改动像素数为 0。证据见 `previews/m5-delivery-report.json` 与 `previews/m5-delivery.png`。
- **待用户验收**：代表帧中手臂主体已移除，但桌面仍有明显手臂阴影及局部模糊；视频时序效果、接触边界和操作物体保留须用户审阅。M5 gate 未勾选，M6–M8 尚未实现。

验收入口：`segmentation/previews/mask-review.mp4`、`inpainting/previews/background-review.mp4`，各自目录中另有 8 张代表帧。程序完成不表示 M5 验收通过；按用户意见修正或进入 M6。

## 状态更新（2026-11-02）：擦除分辨率口径统一到 960×544

- 上面 2026-09-27/09-28 的记录保留为历史：当时的正式档是 640×360 预览分辨率，四集里只有 episode 000 后来被单独提到 960×544，其余三集仍停在 640×360 / subvideo 80，`configs/pipeline/m5.yaml` 与 `m5-inpaint.yaml` 也互相分叉（后者留空 = 原生 1080p，本机必 OOM）。同一批视觉产物用两种推理分辨率无法横向比较。
- 现按 [ADR-0011 Amendment](decisions/0011-m5-runtime-and-mask-contract.md)（依据历史整集 52 帧单变量扫描）统一为 **960×544 / subvideo 20 / dil 4**，两个 profile 写同一组值；640×360 降为快速预览档。
- 三集（`egodex_pour_000`、`egodex_vertical_pick_place_049`、`egodex_vertical_pick_place_107`）已按新档重跑 `inpainting`，并级联重跑 `scene_depth` 与 `composite`。四集的 `inpainting/stage-manifest.json` 推理尺寸一致，`segmentation/bundle/masks/refined` 未被改动（重跑前后哈希一致）。
- **仍然不构成 M5 或 M6 验收。** 960×544 低于 1920×1080 源画布，`preview_only` 仍为 true；擦除残留、接触边界与操作物体保留仍须用户看视频判定。M5/M6 gate 未勾选。

## 开工依据与边界

- 用户已在本任务审阅 `egodex_vertical_pick_place_000/previews/projection-overlay.mp4`，明确反馈“从视频来看是合格的”；随后明确要求继续 M5–M8。
- 本轮以 episode 000 的已审阅几何结果为开发基线。这不扩大为全部合成样例或其他 episode 的验收。
- 先落实 M5，再按 M6 → M7 → M8 推进。每阶段提供真实结果图、视频（时序阶段）、数值和来源记录；程序运行完成与用户验收分开记录。
- 沿用已有 52 帧 episode 000，不下载完整训练集。不自行编造验收输入，不创建自动化测试用例。
- 尚未安装的模型、权重、授权与硬件适配明确记为依赖，不能以空实现、人工背景或伪造结果宣称完成。

## 任务队列

| 顺序 | 任务 | 具体交付 | 进入下阶段需要的证据 |
|---|---|---|---|
| M5.1 | 核对 SAM 3 / ProPainter 官方接口与本机条件 | `m5-runtime-readiness.md`，官方来源、版本与权重路径、运行环境方案 | 官方推理调用可追溯；缺失依赖明确 |
| M5.2 | 分割输入输出与 mask 后处理 | 按 PTS 配对的源帧；400/50 分块；重叠 OR；短缺口、面积异常、close；raw/refined masks | 原图、raw/refined 叠加与面积曲线 |
| M5.3 | SAM 3 真实分割适配 | `segmentation` stage，文本 person，中间帧 anchor，独立模型进程 | episode 000 的完整 mask 与代表帧 |
| M5.4 | ProPainter 时序修复适配 | `inpainting` stage，论文参数，PNG 背景帧、预览视频、修复区对照 | 手臂被覆盖、物体保留、背景时序合理，由用户审阅 |
| M6.1 | 深度来源与米制对齐决策 | 官方 DA3 适配核对、ADR、raw depth/confidence、明确 metric/aligned/relative | 可信的尺度依据；不能拿机器人表面深度冒充同像素真实场景深度 |
| M6.2 | 场景深度与遮挡合成 | `scene_depth` / `composite`，arm/gripper 分开处理、decision map、遮挡 mask | **已实现，视觉审查待进行**（2026-10-28，ADR-0020）：episode 000 两个 stage 跑通、锚点残差已记录、三情形对照图已出；未被用户验收 |
| M7.1 | L1/L2 与统计版本 | 真实计算指标、逐源统计、invalid reason、版本化阈值 | M6 结果齐全；样本不足时不虚构 dataset statistics |
| M7.2 | 数据包与读取导出 | 校验和清单、canonical reader、至少一个明确映射的 exporter、图文报告 | 用户可逐帧检查 image + 7D action，数据与报告一致 |
| M8.1 | WiLoR 与 tracking | 纯 RGB 检测/MANO、身份关联、SAM hand mask 过滤、wrist jump filter | M7 完成；MANO/模型授权与运行环境就绪 |
| M8.2 | DynHaMR 与尺度处理 | 时序优化、优化前后轨迹、metric scale 来源、统一 HandSequence | 纯 RGB 使用同一片段与 Path A 对照，重新提供 M1–M3 证据并由用户验收 |

M9、多机器人、训练与 policy evaluation 不在本轮任务中。

## Luna 委派与文件所有权

| Worker | 首轮范围 | 完成条件 |
|---|---|---|
| `luna_m5_readiness` | 官方接口与环境调研；`docs/m5-runtime-readiness.md` | 核对实际 API/CLI、依赖、授权及资源约束；不猜接口 |
| `luna_m5_masks` | `vision/mask_refine.py` | 独立可调用的分块/合并/后处理；确定性与输入校验 |
| 后续 M5 adapter worker | `vision` 下模型 worker 与桥接模块 | 实际官方模型调用、缺依赖明确失败、输出校验 |
| 后续 M5 stage worker | 配置、stage、registry 与预览 | 接入 runner，完整 manifest、fingerprint/resume、时序和尺寸检查 |

主 Agent 在委派前固定跨模块接口；相互依赖的实现顺序交接，同期 worker 不改相同文件。主 Agent 检查实际变更与产物后，再呈交用户。

## M5 新增契约与兼容策略

1. 新增 `segmentation` 与 `inpainting` 两个 runtime stage，各自拥有独立目录及 stage manifest；已有 `mvp` / `repro` 默认阶段列表保持有效。
2. 新增显式 M5 profile，直接复用 episode 000 已有摄取/几何产物。重模型运行前核对文件、时间轴、尺寸、设备与权重。
3. 持久化逐帧索引：`schema_version`、`frame_index`、`timestamp_s`、`valid`、图像尺寸、PNG 文件及哈希、模型来源/版本、配置、seed 和运行环境。背景帧以 PNG 为正式产物，视频仅为预览；视频的固定预览帧率不改变源 PTS。
4. 模型依赖采用独立 Python 进程，主几何环境保持轻量。显式配置解释器、仓库、checkpoint 与设备；不自动执行任意 shell 字符串、不静默下载模型，不把依赖缺失降级为假结果。
5. `segmentation` 保存 raw/refined masks、面积曲线、代表帧叠加；`inpainting` 保存按相同索引配对的背景帧、前后对照与视频。
6. 新字段有默认值，旧配置和 v1 几何产物继续可读；旧 episode 通过新 profile 增量添加视觉目录。已有产物不就地改变 schema 或坐标语义。M6–M8 的新增 schema 在对应阶段开工前补充两份规范和迁移说明。

工程选择见 ADR-0011；运行/模型验收尚未完成时，状态只能记为实现中或依赖待就绪。

## 当前依赖与未结项

- 本机：Windows、Python 3.13.5、RTX 5070 12GB，核心环境未安装 torch/SAM3/DA3/WiLoR。真实推理能力需官方依赖核对后落实。
- 用户已明确选择首轮 ProPainter 640×360 预览；配置与产物保存推理尺寸、原始尺寸和 preview_only 标记，详见 ADR-0011 补充。WSL Ubuntu 22.04 可用于独立模型环境。
- SAM 3 与 MANO 权重若要求用户授权或登录，只在确实需要用户操作时提出具体事项。
- 合成样例关节分支跳变、投影检查未纳入硬 gate、confidence 与相机 provenance 问题继续单独跟踪；不得把本轮用户对 episode 000 视频的认可扩展为这些问题已解决。
- M6 的深度尺度与 M7 的跨数据统计需要真实证据；M8 需要纯 RGB 重建的模型资产。不得为了让所有阶段显示 succeeded 而填入占位数据。
