# ADR-0011：M5 独立模型进程、逐帧产物与 mask 后处理

- 状态：工程实现决策；模型结果待用户验收
- 日期：2026-09-26
- 依据：system-design.md 9.2/9.3、implementation-plan.md M5；固定论文副本 docs/references/ego2robot-paper.html 附录 A.4。

论文明确给出 SAM 3 的 person / middle anchor / 400 帧与 50 overlap / OR 合并，以及 3 帧缺口、11 帧局部面积中位数 50%、5×5 ellipse close。ProPainter 参数为 fp16、neighbor_length=10、ref_stride=10、subvideo_length=80、mask_dilation=4、RAFT iterations=20。

论文没有规定本仓库的进程部署、PNG 索引格式、mask 缺口插值具体实现和所有确定性设置。决定如下：

1. 在独立解释器中执行官方模型适配器，通过受校验的 JSON 请求/结果传递路径与配置，使用 argv 调用进程。记录模型代码版本、checkpoint SHA-256、解释器/库版本、设备、随机种子。seed=0 是可覆盖的工程默认值，不能标作论文参数。启用模型支持的确定性选项；不支持时明确报错或记录限制，不能保证未经实测的跨设备位级复现。
2. 正式时序由源 PTS 和 frame_index 定义；PNG 帧与 bool mask 各有逐帧索引。视频是人工预览，不能取代原始时间戳。缺失帧或尺寸不符必须失败，不能截断到较短一边。
3. 对内侧长度 <=3 的空 mask 缺口，按两端 mask 的有符号距离场线性插值，零阈值二值化；不外推首尾空白或填长缺口。此选择比 RGB 插值后任意阈值更明确，但不能恢复真实运动，须保留 raw mask 和修复统计供人工检查。
4. 低面积 mask 替换的候选集合固定在该处理步骤开始时，最近帧距离相同选较早一帧，避免遍历顺序造成连锁替换。长缺口不可经面积替换间接补满。
5. 每个 stage 保存 manifest、输入/模型/配置指纹、全部输出清单和图片证据，成功发布后才供后续使用。依赖或模型缺失不能生成带正式 succeeded 的占位结果。

新配置采用默认禁用/显式新 profile 进入，旧 v1 几何产物不改动。M5 产物采用新的版本化索引；不要求重跑已验收的几何产物。详见 docs/m5-m8-workplan.md。

## 首轮低分辨率预览（用户于 2026-09-26 明确选择）

用户在获知本机 12GB 显存约束后选择“先用 640×360 生成预览，记录分辨率差异”。因此首轮 ProPainter 配置明确设 `inference_width=640`、`inference_height=360`。源视频和几何相机仍为 1920×1080；mask 使用 nearest-neighbor 缩放，RGB 使用 area 缩小，修复结果使用 cubic 回到源画布，仅在实际膨胀修复 mask 的区域写入，其他像素保留原图。保存处理分辨率结果与源画布结果，来源中写入 `preview_only=true` 和两套尺寸。

该选择是受显存限制的工程预览，不是论文分辨率设定，也不能宣称为原始 1080p 推理。后续正式视觉数据包必须识别这个标记，不能静默按 full-resolution 结果消费。

## Amendment（2026-11-02）：本地正式档由 640×360 改为 960×544，subvideo 由 80 改为 20

### 触发

四集联调时发现口径不一致：`egodex_vertical_pick_place_000` 的擦除已按 960×544 / subvideo 20 跑，
而 `egodex_pour_000`、`egodex_vertical_pick_place_049`、`egodex_vertical_pick_place_107` 仍是 640×360 / subvideo 80；
`configs/pipeline/m5.yaml` 仍写 640×360，`configs/pipeline/m5-inpaint.yaml` 则把 `inference_width/height` 留空
（等于原生 1920×1080，本机必然 OOM）。同一批视觉产物用两种分辨率不可比较，因此统一。

### 依据

2026-10-27 在租机 L40S 上做的**整集 52 帧单变量扫描**（`historical-local-evidence`、同目录 `metrics.json`）：
`ring_dark`（mask 外 4–20 px 环带残影）与 `core`（mask 内偏离时序底板）两个代理指标下，

- 时间维度参数（`ref_stride` 5/20、`neighbor_length` 20、真正触发切块的 `subvideo_length` 20）全部落在 1.05 / 16.2 附近，差异 ≤0.3，落在噪声内；
- `mask_dilation` 4→8→12 只把 `ring_dark` 从 1.22 降到 1.05→0.87（约 0.35 个灰阶，肉眼不可见），而 `core` 从 15.42 升到 16.22→18.51，即靠生成内容覆盖、保真度下降；
- 分辨率是唯一有明显正向作用的旋钮：960×540 的 `core` 15.00 优于 640×360 的 15.42。

（同一批对照中还有一组同时改了分辨率、subvideo、neighbor、ref_stride 的实验，得出了相反的
"时间窗决定质量"结论；该归因错误已作反例记录在学习工作区 `pits/PIT-047-擦除调参对照必须单变量.md`，
本 ADR 不采用该结论。）

### 决定

1. 本地正式档：`inference_width=960`、`inference_height=544`、`subvideo_length=20`、`mask_dilation=4`（保持论文默认）、其余 ProPainter 参数不变。两个 profile（`m5.yaml`、`m5-inpaint.yaml`）写成同一组值，避免再次分叉。
2. 高度取 544 而非扫描里的 540：`InpaintingConfig` 要求 `inference_width/height` 为 8 的倍数（`vision/m5_contracts.py`），540 会被配置校验直接拒绝。960×544 与 960×540 差 4 行像素，属同一档。
3. 把 `subvideo_length` 压到 20 是**换显存**的保守选择，不是质量选择：扫描证明时间参数对结果无影响，所以压缩它是安全的。
4. 640×360 降为**快速预览档**，仅在需要几分钟内看结构时使用，不作为可比较的正式产物。

### 已知代价与边界

- 本地 RTX 5070 只有 11.9 GiB：960×544 / sub20 可用（52 帧 344 s），1280×720 / sub40 直接 OOM（单次申请 4.95 GiB、累计 21.89 GiB）。**本地没有第二档可用分辨率**。
- 960×544 仍低于 1920×1080 源画布，因此 `inpainting` stage 会继续写 `preview_only=true`（判定条件见 `vision/m5_stages.py`：推理尺寸 ≠ 源尺寸）。这个标记的语义是"推理分辨率低于源分辨率"，**不表示"只能预览、不可用于后续比较"**；四集统一到同一推理分辨率后才可横向比较。
- `960×540 × dil=4` 这个交叉当时没有测（扫描的 H 组是 `dil=8`）。按"dil 越小 core 越好"的方向，本次配置应不差于 15.00，但**这是推断，不是实测**。
- 上述代理指标只用了一个 episode、一个场景，且没有人工真值；判断残留应看整段视频，不能看放大截图。
