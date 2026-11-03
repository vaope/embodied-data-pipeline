# Ego2Robot 数据链路实施步骤

状态：`v0.2 / normative`  
更新时间：`2026-09-20`  
依赖设计：[`system-design.md`](system-design.md)

> **2026-09-26 补充**：用户已验收 episode 000 的 M3 投影视频并明确要求继续 M5–M8，任务拆分及交付顺序见 [M5–M8 实施计划](m5-m8-workplan.md)。先由 Luna worker 落实 M5；后续阶段按依赖与用户验收推进。旧报告的程序 gate pass 不等于用户验收；全部合成样例的 M3 问题尚未关闭。M5 新增契约与旧数据迁移见该计划和 ADR-0011。下列 2026-09-20 内容作为历史实现记录。
>
> **历史实现记录（2026-09-20）**：M0–M3 已实现，并在 5 个 synthetic_geometry case 上
> 通过各自 gate（见 `data/fixtures/report.md`）。M4–M9 未开始。
> 代码位于 `src/ego2robot/`，运行入口为 `e2r`（见 [`../README.md`](../README.md)）。
> 本文档的 checkbox 已按实际实现勾选；与实现冲突的条款应修改本文档，而不是让实现
> 默默偏离。
>
> 本轮实现中论文未公开、需要工程决策的点全部写入
> [`decisions/`](decisions/README.md)（ADR-0001 … ADR-0010），并在配置中可追溯。


## 1. 执行目标

按“先数据契约、再几何闭环、后视觉链、最后规模化”的顺序，实现：

```text
1 个可控 ego clip
  -> hand pose
  -> canonical TCP/action
  -> Franka qpos
  -> robot-only render
  -> human-free background
  -> depth-aware synthetic video
  -> QC + canonical dataset
```

训练、policy evaluation、15 morphology 并行和自动 VLM 清洗不属于 MVP。任何下游阶段必须等上游 gate 通过，不能用“先把流程跑完”替代几何正确性。

## 2. 里程碑总览

| 阶段 | 主要交付 | 进入条件 | 退出 Gate |
|---|---|---|---|
| M0 基础骨架 | 包、配置、schema、CLI、stage runner | 文档批准 | schema/transform/unit tests 全绿 |
| M1 受控输入 | fixture、Path A、相机与 hand overlay | M0 | 视频/hand/timebase 对齐 |
| M2 Canonical action | retarget、过滤、补帧、平滑 | M1 | TCP/orientation/gripper golden tests |
| M3 几何闭环 | fixed base、Franka IK/FK、robot render | M2 | 投影对齐 + IK/FK 指标达标 |
| M4 Base search | 论文式候选、keyframe、score | M3 | 自动 placement 达到固定 base 基线 |
| M5 Human removal | SAM 3 adapter、mask refine、ProPainter | M3 | 时序 mask/inpainting 通过人工审查 |
| M6 Depth composite | scene depth、scale align、occlusion | M4/M5 | 遮挡测试和 synthetic video 通过 |
| M7 QC/export | L1/L2、canonical package、export adapter | M6 | 端到端数据契约完整、可重跑 |
| M8 Path B | WiLoR、tracking、DynHaMR | M7 | RGB-only clip 达到 M2/M3 同等 gate |
| M9 扩展 | 双臂、多 robot、L3、吞吐 | M8 | 单独立项与指标 |

## 3. M0：工程基础与可执行骨架

目标：在接入重模型前，把坐标、schema、stage 生命周期和测试规则固定下来。

### 任务

- [x] 建立 `pyproject.toml`，采用 `src/` layout；固定 Python 版本。lockfile 见 `docs/environment-report.md`（pip freeze 全量列表）；实测环境为 Python 3.13.5，与设计文档的 3.11 基线不同，差异记录在 environment-report。
- [~] `pyproject.toml` 已配置 ruff / mypy / pytest。按 AGENTS.md rule 7，本仓库不写自动化测试，因此 CI 未接入；验证改由 `e2r fixtures verify` 产出可人工审查的证据。
- [x] 实现 `core.frames`：frame 枚举、`T_A_B` 解析与校验、frame 约定注册表（`e2r frames`）。
- [x] 实现 `core.schema`：上述全部 schema，另加 `CameraFile`、`RobotConfig`、`StageManifest`。数组 schema 的 shape/dtype/frame/unit 在 `ARRAY_SPECS` 中声明并在 stage 边界强制校验。
- [x] 实现 `core.artifacts`：episode 目录契约、fingerprint、sha256、`TempPublish` 原子发布、partial 清理与 resume 判定。
- [x] 实现配置分层解析（`core.config`），resolved config 落盘到 `configs/resolved.json` 并写入 fingerprint。
- [x] 实现 `pipeline/stage.py`、`runner.py`、`registry.py`：cache、`--until`、`--force-stage`、失败 manifest、上游 payload 传递。
- [x] 建立 CLI（e2r）；除设计要求的四条命令外，另有 rames、stages、ixtures build/verify、etch（单 episode 按 HTTP range 从官方 EgoDex zip 抽取）：

```text
e2r inspect <episode>
e2r run <episode> --profile mvp --until <stage>
e2r validate <episode>
e2r viz <episode> --view <name>
```

- [x] 建立 `docs/decisions/`，含索引与 10 篇 ADR。

### 测试

- transform inverse/composition/property tests；
- quaternion `xyzw` round-trip、hemisphere correction；
- schema 对 shape、dtype、unit/frame metadata 的失败用例；
- stage 同 fingerprint 命中 cache；配置变化只使当前及下游 stage 失效；
- stage failure 不发布半成品。

### Gate M0

- `pytest`、type check、lint 全绿；
- CLI 能对 synthetic fixture 做 dry-run，并打印完整 DAG；
- transform 测试覆盖已知 90° 旋转和非交换乘法用例；
- 任意 persisted pose 都能回答“从哪个 frame 到哪个 frame、单位是什么”。

## 4. M1：EgoDex 公开数据、视频摄取与 Path A

目标：建立唯一、可重复的小视频基准，并证明视频、相机、3D hand pose 的时间对齐正确。

### 初始公开数据源

使用 [Apple EgoDex 官方仓库](https://github.com/apple/ml-egodex) 提供的公开测试集。测试集约 16GB，是官方最小切分；MP4 与同名 HDF5 成对，HDF5 提供逐帧 3D 手部/身体关节、置信度和相机变换。初期只抽取一个短 episode，不下载约 725 小时的完整训练集。

### 首个 episode 选择规范

- 5–10 秒、30 fps 左右；
- 单手，手在绝大多数帧可见；
- 简单 pick–move–place；
- camera motion 小；
- 优先选择相机运动较小、标注置信度较高的 episode；
- 使用 EgoDex 提供的相机变换和 metric 3D hand joints，并由 adapter 转换到本项目 camera optical frame。

不要把私人原视频提交到 Git；测试仓库只放可公开的小型派生 fixture 或程序生成数据。

### 任务

- [x] 视频 probe（PyAV 为主、OpenCV 为后备）：codec、分辨率、PTS、measured fps、frame count、首帧 sha256、PTS 单调性检查。
- [x] `ingest` stage 生成 `episode.json` 与 `camera.json`，并保留 source 已声明的 `T_C_B`。
- [x] 定义 `perception/hand/base.py::HandPoseProvider`。
- [x] 实现 `ingest/egodex.py::EgoDexHandAdapter`：world→camera、ARKit24→MANO-21、confidence、重投影自检。映射与缺失关节策略见 ADR-0003；论文未提及 ARKit，故该步为 [ENGINEERING]。
- [x] `hand/sequence.npz` + `hand/provenance.json` 保存来源 frame、单位、joint order 映射、confidence 来源与投影自检结果。
- [x] `viz/hand_overlay.py` + `perception/hand/preview.py`：skeleton、handedness、confidence、missing 状态、TCP 轴、`previews/hand-overlay.mp4` 与 stills。
- [x] `viz/frames_3d.py` + `previews/hand-3d.png`：相机视锥、wrist 轨迹、四个 retarget 关键点、TCP 坐标系。

### 测试

- joint name/order mapping；
- millimeter -> meter 或 normalized -> metric 转换必须显式且可测；
- PTS 与 hand timestamp 最近邻/插值规则；
- 投影 `K * p_C` 与提供 2D keypoints 的重投影误差。

### Gate M1

- 所有 frame 都能追溯到 source PTS；
- 关键点 overlay 与手部像素位置一致，无 x/y mirror；
- metric depth 的方向符合 `+z forward`；
- missing/low-confidence frame 被标记，不被零向量代替；
- 产出 `previews/hand-overlay.mp4` 和 `hand/stage-manifest.json`。

## 5. M2：Hand -> canonical gripper

目标：完全脱离具体机器人，得到连续、可检查的 `T_C_TCP + gripper_width` 和 7D relative action。

### 任务

- [x] `retarget/hand_to_gripper.py`：virtual fingertip、TCP、width、handedness sign（论文 eq. 1-3）。
- [x] canonical `R=[x y z]` 经 SVD 投影回 SO(3)，修正量写入 `so3_correction_rad`。
- [x] 实现两类 degeneracy fallback（`w < 10 mm`、`||z x d|| ~ 0`）与 `FallbackCode` 位标志；无数值 epsilon 的判据见 ADR-0004。
- [x] `retarget/outliers.py`：论文 eq. 7 阈值、严格 2 轮、逐轮拒绝数入库。
- [x] `retarget/gaps.py`：≤10 帧插值、>10 帧 home 混合，blend length 公式逐字实现；home 语义见 ADR-0006。
- [x] `retarget/smoothing.py`：窗口 `min(21,n)`、polyorder `min(3,window-1)`，并把 window 自动修正为合法奇数（短序列/偶数窗口）。
- [x] `core.transforms.hemisphere_correct` + `retarget/smoothing.py` 的切空间高斯加权 SLERP。
- [x] `CanonicalActionSequence.relative_action`：左乘约定 `dR = R[t+1] R[t]^T`，输出 `canonical/actions-delta.npz`。
- [x] `previews/gripper-frame-*.png`（TCP 三轴 + jaw line + width）与手部 3D viewer。

### 单元测试矩阵

| 用例 | 预期 |
|---|---|
| right/left mirror hand | 两者映射到一致 canonical jaw axis |
| 静止 hand | delta action 接近 0 |
| 沿 camera +x 平移 | 只有 position delta x 为正 |
| 已知 90° rotation | rotvec 方向和幅度正确 |
| `w < 0.01 m` | 冻结上一 orientation 并记录 fallback |
| `z` 与 `d` 平行 | 不产生 NaN/Inf |
| quaternion 正负号交替 | smoothing 后无 180° 假跳变 |
| 短序列/偶数 window | Savitzky–Golay 参数自动合法化 |
| 5 帧 gap / 20 帧 gap | 分别走插值/home blend 路径 |

### Gate M2

- 每个有效 rotation 满足 `R.T @ R ~= I`、`det(R) ~= 1`；
- 没有非预期 NaN/Inf；
- 合成轨迹 golden tests 的 position/orientation/gripper 误差在定义容差内；
- fixture preview 无 handedness flip、axis flip 和明显高频 jitter；
- 原始、filtered、smoothed 三条轨迹都被保留用于对比，正式输出只引用 smoothed 版本。

### 限定扩展：双手数据（2026-11-02，ADR-0021）

不属于 M2 的验收，也不解除 M3–M9 的任何依赖；它是 PLAN-02 B0/B1 的**受限交付**，用来先拿到真实双手数据证据：

- [x] `HandConfig.selection`（`legacy|both|explicit`）+ `sides`，`hand_count` 必须与 sides 一致；`configs/pipeline/bimanual.yaml` 只到 `retarget`。
- [x] `EgoDexHandAdapter` 一次读两侧（`hands=(right,left)`），共享 PTS/相机，`keypoints_C_m` 等为 `[T,2,...]`；缺失任一侧的必需关节直接失败。
- [x] `core/capabilities.py` + 各 stage `validate_inputs` 守卫：robot/render/projection/quality/export/segmentation/scene_depth/composite 拒绝 `H=2`，不再 `[:,0]` 静默丢手。
- [x] `retarget` 逐手执行过滤/补洞/平滑/SO(3) 修正，输出带 `hand_ids` 的 `[T,2,...]` raw/filtered/final/delta。
- [x] 证据图：两手骨架 + 四点 + TCP 三轴叠图（≥6 帧）、逐手开度曲线、跨手距离/相对旋转（raw vs final）、双手 TCP 3D、逐手有效性 JSON。
- [x] 回归：四个既有 `H=1` episode 用新代码重跑 `hand`+`retarget`，数组与改动前逐值一致（逐值比对，不是“能跑”）。
- [ ] 用户审阅 B1 证据（骨架与像素对齐程度、跨手关系是否可信）。
- [ ] B2/B3（双臂 rig、联合场景、owner map、按臂质量门、导出）按 ADR-0022 立项后再做。

Gate（B1，机器证据，不等于验收）：`H=2` 两侧共享时间轴与身份；两手的 `hand_ids`/`handedness` 一致；有效帧四点有限、旋转合法；`H>1` 被所有未迁移入口拒绝；`H=1` 输出不变。

### 限定扩展：双臂 rig（2026-11-02，ADR-0023，B2 前一半）

同样不属于 M3–M9 的验收，只交付"两臂各自 IK/FK + 联合碰撞 + 投影证据"：

- [x] `robots/ik.py`：把单臂逐帧 IK 循环抽成共享函数；`robot` stage 改调它，`egodex_pour_000` 190 帧 `trajectory.npz` 逐值不变（14 个数组）。
- [x] `robots/rig.py`：由 vendored Panda 生成只含碰撞几何的双臂场景（body/joint/tendon/equality/actuator/exclude 全前缀），世界系=右臂基座系；拒绝 det=-1 的"镜像基座"。
- [x] `robots/rig_stage.py`（stage `rig`）：逐臂 `RobotTrajectory`、`rig.json`（绑定/双基座/相对放置/场景哈希）、`contacts.npz`（逐帧跨臂接触点数、去重 pair 数、FK 偏差、可见性代理）。
- [x] `core/config.py::RigConfig` + `configs/pipeline/bimanual-rig.yaml`；`base_mode: per_arm_search|fixed`，搜索后把两套 `T_C_B` 冻结进 `rig.json`。
- [x] `BaseSearchConfig.min_forward_depth_m`（默认 0 不影响单臂）：第一次 rig 运行把右臂基座放在相机前 0.117 m，臂横跨画面；加 0.30 m 约束后回到 0.459 m。
- [x] 证据：两集逐臂 IK 100%、p95 ≈3e-6 m、跨臂接触 0、联合场景 FK 与独立 FK 差 ≤4.5e-9 m、两臂同时有效 100%；joint jump 显式 gate **未通过**（左 3 帧 / 右 1 帧 >1.5 rad）。
- [ ] 用户审阅 B2 几何证据（投影叠图、3D、rig.json 数字）。
- [ ] 联合双基座 pairwise 搜索（top_k×top_k + 可见性/侧别/间距/跨臂碰撞约束）——用当前结果作 baseline。
- [ ] B3（渲染/擦除/合成/owner map）与 B4（导出/reader）按 ADR-0022 另立。

Gate（B2-v1，机器证据）：两臂 IK 各自 ≥95%、联合场景 FK 与独立 FK 一致、跨臂接触不超阈值、逐臂与联合有效率分开报告、joint jump 进显式 gate；不通过项必须出现在 metrics 而不是被平均掉。

## 6. M3：Fixed-base Franka 几何闭环

目标：完成本项目最重要的 gate：`T_C_TCP -> T_B_TCP -> qpos -> FK -> image projection`。

### 任务

- [x] 采用 `mujoco_menagerie/franka_emika_panda/panda.xml`（Apache-2.0），`RobotConfig` 记录 license/source/sha256/joint 名/限位；见 ADR-0005。
- [x] `T_E_TCP` 由模型几何导出（finger_joint=width/2），见 ADR-0005。
- [x] `robots/franka/adapter.py`：loader、FK、joint limits、contact 分类（adjacent/self/scene）。
- [x] 接入 Mink（quadprog，100 iterations，1e-5）+ 显式外层循环；语义见 ADR-0007。
- [x] `q[t-1]` 作为 `q[t]` 初值；失败时用确定性 seed 序列恢复。
- [x] `T_C_B` 从 episode / profile / `base.json` 读入，缺失时 stage 拒绝运行；见 ADR-0009。
- [x] `robot/trajectory.npz` 逐帧保存 target、FK、双误差、solver_status、collision_code、joint jump、ik iterations。
- [x] `vision/render.py` 输出 RGB、robot mask（arm body）、gripper mask、metric depth（无效为 NaN）。
- [x] `viz/projection_debug.py`：target 轴、FK 十字、渲染臂混合、逐帧 projected gap、3D 全景图与两种视频证据。

### 测试

- 简单可达 pose 的 IK/FK round-trip；
- 不可达 pose、joint limit、碰撞的错误分类；
- 连续轨迹没有大幅 joint branch jump；
- renderer segmentation IDs 正确；
- depth 的 near/far 和无效像素语义正确；
- 已知 3D 点的 pinhole projection golden test。

### Gate M3：禁止绕过

- fixture `ik_success_ratio >= 95%`；
- p95 FK position error `<0.01 m`；
- p95 FK rotation error `<0.10 rad`；
- robot EEF 与 target TCP 在抽检 overlay 中方向、尺度、位置一致；
- robot-only video 连续，无 elbow-up/elbow-down 频繁切换；
- 失败帧有明确 reason code，而非被上一帧静默覆盖。

若 projection 不对，按以下顺序排查：`K/distortion -> T_C_B -> T_E_TCP -> frame convention -> IK`。M3 未通过时暂停视觉合成开发。

> M4–M9 未在本轮实现。`base_search.py` 已按论文 A.4 实现候选生成、关键帧选择与 `S = FR - 5|rho - 0.65|` 评分，但尚未在 `repro` profile 上跑通并出报告，因此不计入 M4 完成。

## 7. M4：自动 Robot base search

目标：从人工 fixed base 升级到论文式 morphology-specific placement。

### 任务

- [ ] 实现 trajectory centroid 与机器人 reach 归一化。
- [ ] 实现最多 20 个空间/旋转极值 keyframes。
- [ ] 按论文枚举 lateral、forward、vertical、pitch/yaw/roll 候选。
- [ ] 预过滤 camera distance、`<0.08 m`、`>0.9 * reach` 候选。
- [ ] 批量运行 keyframe IK，计算 feasibility rate 和 mean reach ratio。
- [ ] 以 `FR - 5 * abs(mean_reach_ratio - 0.65)` 排序。
- [ ] 对 top candidates 跑全轨迹，不能只依据稀疏 keyframes 宣告成功。
- [ ] 缓存 candidate 结果并生成 search heatmap/table。

### 测试

- 人工构造的可达/不可达轨迹；
- score 计算与 tie-break 稳定；
- camera distance/reach filter 边界；
- fixed random seed 时结果确定；
- selected candidate 全轨迹验证失败时能回退到次优候选。

### Gate M4

- 在 fixture 上自动 placement 的 IK success 不低于人工 fixed-base 基线 2 个百分点以上；
- 达到 M3 的 FK error gate；
- 选中 base 的 mean reach ratio 接近 0.65，或报告为什么 feasibility 优先导致偏离；
- search report 包含候选数、过滤原因、top-N score 和最终全轨迹指标。

## 8. M5：Human-arm removal

M5 可在 M4 开发期间并行，但正式合成仍须等待 M3 gate。

2026-09-26 执行补充：本轮使用用户已验收投影视频的 episode 000。新增 schema 与字段以 `system-design.md` 6.7 和 `vision/m5_contracts.py` 为准；旧配置自动补默认 M5 字段，旧几何数据无须转换，独立 M5 profile 追加 segmentation/inpainting 两个 stage。低分辨率预览和独立 WSL 模型进程见 ADR-0011。实际运行证据与实现状态维护在 `m5-m8-workplan.md`，用户验收后再勾选 gate。

### 任务

- [x] 以类型化 `ModelRequest` / `ModelResult` 独立进程契约实现 `ArmSegmenter` 适配边界和 device/resource contract（ADR-0011）；不要求核心进程导入模型框架。
- [x] 接入 SAM 3：`person` prompt、中间帧 anchor、400/50 chunk/overlap。
- [x] 实现 3-frame gap、area median 和 morphology close 后处理。
- [x] 输出 raw/refined/dilated masks 和面积曲线；dilated masks 保存在 inpainting 的 model-output 下并由 provenance 索引。
- [x] 以相同类型化进程契约实现 `VideoInpainter` 适配边界。
- [x] 接入 ProPainter 及论文公开参数；首轮按用户选择以 640×360 预览运行并记录分辨率差异。
- [x] 保存 background review video、逐帧修复区域和时序 debug montage。

2026-09-27：episode 000 两段真实推理均已完成，52 帧全部产出，重复运行命中缓存。程序交付已完成；代表帧仍有手臂阴影及局部模糊，**M5 人工 gate 待用户审阅**，不得据此启动 M6 合成。详细证据见 `m5-m8-workplan.md`。

### 测试

- chunk overlap 合并；
- 空 mask、全图 mask、断裂 mask、面积骤降；
- 被操作物体附近 mask 的回归 fixture；
- inpainting 帧数、分辨率、PTS 不变化；
- 模型不可用/OOM 时 stage 明确失败并可恢复。

### Gate M5

- 手臂主体在 `>=95%` 抽检帧被覆盖；
- mask 不得持续吞掉核心操作物体；
- 无明显 mask flicker 或 chunk seam；
- background video 不出现大面积时序闪烁；
- 人工评审记录与 preview 一起进入 stage manifest。

## 9. M6：Scene depth 与 depth-aware compositing

目标：产生遮挡关系可解释的 synthetic robot video。

### 先决 ADR

在实现正式比较前，必须在 `docs/decisions/` 选择一个 metric alignment 策略，例如：

- 相机/深度传感器直接提供 metric depth；
- 用已知桌面/物体/手深度 anchors 拟合 scale + shift；
- 只做 debug alpha composite，不宣称 depth-aware 正确。

### 任务

- [ ] 定义 `SceneDepth`，严格区分 relative/metric/aligned。
- [ ] 接入 Depth Anything V3 adapter，保留 raw depth/confidence。
- [ ] 实现选定的 `DepthScaleAligner` 和拟合诊断图。
- [ ] 分开处理 arm body mask 与 gripper mask。
- [ ] arm body 总是合成；gripper 按 scene/robot depth 和 dilated hand mask 判定遮挡。
- [ ] 输出 final RGB、visible robot mask、occluded mask、decision map。
- [ ] 提供 alpha-only debug compositor，与正式结果分开命名。

### 测试

- 合成平面场景：robot 全在前、全在后、交叉深度；
- NaN/无效 scene depth 的保守策略；
- hand-mask 保护区逻辑；
- metric/relative 类型混用必须直接报错；
- 颜色空间和 mask edge 无明显 halo。

### 实施状态（2026-10-28，ADR-0020）

M6 的两个 runtime stage 已实现并可在 episode 000 上跑通，**但视觉验收尚未进行**
（AGENTS.md 规则 9：勾选与 Gate 结论须由用户查看产物后给出）。

| 任务 | 状态 |
|---|---|
| 定义 `SceneDepth`，区分 relative/metric/aligned | 已实现 `vision/scene_depth.py` |
| 接入 Depth Anything V3 adapter，保留 raw depth/confidence | 已实现 `vision/da3_bridge.py` + `tools/da3_model_worker.py`（隔离进程） |
| 选定的 `DepthScaleAligner` 与拟合诊断 | `align_relative_depth` 已实现；本 profile 走 `metric`，于是它只当后备路径，尚未在本集触发 |
| arm body 与 gripper 分开处理 | 已实现 `vision/composite.py` |
| arm 总是合成；gripper 按 scene/robot depth + dilated hand mask 判断 | 已实现，`5×5` 膨胀 |
| 输出 final RGB、visible mask、occluded mask、decision map | 已实现，decision map 为固定整数编码 |
| alpha-only debug compositor，与正式结果分开命名 | 已实现 `mode: debug_alpha` → `frames-alpha/`、`valid_episode=false` |

M6 的"测试"条目按规则 7 不写成自动化测试；等价证据是
`tools/composite_sanity_cases.py` 产出的前后关系三情形对照图
（`previews/composite-occlusion-cases.png`）以及 stage manifest 里的锚点残差。

### Gate M6

- synthetic video 中 robot EEF 仍保持 M3 投影对齐；
- 三类遮挡 golden scene 像素级通过；
- fixture 抽检中前后关系符合交互物体几何；
- 所有正式输出可追溯到 depth mode 和 alignment residual；
- 未对齐 relative depth 不能生成标记为 `valid` 的正式 episode。

## 10. M7：Quality、canonical packaging 与导出

目标：把“能跑”升级为“可被后续训练安全消费”。

### 任务

- [ ] 实现 Preflight 和稳定 reason-code registry。
- [ ] 实现论文 L1 条件、cross-arm/area 条件和 stability erosion。
- [ ] 实现离线 dataset statistics job，生成版本化 Q1/Q99 和 motion thresholds。
- [ ] 实现 L2 outlier、residual/acceleration/jerk 和 episode `>60% invalid` 过滤。
- [ ] L3 先实现 Protocol、schema 和 `manual_review` backend；不调用 VLM。
- [ ] 组装 canonical episode，写 checksum 和 schema version。
- [ ] 实现 canonical reader，证明所有字段可 round-trip。
- [ ] 实现至少一个 exporter skeleton；只有 canonical 稳定后再补 LeRobot mapping。
- [ ] 生成 episode HTML/Markdown summary 或静态 report。

### 测试

- 每条 L1 reason 的正反例；
- stability erosion 边界；
- Q1/Q99 使用 L1 valid 数据且按 dataset/source 隔离；
- invalid ratio 刚好 60% 与大于 60%；
- canonical write/read round-trip；
- exporter 不得修改 canonical source。

### Gate M7：端到端 MVP 完成

- 从 source manifest 一条命令运行到 canonical episode；
- `e2r validate` 对全部 artifacts、checksums、schema 和 QC 返回成功；
- 第二次执行命中 cache，且不重复运行重模型；
- 删除任一中间 stage 后可从最近有效上游恢复；
- canonical package 含 source、hand、action、robot、visual、quality、config、provenance；
- 至少一个干净进程可以读取 episode 并逐帧返回 image + 7D action。

## 11. M8：Path B（纯 RGB 输入）

目标：在主链稳定后，把 hand pose 来源替换为论文 Path B。

### 任务

- [ ] WiLoR adapter：per-frame detection、MANO 和 21 keypoints。
- [ ] SAM 3 hand-mask consistency：超过 80% projected keypoints 在 mask 外的检测丢弃。
- [ ] 以左右手 combined score 最高帧为 seed，按 wrist image-space nearest neighbor 双向关联。
- [ ] 冲突时保留高分 detection。
- [ ] 按 `max(4 * median velocity, 0.003 m/frame)` 做 wrist jump filter。
- [ ] DynHaMR temporal refinement adapter；保存优化前后轨迹和 loss diagnostics。
- [ ] hand depth 限制 `[0.05, 0.4] m`。
- [ ] 重新运行 M1/M2/M3 gates，对比 Path A 基准。
- [ ] 长视频 60 秒预切和 VLM subtask segmentation 延后到 Path B 短视频通过后。

### Gate M8

- RGB-only fixture 的 hand overlay 稳定、identity 不跳；
- Path B canonical trajectory 不出现系统尺度漂移；
- 相比 Path A/人工基准，关键点/TCP/投影误差有量化报告；
- 仍满足 M3 IK/FK gate，或失败被清晰归因而非降低 gate。

## 12. M9：扩展项，必须单独立项

**局部例外（2026-09-28，ADR-0019）**：用户明确要求先比较多种机器人形态，
因此允许在已审阅的 52 帧 source/canonical 上单独试验 UR5e、xArm7、
ViperX-300 的 robot binding、render 和 projection。各型号必须有独立模型、
配置、IK/FK 指标与图像产物；该试验不等于 M9 验收，也不解除以下大规模
扩展项的依赖。UR5e 官方模型没有夹爪，仅可作为末端位姿对照。

以下工作只有 M7/M8 完成后启动：

- 双手与双臂 top-5 × top-5 联合 base verification；
- 其余 14 robot morphologies；
- Qwen3.5 subtask segmentation 和 L3 consistency audit；
- 大规模 episode scheduler、GPU resource queues 和失败重试；
- Zarr/Parquet 或对象存储迁移；
- LeRobot/RLDS/HDF5 完整 exporter；
- 每个数据源的 action-speed alignment profile；
- dataset-level dashboard 和人工抽检工具。

每项扩展必须有独立 ADR、资源预算和回归集，不能直接塞进 MVP profile。

## 13. 建议的首批 issue 拆分

| ID | Issue | 依赖 | 估算 |
|---|---|---|---|
| E2R-001 | Project skeleton, tooling, CI | 无 | 1–2 人日 |
| E2R-002 | Frame/SE3/quaternion primitives | 001 | 2 人日 |
| E2R-003 | Versioned schemas and artifact store | 001/002 | 2–3 人日 |
| E2R-004 | Stage runner, fingerprint, resume | 003 | 2–3 人日 |
| E2R-005 | EgoDex episode ingest and Path A adapter | 003 | 2–3 人日 |
| E2R-006 | Hand overlay and 3D debug viewer | 005 | 2 人日 |
| E2R-007 | Hand-to-gripper retargeting | 002/005 | 2–3 人日 |
| E2R-008 | Filtering, gaps and smoothing | 007 | 3 人日 |
| E2R-009 | Franka/MuJoCo model adapter | 002 | 3 人日 |
| E2R-010 | Mink IK/FK/collision trajectory | 008/009 | 4–6 人日 |
| E2R-011 | Robot RGB/mask/depth renderer | 009/010 | 3–4 人日 |
| E2R-012 | Projection debug gate | 005/010/011 | 2–3 人日 |
| E2R-013 | Base candidate/keyframe/search | 010/012 | 4–6 人日 |
| E2R-014 | SAM 3 segmentation adapter | 005 | 3–5 人日 |
| E2R-015 | ProPainter adapter | 014 | 3–5 人日 |
| E2R-016 | Scene depth and scale ADR/adapter | 012 | 4–7 人日 |
| E2R-017 | Depth-aware compositor | 011/015/016 | 3–4 人日 |
| E2R-018 | L1/L2 curation | 010/017 | 3–5 人日 |
| E2R-019 | Canonical packager and reader | 003/018 | 2–3 人日 |
| E2R-020 | End-to-end CLI and recovery tests | 全部 MVP | 3–5 人日 |

估算只用于排序，不作为承诺。模型权重、CUDA/MuJoCo 兼容和 fixture 质量可能是主要不确定性。

## 14. Pull request / 代码变更准入规则

每个 PR/变更应回答：

1. 对应哪个里程碑和 issue？
2. 输入/输出 schema 是否变化？
3. 使用的是论文明确参数、工程决策还是开放假设？
4. 新增了哪些 unit/integration/golden tests？
5. 哪个 gate 提供了什么证据（metrics、preview、report）？
6. 对已有 artifact 是否需要 migration/cache invalidation？
7. CPU/GPU/VRAM/时间成本有什么变化？

以下情况禁止合并：

- 裸 `pose`/`transform` 变量跨模块传递；
- 未说明 quaternion order、单位或 frame；
- perception 直接依赖 Franka/MuJoCo；
- visual stage 绕过 M3 geometry gate；
- 相对 scene depth 与 metric robot depth 直接比较；
- 用第三方复现参数却标成论文参数；
- 测试只断言“没有异常”，不验证数值或几何语义。

## 15. 实施中的验证节奏

每完成一个里程碑：

1. 跑该阶段 unit tests；
2. 跑所有已完成阶段的 regression tests；
3. 在固定 fixture 上生成 metrics 和 preview；
4. 将证据路径写入 stage/episode report；
5. 更新本文件的 checkbox 和当前状态；
6. 如出现论文未说明的新选择，先写 ADR，再继续实现。

推荐长期保留三类 fixture：

- `synthetic_geometry`：完全已知 pose/camera，检查公式和坐标；
- `single_hand_pick_place`：真实短视频，检查感知与几何；
- `occlusion_tabletop`：具有明确前后遮挡，检查视觉合成。

## 16. 第一轮开发的停止条件

第一轮以 M3 为明确停止点：先证明 hand -> canonical -> Franka -> projection 的几何链正确。到达 M3 后进行一次设计复审，确认：

- 数据 schema 是否足够；
- `T_C_B`、`T_E_TCP` 和相机模型是否稳定；
- IK/FK 指标与视觉 overlay 是否一致；
- 是否需要修改 base-search 或视觉阶段接口。

只有复审通过，才进入 M4–M7。这样可以避免在错误坐标或尺度之上堆叠 SAM、ProPainter 和 depth 模型。

## 17. 当前第一步

下一次开始代码实现时，只启动 M0，并按以下顺序提交：

```text
E2R-001 project skeleton
  -> E2R-002 transform primitives
  -> E2R-003 schemas/artifacts
  -> E2R-004 stage runner
```

在 M0 Gate 通过前，不下载 WiLoR、DynHaMR、SAM 3、ProPainter 或 Depth Anything 权重，也不实现 IK。这个约束保证后续重模块都落在已经验证的数据契约和坐标约定上。
