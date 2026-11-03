# 连续 Subtask 检索 V1 开发规格

2026-10-30增补：Robot→Ego已新增可选末端运动重排、重排候选池及真实3k对照，见[MOTION_RERANKING_V2.md](MOTION_RERANKING_V2.md)。下文保留原纯语义V1设计与运行历史；原语义默认仍可独立运行。

状态：2026-10-29，用户授权开发。目标是对选定 Robot 动作链，在单一 Ego 视频中返回覆盖该链的连续区间及可解释的步骤对应。代码位于 packages/pipeline/src/retrieval/sequence/，作为独立入口 edp-sequence；历史单步入口保持可用。

实施状态：V1已实现并非editable安装验证，79测试通过/2私人数据旧项跳过。真实300来源/3,041事件完成两条原标注查询和一条明确声明的文字探针，探针返回55–60.5秒两步连续候选；两条原查询无完整候选。运行记录位于工作区experiments/sequence_retrieval_20261029/README.md。用户效果验收未完成，阈值及规则泛化待评估。

## 输入与输出契约

查询JSON：`query_id`, `key_step_id`, `steps`。每步：`step_id`, `source_id`, `start_s`, `end_s`, `caption`，可含`video_path`、`provenance`。同一来源、按时间有序、不重叠、非空描述；时间统一左闭右开秒。所有查询步骤必需，key必须属于查询。可加`task`、`warnings`记录上下文，不隐式用于匹配。显式构造的语义探针与原始自动标注查询分开。

可选`video_reference`含`source_id/video_path/start_s/end_s/note`，用于关联任务参考视频而不宣称逐步时间对齐；路径相对查询JSON解析。此字段只参与展示，不进入检索。存在时页面隐藏文字步骤的符号时间，标注Robot边界待核实，步骤按钮仅定位Ego；两侧仍可从片段起点原速播放。

候选：JSONL支持相同规范，或现有event manifest字段。优先使用`annotation_start_s/annotation_end_s`，而不是旧的末帧播放时间。`clip_id/episode_id`分别映射步骤/来源ID。单一来源的完整步骤序列必须读入，不提前每视频限额；拒绝重复ID、同源重叠和异常时间。

输出JSON版本`sequence.v1`：查询、配置、输入/代码/向量缓存来源、锚点候选、完整/部分结果、基线。每个结果含source_id、起止秒、anchor_step_id、对应块及逐查询步骤分、missing_query_ids、skipped_candidate_ids、未匹配时长、coverage、weakest_similarity、score。complete仅表示算法约束通过，不表示人工语义验收。部分结果只供诊断，展示页默认仅完整结果。

## 检索与对齐

1. 单个指定key用现有冻结语义编码器召回多个候选位置；排名稳定排序，不提前按视频去重。
2. 每个位置前后取邻域，限制步骤距离及秒数。顺序对齐必须经过query key与该候选anchor的对应块。
3. 状态包含已消费查询/候选、是否经过锚点、插入步骤数、缺失查询数。枚举局部开始位置，动态规划保留不同缺失数路径；按覆盖数优先、再按归一化分排序。区间外前后缀免费，区间内插入段计成本；查询缺失必须显式输出，key不允许缺失。
4. 转移支持1:1、1:2、2:1，不支持2:2及任意大块。组合文本按原顺序使用`Then`连接，实际重新编码，不平均单步向量。块相似度与块内每个查询步骤对候选块的相似度均需过门槛；1:2还检查两个候选步骤各自与查询的相似度，避免吞入无关动作。此规则为启发式，不能证明细粒度语义包含关系。
5. 同源单调、禁止步骤重复消费；跨度/相邻时间缺口受限。score=(每查询步骤贡献之和－跳过步骤惩罚－未匹配秒数惩罚)/查询步数。弱步骤不能靠其他高分越过门槛。按同源重叠区间NMS去重，完整结果优先。候选只来自key召回，因此不是全库最优保证。
6. 默认开发参数：anchor_top_k=50、top_k=10、neighbor_steps=8、neighbor_seconds=45、max_span_s=45、max_step_gap_s=5、max_skips=2、min_similarity=0.45、anchor_min_similarity=0.45、skip_penalty=0.10、gap_penalty_per_s=0.01、merge_penalty=0.05、nms_iou=0.8。全部可配置，未校准，不作为效果结论。

## 模块与执行

实跑修订：2026-10-29首轮余弦方案错误地将“拿笔”对“拆橡皮筋”、将拿取+放置对仅放置。新增semantics.py保守英文动作/直接对象冲突检查：已解析两侧动作/对象序列必须一致（相邻重复可折叠）；合并块的所有成员必须可解析，否则禁止合并。单步无法解析时保留语义路径并标unparsed。来源/目标位置目前不硬过滤，完整结果仍需复核。规则不是通用语言理解，可能漏掉同义表达；不能把它称为已解决语义匹配。首轮结果保留在实验pen_run，仅作诊断，不作为展示推荐。

- models.py：规范/适配/验证。
- engine.py：锚点、DP、重叠去重、基线；编码器通过encode接口注入，便于受控验证。
- cli.py：JSON/JSONL输入、固定模型配置、本地缓存、JSON与HTML导出。
- report.py：逐步映射、key高亮、两侧视频；可选FFmpeg精确重编码裁剪，保留插入段，播放原速。
- `edp-sequence --manifest events.jsonl --query query.json --out new-dir [--config config.json] [--render-clips 3]`。输出必须新目录，避免覆盖历史。默认模型离线加载，显式`--allow-download`才允许拉取；不调用外部VLM。
- 比较基线：key单步检索、所有邻接1..2m步且跨度受限的连续caption拼接检索。两者与序列方案使用同一编码器。报告排序与时间范围，不在缺乏金标时自动产生准确率。

## 验证与验收

工程测试必须覆盖：同源完整链、反序、缺必需步骤、key不可丢、插入成本/上限、1:2与2:1、合并不能掩盖缺步、局部起止、多锚点/重复操作、重叠去重、输入异常、缓存身份变化及HTML转义。与小规模穷举比较DP最优分数，防止仅写实现镜像测试。

非editable干净安装回归既有测试；真实3,041事件执行烟测，保存命令、输入哈希和结果。真实自动标注不等于GT，语义探针不等于已验证Robot边界。首版开发完成标准为可安装/可运行/映射与视频可复查；效果验收仍待人工确认。运动重排和视觉embedding在此纯语义基线评估后另行开发。
