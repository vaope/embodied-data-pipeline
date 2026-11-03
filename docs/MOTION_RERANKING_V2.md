# Robot→Ego：语义候选后末端运动重排

2026-10-30，已实现、干净安装及真实数据运行。入口`edp-sequence --robot-motion <steps.csv>`；Ego→Ego的`pose_language_active`不变。此处“微调”是候选重排，无模型训练或外部API。

## 输入与候选池

沿用连续查询JSON；只有一个步骤时也可运行。新增search配置`rerank_pool_size`（默认0，不开重排池）与`concatenated_baseline`（默认true，保留旧基线）。开启运动需`rerank_pool_size>0`，例如50；`top_k`独立，例如5。`--robot-motion`显式指定观测末端CSV。manifest保留`clip_id/hdf5_path/start_frame/end_frame/start_time/end_time/fps`，`end_frame`为含端点；annotation秒区间仍为半开。

关键步骤单步基线保留文字语义Top N、不在池前加来源限额，输出`single_key_motion`。连续流程是key语义锚点→同源邻域原顺序DP→语义排序并按IoU去重保留N个区间→运动评分→最终Top K。N是已有语义路径的候选池上限，不保证全库最优；重排不会重新搜索被DP舍弃的其他映射。

## 运动与融合

`sequence/motion.py`读取Robot观测EE和Ego原生腕根Pose。固定真实事件跨度：Robot按半开帧裁切，Ego读到含端点末帧。1:2/2:1读取各自所覆盖的原始连续区间，包括间隔，避免直接拼坐标导致假跳变。按已有算法归一进度到32点，比较速度幅值、角速度幅值、停顿比例与路径比；使用真实采样时长计算速度，Robot无实测时间戳时明确名义15Hz。

操作手按候选活动量独立于query选择，保留各手活动量和低活动警告；不是已核实操作手。固定初始自身系17点XYZ轨迹保留作诊断，不作为已校准跨具身方向分；抓握不计分。

```text
B = 1/(1+d)
F = S + weight × Σ[有效匹配块所含query步数 × (B−0.5)] / 必需query步数
weight = 0.03（本轮预先冻结，未调参）
```

每个必需query步骤至多投一次票；单个结果调分范围±0.015。缺失轨迹贡献0，保留原语义分和失败原因，不能伪装为距离0或默认错误。部分有效时按覆盖率衰减调整。与旧实验`S−0.03*d/(1+d)`不同，采用中性回退，不将未知证据扣满惩罚；此融合仍是未校准工程默认，不是概率。

输出保留`semantic_complete`、`candidate_pools`、`motion_pools`以及各候选`semantic_score/semantic_rank/fusion_rank/motion_adjustment/motion_coverage/motion_status/motion_pairs`。`complete`为融合后最终列表，仍仅表示算法路径完整；`motion_pairs`含逐组分解、源Pose摘要哈希、手别、有效率与轨迹诊断。`motion_provenance`记录CSV哈希、配置、公式和时钟来源。

## 验证与边界

新增真实合成Pose读回测试：人手y轴运动与Robot x轴运动在计分特征上等价，正确运动候选可从语义第2升到最终第1；同时验证缺失中性、多步权重及无效参数。干净非editable安装后82 passed、2私人项skipped。新增3k库的单步CLI真实运行成功，输出50候选运动分及最终5结果；实验另完成48次检索和缓存重放验证。

本轮15已见评测查询的新3k库连续结构Success@5为语义3/15、运动3/15，精确确认均0/15；没有效果提升结论。单步仅Top1追加AI复核，完整Top5质量尚未评测。后续重点是已知正例、步骤/角色语义与有向轴/参考点校准，不能靠扩大运动权重替代。

真实运行与参数见工作区`experiments/sequence_retrieval_20261029/motion_3k_20261030/README.md`。本轮不改变Ego→Ego检索算法，不修改原标注。
