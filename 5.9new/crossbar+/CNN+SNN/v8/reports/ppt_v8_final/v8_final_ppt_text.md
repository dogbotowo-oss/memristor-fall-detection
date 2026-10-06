# v8 final：PPT 汇报版文案

## 第 1 页：研究目标

标题：面向真实视频泛化的跌倒检测模型

核心问题：
- 新真人视频上容易出现泛化不足。
- ADL 与真实跌倒存在混淆。
- 坐下、蹲下、弯腰、躺下等接近地面动作容易被误报为 Fall。

本阶段目标：
- 在保持 Fall recall 的同时提升 ADL specificity。
- 使用 Crossbar / 忆阻器动态映射增强模型对时序状态变化的表达。
- 使用严格 subject-level 划分保证最终测试可信。

## 第 2 页：v8 final 整体流程

标题：v8 final 模型流程

流程：
1. 输入视频切成 16-frame 窗口。
2. CNN 提取空间姿态特征。
3. GRU 建模短时动作演化。
4. SNN temporal branch 强化跌倒瞬间的动态响应。
5. Crossbar / 忆阻器映射引入器件非理想性和动态响应。
6. Teacher-student Gaussian noise 提升噪声鲁棒性。
7. SNN ADL gate penalty 抑制 ADL 误报。
8. 在 Subject3 + external val 上选择阈值。
9. Subject4 只作为最终独立测试。

## 第 3 页：严格数据划分

标题：严格避免 Subject4 污染

划分规则：
- GMDCSA Subject 1-2：训练。
- GMDCSA Subject 3：验证 + 阈值校准。
- GMDCSA Subject 4：独立最终测试。
- Zenodo / 外部数据：按 video / person / subject 划分，禁止窗口随机混合。

论文口径：
- 阈值只能在 Subject3 + external val 上选择。
- Subject4 不能用于选模型、调阈值、调 sampler、调 penalty。

## 第 4 页：v8 final 的关键机制

标题：为什么 v8 final 是当前主结果

关键设计：
- CNN/GRU 主干负责姿态与短时动作建模。
- SNN temporal branch 对跌倒瞬间的快速动态更敏感。
- Crossbar / device dynamics 引入忆阻器器件响应，贴合硬件动态映射思路。
- Teacher-student noise training 提升对真实视频噪声与扰动的鲁棒性。
- SNN ADL gate penalty 控制 ADL 误报，避免把接近地面的非跌倒动作过度判为 Fall。

## 第 5 页：Subject4 最终结果

标题：v8 final p006 是当前综合最优主结果

Subject4 最终独立测试：
- Threshold：0.70
- Accuracy：78.38%
- Balanced Accuracy：78.24%
- ADL specificity：80.00%
- Fall recall：76.47%
- Precision：76.47%
- TP / TN / FP / FN：13 / 16 / 4 / 4

结论：
- v8 final 在 ADL specificity 与 Fall recall 之间取得目前最稳定平衡。
- 后续 v14/v15 虽然 Fall recall 可提高到 82.35%，但 ADL specificity 下降到约 60.00%。
- route-A subtype penalty 能降低部分误报，但 Fall recall 明显下降，暂不能替代 v8 final。

## 第 6 页：当前结论与下一步

标题：当前主线与后续优化方向

当前结论：
- v8 final p006 仍作为论文主结果。
- 单纯增强 ADL penalty 容易让模型变保守，导致 Fall 漏检。
- 后续重点不应继续强压 ADL，而应增强真实跌倒瞬间动态识别。

下一步方向：
- 继续探索 Fall event positive mining。
- 用 center-of-mass drop、姿态快速变化、运动峰值等动态特征强化真实跌倒窗口。
- 将 ADL subtype 信息用于 gate / post-score 校正，而不是直接压低主分类 Fall 概率。
