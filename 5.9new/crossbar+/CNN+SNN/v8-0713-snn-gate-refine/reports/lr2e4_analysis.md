# CNN 冻结与学习率实验综合分析

## 验证口径

- 训练集：GMDCSA Subject 1-2 + Zenodo train，共 3539 个窗口
- 验证集：GMDCSA Subject 3 + Zenodo val，共 1076 个窗口
- Crossbar 与 device dynamics：启用
- Subject4：未使用

## 对比结果

| 方案 | 最佳 epoch | Balanced Accuracy | ADL specificity | Fall recall |
|---|---:|---:|---:|---:|
| group-only | 1 | 74.1% | 72.9% | 75.3% |
| 统一 lr=1e-4 | 4 | 74.2% | 68.1% | 80.3% |
| GRU-only corrected | 5 | 73.9% | 68.8% | 79.1% |
| CNN 冻结 + 分层学习率 | 5 | 74.2% | 68.6% | 79.7% |
| CNN 冻结 + 统一 lr=2e-4 | 5 | **74.4%** | 68.6% | 80.1% |

## 当前低准确率的主要原因

1. **Fall 偏置明显**：最佳点 ADL specificity 只有 68.6%，而 Fall recall 达到 80.1%，说明模型把一部分接近地面、姿势变化或快速移动的 ADL 判成 Fall。
2. **时间顺序 gate 增强了敏感性但不够区分**：冲击、中心下降、姿势变化、触地等特征在主动坐下、躺下、蹲下等 ADL 中也可能出现，因而提高 Fall recall 的同时带来 ADL false positive。
3. **微调对象受限**：CNN 全程冻结后空间特征稳定，但模型只能调整 GRU、SNN 和 gate，无法修正器件噪声下的底层姿态表征；这解释了为什么稳定性提高，但 ADL specificity 没有回到 group-only 水平。
4. **从成熟 checkpoint 继续微调**：模型从已经训练好的 group-only 初始化，继续用较高学习率会改变原有判别边界，因此最佳点集中在早期 epoch。
5. **器件动态和 teacher-student 噪声叠加**：当前训练同时启用了 Crossbar readout noise、device dynamics、Gaussian teacher-student noise 和 frame drop。它们有助于鲁棒性，但也可能削弱边界样本的姿态细节；这需要单独消融确认，不能直接归因于 SNN。

## 推荐下一步

1. 保留 `CNN 冻结 + lr=2e-4` 作为稳定训练基线，不再优先搜索更高学习率。
2. 在 Subject3 + Zenodo val 上做轻量 gate 校正：固定模型，只扫描 ordered gate 的增益/下限，优先约束 ADL false positive，同时要求 Fall recall >= 75%。
3. 加入更严格的 ground-hold 条件：冲击后必须同时满足连续静止、人体高度保持低位、中心位置变化小，避免把主动坐下/躺下当作跌倒。
4. 对 `posture/ground-hold` 与 `impact/center-drop` 分开加权，而不是把整个 ordered score 统一提高；先使用小增益，再比较 ADL specificity 与 Fall recall。
5. 做最小噪声消融：固定 CNN 冻结和 lr=2e-4，仅分别关闭 device dynamics、teacher-student noise、frame drop，确认准确率低究竟来自 SNN gate 还是噪声叠加。
6. 最终模型选择继续使用 Subject3 + external val，采用“Fall recall >= 75% 后最大化 Balanced Accuracy，再以 ADL specificity 优先”的规则；Subject4 只做一次最终测试。
