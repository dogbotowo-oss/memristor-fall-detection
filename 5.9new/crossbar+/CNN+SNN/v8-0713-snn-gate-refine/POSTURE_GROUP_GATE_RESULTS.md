# Posture-aware Group Gate 结果

验证集：GMDCSA Subject3 + Zenodo val。Subject4 未参与调参。

## 最佳结果

- Best epoch：2
- Accuracy：72.5%
- Balanced Accuracy：72.9%
- ADL specificity：66.5%
- Fall recall：79.3%

姿势与 ground-hold 上下文提高了 Fall recall，但没有改善 ADL-Fall 平衡。相比 group-only，Fall recall 上升 4.0 个百分点，ADL specificity 下降 6.4 个百分点，Balanced Accuracy 下降 1.2 个百分点。

当前 50% learned gate + 50% guided gate 的引导强度过高，且坐下、主动躺下等 ADL 也可能包含低位、横向姿势和尾段静止。因此该版本不作为下一阶段候选，继续保留 group-only 为综合最优结构。
