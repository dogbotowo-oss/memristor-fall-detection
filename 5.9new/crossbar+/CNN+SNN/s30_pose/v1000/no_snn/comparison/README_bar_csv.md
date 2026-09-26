# v1000 no-SNN 与 SNN 柱状图数据

依据 F.docx 第 5 节及当前目录原始逐折表和汇总表，采用 balanced（gtest）协议、seed 11、四折 LOSO 宏平均。四个被试等权，共 160 个留出视频；不混用 pooled 汇总或 v8、v999 结果。

- v1000_bar_comparison_macro.csv：第一列是横轴指标，第二、三列是 no-SNN 与 SNN 的柱高，单位 %。第四、五列是四折样本标准差（ddof=1），可作为对应误差棒；不是额外柱高。
- v1000_bar_improvement_macro.csv：第一列是横轴指标，第二列是 SNN 宏平均减去 no-SNN 宏平均，单位为百分点（percentage points），不是相对增长率。

指标顺序为 Accuracy、Balanced Accuracy、ADL specificity、Fall recall。绘图显示两位小数，CSV 保留十位小数；提升由未舍入数据计算。准确率提升为 19.49 个百分点，不能用已舍入的 82.12−62.62 替代。

数据来源：v1000_loso_snn_vs_no_snn_summary.csv 和 v1000_loso_snn_vs_no_snn_fold_metrics.csv 中 scheme=balanced 的记录。export_bar_csv.py 从逐折 TN/FP/FN/TP 重新计算四项指标及宏平均、样本标准差，与原始汇总校验后导出两份 CSV。没有重新训练或重新选择阈值。

用户示例图片用作图形样式参考，图内数值不是当前 v1000 数据。建议第一张纵轴为 Percentage (%)，第二张为 Change (percentage points)。
