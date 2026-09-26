# 代码版本登记

论文实验对应的代码版本。规则：**已用于论文结果的版本目录保持冻结，新改动开新版本号**。

| 版本 | 位置 | 内容 | 对应论文结果 |
|---|---|---|---|
| v1000 | `5.9new/crossbar+/CNN+SNN/s30_pose/`（主代码 + `v1000/`） | 原始训练/评估管线；ideal vs 非理想 LOSO 消融队列 | 主结果（86.22% BA）、编码端消融 |
| v1001 | `s30_pose/v1001/` | **仅数据，无代码**：主结果表（per_fold.csv + 各折指标）；协议 = Zenodo val + 折内被试，F1-max（cap 0.70）。生成脚本已遗失（旧 E 盘），复现用 `v1002/scripts/eval_v1001_protocol.py` | 论文主表 |
| v1002 | `s30_pose/v1002/` | SNN 分支权重量化到实测电导态：`--snn-binary-weights`（二值）、`--snn-weight-levels`（多态/差分对，含负数电平即差分模式）；多种子队列 | 权重映射对比（bw/bw4/bw4d） |

## 对应器件数据

- `F:\12.18-2\data1\`：HRS/LRS 电导、EPSC、LTP/LTD 脉冲数据（注意：7_28 的 LTP/LTD 阶梯数据读电导为平线，无多级态，勿用作多级证据）；
- `C:\Users\lin\Desktop\data\xianliu-data\`：限流系列整理（4 个可分辨 LRS 档：847 Ω / 8.1 kΩ / 40 kΩ / 90 kΩ，对应 bw4 量化电平 {1.0, 0.104, 0.021, 0.0094}）。

## 运行环境

- Python：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`（torch 2.5.1+cu121，GTX 1650）
