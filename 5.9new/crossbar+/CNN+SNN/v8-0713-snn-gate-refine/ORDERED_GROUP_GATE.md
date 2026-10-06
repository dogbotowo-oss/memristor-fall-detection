# Ordered-event Group Gate

本实验保留 group-only 结构和 checkpoint，只将 group gate 使用的事件分数改为有时间顺序的确认分数。

要求的顺序为：

`快速冲击 -> 中心下降 -> 姿势变化/接触地面 -> 尾段持续静止`

实现方式：

- 找到 fall-event 序列的冲击峰值位置。
- 峰值前统计初始中心、人体高度和宽高比。
- 峰值后统计中心下降、高度降低、横向姿势变化。
- 峰值后至少延迟两帧，再判断运动量是否持续降低。
- 冲击位置必须保留足够的峰值前和峰值后帧，否则降低确认分数。

最终 ordered event score 仍作为一个标量输入现有 group gate，不新建 channel gate，不加入 ADL penalty。初始化使用已完成的 group-only checkpoint。
