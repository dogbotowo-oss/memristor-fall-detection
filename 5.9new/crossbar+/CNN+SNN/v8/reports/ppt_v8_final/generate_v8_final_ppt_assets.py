# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import html


OUT_DIR = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\ppt_v8_final")
DRAWIO_PATH = OUT_DIR / "v8_final_flowchart.drawio"
PPT_TEXT_PATH = OUT_DIR / "v8_final_ppt_text.md"


def esc(text: str) -> str:
    return html.escape(text, quote=True).replace("\n", "&#xa;")


def style(fill: str, stroke: str, font: int = 18, bold: bool = True, rounded: bool = True) -> str:
    return (
        f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontSize={font};fontStyle={1 if bold else 0};fontColor=#17324D;"
        "align=center;verticalAlign=middle;spacing=10;arcSize=14;"
    )


def cell(cid: str, value: str, x: int, y: int, w: int, h: int, fill: str, stroke: str, font: int = 18) -> str:
    return (
        f'    <mxCell id="{cid}" value="{esc(value)}" style="{style(fill, stroke, font)}" '
        f'vertex="1" parent="1">\n'
        f'      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
        f"    </mxCell>\n"
    )


def swimlane(cid: str, value: str, x: int, y: int, w: int, h: int, fill: str, stroke: str) -> str:
    st = (
        "swimlane;html=1;rounded=1;startSize=42;collapsible=0;container=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        "fontSize=20;fontStyle=1;fontColor=#17324D;align=center;"
    )
    return (
        f'    <mxCell id="{cid}" value="{esc(value)}" style="{st}" vertex="1" parent="1">\n'
        f'      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
        f"    </mxCell>\n"
    )


def edge(
    cid: str,
    source: str,
    target: str,
    label: str = "",
    color: str = "#567C8D",
    exit_x: float | None = None,
    exit_y: float | None = None,
    entry_x: float | None = None,
    entry_y: float | None = None,
    dashed: bool = False,
    points: list[tuple[int, int]] | None = None,
) -> str:
    pins = ""
    if exit_x is not None and exit_y is not None:
        pins += f"exitX={exit_x};exitY={exit_y};exitDx=0;exitDy=0;"
    if entry_x is not None and entry_y is not None:
        pins += f"entryX={entry_x};entryY={entry_y};entryDx=0;entryDy=0;"
    dash = "dashed=1;dashPattern=8 6;" if dashed else ""
    st = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
        f"strokeColor={color};strokeWidth=2;endArrow=block;endFill=1;"
        "fontSize=14;fontColor=#36515F;labelBackgroundColor=#FFFFFF;"
        f"{pins}{dash}"
    )
    if points:
        point_xml = "        <Array as=\"points\">\n"
        point_xml += "".join(f'          <mxPoint x="{x}" y="{y}" />\n' for x, y in points)
        point_xml += "        </Array>\n"
        geometry = f'      <mxGeometry relative="1" as="geometry">\n{point_xml}      </mxGeometry>\n'
    else:
        geometry = '      <mxGeometry relative="1" as="geometry" />\n'
    return (
        f'    <mxCell id="{cid}" value="{esc(label)}" style="{st}" edge="1" parent="1" '
        f'source="{source}" target="{target}">\n'
        f"{geometry}"
        "    </mxCell>\n"
    )


def make_drawio() -> str:
    cells = []
    cells.append(swimlane("lane_data", "数据与划分", 40, 80, 250, 500, "#EAF4FF", "#5B8DEF"))
    cells.append(swimlane("lane_model", "v8 final 模型主干", 330, 80, 500, 500, "#F0F7EF", "#6DAA6B"))
    cells.append(swimlane("lane_train", "训练与校准", 870, 80, 310, 500, "#FFF5E6", "#E8A33E"))
    cells.append(swimlane("lane_eval", "最终测试输出", 1220, 80, 340, 500, "#F8ECF6", "#B66AA6"))

    cells.append(cell("n1", "GMDCSA\nSubject 1-2\n训练主体", 75, 150, 180, 82, "#D8EBFF", "#5B8DEF", 18))
    cells.append(cell("n2", "Zenodo / 外部视频\n按视频/人划分", 75, 270, 180, 82, "#D8EBFF", "#5B8DEF", 17))
    cells.append(cell("n3", "Subject 3\n验证 + 阈值校准", 75, 390, 180, 82, "#D8EBFF", "#5B8DEF", 18))

    cells.append(cell("n4", "视频帧窗口\n16-frame clips\nimage 64×64", 365, 145, 190, 86, "#DAF0D6", "#6DAA6B", 17))
    cells.append(cell("n5", "CNN 编码器\n空间姿态特征", 620, 145, 180, 86, "#DAF0D6", "#6DAA6B", 18))
    cells.append(cell("n6", "GRU 时序主干\n短时动作演化", 365, 305, 190, 86, "#DAF0D6", "#6DAA6B", 18))
    cells.append(cell("n7", "SNN temporal branch\n跌倒瞬间动态响应", 620, 305, 180, 86, "#D5F0EA", "#44A08D", 16))
    cells.append(cell("n8", "Crossbar / 忆阻器映射\nconductance + device dynamics", 490, 455, 220, 90, "#E2F4D9", "#7BAF58", 16))

    cells.append(cell("n9", "Teacher-student\nGaussian noise 训练", 915, 145, 220, 82, "#FFE8C2", "#E8A33E", 17))
    cells.append(cell("n10", "SNN ADL gate penalty\np006 = 0.06", 915, 270, 220, 82, "#FFE8C2", "#E8A33E", 18))
    cells.append(cell("n11", "阈值选择\nSubject3 + external val\nthreshold = 0.70", 915, 405, 220, 92, "#FFE8C2", "#E8A33E", 17))

    cells.append(cell("n12", "Subject4\n独立最终测试\n绝不参与调参", 1260, 145, 250, 92, "#F6DAF0", "#B66AA6", 18))
    cells.append(cell("n13", "视频级输出\nADL / Fall", 1260, 285, 250, 82, "#F6DAF0", "#B66AA6", 20))
    cells.append(cell("n14", "v8 final p006 结果\nAcc 78.38% | Bal Acc 78.24%\nADL spec 80.00% | Fall recall 76.47%\nTP/TN/FP/FN = 13/16/4/4", 1245, 425, 280, 112, "#F2D7EE", "#B66AA6", 16))

    edges = [
        edge("e1", "n1", "n4", exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.35),
        edge("e2", "n2", "n6", exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5),
        edge("e3", "n4", "n5", exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5),
        edge("e4", "n5", "n7", exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0),
        edge("e5", "n6", "n7", exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5),
        edge("e6", "n7", "n8", exit_x=0.5, exit_y=1, entry_x=0.85, entry_y=0),
        edge("e7", "n8", "n9", exit_x=1, exit_y=0.35, entry_x=0, entry_y=0.5, points=[(815, 500), (815, 186)]),
        edge("e8", "n9", "n10", exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0),
        edge("e9", "n3", "n11", "只用于校准", color="#7B8A99", exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5, dashed=True, points=[(290, 431), (860, 431)]),
        edge("e10", "n10", "n11", exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0),
        edge("e11", "n11", "n12", "锁定阈值", exit_x=1, exit_y=0.45, entry_x=0, entry_y=0.5, points=[(1190, 450), (1190, 191)]),
        edge("e12", "n12", "n13", exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0),
        edge("e13", "n13", "n14", exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0),
    ]

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="drawio" version="26.0.0">
  <diagram name="v8-final-flowchart">
    <mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="900" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
"""
    xml += "".join(cells)
    xml += "".join(edges)
    xml += """      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    return xml


PPT_TEXT = """# v8 final：PPT 汇报版文案

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
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAWIO_PATH.write_text(make_drawio(), encoding="utf-8")
    PPT_TEXT_PATH.write_text(PPT_TEXT, encoding="utf-8")
    print(DRAWIO_PATH)
    print(PPT_TEXT_PATH)


if __name__ == "__main__":
    main()
