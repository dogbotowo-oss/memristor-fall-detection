# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import math
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np


OUT_DIR = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\ppt_v8_final")
PNG_PATH = OUT_DIR / "oxygen_vacancy_gradient_band_bending.png"
SVG_PATH = OUT_DIR / "oxygen_vacancy_gradient_band_bending.svg"


def choose_font() -> str:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Source Han Sans CN",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


FONT = choose_font()
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False


def add_panel(ax, x, y, w, h, title, subtitle, edge, fill):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=2.2,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="center", fontsize=18, fontweight="bold", color="#17324D")
    ax.text(x + w / 2, y + h - 0.082, subtitle, ha="center", va="center", fontsize=12.5, color="#415A6B")


def draw_material(ax, x, y, w, h, gradient: bool):
    rect = Rectangle((x, y), w, h, linewidth=1.8, edgecolor="#4E6E81", facecolor="#F8FBFD")
    ax.add_patch(rect)

    if gradient:
        # Background density gradient.
        nx = 120
        for i in range(nx):
            alpha = 0.08 + 0.33 * (1 - i / (nx - 1))
            ax.add_patch(
                Rectangle(
                    (x + w * i / nx, y),
                    w / nx + 0.001,
                    h,
                    linewidth=0,
                    facecolor="#FFB35C",
                    alpha=alpha,
                )
            )
        ax.text(x + 0.02, y + h - 0.018, "氧空位浓度：高 → 低（连续梯度）", ha="left", va="top", fontsize=11.5, color="#8A4B00")
    else:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=0, facecolor="#FFB35C", alpha=0.15))
        ax.text(x + 0.02, y + h - 0.018, "氧空位浓度近似均匀 / 缺少梯度", ha="left", va="top", fontsize=11.5, color="#8A4B00")

    rng = random.Random(7 if gradient else 3)
    for _ in range(95):
        if gradient:
            # Rejection sample: more vacancies on the left.
            for _try in range(20):
                px = rng.random()
                if rng.random() < (1.04 - px) ** 1.55:
                    break
            cx = x + 0.035 * w + px * 0.93 * w
        else:
            cx = x + (0.05 + 0.90 * rng.random()) * w
        cy = y + (0.08 + 0.84 * rng.random()) * h
        r = 0.0055 if not gradient else 0.0048 + 0.0022 * (1 - (cx - x) / w)
        ax.add_patch(Circle((cx, cy), r, facecolor="#E67E22", edgecolor="#B85B12", linewidth=0.4, alpha=0.92))

    ax.text(x + 0.02, y + 0.018, "● 氧空位", ha="left", va="bottom", fontsize=11, color="#B85B12")


def draw_band(ax, x, y, w, h, smooth: bool):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.5, edgecolor="#AAB7C4", facecolor="#FFFFFF", alpha=0.96))
    ax.annotate("", xy=(x + 0.04, y + h - 0.02), xytext=(x + 0.04, y + 0.03), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#4E5D6C"))
    ax.annotate("", xy=(x + w - 0.02, y + 0.055), xytext=(x + 0.035, y + 0.055), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#4E5D6C"))
    ax.text(x + 0.012, y + h - 0.01, "E", fontsize=12, color="#4E5D6C")
    ax.text(x + w - 0.02, y + 0.01, "位置 x", fontsize=11, ha="right", color="#4E5D6C")

    t = np.linspace(0, 1, 240)
    if smooth:
        curve = 0.70 - 0.33 * (1 / (1 + np.exp(-5.2 * (t - 0.52))))
        label = "平缓弯曲"
        color = "#2E86AB"
    else:
        curve = 0.72 - 0.36 * (1 / (1 + np.exp(-22 * (t - 0.50))))
        label = "陡峭弯曲"
        color = "#C0392B"

    xs = x + 0.08 * w + t * 0.84 * w
    ys = y + curve * h
    ax.plot(xs, ys, lw=4.0, color=color, solid_capstyle="round")
    ax.plot(xs, ys - 0.105 * h, lw=2.2, color=color, alpha=0.38, linestyle="--")

    if smooth:
        ax.text(x + 0.53 * w, y + 0.74 * h, label, fontsize=15, color=color, fontweight="bold")
    else:
        ax.text(x + 0.50 * w, y + 0.76 * h, label, fontsize=15, color=color, fontweight="bold")


def add_result_card(ax, x, y, w, h, title, lines, fill, edge):
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(card)
    ax.text(x + 0.03, y + h - 0.035, title, ha="left", va="top", fontsize=14.5, fontweight="bold", color="#17324D")
    for i, line in enumerate(lines):
        ax.text(x + 0.035, y + h - 0.080 - i * 0.036, line, ha="left", va="top", fontsize=11.5, color="#2E4050")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")

    ax.text(
        0.5,
        0.965,
        "氧空位浓度梯度对能带弯曲与模拟型电导调节的影响",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#17324D",
    )
    add_panel(ax, 0.045, 0.135, 0.43, 0.72, "无氧空位浓度梯度", "局域势垒突变，能带弯曲陡峭", "#C0392B", "#FFF1EF")
    add_panel(ax, 0.525, 0.135, 0.43, 0.72, "有氧空位浓度梯度", "势垒连续过渡，能带弯曲平缓", "#2E86AB", "#EEF8FF")

    draw_material(ax, 0.078, 0.560, 0.365, 0.155, gradient=False)
    draw_material(ax, 0.558, 0.560, 0.365, 0.155, gradient=True)

    draw_band(ax, 0.078, 0.305, 0.365, 0.215, smooth=False)
    draw_band(ax, 0.558, 0.305, 0.365, 0.215, smooth=True)

    add_result_card(
        ax,
        0.078,
        0.145,
        0.365,
        0.135,
        "宏观表现",
        ["• 电场/势垒集中在局部区域", "• 导电通道形成/断裂更突变"],
        "#FFE4DF",
        "#C0392B",
    )
    add_result_card(
        ax,
        0.558,
        0.145,
        0.365,
        0.135,
        "宏观表现",
        ["• 电场与势垒沿厚度方向渐变", "• 导电通道逐步增强/削弱"],
        "#DFF1FF",
        "#2E86AB",
    )

    ax.text(
        0.5,
        0.060,
        "用于PPT表达：氧空位浓度不是简单“越多越好”，关键是形成空间梯度，使内部势垒连续过渡，从而支撑多级/模拟电导调制。",
        ha="center",
        va="center",
        fontsize=13,
        color="#17324D",
        bbox=dict(boxstyle="round,pad=0.55", fc="#FFFFFF", ec="#D0DAE3", lw=1.5),
    )

    fig.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.08)
    print(PNG_PATH)
    print(SVG_PATH)
    print(FONT)


if __name__ == "__main__":
    main()
