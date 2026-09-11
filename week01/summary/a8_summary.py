"""
week08 —— 总结报告图

任务：把 A1–A5 五个动作的核心结论压缩到一张总览图上。
产出：
    figures/a8_summary_dashboard.png  五动作总结总览（代表图）
    data/a8_conclusions.md            结论清单（同时写入 README）
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
import trajlib as T                                    # noqa: E402

plt = T.setup_matplotlib()

HERE = os.path.dirname(os.path.abspath(__file__))   # week01/summary
WK = HERE    # 图放在本目录
ROOT = os.path.dirname(HERE)                        # week01/
FIG = os.path.join(WK, "figures")
DAT = os.path.join(WK, "data")

COLORS = {
    "seq_eth":    "#d62728",
    "seq_hotel":  "#1f77b4",
    "zara01":     "#2ca02c",
    "zara02":     "#9467bd",
    "students03": "#ff7f0e",
}


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)

    m = pd.read_csv(os.path.join(ROOT, "compare", "data", "a7_master.csv"))

    fig = plt.figure(figsize=(21, 13.5))
    gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.28,
                          height_ratios=[1, 1, 1.05])

    def style(ax, title, xl=None, yl=None):
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        if xl:
            ax.set_xlabel(xl, fontsize=9.5)
        if yl:
            ax.set_ylabel(yl, fontsize=9.5)
        ax.tick_params(labelsize=8.5)

    # ---------- A1：轨迹（用统计代替，画轨迹太占地方）----------
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(m))
    ax.bar(xs - 0.2, m["个体数"], 0.4, label="个体数",
           color=[COLORS[s] for s in m["scene"]], alpha=0.9)
    ax.bar(xs + 0.2, m["时长_s"], 0.4, label="时长 (s)",
           color=[COLORS[s] for s in m["scene"]], alpha=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([T.SCENES[s][0].split()[0] for s in m["scene"]],
                       fontsize=8, rotation=25)
    style(ax, "A1 轨迹：规模与时长", yl="计数")
    ax.legend(fontsize=8)

    # ---------- A2：速度分布 ----------
    ax = fig.add_subplot(gs[0, 1])
    for i, s in enumerate(m["scene"]):
        ax.errorbar(m["平均速度_m/s"].iloc[i],
                    i,
                    xerr=[[m["平均速度_m/s"].iloc[i] - 0],
                          [m["速度P95"].iloc[i] - m["平均速度_m/s"].iloc[i]]],
                    fmt="o", ms=9, color=COLORS[s], capsize=5, lw=2)
    ax.axvline(1.34, color="k", ls="--", lw=1.3)
    ax.text(1.37, -0.6, "1.34 m/s\n自由步速", fontsize=8)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([T.SCENES[s][0].split()[0] for s in m["scene"]], fontsize=8)
    style(ax, "A2 速度：均值与 P95", xl="速度 (m/s)")

    # ---------- A2b：加速度分解 ----------
    ax = fig.add_subplot(gs[0, 2])
    xs = np.arange(len(m))
    ax.bar(xs - 0.2, m["横向加速度均值"], 0.4, label="横向（转向）",
           color="#d62728", alpha=0.85)
    ax.bar(xs + 0.2, [abs(v) for v in m["横向加速度均值"] * 0 + 0.034] if False
           else m["加速度均值_m/s2"], 0.4, label="总加速度", color="#888", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([T.SCENES[s][0].split()[0] for s in m["scene"]],
                       fontsize=8, rotation=25)
    style(ax, "A2 加速度：横向 > 纵向", yl="m/s²")
    ax.legend(fontsize=8)

    # ---------- A2c：方向集中度 ----------
    ax = fig.add_subplot(gs[0, 3])
    ax.barh(range(len(m)), m["方向集中度R"],
            color=[COLORS[s] for s in m["scene"]], alpha=0.88)
    for i, v in enumerate(m["方向集中度R"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([T.SCENES[s][0].split()[0] for s in m["scene"]], fontsize=8)
    ax.invert_yaxis()
    ax.margins(x=0.25)
    style(ax, "A2 方向集中度（越低越双向）", xl="R")

    # ---------- A3：NN + TTC ----------
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(m["NN中位数_m"], m["NN<0.5m占比"], s=230,
               c=[COLORS[s] for s in m["scene"]], edgecolor="k", lw=1.2, zorder=3)
    for _, r in m.iterrows():
        ax.annotate(T.SCENES[r["scene"]][0].split()[0],
                    (r["NN中位数_m"], r["NN<0.5m占比"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8.5)
    style(ax, "A3 间距 vs 贴身接触率",
          xl="最近邻间距中位数 (m)", yl="NN<0.5m 占比")

    ax = fig.add_subplot(gs[1, 1])
    xs = np.arange(len(m))
    ax.bar(xs, m["TTC<2s占比"] * 100,
           color=[COLORS[s] for s in m["scene"]], alpha=0.88)
    for i, v in enumerate(m["TTC<2s占比"] * 100):
        ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([T.SCENES[s][0].split()[0] for s in m["scene"]],
                       fontsize=8, rotation=25)
    style(ax, "A3 碰撞风险（TTC<2s 占比）", yl="%")

    # ---------- A4：q-k 三个代表点 ----------
    ax = fig.add_subplot(gs[1, 2])
    for _, r in m.iterrows():
        ax.scatter(r["峰值流率对应密度"], r["峰值流率_通行能力"],
                   s=250, c=COLORS[r["scene"]], edgecolor="k", lw=1.2, zorder=3)
        ax.annotate(T.SCENES[r["scene"]][0].split()[0],
                    (r["峰值流率对应密度"], r["峰值流率_通行能力"]),
                    textcoords="offset points", xytext=(9, 6), fontsize=8.5)
    style(ax, "A4 基本图峰值（通行能力）",
          xl="峰值对应密度 (人/m²)", yl="通行能力 人/(m·s)")

    ax = fig.add_subplot(gs[1, 3])
    xs = np.arange(len(m))
    vals = m["自由流斜率"].astype(float)
    ax.bar(xs, vals,
           color=["#c00" if v < 0 else "#0a0" for v in vals], alpha=0.85)
    ax.axhline(0, color="k", lw=1.1)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.03 if v >= 0 else -0.07), f"{v:.2f}",
                ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([T.SCENES[s][0].split()[0] for s in m["scene"]],
                       fontsize=8, rotation=25)
    style(ax, "A4 速度随密度的衰减斜率", yl="m/s per (人/m²)")

    # ---------- A5：成行 / 避让 ----------
    ax = fig.add_subplot(gs[2, 0])
    xs = np.arange(len(m))
    ax.bar(xs - 0.2, m["成行事件数"], 0.4, label="成行",
           color="#1f77b4", alpha=0.88)
    ax.bar(xs + 0.2, m["避让事件数"], 0.4, label="避让",
           color="#d62728", alpha=0.88)
    ax.set_xticks(xs)
    ax.set_xticklabels([T.SCENES[s][0].split()[0] for s in m["scene"]],
                       fontsize=8, rotation=25)
    ax.set_yscale("log")
    style(ax, "A5 行为事件量（对数）", yl="事件数")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(m["密度中位数"], m["避让平均转角_deg"], s=240,
               c=[COLORS[s] for s in m["scene"]], edgecolor="k", lw=1.2, zorder=3)
    for _, r in m.iterrows():
        ax.annotate(T.SCENES[r["scene"]][0].split()[0],
                    (r["密度中位数"], r["避让平均转角_deg"]),
                    textcoords="offset points", xytext=(9, 5), fontsize=8.5)
    style(ax, "A5 密度越大→避让转角越大",
          xl="密度中位数 (人/m²)", yl="避让平均转角 (°)")

    ax = fig.add_subplot(gs[2, 2])
    xs = np.arange(len(m))
    ax.bar(xs, m["热点密度倍数"],
           color=[COLORS[s] for s in m["scene"]], alpha=0.88)
    ax.axhline(1.0, color="k", ls="--", lw=1.3)
    ax.text(0.1, 1.06, "1.0 = 无瓶颈", fontsize=8.5)
    for i, v in enumerate(m["热点密度倍数"]):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([T.SCENES[s][0].split()[0] for s in m["scene"]],
                       fontsize=8, rotation=25)
    style(ax, "A5 瓶颈强度（热点倍数）", yl="倍数")

    # ---------- 结论文字块 ----------
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    txt = (
        "核心结论（一句话版）\n"
        "────────────────────────\n"
        "A1  五个场景 = 五种空间结构，\n"
        "    看轨迹图就能读出人流走向\n\n"
        "A2  速度/方向都是多峰 → 均值会骗人；\n"
        "    加速度主要在横向（转向避让）\n\n"
        "A3  拥挤看 NN<0.5m（zara02 16%最挤）；\n"
        "    危险看 TTC<2s（ETH 26%最高，因为快）\n\n"
        "A4  q=k·v：通行能力由速度主导。\n"
        "    ETH 靠快拿 1.56，Hotel 两头低只有 0.55\n\n"
        "A5  三个普适常数：\n"
        "      避让距离 0.6–0.7 m\n"
        "      避让转角 15–17°\n"
        "      跟随时距 1.5–2.0 m"
    )
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=10.5,
            linespacing=1.75, transform=ax.transAxes,
            bbox=dict(fc="#f7f7f2", ec="#bbbbaa", lw=1.2,
                      boxstyle="round,pad=0.7"))

    fig.suptitle("A1–A5 行人轨迹分析 总览  |  ETH (seq_eth, seq_hotel) + "
                 "UCY (zara01, zara02, students03)",
                 fontsize=16, fontweight="bold", y=0.985)
    T.savefig(fig, os.path.join(FIG, "a8_summary_dashboard.png"))
    plt.close(fig)

    print("A8 完成")


if __name__ == "__main__":
    print("A8 总结报告图：开始")
    main()
