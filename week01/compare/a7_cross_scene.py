"""
week07 —— 跨场景横向对比

任务：把前六周的指标汇到一张表/一组图里，横向比较五个场景，
      找出"哪些指标是场景决定的、哪些是普适常数"。

产出：
    figures/a7_radar.png            五场景多维雷达图（代表图）
    figures/a7_matrix.png           指标相关矩阵散点图
    figures/a7_ranking.png          各指标排序条形图
    data/a7_master.csv              汇总总表（所有指标）
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
import trajlib as T                                    # noqa: E402

plt = T.setup_matplotlib()

HERE = os.path.dirname(os.path.abspath(__file__))   # week01/compare
WK = HERE    # 图与数据放在本目录
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


def load_csv(sub, name):
    """读取 week01/<sub>/data/<name> 下的中间结果。"""
    p = os.path.join(ROOT, sub, "data", name)
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)

    # ---- 汇总各周的表
    base = load_csv("a1", "a1_scene_stats.csv")
    kine = load_csv("a2", "a2_kinematics.csv")
    nns = load_csv("a3", "a3_nn_stats.csv")
    ttcs = load_csv("a3", "a3_ttc_stats.csv")
    fund = load_csv("a4", "a4_fundamental.csv")
    beh = load_csv("a5", "a5_behavior_summary.csv")
    osc = load_csv("a5", "a6_oscillation.csv")
    bnk = load_csv("a5", "a6_bottleneck.csv")

    m = pd.DataFrame({"scene": T.SCENE_ORDER})
    m["场景"] = [T.SCENES[s][0] for s in T.SCENE_ORDER]

    def merge(df, cols):
        nonlocal m
        if df is None:
            return
        use = ["scene"] + [c for c in cols if c in df.columns]
        m = m.merge(df[use], on="scene", how="left")

    merge(base, ["个体数", "时长_s", "平均速度_m/s", "最大速度_m/s"])
    merge(kine, ["速度中位数", "速度P95", "加速度均值_m/s2", "横向加速度均值", "方向集中度R"])
    merge(nns, ["NN中位数_m", "NN<0.5m占比"])
    merge(ttcs, ["危险对_帧数", "TTC中位数_s", "TTC<1s对数", "TTC<2s占比"])
    merge(fund, ["密度中位数", "峰值流率_通行能力", "峰值流率对应密度", "自由流斜率"])
    merge(beh, ["成行事件数", "避让事件数", "避让平均最近距离_m", "避让平均转角_deg"])
    merge(osc, ["速度变异系数CV", "stop比例_%", "密度时空条纹对比度"])
    merge(bnk, ["热点密度倍数"])

    m.to_csv(os.path.join(DAT, "a7_master.csv"),
             index=False, encoding="utf-8-sig")
    print("=== A7 汇总总表 ===")
    print(m.to_string(index=False))

    # ================= 雷达图 =================
    dims = [
        ("平均速度_m/s", "速度快", False),
        ("密度中位数", "密度高", False),
        ("NN<0.5m占比", "贴身接触多", False),
        ("TTC<2s占比", "碰撞风险高", False),
        ("峰值流率_通行能力", "通行能力大", False),
        ("stop比例_%", "停顿比例高", False),
        ("速度变异系数CV", "速度波动大", False),
        ("避让平均转角_deg", "避让幅度大", False),
    ]
    labels = [d[1] for d in dims]
    # 归一化到 0-1（各维独立 min-max）
    norm = []
    for key, _, _ in dims:
        v = m[key].astype(float).to_numpy()
        lo, hi = np.nanmin(v), np.nanmax(v)
        norm.append((v - lo) / (hi - lo + 1e-12))
    norm = np.array(norm)                 # shape (ndim, nscene)

    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    fig, axes = plt.subplots(1, 2, figsize=(17, 8),
                             subplot_kw=dict(polar=True))
    for ax, which in zip(axes, ["orig", "norm"]):
        for i, s in enumerate(T.SCENE_ORDER):
            vals = norm[:, i] if which == "norm" else np.array(
                [(m[key].astype(float).iloc[i] /
                  (np.nanmax(m[key].astype(float)) + 1e-12)) for key, _, _ in dims])
            v = np.concatenate([vals, vals[:1]])
            ax.plot(ang_c, v, "o-", lw=2.2, ms=5, color=COLORS[s],
                    label=f"{T.SCENES[s][0]}")
            ax.fill(ang_c, v, color=COLORS[s], alpha=0.10)
        ax.set_xticks(ang)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_title("各维按场景间 min-max 归一化"
                     if which == "norm" else "各维按全局最大值归一化",
                     fontsize=12, fontweight="bold", pad=18)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=9.5)
    fig.suptitle("A7 五场景多维特征雷达图", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    T.savefig(fig, os.path.join(FIG, "a7_radar.png"))
    plt.close(fig)

    # ================= 排序条形图 =================
    show = [
        ("平均速度_m/s", "平均速度 (m/s)", "%.2f", False),
        ("密度中位数", "密度中位数 (人/m²)", "%.3f", False),
        ("NN中位数_m", "最近邻间距中位数 (m)", "%.2f", True),
        ("TTC<2s占比", "TTC<2s 占比", "%.3f", False),
        ("峰值流率_通行能力", "通行能力 人/(m·s)", "%.2f", False),
        ("stop比例_%", "停顿比例 (%)", "%.1f", False),
        ("速度变异系数CV", "速度变异系数", "%.3f", False),
        ("避让平均转角_deg", "避让平均转角 (°)", "%.1f", False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (key, title, fmt, asc) in zip(axes.ravel(), show):
        d = m[["scene", key]].dropna().sort_values(key, ascending=asc)
        y = np.arange(len(d))
        ax.barh(y, d[key].astype(float),
                color=[COLORS[s] for s in d["scene"]], alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels([T.SCENES[s][0] for s in d["scene"]], fontsize=9)
        for yi, v in zip(y, d[key].astype(float)):
            ax.text(v, yi, " " + (fmt % v), va="center", fontsize=8.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.margins(x=0.22)
    fig.suptitle("A7 各指标的场景排序", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    T.savefig(fig, os.path.join(FIG, "a7_ranking.png"))
    plt.close(fig)

    # ================= 相关矩阵 =================
    keys = [k for k, _, _, _ in show] + ["个体数"]
    sub = m[keys].astype(float)
    labs = ["平均速度", "密度中位数", "NN间距", "TTC<2s占比",
            "通行能力", "停顿比例", "速度CV", "避让转角", "个体数"]
    C = sub.corr(method="spearman").to_numpy()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 7.2))
    im = a1.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    a1.set_xticks(range(len(labs))); a1.set_xticklabels(labs, rotation=42,
                                                        ha="right", fontsize=9.5)
    a1.set_yticks(range(len(labs))); a1.set_yticklabels(labs, fontsize=9.5)
    for i in range(len(labs)):
        for j in range(len(labs)):
            a1.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if abs(C[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=a1, fraction=0.046, label="Spearman 相关系数")
    a1.set_title("指标间的相关矩阵（秩相关）", fontsize=12, fontweight="bold")

    # 关键关系：速度 vs 通行能力
    a2.scatter(m["平均速度_m/s"], m["峰值流率_通行能力"],
               s=220, c=[COLORS[s] for s in m["scene"]],
               edgecolor="k", lw=1.2, zorder=3)
    for _, r in m.iterrows():
        a2.annotate(T.SCENES[r["scene"]][0],
                    (r["平均速度_m/s"], r["峰值流率_通行能力"]),
                    textcoords="offset points", xytext=(9, 6), fontsize=9)
    a2.set_xlabel("平均速度 (m/s)")
    a2.set_ylabel("通行能力 人/(m·s)")
    a2.set_title("速度 vs 通行能力（q=k·v 的体现）",
                 fontsize=12, fontweight="bold")
    if len(m) > 2:
        z = np.polyfit(m["平均速度_m/s"], m["峰值流率_通行能力"], 1)
        xs = np.linspace(m["平均速度_m/s"].min(), m["平均速度_m/s"].max(), 30)
        rs = m[["平均速度_m/s", "峰值流率_通行能力"]].corr(
            method="spearman").iloc[0, 1]
        a2.plot(xs, np.polyval(z, xs), "k--", lw=1.6,
                label=f"Spearman ρ = {rs:.2f}")
        a2.legend(fontsize=10)

    fig.suptitle("A7 指标间关系", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    T.savefig(fig, os.path.join(FIG, "a7_matrix.png"))
    plt.close(fig)

    print("\nA7 完成")


if __name__ == "__main__":
    print("A7 跨场景对比：开始")
    main()
