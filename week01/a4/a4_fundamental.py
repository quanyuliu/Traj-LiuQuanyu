"""
week04 / A4 —— 基本图：密度 k – 流率 q – 速度 v

任务：算出宏观量（密度、流率、速度），画基本图 q-k-v。

产出：
    figures/a4_fundamental_all.png   五场景 q-k / v-k / q-v 三联（代表图）
    figures/a4_qk_<scene>.png        单场景 q-k 详图（含峰值与分支标注）
    figures/a4_density_map_<scene>.png 密度场热力图（空间分布，配合看）
    data/a4_fundamental.csv          宏观量汇总
    data/a4_local_<scene>.csv        逐人逐帧局部量明细

======================================================================
方法说明 —— 这是本作业技术含量最高的一步，值得写清楚
======================================================================

【为什么不能用固定网格法】
最容易想到的做法是在场景上铺 1 m × 1 m 网格，数每格人数。
但 ETH/UCY 的数据**非常稀疏**：每帧在场人数只有
    seq_eth 6.1 / seq_hotel 5.4 / zara01 5.9 / zara02 9.2 / students03 33.1 人，
而场景面积 111–362 m²。也就是说**全局密度只有 0.017–0.151 人/m²**。

在这种密度下，1 m² 网格里绝大多数被占用的格子只有 1 个人，
密度被强行量化成 1 或 2 人/m²，密度范围被压窄，
q-k 曲线的曲率消失、速度衰减变成 0%——**基本图就废了**。
（本仓库第一版就是这么做的，结果速度降幅全部为 0%，因此推倒重来。）

【本仓库采用的方法：k 近邻密度估计】
对每一帧的每个人 i，找它到**第 k 近邻**的距离 r_k，则

    局部密度   k_i = k / (π · r_k²)        单位：人/m²
    局部速度   v_i = |velocity_i|           单位：m/s
    局部流率   q_i = k_i · v_i              单位：人/(m·s)

取 k=3。含义很直白：**r_k 小 ⇔ 邻居挤得近 ⇔ 密度高**。
这个估计量对稀疏数据稳健，且对孤立个体自然给出很低的密度（不会虚高）。

【与 Voronoi 法的关系】
行人流文献（如 arXiv:2409.11857）推荐 Voronoi 自由面积法：
把每个人的"专属面积" A_i 取为到所有邻居中垂线围成的多边形，则 k_i = 1/A_i。
本仓库在 `trajlib.local_density_voronoi()` 里也实现了 Voronoi 法，
两者量级一致（见 week04 结论中的交叉验证），
但 Voronoi 法计算慢约 40 倍且有边界退化问题，
所以正式出图用 k 近邻法，Voronoi 法保留作为验证。

【绘图约定】
    · q–k 图是"基本图"本体。自由流支斜率 ≈ 自由速度；拥堵支 q 反而下降。
    · 峰值流率（capacity）就是基本图的最高点，是通行能力的上限。
    · 用密度分箱求均值曲线（原始散点太散，看不清趋势）。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
import trajlib as T                                    # noqa: E402

plt = T.setup_matplotlib()

HERE = os.path.dirname(os.path.abspath(__file__))
WK = HERE   # 脚本自己所在目录（week01/aN），图与数据同级存放
FIG = os.path.join(WK, "figures")
DAT = os.path.join(WK, "data")

COLORS = {
    "seq_eth":    "#d62728",
    "seq_hotel":  "#1f77b4",
    "zara01":     "#2ca02c",
    "zara02":     "#9467bd",
    "students03": "#ff7f0e",
}

KNN_K = 3          # 近邻数
NBIN = 18          # 密度分箱数


def binned(g, kcol="k_density", qcol="q", nbin=NBIN, min_count=25):
    """按密度分箱，得到 q(k)、v(k) 的均值曲线（含标准误）。"""
    g = g.dropna(subset=[kcol, qcol, "speed"])
    if len(g) < 50:
        return None
    kmax = np.percentile(g[kcol], 99)
    edges = np.linspace(0, kmax, nbin + 1)
    idx = np.digitize(g[kcol], edges) - 1
    out = []
    for b in range(nbin):
        m = idx == b
        if m.sum() < min_count:
            continue
        sub = g[m]
        out.append({
            "k_mid": (edges[b] + edges[b + 1]) / 2,
            "k": sub[kcol].mean(),
            "v": sub["speed"].mean(),
            "v_sem": sub["speed"].std() / np.sqrt(len(sub)),
            "q": sub[qcol].mean(),
            "q_sem": sub[qcol].std() / np.sqrt(len(sub)),
            "n": int(len(sub)),
        })
    return pd.DataFrame(out) if len(out) >= 3 else None


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)

    local, curves, rows = {}, {}, []

    for s in T.SCENE_ORDER:
        df = T.load_scene(s)
        print(f"  局部量计算 {s} ...")
        d = T.local_density_knn(df, k=KNN_K)
        d["scene"] = s
        local[s] = d
        d.to_csv(os.path.join(DAT, f"a4_local_{s}.csv"), index=False)

        c = curves[s] = binned(d)
        if c is None:
            print("    !! 数据不足")
            continue

        ip = c["q"].idxmax()          # 峰值流率 = 通行能力
        # 自由流段：密度最低的 1/3
        n1 = max(3, len(c) // 3)
        z = np.polyfit(c["k"].iloc[:n1], c["v"].iloc[:n1], 1)

        rows.append({
            "scene": s, "场景": df.attrs["scene_cn"],
            "平均密度_人/m2": round(float(d["k_density"].mean()), 4),
            "密度中位数": round(float(d["k_density"].median()), 4),
            "密度P95": round(float(d["k_density"].quantile(0.95)), 3),
            "平均速度_m/s": round(float(d["speed"].mean()), 3),
            "平均流率_人/(m·s)": round(float(d["q"].mean()), 4),
            "峰值流率_通行能力": round(float(c.loc[ip, "q"]), 4),
            "峰值流率对应密度": round(float(c.loc[ip, "k"]), 3),
            "自由流速度_m/s": round(float(z[1]), 3),
            "自由流斜率": round(float(z[0]), 3),
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(DAT, "a4_fundamental.csv"),
               index=False, encoding="utf-8-sig")
    print("\n=== A4 基本图统计 ===")
    print(tab.to_string(index=False))

    # ================= 图 1：五场景三联 =================
    fig, axes = plt.subplots(3, 5, figsize=(20.5, 13))

    for j, s in enumerate(T.SCENE_ORDER):
        g, c, col = local[s], curves[s], COLORS[s]
        nm = f"{T.SCENES[s][0]}\n({s})"

        # 抽样画散点（避免图太重）
        gg = g.sample(min(len(g), 3000), random_state=0) if len(g) > 3000 else g

        # --- 行 1：q–k 基本图
        ax = axes[0, j]
        ax.scatter(gg["k_density"], gg["q"], s=4, color=col, alpha=0.12)
        if c is not None:
            ax.plot(c["k"], c["q"], "o-", color="k", ms=4.5, lw=1.9)
            ip = c["q"].idxmax()
            ax.plot(c.loc[ip, "k"], c.loc[ip, "q"], "*", ms=17,
                    color="#ffcc00", mec="k", mew=0.9, zorder=5)
        ax.set_title(nm, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("密度 k (人/m²)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("流率 q (人/(m·s))", fontsize=10)
            ax.text(0.5, 1.05, "q–k 基本图", transform=ax.transAxes,
                    ha="center", fontsize=10.5, fontweight="bold")
        ax.tick_params(labelsize=8)

        # --- 行 2：v–k
        ax = axes[1, j]
        ax.scatter(gg["k_density"], gg["speed"], s=4, color=col, alpha=0.12)
        if c is not None:
            ax.plot(c["k"], c["v"], "o-", color="k", ms=4.5, lw=1.9)
            ax.fill_between(c["k"], c["v"] - c["v_sem"], c["v"] + c["v_sem"],
                            color="k", alpha=0.15)
        ax.set_xlabel("密度 k (人/m²)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("速度 v (m/s)", fontsize=10)
            ax.text(0.5, 1.05, "v–k 速度衰减", transform=ax.transAxes,
                    ha="center", fontsize=10.5, fontweight="bold")
        ax.tick_params(labelsize=8)

        # --- 行 3：q–v
        ax = axes[2, j]
        ax.scatter(gg["speed"], gg["q"], s=4, color=col, alpha=0.12)
        if c is not None:
            ax.plot(c["v"], c["q"], "o-", color="k", ms=4.5, lw=1.9)
        ax.set_xlabel("速度 v (m/s)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("流率 q (人/(m·s))", fontsize=10)
            ax.text(0.5, 1.05, "q–v", transform=ax.transAxes,
                    ha="center", fontsize=10.5, fontweight="bold")
        ax.tick_params(labelsize=8)

    fig.suptitle("A4 基本图（q–k–v）—— 由 k 近邻局部密度估计 "
                 f"(k={KNN_K})，五场景 × 三视图",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    T.savefig(fig, os.path.join(FIG, "a4_fundamental_all.png"))
    plt.close(fig)

    # ================= 图 2：单场景 q–k 详图 =================
    for s in T.SCENE_ORDER:
        g, c = local[s], curves[s]
        if c is None:
            continue
        col = COLORS[s]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.4))

        ax.scatter(g["k_density"], g["q"], s=4, color=col, alpha=0.11,
                   label=f"逐个体逐帧数据 (n={len(g)})")
        ax.plot(c["k"], c["q"], "o-", color="k", lw=2.3, ms=6.5,
                label="密度分箱均值")
        ax.fill_between(c["k"], c["q"] - c["q_sem"], c["q"] + c["q_sem"],
                        color="k", alpha=0.15)
        ip = c["q"].idxmax()
        ax.plot(c.loc[ip, "k"], c.loc[ip, "q"], "*", ms=22, color="#ffcc00",
                mec="k", mew=1.1, zorder=5,
                label=f"通行能力 {c.loc[ip,'q']:.2f} 人/(m·s)")
        ax.axvline(c.loc[ip, "k"], color="gray", ls="--", lw=1.4)
        ax.annotate("自由流支", (c["k"].iloc[1], c["q"].iloc[1]),
                    textcoords="offset points", xytext=(-6, 22),
                    fontsize=10, color="#0a0",
                    arrowprops=dict(arrowstyle="->", color="#0a0", lw=1.4))
        if c.loc[ip, "k"] < c["k"].max() * 0.95:
            ax.annotate("拥堵支", (c["k"].iloc[-1], c["q"].iloc[-1]),
                        textcoords="offset points", xytext=(-30, -26),
                        fontsize=10, color="#c00",
                        arrowprops=dict(arrowstyle="->", color="#c00", lw=1.4))
        ax.set_xlabel("密度 k (人/m²)")
        ax.set_ylabel("流率 q (人/(m·s))")
        ax.set_title("q–k 基本图", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")

        ax2.scatter(g["k_density"], g["speed"], s=4, color=col, alpha=0.11)
        ax2.plot(c["k"], c["v"], "o-", color="k", lw=2.3, ms=6.5)
        ax2.fill_between(c["k"], c["v"] - c["v_sem"], c["v"] + c["v_sem"],
                         color="k", alpha=0.15, label="±1 标准误")
        n1 = max(3, len(c) // 3)
        z = np.polyfit(c["k"].iloc[:n1], c["v"].iloc[:n1], 1)
        xs = np.linspace(0, c["k"].iloc[n1], 30)
        ax2.plot(xs, np.polyval(z, xs), "r--", lw=1.9,
                 label=f"自由流拟合 v≈{z[1]:.2f}{z[0]:+.2f}k")
        ax2.set_xlabel("密度 k (人/m²)")
        ax2.set_ylabel("速度 v (m/s)")
        ax2.set_title("v–k 速度随密度衰减", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)

        fig.suptitle(f"A4 基本图 —— {T.SCENES[s][0]} ({s})",
                     fontsize=13.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        T.savefig(fig, os.path.join(FIG, f"a4_qk_{s}.png"))
        plt.close(fig)

    # ================= 图 3：密度场空间热力图 =================
    for s in T.SCENE_ORDER:
        g = local[s]
        df = T.load_scene(s)
        col = COLORS[s]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))

        sc = a1.scatter(g["x"], g["y"], c=g["k_density"], s=9,
                        cmap="YlOrRd", vmin=0,
                        vmax=np.percentile(g["k_density"], 97))
        fig.colorbar(sc, ax=a1, label="局部密度 k (人/m²)", fraction=0.045)
        a1.set_aspect("equal", adjustable="datalim")
        a1.set_xlabel("x (m)")
        a1.set_ylabel("y (m)")
        a1.set_title("密度场空间分布", fontsize=11.5, fontweight="bold")

        # 平均密度随时间的演化
        ts = g.groupby("t")["k_density"].mean()
        a2.plot(ts.index, ts.values, "-", color=col, lw=1.8)
        a2.axhline(ts.mean(), color="k", ls="--", lw=1.2,
                   label=f"全时段均值 {ts.mean():.3f}")
        a2.fill_between(ts.index, 0, ts.values, color=col, alpha=0.15)
        a2.set_xlabel("时间 (s)")
        a2.set_ylabel("平均局部密度 (人/m²)")
        a2.set_title("密度随时间的波动", fontsize=11.5, fontweight="bold")
        a2.legend(fontsize=9)
        a2.set_ylim(bottom=0)

        fig.suptitle(f"A4 密度场 —— {df.attrs['scene_cn']} ({s})",
                     fontsize=13.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        T.savefig(fig, os.path.join(FIG, f"a4_density_map_{s}.png"))
        plt.close(fig)

    print("\nA4 完成")


if __name__ == "__main__":
    print("A4 基本图 q-k-v：开始")
    main()
