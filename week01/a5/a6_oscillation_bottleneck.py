"""
week06 / A5(下) —— 行为分析：震荡（stop-and-go 波） 与 瓶颈

任务：检测并可视化另两类现象：震荡（拥堵波）和瓶颈。

产出：
    figures/a6_oscillation_all.png   震荡：密度时空图 + 速度波动 + 频谱
    figures/a6_bottleneck_all.png    瓶颈：高密度区识别 + 通过率对比
    data/a6_oscillation.csv          震荡指标
    data/a6_bottleneck.csv           瓶颈指标
    data/a6_spacetime_<scene>.csv    密度时空场（供复现）

======================================================================
方法
======================================================================

【震荡 / stop-and-go 波】
现象：人群密度高的地方，行人会"走一下停一下"，形成向后传播的密度波。
检测：
    1. 把空间切成 1 m 的纵向条带（按主运动方向），时间切 1 s 窗口，
       作**密度时空图**（x 轴时间，y 轴位置，颜色=密度）。
    2. 在时空图上，震荡表现为**斜向的明暗条纹**。
    3. 量化：对每个空间位置取密度时间序列，算
       · 波动强度 = 时间标准差 / 时间均值（变异系数 CV）
       · 主周期 = 自相关函数第一个峰的滞后
       · stop-and-go 比例 = 速度<0.3 m/s 的样本占比

【瓶颈】
现象：某处的通行能力明显低于别处，人在这里排队堆积。
检测：
    1. 用 Week 04 的局部密度，找**高密度聚集区**（密度 > P90）。
    2. 用 KDE/直方图找密度的空间热点（局部极大值）。
    3. 量化：热点区域的通过率（每秒通过人数）与场景平均对比，
       明显偏低者判为瓶颈。

注意：本数据集没有"门"的标注，所以瓶颈是**从数据里发现的**，
不是预先指定的。这也是限制：只能发现"数据中表现出拥堵的位置"。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
import trajlib as T                                    # noqa: E402

plt = T.setup_matplotlib()
from scipy.ndimage import gaussian_filter               # noqa: E402

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

STILL_SPEED = 0.30      # 低于此速度视为"停"（m/s）
BAND = 1.0              # 纵向条带宽度（m）


def spacetime(df, local, band=BAND):
    """构造密度时空场。

    为了让"纵向"有统一含义，把坐标系旋转到主运动方向（PCA 第一主成分）。
    返回 (X轴=时间, Y轴=纵向位置, Z=密度均值) 三个数组 + 位置网格。
    """
    # 主方向：所有速度向量的主成分
    v = df.dropna(subset=["vx", "vy"])[["vx", "vy"]].to_numpy()
    if len(v) < 10:
        return None
    v = v - v.mean(0)
    # 用 SVD 求主方向
    _, _, vt = np.linalg.svd(v, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])          # 法向

    loc = local.copy()
    loc = loc.merge(df[["ped_id", "frame", "x", "y"]],
                    on=["ped_id", "frame"], how="left", suffixes=("", "_d"))
    xs = loc["x_d"].to_numpy()
    ys = loc["y_d"].to_numpy()
    s = xs * d[0] + ys * d[1]            # 沿主方向坐标
    s = s - s.min()

    nb = int(np.ceil(s.max() / band)) + 1
    times = np.sort(loc["t"].unique())
    tidx = {t: i for i, t in enumerate(times)}
    M = np.full((nb, len(times)), np.nan)
    bidx = np.floor(s / band).astype(int)
    ti = loc["t"].map(tidx).to_numpy()
    k = loc["k_density"].to_numpy()
    for b, t_, kv in zip(bidx, ti, k):
        if not np.isnan(kv):
            cur = M[b, t_]
            M[b, t_] = kv if np.isnan(cur) else (cur + kv) / 2
    # 平滑（时空二维）
    Ms = np.where(np.isnan(M), 0, M)
    W = (~np.isnan(M)).astype(float)
    Ms = gaussian_filter(Ms, sigma=1.2)
    Ws = gaussian_filter(W, sigma=1.2)
    Msm = np.divide(Ms, Ws, out=np.full_like(Ms, np.nan), where=Ws > 0.05)
    return Msm, times, np.arange(nb) * band


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)

    stor = {}
    osc_rows, bn_rows = [], []

    for s in T.SCENE_ORDER:
        df = T.load_scene(s)
        print(f"  分析 {s} ...")
        local = T.local_density_knn(df, k=3)

        # ---------- 震荡 ----------
        st = spacetime(df, local)
        stor[s] = (st, df, local)

        d = df.dropna(subset=["speed"])
        # 每个时间窗的均值速度序列
        ts = d.groupby("t")["speed"].mean()
        v = ts.to_numpy()
        cv = float(v.std() / v.mean()) if v.mean() > 0 else np.nan

        # 自相关主周期
        period = np.nan
        if len(v) > 12:
            vc = v - v.mean()
            ac = np.correlate(vc, vc, mode="full")[len(vc) - 1:]
            ac = ac / ac[0]
            # 找第一个过零点后的第一个峰
            pk = None
            for i in range(2, min(len(ac) - 1, 60)):
                if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.15:
                    pk = i
                    break
            if pk:
                period = float(pk * T.DT)

        still = float((d["speed"] < STILL_SPEED).mean())
        # 强烈震荡指标：密度时空场的条纹对比度
        stripe = np.nan
        if st is not None:
            M = st[0]
            stripe = float(np.nanstd(M) / max(np.nanmean(M), 1e-9))

        osc_rows.append({
            "scene": s, "场景": df.attrs["scene_cn"],
            "速度变异系数CV": round(cv, 4),
            "速度自相关主周期_s": round(period, 2) if period == period else None,
            "stop比例_%": round(still * 100, 1),
            "密度时空条纹对比度": round(stripe, 3) if stripe == stripe else None,
            "场景时长_s": round(float(ts.index.max()), 1),
        })

        # ---------- 瓶颈 ----------
        # 思路：瓶颈 = 空间上密度显著高于其它地方的"热点"。
        # 比较基准必须是**被行人占用的区域**的典型密度，
        # 而不是把整个包围盒（含大片空地）一起平均 —— 否则热点会被空地稀释，
        # 甚至出现"热点密度低于全局均值"的假象。
        # 因此基准取"被占用网格密度的 P50"，热点取 P95。
        d2 = df.dropna(subset=["speed"])

        # 空间 2D 直方图（不加权），找出"被占用"的格子
        Hc, xe, ye = np.histogram2d(local["x"], local["y"], bins=26)
        Hk, _, _ = np.histogram2d(local["x"], local["y"], bins=26,
                                  weights=local["k_density"])
        Hm = np.divide(Hk, np.maximum(Hc, 1))
        Hm[Hc < 3] = np.nan                 # 样本太少的格子不参与
        Hm = gaussian_filter(np.nan_to_num(Hm), sigma=1.4)

        occ = Hm[Hc > 0]
        occ = occ[occ > 0]
        k_typ = float(np.median(occ)) if len(occ) else np.nan     # 典型占用密度
        k_hotP95 = float(np.percentile(local["k_density"], 95))

        iy, ix = np.unravel_index(np.argmax(Hm), Hm.shape)
        hot_x = (xe[ix] + xe[ix + 1]) / 2
        hot_y = (ye[iy] + ye[iy + 1]) / 2
        # 热点附近 2 m 内的实际密度
        near = np.hypot(local["x"] - hot_x, local["y"] - hot_y) < 2.0
        k_hot = float(local.loc[near, "k_density"].mean()) if near.sum() else np.nan

        bn_rows.append({
            "scene": s, "场景": df.attrs["scene_cn"],
            "热点位置x": round(hot_x, 1), "热点位置y": round(hot_y, 1),
            "热点平均密度": round(k_hot, 3) if k_hot == k_hot else None,
            "占用区典型密度_P50": round(k_typ, 3) if k_typ == k_typ else None,
            "全场景密度P95": round(k_hotP95, 3),
            "热点密度倍数": round(k_hot / max(k_typ, 1e-9), 2) if k_hot == k_hot else None,
            "低速度阈值_P10": round(d2["speed"].quantile(0.10), 3),
        })

    osc = pd.DataFrame(osc_rows)
    bn = pd.DataFrame(bn_rows)
    osc.to_csv(os.path.join(DAT, "a6_oscillation.csv"),
               index=False, encoding="utf-8-sig")
    bn.to_csv(os.path.join(DAT, "a6_bottleneck.csv"),
              index=False, encoding="utf-8-sig")

    print("\n=== A6 震荡指标 ===")
    print(osc.to_string(index=False))
    print("\n=== A6 瓶颈（密度热点）===")
    print(bn.to_string(index=False))

    # ================= 图 1：震荡 =================
    fig, axes = plt.subplots(3, 5, figsize=(21, 12))

    for j, s in enumerate(T.SCENE_ORDER):
        st, df, local = stor[s]
        col = COLORS[s]
        nm = f"{T.SCENES[s][0]}\n({s})"

        # 行 1：密度时空图
        ax = axes[0, j]
        if st is not None:
            M, times, pos = st
            im = ax.pcolormesh(times, pos, M, shading="auto",
                               cmap="YlOrRd", vmin=0,
                               vmax=np.nanpercentile(M, 97))
            fig.colorbar(im, ax=ax, fraction=0.045, label="密度 人/m²")
        ax.set_title(nm, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("时间 (s)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("沿主方向位置 (m)", fontsize=9.5)
            ax.text(0.5, 1.05, "密度时空图\n(斜条纹=震荡波)",
                    transform=ax.transAxes, ha="center",
                    fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

        # 行 2：平均速度时间序列
        ax = axes[1, j]
        d = df.dropna(subset=["speed"])
        ts = d.groupby("t")["speed"].mean()
        ax.plot(ts.index, ts.values, "-", color=col, lw=1.3)
        ax.axhline(ts.mean(), color="k", ls="--", lw=1.2)
        cv = ts.std() / ts.mean()
        ax.text(0.97, 0.94, f"CV={cv:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        ax.set_xlabel("时间 (s)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("平均速度 (m/s)", fontsize=9.5)
            ax.text(0.5, 1.05, "速度时间波动", transform=ax.transAxes,
                    ha="center", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

        # 行 3：自相关
        ax = axes[2, j]
        v = ts.to_numpy()
        if len(v) > 12:
            vc = v - v.mean()
            ac = np.correlate(vc, vc, mode="full")[len(vc) - 1:]
            ac = ac / ac[0]
            lags = np.arange(len(ac)) * T.DT
            m = lags <= 40
            ax.plot(lags[m], ac[m], "-", color=col, lw=1.6)
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(0.15, color="gray", ls=":", lw=1)
            ax.fill_between(lags[m], 0, ac[m], color=col, alpha=0.15)
        ax.set_xlabel("滞后 (s)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("自相关", fontsize=9.5)
            ax.text(0.5, 1.05, "速度自相关(找周期)", transform=ax.transAxes,
                    ha="center", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

    fig.suptitle("A5（下）震荡 / stop-and-go 波 —— 密度时空场、速度波动、自相关",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    T.savefig(fig, os.path.join(FIG, "a6_oscillation_all.png"))
    plt.close(fig)

    # ================= 图 2：瓶颈 =================
    fig, axes = plt.subplots(2, 6, figsize=(22, 7.6))

    for j, s in enumerate(T.SCENE_ORDER):
        st, df, local = stor[s]
        col = COLORS[s]
        row = bn[bn["scene"] == s].iloc[0]

        # 上行：密度场 + 热点标注
        ax = axes[0, j]
        sc = ax.scatter(local["x"], local["y"], c=local["k_density"], s=7,
                        cmap="YlOrRd", vmin=0,
                        vmax=np.percentile(local["k_density"], 96))
        ax.plot(row["热点位置x"], row["热点位置y"], "o", ms=15,
                mfc="none", mec="#0044ff", mew=2.4)
        ax.annotate(f"热点\n×{row['热点密度倍数']:.1f}",
                    (row["热点位置x"], row["热点位置y"]),
                    textcoords="offset points", xytext=(12, 8),
                    fontsize=8.5, color="#0044ff", fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{T.SCENES[s][0]}\n({s})", fontsize=9.5,
                     fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            fig.colorbar(sc, ax=ax, fraction=0.05, label="密度 人/m²")

        # 下行：密度分布的直方图（看有没有"长尾高密度区"）
        ax = axes[1, j]
        ax.hist(local["k_density"], bins=np.arange(0, 3.05, 0.1),
                color=col, alpha=0.85, edgecolor="white", lw=0.4)
        p90 = np.percentile(local["k_density"], 90)
        ax.axvline(p90, color="k", ls="--", lw=1.5)
        ax.text(p90 * 1.05, ax.get_ylim()[1] * 0.85,
                f"P90={p90:.2f}", fontsize=8.5)
        ax.set_xlabel("局部密度 (人/m²)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("计数", fontsize=9.5)
        ax.tick_params(labelsize=8)

    # 最后一格：瓶颈强度对比
    ax = axes[0, 5]
    vals = [float(bn[bn["scene"] == s]["热点密度倍数"].iloc[0])
            for s in T.SCENE_ORDER]
    ax.barh(range(len(T.SCENE_ORDER)), vals,
            color=[COLORS[s] for s in T.SCENE_ORDER], alpha=0.88)
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.1f}×", va="center", fontsize=9)
    ax.set_yticks(range(len(T.SCENE_ORDER)))
    ax.set_yticklabels(T.SCENE_ORDER, fontsize=8)
    ax.set_xlabel("热点密度 / 全局平均密度")
    ax.set_title("瓶颈强度", fontsize=10, fontweight="bold")
    ax.invert_yaxis()

    ax = axes[1, 5]
    ax.bar(range(len(T.SCENE_ORDER)),
           [float(osc[osc["scene"] == s]["stop比例_%"].iloc[0])
            for s in T.SCENE_ORDER],
           color=[COLORS[s] for s in T.SCENE_ORDER], alpha=0.88)
    for i, s in enumerate(T.SCENE_ORDER):
        v = float(osc[osc["scene"] == s]["stop比例_%"].iloc[0])
        ax.text(i, v + 0.3, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(T.SCENE_ORDER)))
    ax.set_xticklabels(T.SCENE_ORDER, fontsize=7.5, rotation=30)
    ax.set_ylabel("速度<0.3 m/s 占比 (%)")
    ax.set_title("完全停下(stop)的比例", fontsize=10, fontweight="bold")

    fig.suptitle("A5（下）瓶颈分析 —— 高密度热点识别与停走比例",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    T.savefig(fig, os.path.join(FIG, "a6_bottleneck_all.png"))
    plt.close(fig)

    print("\nA5（下）完成")


if __name__ == "__main__":
    print("A5 行为分析（震荡 / 瓶颈）：开始")
    main()
