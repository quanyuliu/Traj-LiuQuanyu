"""
week02 / A2 —— 个体运动量：速度、方向、加速度分布

任务：对每个个体计算速度 / 方向 / 加速度，画出分布图。
产出：
    figures/a2_speed_all.png       五场景速度分布对比
    figures/a2_kinematics_all.png  五场景 速度/方向/加速度 三联分布
    figures/a2_speed_hist_<scene>.png  单场景速度直方图（含分位数标注）
    data/a2_kinematics.csv         各场景运动学统计
    data/a2_speed_pooled.csv       逐点速度明细（供 Week 07 复用）

速度是用相邻标注点差分算的。注意 t = frame / 25（见 trajlib 顶部说明）。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
import trajlib as T                                    # noqa: E402

plt = T.setup_matplotlib()
from matplotlib.patches import Patch                   # noqa: E402

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


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)

    data = {s: T.load_scene(s) for s in T.SCENE_ORDER}

    # ---------------- 统计表 ----------------
    rows = []
    pooled = []
    for s in T.SCENE_ORDER:
        df = data[s]
        d = df.dropna(subset=["speed"])
        rows.append({
            "scene": s,
            "场景": df.attrs["scene_cn"],
            "个体数": int(df["ped_id"].nunique()),
            "速度均值_m/s": round(float(d["speed"].mean()), 3),
            "速度中位数": round(float(d["speed"].median()), 3),
            "速度P5": round(float(d["speed"].quantile(0.05)), 3),
            "速度P95": round(float(d["speed"].quantile(0.95)), 3),
            "速度标准差": round(float(d["speed"].std()), 3),
            "个体间速度标准差": round(float(
                d.groupby("ped_id")["speed"].mean().std()), 3),
            "加速度均值_m/s2": round(float(d["acc"].mean()), 3),
            "加速度P95_m/s2": round(float(d["acc"].quantile(0.95)), 3),
            "切向加速度均值": round(float(d["acc_along"].mean()), 4),
            "横向加速度均值": round(float(d["acc_lat"].abs().mean()), 4),
            # 方向一致性：方向角的标准差（圆统计）
            "方向集中度R": round(float(np.abs(np.exp(
                1j * np.radians(d["heading_deg"])).mean())), 3),
        })
        for _, r in d.iterrows():
            pooled.append((s, r["speed"]))

    stat = pd.DataFrame(rows)
    stat.to_csv(os.path.join(DAT, "a2_kinematics.csv"),
                index=False, encoding="utf-8-sig")
    pd.DataFrame(pooled, columns=["scene", "speed"]).to_csv(
        os.path.join(DAT, "a2_speed_pooled.csv"), index=False)

    print("=== A2 运动学统计 ===")
    print(stat.to_string(index=False))

    # ---------------- 图 1：速度分布对比（KDE + 直方图） ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))

    bins = np.arange(0, 3.05, 0.05)
    for s in T.SCENE_ORDER:
        v = data[s]["speed"].dropna().to_numpy()
        ax1.hist(v, bins=bins, histtype="step", lw=2.0,
                 color=COLORS[s], density=True,
                 label=f"{data[s].attrs['scene_cn']}  μ={v.mean():.2f}")
    ax1.set_xlabel("速度 (m/s)")
    ax1.set_ylabel("概率密度")
    ax1.set_title("速度分布对比（五场景，归一化直方图）",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.axvline(1.34, color="k", ls="--", lw=1.2, alpha=0.7)
    ax1.text(1.36, ax1.get_ylim()[1] * 0.88,
             "1.34 m/s\n(Weidmann 自由步速)", fontsize=8.5, va="top")

    # 小提琴图更直观看出"多峰"
    vp = ax2.violinplot([data[s]["speed"].dropna().to_numpy()
                         for s in T.SCENE_ORDER],
                        showmeans=True, showmedians=True, widths=0.85)
    for i, s in enumerate(T.SCENE_ORDER):
        vp["bodies"][i].set_facecolor(COLORS[s])
        vp["bodies"][i].set_alpha(0.55)
    ax2.set_xticks(range(1, 6))
    ax2.set_xticklabels([f"{data[s].attrs['scene_cn']}\n({s})"
                         for s in T.SCENE_ORDER], fontsize=8.5)
    ax2.set_ylabel("速度 (m/s)")
    ax2.set_title("速度分布的小提琴图（看形状与离散度）",
                  fontsize=12, fontweight="bold")

    fig.suptitle("A2 个体速度分布 —— 五场景对比",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    T.savefig(fig, os.path.join(FIG, "a2_speed_all.png"))
    plt.close(fig)

    # ---------------- 图 2：速度 / 方向 / 加速度 三联 ----------------
    fig, axes = plt.subplots(3, 5, figsize=(20, 11.5))

    for j, s in enumerate(T.SCENE_ORDER):
        df = data[s]
        d = df.dropna(subset=["speed"])
        c = COLORS[s]
        nm = f"{df.attrs['scene_cn']}\n({s})"

        # 第 1 行：速度
        ax = axes[0, j]
        ax.hist(d["speed"], bins=np.arange(0, 3.05, 0.06),
                color=c, alpha=0.8, edgecolor="white", lw=0.4)
        ax.axvline(d["speed"].mean(), color="k", ls="--", lw=1.5)
        ax.text(0.97, 0.95, f"μ={d['speed'].mean():.2f}\n"
                            f"σ={d['speed'].std():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title(nm, fontsize=9.5, fontweight="bold")
        if j == 0:
            ax.set_ylabel("速度计数", fontsize=10)
        ax.set_xlabel("速度 (m/s)", fontsize=8.5)
        ax.tick_params(labelsize=8)

        # 第 2 行：方向（极坐标风格的玫瑰图）
        ax = axes[1, j]
        h = np.radians(d["heading_deg"])
        bins = np.linspace(-np.pi, np.pi, 37)
        cnt, edges = np.histogram(h, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, cnt, width=(edges[1] - edges[0]) * 0.92,
               color=c, alpha=0.8, edgecolor="white", lw=0.3)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["−180°", "−90°", "0°", "+90°", "180°"], fontsize=7.5)
        if j == 0:
            ax.set_ylabel("方向计数", fontsize=10)
        ax.set_xlabel("运动方向 (0°=+x 轴)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.text(0.5, 1.04, "方向分布", transform=ax.transAxes,
                    ha="center", fontsize=10, fontweight="bold")

        # 第 3 行：加速度
        ax = axes[2, j]
        ax.hist(d["acc"].clip(0, 1.5), bins=np.arange(0, 1.52, 0.03),
                color=c, alpha=0.8, edgecolor="white", lw=0.4)
        ax.axvline(d["acc"].mean(), color="k", ls="--", lw=1.5)
        ax.text(0.97, 0.95, f"μ={d['acc'].mean():.3f}\n"
                            f"P95={d['acc'].quantile(0.95):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        if j == 0:
            ax.set_ylabel("加速度计数", fontsize=10)
        ax.set_xlabel("加速度 |a| (m/s²)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.text(0.5, 1.04, "加速度分布", transform=ax.transAxes,
                    ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("A2 个体运动学量分布 —— 速度 / 方向 / 加速度（五场景 × 三视图）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    T.savefig(fig, os.path.join(FIG, "a2_kinematics_all.png"))
    plt.close(fig)

    # ---------------- 图 3：单场景速度直方图（含分位数） ----------------
    for s in T.SCENE_ORDER:
        d = data[s].dropna(subset=["speed"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(d["speed"], bins=np.arange(0, 3.05, 0.05),
                color=COLORS[s], alpha=0.8, edgecolor="white")

        q = d["speed"].quantile([0.05, 0.5, 0.95])
        for val, lab, cl in [(q[0.05], f"P5={q[0.05]:.2f}", "gray"),
                             (q[0.5], f"中位数={q[0.5]:.2f}", "black"),
                             (q[0.95], f"P95={q[0.95]:.2f}", "gray")]:
            ax.axvline(val, color=cl, ls="--", lw=1.4)

        # 静止比例：速度 < 0.1 m/s 视为基本静止
        frac_still = float((d["speed"] < 0.1).mean())
        ax.axvspan(0, 0.1, color="red", alpha=0.10)
        ax.text(0.11, ax.get_ylim()[1] * 0.80,
                f"速度<0.1 m/s 占比 {frac_still*100:.0f}%\n(近似静止)",
                fontsize=9, color="#a00")

        ax.set_xlabel("速度 (m/s)")
        ax.set_ylabel("计数")
        ax.set_title(f"A2 速度分布 —— {data[s].attrs['scene_cn']} ({s})\n"
                     f"N={data[s]['ped_id'].nunique()} 个体，"
                     f"μ={d['speed'].mean():.2f} m/s",
                     fontsize=12, fontweight="bold")
        # 图内标注分位数
        txt = (f"P5   = {q[0.05]:.2f} m/s\n"
               f"中位数 = {q[0.5]:.2f} m/s\n"
               f"P95  = {q[0.95]:.2f} m/s")
        ax.text(0.975, 0.62, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(fc="white", ec="gray", alpha=0.9))
        T.savefig(fig, os.path.join(FIG, f"a2_speed_hist_{s}.png"))
        plt.close(fig)

    print("\nA2 完成")


if __name__ == "__main__":
    print("A2 个体运动量分布：开始")
    main()
