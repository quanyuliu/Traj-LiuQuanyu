"""
week03 / A3 —— 对子量：最近邻间距 + TTC（碰撞时间）

任务：计算每个个体的最近邻距离，以及行人两两之间的碰撞时间 TTC，画分布图。

产出：
    figures/a3_nn_ttc_all.png        五场景 NN + TTC 分布对比（代表图）
    figures/a3_nn_detail_<scene>.png 单场景最近邻详情（分布 + 随密度变化）
    data/a3_nn_stats.csv             最近邻统计
    data/a3_ttc_stats.csv            TTC 统计
    data/a3_nn_pooled.csv            最近邻明细

TTC 定义（见 trajlib.compute_ttc）：
    对每一对行人做相对运动外推，若最近接近距离 < 两倍人体半径(0.5 m)，
    则认为会碰撞，TTC = 距离最近点的时刻。-1 表示 TTC 很大（不会碰撞）。
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

# TTC 危险阈值（秒）：常用于行人与车辆；行人之间取 2s 作为"需要避让"的界
TTC_WARN = 2.0


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)

    nn_all, ttc_all, nn_stat, ttc_stat = {}, {}, [], []
    nn_pooled = []

    for s in T.SCENE_ORDER:
        df = T.load_scene(s)
        nm = df.attrs["scene_cn"]

        print(f"  处理 {s} ...")
        nn = T.nearest_neighbor(df)
        ttc = T.compute_ttc(df)
        nn_all[s], ttc_all[s] = nn, ttc

        # --- 最近邻统计
        nn_stat.append({
            "scene": s, "场景": nm,
            "个体数": int(df["ped_id"].nunique()),
            "NN均值_m": round(float(nn["nn_dist"].mean()), 3),
            "NN中位数_m": round(float(nn["nn_dist"].median()), 3),
            "NN_P5_m": round(float(nn["nn_dist"].quantile(0.05)), 3),
            "NN_P95_m": round(float(nn["nn_dist"].quantile(0.95)), 3),
            # 小于 0.5 m（两人身体尺度）的比例 = 拥挤接触率
            "NN<0.5m占比": round(float((nn["nn_dist"] < 0.5).mean()), 3),
            "NN<0.8m占比": round(float((nn["nn_dist"] < 0.8).mean()), 3),
        })

        # --- TTC 统计
        if len(ttc):
            ttc_stat.append({
                "scene": s, "场景": nm,
                "危险对_帧数": int(len(ttc)),
                "TTC中位数_s": round(float(ttc["ttc"].median()), 3),
                "TTC_P10_s": round(float(ttc["ttc"].quantile(0.10)), 3),
                "TTC<1s对数": int((ttc["ttc"] < 1.0).sum()),
                "TTC<2s对数": int((ttc["ttc"] < 2.0).sum()),
                "TTC<2s占比": round(float((ttc["ttc"] < 2.0).mean()), 4),
                "最小TTC_s": round(float(ttc["ttc"].min()), 3),
            })
        else:
            ttc_stat.append({"scene": s, "场景": nm, "危险对_帧数": 0})

        for _, r in nn.iterrows():
            nn_pooled.append((s, r["ped_id"], r["nn_dist"]))

    nn_tab = pd.DataFrame(nn_stat)
    ttc_tab = pd.DataFrame(ttc_stat)
    nn_tab.to_csv(os.path.join(DAT, "a3_nn_stats.csv"),
                  index=False, encoding="utf-8-sig")
    ttc_tab.to_csv(os.path.join(DAT, "a3_ttc_stats.csv"),
                   index=False, encoding="utf-8-sig")
    pd.DataFrame(nn_pooled, columns=["scene", "ped_id", "nn_dist"]).to_csv(
        os.path.join(DAT, "a3_nn_pooled.csv"), index=False)

    print("\n=== A3 最近邻间距 ===")
    print(nn_tab.to_string(index=False))
    print("\n=== A3 TTC ===")
    print(ttc_tab.to_string(index=False))

    # ================= 图 1：NN + TTC 总览 =================
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))

    # (0,0) 最近邻距离分布
    ax = axes[0, 0]
    bins = np.arange(0, 4.05, 0.05)
    for s in T.SCENE_ORDER:
        ax.hist(nn_all[s]["nn_dist"], bins=bins, histtype="step", lw=2,
                color=COLORS[s], density=True,
                label=f"{T.SCENES[s][0]}  μ={nn_all[s]['nn_dist'].mean():.2f}")
    ax.axvline(0.5, color="k", ls="--", lw=1.3, alpha=0.7)
    ax.text(0.52, ax.get_ylim()[1] * 0.93, "0.5 m\n(身体重叠)", fontsize=8.5, va="top")
    ax.axvline(1.2, color="gray", ls=":", lw=1.3)
    ax.text(1.22, ax.get_ylim()[1] * 0.93, "1.2 m\n(舒适间距)", fontsize=8.5, va="top")
    ax.set_xlabel("最近邻距离 (m)")
    ax.set_ylabel("概率密度")
    ax.set_title("最近邻间距分布（五场景）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5)

    # (0,1) 最近邻的小提琴
    ax = axes[0, 1]
    vp = ax.violinplot([nn_all[s]["nn_dist"].to_numpy() for s in T.SCENE_ORDER],
                       showmeans=True, showmedians=True, widths=0.85)
    for i, s in enumerate(T.SCENE_ORDER):
        vp["bodies"][i].set_facecolor(COLORS[s])
        vp["bodies"][i].set_alpha(0.55)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([T.SCENES[s][0] for s in T.SCENE_ORDER], fontsize=8.5)
    ax.set_ylabel("最近邻距离 (m)")
    ax.set_title("最近邻间距的小提琴图（离散度）", fontsize=12, fontweight="bold")

    # (1,0) TTC 分布（对数横轴，只看 <60s）
    ax = axes[1, 0]
    bins = np.logspace(np.log10(0.05), np.log10(60), 55)
    for s in T.SCENE_ORDER:
        t = ttc_all[s]
        if not len(t):
            continue
        ax.hist(t["ttc"].clip(0.05, 60), bins=bins, histtype="step", lw=2,
                color=COLORS[s], density=True,
                label=f"{T.SCENES[s][0]}  n={len(t)}")
    ax.axvline(TTC_WARN, color="k", ls="--", lw=1.3)
    ax.text(TTC_WARN * 1.1, ax.get_ylim()[1] * 0.85,
            f"TTC={TTC_WARN}s\n(需避让)", fontsize=8.5, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("TTC 碰撞时间 (s，对数轴)")
    ax.set_ylabel("概率密度")
    ax.set_title("TTC 分布（仅统计会接近到 0.5 m 以内的对子）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5)

    # (1,1) 危险对占比条形图
    ax = axes[1, 1]
    x = np.arange(len(T.SCENE_ORDER))
    w = 0.38
    v_lt1 = [float((ttc_all[s]["ttc"] < 1.0).mean()) * 100 if len(ttc_all[s]) else 0
             for s in T.SCENE_ORDER]
    v_lt2 = [float((ttc_all[s]["ttc"] < 2.0).mean()) * 100 if len(ttc_all[s]) else 0
             for s in T.SCENE_ORDER]
    ax.bar(x - w / 2, v_lt1, w, label="TTC < 1 s（紧急）",
           color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, v_lt2, w, label="TTC < 2 s（需关注）",
           color="#ff9f4a", alpha=0.85)
    for xi, (a, b) in enumerate(zip(v_lt1, v_lt2)):
        ax.text(xi - w / 2, a + 1, f"{a:.0f}%", ha="center", fontsize=8.5)
        ax.text(xi + w / 2, b + 1, f"{b:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{T.SCENES[s][0]}\n({s})" for s in T.SCENE_ORDER],
                       fontsize=8.5)
    ax.set_ylabel("占所有危险对的比例 (%)")
    ax.set_title("危险对中短 TTC 的比例", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(v_lt2 + [1]) * 1.25)

    fig.suptitle("A3 对子量分析 —— 最近邻间距 与 碰撞时间 TTC",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    T.savefig(fig, os.path.join(FIG, "a3_nn_ttc_all.png"))
    plt.close(fig)

    # ================= 图 2：单场景 NN 详情 =================
    for s in T.SCENE_ORDER:
        nn = nn_all[s]
        df = T.load_scene(s)
        c = COLORS[s]

        fig, axes2 = plt.subplots(1, 3, figsize=(17, 4.9))

        # 左：NN 直方图
        ax = axes2[0]
        ax.hist(nn["nn_dist"], bins=np.arange(0, 4.05, 0.06),
                color=c, alpha=0.82, edgecolor="white", lw=0.4)
        med = nn["nn_dist"].median()
        ax.axvline(med, color="k", ls="--", lw=1.6)
        ax.axvspan(0, 0.5, color="red", alpha=0.10)
        ax.text(med * 1.06, ax.get_ylim()[1] * 0.92,
                f"中位数 {med:.2f} m", fontsize=9.5)
        ax.text(0.05, ax.get_ylim()[1] * 0.66,
                f"<0.5 m 占比\n{(nn['nn_dist']<0.5).mean()*100:.0f}%",
                fontsize=9, color="#a00")
        ax.set_xlabel("最近邻距离 (m)")
        ax.set_ylabel("计数")
        ax.set_title("最近邻间距分布", fontsize=11.5, fontweight="bold")

        # 中：NN 随时间变化（看拥堵演化）
        ax = axes2[1]
        per_frame = nn.groupby("t")["nn_dist"].mean()
        # 同帧人数
        cnt = df.groupby("t")["ped_id"].count()
        ax.plot(per_frame.index, per_frame.values, "-", color=c, lw=1.5,
                label="平均最近邻距离")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("平均最近邻距离 (m)", color=c)
        ax.tick_params(axis="y", labelcolor=c)
        ax2b = ax.twinx()
        ax2b.plot(cnt.index, cnt.values, "-", color="gray", lw=1.1, alpha=0.75,
                  label="在场人数")
        ax2b.set_ylabel("在场人数", color="gray")
        ax2b.tick_params(axis="y", labelcolor="gray")
        ax2b.grid(False)
        ax.set_title("间距随时间的演化", fontsize=11.5, fontweight="bold")

        # 右：NN vs 同帧人数（拥挤度关系）
        ax = axes2[2]
        merged = pd.DataFrame({"t": per_frame.index,
                               "nn": per_frame.values}).merge(
            pd.DataFrame({"t": cnt.index, "n": cnt.values}), on="t")
        ax.scatter(merged["n"], merged["nn"], s=14, color=c, alpha=0.5)
        if len(merged) > 3:
            z = np.polyfit(merged["n"], merged["nn"], 1)
            xs = np.linspace(merged["n"].min(), merged["n"].max(), 50)
            ax.plot(xs, np.polyval(z, xs), "k--", lw=1.6,
                    label=f"拟合 斜率={z[0]:+.4f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("同帧在场人数")
        ax.set_ylabel("平均最近邻距离 (m)")
        ax.set_title("人越多 → 间距越近？", fontsize=11.5, fontweight="bold")

        fig.suptitle(f"A3 最近邻间距详情 —— {df.attrs['scene_cn']} ({s})",
                     fontsize=13.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        T.savefig(fig, os.path.join(FIG, f"a3_nn_detail_{s}.png"))
        plt.close(fig)

    print("\nA3 完成")


if __name__ == "__main__":
    print("A3 对子量分析：开始")
    main()
