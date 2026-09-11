"""
week01 / A1 —— 轨迹可视化：每个个体一条线

任务：把五个场景里每个人的运动轨迹画出来（一人一条线）。
产出：
    figures/a1_trajectories_all.png     五场景总览（2x3 布局）
    figures/a1_<scene>.png              每个场景单独一张
    data/a1_scene_stats.csv             各场景轨迹统计
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
WK = HERE   # 脚本自己所在目录（week01/a1），图与数据同级存放
FIG = os.path.join(WK, "figures")
DAT = os.path.join(WK, "data")

# 每个场景一个主色调，方便和后面几周的图对应
COLORS = {
    "seq_eth":    "#d62728",
    "seq_hotel":  "#1f77b4",
    "zara01":     "#2ca02c",
    "zara02":     "#9467bd",
    "students03": "#ff7f0e",
}


def draw_one(ax, scene, df, show_ped_count=True):
    """在指定 ax 上画一个场景的全部轨迹。"""
    # 按行人 ID 循环，一人一条线
    # 若同一人有帧间隔 > 1 步，说明轨迹断开，需要拆成多段分别画
    step = T.ANNOT_STRIDE          # 正常帧步长 = 10
    for pid, g in df.groupby("ped_id", sort=True):
        g = g.sort_values("frame")
        fx, x, y = g["frame"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy()
        # 找出断点位置
        brk = np.where(np.diff(fx) > step)[0] + 1
        for seg in np.split(np.arange(len(fx)), brk):
            if len(seg) < 2:
                continue
            ax.plot(x[seg], y[seg], "-", lw=0.7, alpha=0.75,
                    color=COLORS[scene], solid_capstyle="round")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = f"{df.attrs['scene_cn']}\n({scene})"
    if show_ped_count:
        title += f"   N={df['ped_id'].nunique()} 个体"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)

    stats = []
    frames = {}

    for scene in T.SCENE_ORDER:
        df = T.load_scene(scene, min_len=4)   # A1 画图可放宽到 4 帧
        frames[scene] = df
        s = T.scene_summary(df)
        s["scene"] = scene
        stats.append(s)
        print(f"  {scene}: {s['个体数']} 个体, {s['轨迹总点数']} 点")

    # ---------- 单场景大图 ----------
    for scene in T.SCENE_ORDER:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        draw_one(ax, scene, frames[scene])
        T.savefig(fig, os.path.join(FIG, f"a1_{scene}.png"))
        plt.close(fig)

    # ---------- 五场景总览 ----------
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    axes = axes.ravel()
    for i, scene in enumerate(T.SCENE_ORDER):
        draw_one(axes[i], scene, frames[scene])
    axes[5].axis("off")
    # 在空白格写一段图注，方便直接看懂
    axes[5].text(0.02, 0.95,
                 "A1 轨迹图 读图说明\n\n"
                 "· 一条线 = 一个行人，线长 = 该人被追踪的时间\n"
                 "· 颜色按场景区分，不是按个体\n"
                 "· 线密集区 = 人流走廊；线的走向 = 主要流向\n"
                 "· 点状散乱的小线段 = 被遮挡后重新识别的短轨迹",
                 va="top", ha="left", fontsize=11, linespacing=1.9,
                 transform=axes[5].transAxes)
    fig.suptitle("A1 行人轨迹图（五场景）—— 每个个体一条线",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    T.savefig(fig, os.path.join(FIG, "a1_trajectories_all.png"))
    plt.close(fig)

    # ---------- 统计表 ----------
    out = pd.DataFrame(stats)[
        ["scene", "场景", "个体数", "时长_s", "平均轨迹长_帧",
         "轨迹总点数", "平均速度_m/s", "最大速度_m/s"]
    ]
    out.to_csv(os.path.join(DAT, "a1_scene_stats.csv"),
               index=False, encoding="utf-8-sig")
    print("\n=== A1 各场景统计 ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    print("A1 轨迹可视化：开始")
    main()
    print("A1 完成")
