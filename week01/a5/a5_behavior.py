"""
week05 / A5(上) —— 行为分析：成行（跟随/排队） 与 避让

任务：检测并可视化两类典型行为现象。
产出：
    figures/a5_following_all.png     成行行为：五场景示意 + 特征分布
    figures/a5_avoidance_all.png     避让行为：五场景示意 + 侧向偏移分布
    data/a5_following_events.csv     成行事件清单
    data/a5_avoidance_events.csv     避让事件清单
    data/a5_behavior_summary.csv     行为统计汇总

======================================================================
行为判据（本文件核心，全部是可量化的定义，不是主观描述）
======================================================================

【成行 / 跟随 (following)】
判据：A 在 B 正后方形成"队列"。
    1. 空间上 A 在 B 后方：A 的速度向量与 (B − A) 的夹角 < 30°
       （即 A 正朝着 B 走，不是并排走）
    2. 纵向间距 d 在 0.5 – 3.0 m（太近是重叠，太远不算跟随）
    3. 横向偏移 lateral 很小（< 0.6 m，几乎在同一"车道"上）
    4. 速度接近：|v_A − v_B| < 0.4 m/s（跟着走，不会追尾也不会被甩开）
    5. 持续 >= 3 帧（约 1.2 s），排除瞬时巧合

【避让 / 绕行 (avoidance)】
判据：A 在接近 B 时发生侧向偏移并改变航向。
    1. 存在接近阶段：距离从 d0 减小到最小值 d_min，且 d_min < 1.2 m
    2. A 的航向角在此期间改变量 > 8°（真的转了向）
    3. 转的方向是"远离 B"的一侧（侧向偏移与相对位置同号）
       —— 这一条是区分"避让"和"单纯拐弯"的关键
    4. 转向过程 <= 5 帧（约 2 s），是短促的反应，不是缓慢漂移

【为什么这样定义】
成行的本质是"速度对齐 + 纵向跟随"，所以用速度夹角和横向偏移来卡；
避让的本质是"为了不撞而侧向让开"，所以用"航向改变 + 方向正确"来卡。
两个判据都要求事件在时间上持续/短促，避免把噪声当行为。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
import trajlib as T                                    # noqa: E402

plt = T.setup_matplotlib()
from matplotlib.lines import Line2D                    # noqa: E402

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

# --- 成行判据参数
FOL_ANGLE = 30.0      # 后方夹角上限（度）
FOL_DMIN, FOL_DMAX = 0.5, 3.0
FOL_LAT = 0.6         # 横向偏移上限（m）
FOL_DV = 0.4          # 速度差上限（m/s）
FOL_MINFRAMES = 3

# --- 避让判据参数
AVO_APPROACH = 1.2    # 最近距离上限（m）
AVO_TURN = 8.0        # 航向改变下限（度）
AVO_MAXFRAMES = 5


def detect_following(df):
    """检测成行/跟随事件。返回事件表。"""
    events = []
    step = T.ANNOT_STRIDE
    # 按帧建索引，便于快速取"下一帧"状态
    by_ped = {pid: g.set_index("frame") for pid, g in df.groupby("ped_id")}

    for frame, g in df.groupby("frame", sort=True):
        g = g.dropna(subset=["vx", "vy", "speed"])
        n = len(g)
        if n < 2:
            continue
        ids = g["ped_id"].to_numpy()
        xy = g[["x", "y"]].to_numpy()
        vel = g[["vx", "vy"]].to_numpy()
        sp = g["speed"].to_numpy()

        for i in range(n):          # i = 跟随者 A
            vi = vel[i]
            si = sp[i]
            if si < 0.25:           # 静止的人不参与成行
                continue
            ux, uy = vi[0] / si, vi[1] / si

            for j in range(n):      # j = 被跟随者 B
                if i == j:
                    continue
                r = xy[j] - xy[i]
                d = float(np.hypot(*r))
                if not (FOL_DMIN <= d <= FOL_DMAX):
                    continue
                if sp[j] < 0.25:
                    continue
                # 条件 1：A 朝向 B
                cosang = float((r @ vi) / (d * si))
                ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                if ang > FOL_ANGLE:
                    continue
                # 条件 3：横向偏移小 —— 相对位移垂直于 A 速度方向的分量
                lat = abs(float(r[0] * (-uy) + r[1] * ux))
                if lat > FOL_LAT:
                    continue
                # 条件 4：速度接近
                if abs(si - sp[j]) > FOL_DV:
                    continue
                events.append((frame, g["t"].iloc[0], int(ids[i]), int(ids[j]),
                               d, lat, ang, si, sp[j]))

    ev = pd.DataFrame(events, columns=["frame", "t", "follower", "leader",
                                       "dist", "lateral", "angle",
                                       "v_follower", "v_leader"])
    if not len(ev):
        return ev

    # 条件 5：同一 (follower, leader) 对持续 >= FOL_MINFRAMES 帧才算
    ev = ev.sort_values(["follower", "leader", "frame"])
    key = ev["follower"].astype(str) + "_" + ev["leader"].astype(str)
    grp = key.ne(key.shift()).cumsum()
    # 只保留连续（帧差 == step）的段
    ev["_seg"] = grp
    keep = []
    for _, seg in ev.groupby("_seg"):
        if len(seg) >= FOL_MINFRAMES and np.all(np.diff(seg["frame"]) == step):
            keep.append(seg)
    if not keep:
        return ev.iloc[0:0]
    return pd.concat(keep, ignore_index=True)


def detect_avoidance(df):
    """检测避让/绕行事件。返回事件表。"""
    events = []
    step = T.ANNOT_STRIDE

    for pid, g in df.groupby("ped_id", sort=True):
        g = g.dropna(subset=["vx", "vy", "speed"]).sort_values("frame")
        if len(g) < 4:
            continue
        fr = g["frame"].to_numpy()
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()
        vx = g["vx"].to_numpy()
        vy = g["vy"].to_numpy()
        sp = g["speed"].to_numpy()
        hd = g["heading_deg"].to_numpy()

        for a in range(len(g) - 1):
            # 找同一时间窗内、其它人的位置：用后续帧的其他人判断
            f0 = fr[a]
            # 该个体接下来 AVO_MAXFRAMES 帧内航向的总变化
            b_end = min(a + AVO_MAXFRAMES, len(g) - 1)
            if b_end <= a:
                continue
            dhd = hd[b_end] - hd[a]
            # 归一到 [-180,180]
            dhd = (dhd + 180) % 360 - 180
            if abs(dhd) < AVO_TURN:
                continue
            if sp[a] < 0.3:
                continue

            # 找窗口内最近的"别人"
            t0, t1 = fr[a], fr[b_end]
            win = df[(df["frame"] >= t0) & (df["frame"] <= t1)
                     & (df["ped_id"] != pid)].dropna(subset=["speed"])
            if not len(win):
                continue
            # 对每帧算距离，取全局最近
            dmin, best = np.inf, None
            for f, wg in win.groupby("frame"):
                dd = np.hypot(wg["x"].to_numpy() - x[a], wg["y"].to_numpy() - y[a])
                k = int(dd.argmin())
                if dd[k] < dmin:
                    dmin = float(dd[k])
                    best = (int(wg["ped_id"].iloc[k]), f,
                            float(wg["x"].iloc[k]), float(wg["y"].iloc[k]))
            if best is None or dmin > AVO_APPROACH:
                continue

            # 条件 3：转向方向与相对位置同号（朝远离对方的一侧转）
            other_id, f_b, ox, oy = best
            rx, ry = ox - x[a], oy - y[a]        # 相对位置
            # 相对位置在 A 本体坐标系里的横向分量符号
            nrm = np.hypot(vx[a], vy[a])
            if nrm < 1e-6:
                continue
            ux_, uy_ = vx[a] / nrm, vy[a] / nrm
            lat_other = rx * (-uy_) + ry * ux_    # 对方在 A 的左侧(+)还是右侧(-)
            if abs(lat_other) < 0.15:
                continue                          # 正对着走，无法判断转向侧
            # A 的转向：dhd>0 为逆时针（向左），<0 为向右
            turned_left = dhd > 0
            other_left = lat_other > 0
            evasive = (turned_left != other_left)   # 转向背离对方
            if not evasive:
                continue

            events.append((int(pid), int(other_id), int(f0), float(g["t"].iloc[a]),
                           dmin, float(abs(dhd)), lat_other, float(sp[a]),
                           "左" if turned_left else "右"))

    ev = pd.DataFrame(events, columns=["ped", "other", "frame", "t", "d_min",
                                       "turn_deg", "lat_other", "speed",
                                       "turn_dir"])
    # 同一个体相邻帧的重复事件去重（保留间隔 > 2 帧的）
    if len(ev):
        ev = ev.sort_values(["ped", "frame"])
        ev = ev[(ev.groupby("ped")["frame"].diff() > step * 2)
                | (ev.groupby("ped")["frame"].diff().isna())]
    return ev


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)

    fol_all, avo_all, summ = {}, {}, []

    for s in T.SCENE_ORDER:
        df = T.load_scene(s)
        print(f"  检测行为 {s} ...")
        f = detect_following(df)
        a = detect_avoidance(df)
        fol_all[s], avo_all[s] = f, a

        n_ped = df["ped_id"].nunique()
        dur = (df["frame"].max() - df["frame"].min()) / T.VIDEO_FPS
        summ.append({
            "scene": s, "场景": df.attrs["scene_cn"],
            "个体数": int(n_ped),
            "成行事件数": int(len(f)),
            "成行_人·分钟": round(len(f) / max(n_ped, 1) / (dur / 60) * 1000, 2),
            "成行平均间距_m": round(float(f["dist"].mean()), 2) if len(f) else None,
            "避让事件数": int(len(a)),
            "避让_人·分钟": round(len(a) / max(n_ped, 1) / (dur / 60) * 1000, 2),
            "避让平均最近距离_m": round(float(a["d_min"].mean()), 2) if len(a) else None,
            "避让平均转角_deg": round(float(a["turn_deg"].mean()), 1) if len(a) else None,
            "左转占比_%": round(float((a["turn_dir"] == "左").mean() * 100), 1) if len(a) else None,
        })

        if len(f):
            f.assign(scene=s).to_csv(
                os.path.join(DAT, f"a5_follow_{s}.csv"), index=False)
        if len(a):
            a.assign(scene=s).to_csv(
                os.path.join(DAT, f"a5_avoid_{s}.csv"), index=False)

    tab = pd.DataFrame(summ)
    tab.to_csv(os.path.join(DAT, "a5_behavior_summary.csv"),
               index=False, encoding="utf-8-sig")
    print("\n=== A5 行为统计 ===")
    print(tab.to_string(index=False))

    # ---------- 导出事件总表 ----------
    if any(len(v) for v in fol_all.values()):
        pd.concat([v.assign(scene=k) for k, v in fol_all.items() if len(v)],
                  ignore_index=True).to_csv(
            os.path.join(DAT, "a5_following_events.csv"), index=False)
    if any(len(v) for v in avo_all.values()):
        pd.concat([v.assign(scene=k) for k, v in avo_all.items() if len(v)],
                  ignore_index=True).to_csv(
            os.path.join(DAT, "a5_avoidance_events.csv"), index=False)

    # =================== 图 1：成行行为 ===================
    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1], hspace=0.30, wspace=0.26)

    # 上行：五场景示意
    for j, s in enumerate(T.SCENE_ORDER):
        ax = fig.add_subplot(gs[0, j])
        df = T.load_scene(s, min_len=6)
        f = fol_all[s]
        # 淡化底图
        for pid, g in df.groupby("ped_id"):
            ax.plot(g["x"], g["y"], "-", color="0.82", lw=0.6, zorder=1)

        drawn = 0
        if len(f):
            # 取出现次数最多的若干成行事件画出来
            top = (f.groupby(["follower", "leader"]).size()
                   .sort_values(ascending=False).head(6))
            step = T.ANNOT_STRIDE
            for (fo, le) in top.index:
                gf = df[df["ped_id"] == fo].sort_values("frame")
                gl = df[df["ped_id"] == le].sort_values("frame")
                # 只画事件发生的那段帧
                evf = f[(f["follower"] == fo) & (f["leader"] == le)]["frame"]
                lo, hi = evf.min(), evf.max()
                gf = gf[(gf["frame"] >= lo - step) & (gf["frame"] <= hi + step)]
                gl = gl[(gl["frame"] >= lo - step) & (gl["frame"] <= hi + step)]
                if len(gf) < 2 or len(gl) < 2:
                    continue
                ax.plot(gf["x"], gf["y"], "-", color="#d62728", lw=2.4, zorder=4)
                ax.plot(gl["x"], gl["y"], "-", color="#1f77b4", lw=2.4, zorder=4)
                # 起点
                ax.plot(gf["x"].iloc[0], gf["y"].iloc[0], "o",
                        color="#d62728", ms=5, zorder=5)
                ax.plot(gl["x"].iloc[0], gl["y"].iloc[0], "o",
                        color="#1f77b4", ms=5, zorder=5)
                # 连一条线表示跟随关系
                ax.annotate("", xy=(gl["x"].iloc[len(gl)//2], gl["y"].iloc[len(gl)//2]),
                            xytext=(gf["x"].iloc[len(gf)//2], gf["y"].iloc[len(gf)//2]),
                            arrowprops=dict(arrowstyle="->", color="#2ca02c",
                                            lw=1.5, alpha=0.9), zorder=6)
                drawn += 1

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{T.SCENES[s][0]}\n({s})  成行 {len(f)} 例",
                     fontsize=9.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            lg = [Line2D([], [], color="#d62728", lw=2.4, label="跟随者 A"),
                  Line2D([], [], color="#1f77b4", lw=2.4, label="被跟随者 B"),
                  Line2D([], [], color="#2ca02c", lw=1.5, label="跟随关系")]
            ax.legend(handles=lg, fontsize=8, loc="best")

    # 下行：成行特征分布
    ax = fig.add_subplot(gs[1, 0])
    allf = pd.concat([v.assign(scene=k) for k, v in fol_all.items() if len(v)],
                     ignore_index=True) if any(len(v) for v in fol_all.values()) else None
    if allf is not None:
        data = [allf[allf["scene"] == s]["dist"].to_numpy() for s in T.SCENE_ORDER]
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data],
                        patch_artist=True, widths=0.6)
        for i, s in enumerate(T.SCENE_ORDER):
            bp["boxes"][i].set_facecolor(COLORS[s])
            bp["boxes"][i].set_alpha(0.6)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([s for s in T.SCENE_ORDER], fontsize=7.5, rotation=30)
        ax.set_ylabel("跟随纵向间距 (m)")
        ax.set_title("成行的间距分布", fontsize=10, fontweight="bold")
        ax.axhspan(FOL_DMIN, FOL_DMAX, color="gray", alpha=0.10)

    ax = fig.add_subplot(gs[1, 1])
    if allf is not None:
        for s in T.SCENE_ORDER:
            v = allf[allf["scene"] == s]["dist"]
            if len(v):
                ax.hist(v, bins=np.arange(0.4, 3.1, 0.12), histtype="step",
                        lw=2, color=COLORS[s], density=True, label=s)
        ax.set_xlabel("跟随间距 (m)")
        ax.set_ylabel("概率密度")
        ax.legend(fontsize=7.5)
    ax.set_title("间距的分布形状", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    if allf is not None:
        ax.scatter(allf["dist"], allf["lateral"], s=8,
                   c=[COLORS[s] for s in allf["scene"]], alpha=0.35)
        ax.axhline(FOL_LAT, color="k", ls="--", lw=1.2)
        ax.text(0.5, FOL_LAT * 1.03, "横向偏移上限 0.6 m", fontsize=8)
        ax.set_xlabel("纵向间距 (m)")
        ax.set_ylabel("横向偏移 (m)")
    ax.set_title("跟随的\"车道\"对齐程度", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 3])
    if allf is not None:
        ax.scatter(allf["v_follower"], allf["v_leader"], s=8,
                   c=[COLORS[s] for s in allf["scene"]], alpha=0.35)
        lim = [0, max(allf["v_follower"].max(), allf["v_leader"].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1.3, label="y=x（速度相同）")
        ax.set_xlabel("跟随者速度 (m/s)")
        ax.set_ylabel("被跟随者速度 (m/s)")
        ax.legend(fontsize=8)
    ax.set_title("速度对齐（靠对角线=成行）", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 4])
    ax.bar(range(len(T.SCENE_ORDER)),
           [len(fol_all[s]) for s in T.SCENE_ORDER],
           color=[COLORS[s] for s in T.SCENE_ORDER], alpha=0.85)
    for i, s in enumerate(T.SCENE_ORDER):
        ax.text(i, len(fol_all[s]) + max(1, max(len(fol_all[x]) for x in T.SCENE_ORDER) * 0.02),
                str(len(fol_all[s])), ha="center", fontsize=9)
    ax.set_xticks(range(len(T.SCENE_ORDER)))
    ax.set_xticklabels(T.SCENE_ORDER, fontsize=7.5, rotation=30)
    ax.set_ylabel("成行事件数")
    ax.set_title("各场景成行总量", fontsize=10, fontweight="bold")

    fig.suptitle("A5（上）成行 / 跟随行为 —— 检测结果与特征",
                 fontsize=15, fontweight="bold")
    T.savefig(fig, os.path.join(FIG, "a5_following_all.png"))
    plt.close(fig)

    # =================== 图 2：避让行为 ===================
    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1], hspace=0.30, wspace=0.26)

    for j, s in enumerate(T.SCENE_ORDER):
        ax = fig.add_subplot(gs[0, j])
        df = T.load_scene(s, min_len=6)
        a = avo_all[s]
        for pid, g in df.groupby("ped_id"):
            ax.plot(g["x"], g["y"], "-", color="0.85", lw=0.6, zorder=1)

        if len(a):
            # 挑转向角度最大的几个事件展示
            top = a.sort_values("turn_deg", ascending=False).head(7)
            for _, r in top.iterrows():
                gp = df[df["ped_id"] == r["ped"]].sort_values("frame")
                seg = gp[(gp["frame"] >= r["frame"] - T.ANNOT_STRIDE)
                         & (gp["frame"] <= r["frame"] + T.ANNOT_STRIDE * 4)]
                if len(seg) < 2:
                    continue
                ax.plot(seg["x"], seg["y"], "-", color="#d62728", lw=2.5, zorder=4)
                ax.plot(seg["x"].iloc[0], seg["y"].iloc[0], "o",
                        color="#d62728", ms=5.5, zorder=6)
                # 标出对方位置
                go = df[(df["ped_id"] == r["other"])
                        & (df["frame"] == r["frame"])]
                if len(go):
                    ax.plot(go["x"].iloc[0], go["y"].iloc[0], "X",
                            color="#1f77b4", ms=10, zorder=7)
                    ax.annotate("", xy=(go["x"].iloc[0], go["y"].iloc[0]),
                                xytext=(seg["x"].iloc[0], seg["y"].iloc[0]),
                                arrowprops=dict(arrowstyle="->", color="#2ca02c",
                                                lw=1.4, alpha=0.85,
                                                ls="dotted"), zorder=5)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{T.SCENES[s][0]}\n({s})  避让 {len(a)} 例",
                     fontsize=9.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            lg = [Line2D([], [], color="#d62728", lw=2.5, label="避让者 A 的路径"),
                  Line2D([], [], color="#1f77b4", marker="X", ls="", ms=9,
                         label="被避让者 B"),
                  Line2D([], [], color="#2ca02c", ls="dotted", lw=1.4,
                         label="接近方向")]
            ax.legend(handles=lg, fontsize=8, loc="best")

    ax = fig.add_subplot(gs[1, 0])
    alla = pd.concat([v.assign(scene=k) for k, v in avo_all.items() if len(v)],
                     ignore_index=True) if any(len(v) for v in avo_all.values()) else None
    if alla is not None:
        data = [alla[alla["scene"] == s]["d_min"].to_numpy() for s in T.SCENE_ORDER]
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data],
                        patch_artist=True, widths=0.6)
        for i, s in enumerate(T.SCENE_ORDER):
            bp["boxes"][i].set_facecolor(COLORS[s])
            bp["boxes"][i].set_alpha(0.6)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(T.SCENE_ORDER, fontsize=7.5, rotation=30)
        ax.set_ylabel("避让时最近距离 (m)")
        ax.axhline(0.5, color="r", ls="--", lw=1.2)
        ax.text(0.6, 0.52, "身体重叠阈值 0.5 m", fontsize=7.5, color="r")
    ax.set_title("避让发生时的最近距离", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    if alla is not None:
        for s in T.SCENE_ORDER:
            v = alla[alla["scene"] == s]["turn_deg"]
            if len(v):
                ax.hist(v, bins=np.arange(8, 90, 3), histtype="step", lw=2,
                        color=COLORS[s], density=True, label=s)
        ax.set_xlabel("避让转角 (度)")
        ax.set_ylabel("概率密度")
        ax.legend(fontsize=7.5)
    ax.set_title("避让需要转多少度？", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    if alla is not None:
        ax.scatter(alla["d_min"], alla["turn_deg"], s=9,
                   c=[COLORS[s] for s in alla["scene"]], alpha=0.4)
        if len(alla) > 5:
            z = np.polyfit(alla["d_min"], alla["turn_deg"], 1)
            xs = np.linspace(alla["d_min"].min(), alla["d_min"].max(), 30)
            ax.plot(xs, np.polyval(z, xs), "k--", lw=1.6,
                    label=f"斜率 {z[0]:+.1f} °/m")
            ax.legend(fontsize=8)
        ax.set_xlabel("最近距离 (m)")
        ax.set_ylabel("转角 (度)")
    ax.set_title("越靠近→转得越多？（反比关系）",
                 fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 3])
    if alla is not None:
        ld = [float((alla[alla["scene"] == s]["turn_dir"] == "左").mean() * 100)
              for s in T.SCENE_ORDER]
        rd = [100 - v for v in ld]
        x = np.arange(len(T.SCENE_ORDER))
        ax.bar(x, ld, 0.6, label="向左避让", color="#4a90d9")
        ax.bar(x, rd, 0.6, bottom=ld, label="向右避让", color="#e08a3c")
        ax.axhline(50, color="k", ls="--", lw=1.2)
        for i, v in enumerate(ld):
            ax.text(i, v / 2, f"{v:.0f}%", ha="center", fontsize=8, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(T.SCENE_ORDER, fontsize=7.5, rotation=30)
        ax.set_ylabel("占比 (%)")
        ax.legend(fontsize=8)
    ax.set_title("避让方向是否有左右偏好", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 4])
    ax.bar(range(len(T.SCENE_ORDER)),
           [len(avo_all[s]) for s in T.SCENE_ORDER],
           color=[COLORS[s] for s in T.SCENE_ORDER], alpha=0.85)
    for i, s in enumerate(T.SCENE_ORDER):
        ax.text(i, len(avo_all[s]) + max(1, max(len(avo_all[x]) for x in T.SCENE_ORDER) * 0.02),
                str(len(avo_all[s])), ha="center", fontsize=9)
    ax.set_xticks(range(len(T.SCENE_ORDER)))
    ax.set_xticklabels(T.SCENE_ORDER, fontsize=7.5, rotation=30)
    ax.set_ylabel("避让事件数")
    ax.set_title("各场景避让总量", fontsize=10, fontweight="bold")

    fig.suptitle("A5（上）避让 / 绕行行为 —— 检测结果与特征",
                 fontsize=15, fontweight="bold")
    T.savefig(fig, os.path.join(FIG, "a5_avoidance_all.png"))
    plt.close(fig)

    print("\nA5（上）完成")


if __name__ == "__main__":
    print("A5 行为分析（成行 / 避让）：开始")
    main()
