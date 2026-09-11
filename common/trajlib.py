"""
trajlib.py -- ETH/UCY 行人轨迹数据集通用分析工具库

数据格式说明（ETH/UCY 标准格式，每行 4 列，制表符或空格分隔）：
    col0 = frame_id   帧号（整数，ETH 用 1/10 秒为步长，UCY 用 1 帧）
    col1 = ped_id     行人 ID（浮点数，同一个人全程唯一）
    col2 = x          世界坐标 x（单位：米）
    col3 = y          世界坐标 y（单位：米）

参考：
    - Pellegrini et al., "You'll Never Walk Alone: Modeling Social Behavior for
      Multi-Target Tracking", ICCV 2009  (ETH / Hotel)
    - Lerner et al., "Crowds by Example", Eurographics 2007  (UCY: Zara / Students)

本文件被 week01-week08 的所有脚本复用，避免重复代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径与常量

# 相对本文件的项目根目录
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo/
DATA_RAW = os.path.join(ROOT, "data", "raw")

# 五个场景：内部代号 -> (中文场景名, 数据文件名)
#
# !! 关键：采样率说明 !!
# ETH/UCY 原始视频为 25 fps，但标注文件只每 10 帧存一个点（frame 字段步长 = 10）。
# 因此两个相邻标注点之间的真实时间间隔 = 10 / 25 = 0.4 秒。
# 数据文件里的 frame 值必须除以 25 才是秒数，不能直接乘 0.4。
SCENES = {
    "seq_eth":    ("ETH 校园路口",   "seq_eth.txt"),
    "seq_hotel":  ("Hotel 酒店大堂",  "seq_hotel.txt"),
    "zara01":     ("Zara 商业街 01",  "zara01.txt"),
    "zara02":     ("Zara 商业街 02",  "zara02.txt"),
    "students03": ("UCY 校园学生",    "students03.txt"),
}

SCENE_ORDER = ["seq_eth", "seq_hotel", "zara01", "zara02", "students03"]

VIDEO_FPS = 25.0          # 原始视频帧率
ANNOT_STRIDE = 10         # 标注抽帧步长（每 10 帧一个标注点）
DT = ANNOT_STRIDE / VIDEO_FPS   # = 0.4 s，相邻标注点时间间隔

# 行人典型人体尺度：用于估算"个人空间"半径，以及把密度换算成 人/m^2
PED_RADIUS_M = 0.25          # 行人近似半径（米），参考 Helbing 社会力模型常用值


# ---------------------------------------------------------------- 数据载入

def load_scene(scene: str, min_len: int = 8, max_speed: float = 4.0) -> pd.DataFrame:
    """读入一个场景，返回透视后的长表 DataFrame。

    参数
    ----
    scene      : 场景代号，见 SCENES
    min_len    : 丢弃轨迹长度（帧数）小于该值的行人，避免噪声
    max_speed  : 速度上限（m/s），超过该值的点视为标注错误并剔除。
                正常步行 1.0-1.8 m/s，跑动约 3-5 m/s，取 4.0 作为宽松阈值。

    返回列
    ------
    ped_id, frame, x, y, t, vx, vy, speed, heading_deg, acc, acc_along, acc_lat
    """
    if scene not in SCENES:
        raise KeyError(f"未知场景 {scene}，可选：{list(SCENES)}")
    name_cn, fname = SCENES[scene]
    path = os.path.join(DATA_RAW, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件 {path}")

    # 自动识别分隔符（ETH 用 tab，UCY 用 tab，个别镜像用空格）
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :4]
    df.columns = ["frame", "ped_id", "x", "y"]
    df["frame"] = df["frame"].round().astype(int)
    df["ped_id"] = df["ped_id"].round().astype(int)

    # frame 起始值各场景不同（ETH 从 780 开始、zara02 从 10 开始），归零到 0
    df["frame"] = df["frame"] - df["frame"].min()
    # frame 步长为 10（抽帧），除以 fps 得到真实秒数
    df["t"] = df["frame"] / VIDEO_FPS

    df = df.sort_values(["ped_id", "frame"]).reset_index(drop=True)

    # ---- 过滤过短轨迹
    counts = df.groupby("ped_id")["frame"].transform("size")
    df = df[counts >= min_len].reset_index(drop=True)

    # ---- 逐个体计算运动学量
    out = []
    for pid, g in df.groupby("ped_id", sort=True):
        g = g.sort_values("frame").copy()
        t = g["t"].to_numpy()
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()

        # 用时间作权重的数值微分（gradient 处理不等间隔更稳）
        if len(g) >= 3 and np.ptp(t) > 0:
            vx = np.gradient(x, t)
            vy = np.gradient(y, t)
        else:  # 点太少时退化为差分
            vx = np.gradient(x)
            vy = np.gradient(y)

        speed = np.hypot(vx, vy)

        # 方向角：x 轴正向为 0，逆时针为正，范围 [-180, 180)
        heading = np.degrees(np.arctan2(vy, vx))

        # 加速度：速度对时间求导，分解为切向/法向
        if len(g) >= 3 and np.ptp(t) > 0:
            ax = np.gradient(vx, t)
            ay = np.gradient(vy, t)
        else:
            ax = np.gradient(vx)
            ay = np.gradient(vy)

        # 切向单位向量（速度方向），法向为其逆时针 90 度
        sp = np.where(speed > 1e-6, speed, 1e-6)
        tx, ty = vx / sp, vy / sp
        acc_along = ax * tx + ay * ty          # 加速 / 减速
        acc_lat = -ax * ty + ay * tx           # 转向（横向加速度）

        g["vx"], g["vy"] = vx, vy
        g["speed"] = speed
        g["heading_deg"] = heading
        g["acc"] = np.hypot(ax, ay)
        g["acc_along"] = acc_along
        g["acc_lat"] = acc_lat
        out.append(g)

    res = pd.concat(out, ignore_index=True)

    # ---- 剔除速度异常点（标注抖动）
    bad = res["speed"] > max_speed
    n_bad = int(bad.sum())
    if n_bad:
        res.loc[bad, ["speed", "vx", "vy", "acc", "acc_along", "acc_lat"]] = np.nan
    res = res.sort_values(["ped_id", "frame"]).reset_index(drop=True)

    res.attrs["scene_cn"] = name_cn
    res.attrs["dt"] = DT
    res.attrs["n_dropped"] = n_bad
    return res


def load_all(min_len: int = 8) -> dict[str, pd.DataFrame]:
    """一次载入全部五个场景。"""
    return {s: load_scene(s, min_len=min_len) for s in SCENE_ORDER}


# ---------------------------------------------------------------- 统计工具

def scene_summary(df: pd.DataFrame) -> dict:
    """汇总一个场景的基础统计量，用于各周的说明表格。"""
    dt = df.attrs.get("dt", 0.4)
    n_ped = df["ped_id"].nunique()
    n_frames = df["frame"].nunique()
    dur = n_frames * dt
    lengths = df.groupby("ped_id")["frame"].size()

    return {
        "场景": df.attrs.get("scene_cn", ""),
        "个体数": int(n_ped),
        "总帧数": int(n_frames),
        "时长_s": round(dur, 1),
        "平均轨迹长_帧": round(float(lengths.mean()), 1),
        "轨迹总点数": int(len(df)),
        "平均速度_m/s": round(float(df["speed"].mean()), 3),
        "速度标准差": round(float(df["speed"].std()), 3),
        "最大速度_m/s": round(float(df["speed"].max()), 3),
        "平均加速度_m/s2": round(float(df["acc"].mean()), 3),
    }


def smooth_series(v: np.ndarray, win: int = 5) -> np.ndarray:
    """移动平均平滑，win 为奇数窗口大小。用于速度曲线去抖。"""
    if win <= 1 or len(v) < win:
        return v
    win = win if win % 2 == 1 else win + 1
    kernel = np.ones(win) / win
    pad = win // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(vp, kernel, mode="valid")


def pair_distances(df: pd.DataFrame) -> pd.DataFrame:
    """计算每一帧内所有行人两两之间的欧氏距离。

    返回列：frame, t, a, b, dist
    复杂度 O(F * N^2)，五个场景帧数不多（<500 帧，单帧最多约 400 人），可接受。
    """
    rows = []
    for frame, g in df.groupby("frame", sort=True):
        ids = g["ped_id"].to_numpy()
        xy = g[["x", "y"]].to_numpy()
        n = len(ids)
        if n < 2:
            continue
        # 向量化计算所有两两距离
        diff = xy[:, None, :] - xy[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        iu, ju = np.triu_indices(n, k=1)
        for i, j in zip(iu, ju):
            rows.append((frame, g["t"].iloc[0], int(ids[i]), int(ids[j]), float(d[i, j])))
    return pd.DataFrame(rows, columns=["frame", "t", "a", "b", "dist"])


def nearest_neighbor(df: pd.DataFrame) -> pd.DataFrame:
    """每个人的最近邻距离（同一帧内）。

    返回列：frame, t, ped_id, nn_dist, nn_id
    """
    rows = []
    for frame, g in df.groupby("frame", sort=True):
        ids = g["ped_id"].to_numpy()
        xy = g[["x", "y"]].to_numpy()
        n = len(ids)
        if n < 2:
            continue
        diff = xy[:, None, :] - xy[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        j = d.argmin(axis=1)
        for k in range(n):
            rows.append((frame, g["t"].iloc[0], int(ids[k]),
                         float(d[k, j[k]]), int(ids[j[k]])))
    return pd.DataFrame(rows, columns=["frame", "t", "ped_id", "nn_dist", "nn_id"])


def compute_ttc(df: pd.DataFrame, horizon: float = 8.0) -> pd.DataFrame:
    """计算碰撞时间 TTC（Time To Collision）。

    方法：对每一帧、每一对相距 < horizon 米的行人，用二人当前位置与速度
    做相对运动求解最近接近时刻（CPA, Closest Point of Approach）：

        p_rel   = p_b - p_a          相对位置
        v_rel   = v_b - v_a          相对速度
        t_cpa   = -(p_rel · v_rel) / |v_rel|^2
        d_min   = |p_rel + v_rel * t_cpa|

    若 t_cpa > 0 且 d_min < 2*PED_RADIUS_M（两人身体会重叠），
    则该 t_cpa 即 TTC；否则 TTC = inf（不会碰撞）。

    返回列：frame, t, a, b, dist, ttc, d_min
    """
    rows = []
    for frame, g in df.groupby("frame", sort=True):
        g = g.dropna(subset=["vx", "vy", "speed"])
        if len(g) < 2:
            continue
        ids = g["ped_id"].to_numpy()
        xy = g[["x", "y"]].to_numpy()
        vel = g[["vx", "vy"]].to_numpy()
        t_now = float(g["t"].iloc[0])
        n = len(ids)
        thr = 2 * PED_RADIUS_M

        for i in range(n):
            for j in range(i + 1, n):
                p_rel = xy[j] - xy[i]
                dist = float(np.hypot(*p_rel))
                if dist > horizon:          # 距离太远，不可能即将碰撞
                    continue
                v_rel = vel[j] - vel[i]
                vv = float(v_rel @ v_rel)
                if vv < 1e-6:               # 相对静止
                    continue
                t_cpa = float(-(p_rel @ v_rel) / vv)
                if t_cpa <= 0:              # 正在远离
                    continue
                d_min = float(np.hypot(*(p_rel + v_rel * t_cpa)))
                if d_min < thr:             # 会撞上，记录 TTC
                    rows.append((frame, t_now, int(ids[i]), int(ids[j]),
                                 dist, t_cpa, d_min))
    return pd.DataFrame(rows, columns=["frame", "t", "a", "b", "dist", "ttc", "d_min"])


# ---------------------------------------------------------------- 宏观量
# 基本图（q-k-v）用的局部密度/速度/流率测量。
#
# 方法选择说明（重要）：
#   常见做法是在场景上铺 1 m × 1 m 固定网格，数每格人数。但这对 ETH/UCY
#   这类"稀疏"数据集是错的：这些场景每帧只有 5-33 人，铺在 100-360 m² 上，
#   绝大多数被占用的格子里只有 1 个人，密度就被量化成 1 或 2 人/m²，
#   密度范围被强行压窄，无法看出 q-k 的曲率。
#
#   本仓库改用 Voronoi 自由面积法（参数自由，不需要人为设平滑半径）：
#   对每一帧的行人位置做 Voronoi 剖分，每个人的"专属面积" A_i 就是他到
#   所有邻居的中垂线围成的多边形面积。则
#        局部密度  k_i = 1 / A_i
#        局部速度  v_i = |velocity_i|
#        局部流率  q_i = k_i * v_i
#   这是行人流文献里公认的做法（Voronoi 密度）。A_i 小说明邻居近 → 密度高。
#
#   同一帧的 Voronoi 元胞面积用 scipy 计算；边界元胞会外扩到无穷，
#   用包围盒裁剪（clip）处理。

def local_density_voronoi(df: pd.DataFrame, clip_to_bbox: bool = True,
                          max_area: float | None = None) -> pd.DataFrame:
    """用 Voronoi 自由面积计算逐人逐帧的局部密度。

    返回列：frame, t, ped_id, x, y, speed, vor_area, k, q

    参数
    ----
    clip_to_bbox : 把外扩的元胞裁剪到场景包围盒内，避免边界处面积虚大
    max_area     : 面积上限（人/m² 下限），用于抑制孤立个体的极端值
    """
    from scipy.spatial import Voronoi, HalfspaceIntersection
    from scipy.spatial import ConvexHull

    rows = []
    for frame, g in df.groupby("frame", sort=True):
        g = g.dropna(subset=["speed"])
        n = len(g)
        if n == 0:
            continue
        ids = g["ped_id"].to_numpy()
        xy = g[["x", "y"]].to_numpy()
        sp = g["speed"].to_numpy()
        t_now = float(g["t"].iloc[0])

        if n == 1:
            # 只有一个人，没有邻居，用数据集的典型尺度给一个保守面积
            rows.append((frame, t_now, int(ids[0]), xy[0, 0], xy[0, 1],
                         sp[0], np.nan, np.nan, np.nan))
            continue

        # ---- 包围盒：用于裁剪无界的边界元胞
        lo = xy.min(axis=0) - 1.0
        hi = xy.max(axis=0) + 1.0
        bbox = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                         [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]])

        try:
            vor = Voronoi(xy)
        except Exception:
            continue

        # 对每个点，用"半空间求交"精确算它到所有邻居的中垂线围成的多边形
        for i in range(n):
            others = [j for j in range(n) if j != i]
            # 半空间约束：对每个 j，点必须满足 |p - p_i| <= |p - p_j|
            # 即 (p_j - p_i)·p <= (|p_j|^2 - |p_i|^2)/2
            A, b = [], []
            for j in others:
                d = xy[j] - xy[i]
                A.append(d)
                b.append((xy[j] @ xy[j] - xy[i] @ xy[i]) / 2.0)
            # 加包围盒的 4 条边界（裁剪）
            A += [[1, 0], [-1, 0], [0, 1], [0, -1]]
            b += [hi[0], -lo[0], hi[1], -lo[1]]
            A = np.asarray(A, float)
            b = np.asarray(b, float)

            # 找一个内点（严格满足所有约束）
            # 用线性规划找 Chebyshev 中心，简单起见先用质心试探
            interior = xy[i].copy()
            try:
                hs = HalfspaceIntersection(
                    np.hstack([A, -b[:, None]]), interior)
                pts = hs.intersections
            except Exception:
                # 退化时退回到用 scipy 的 vor 顶点近似
                pts = None

            if pts is None or len(pts) < 3:
                area = np.nan
            else:
                try:
                    area = ConvexHull(pts).volume     # 2D 的 volume 即面积
                except Exception:
                    area = np.nan

            k = 1.0 / area if (area and area > 1e-9) else np.nan
            rows.append((frame, t_now, int(ids[i]), xy[i, 0], xy[i, 1],
                         sp[i], area, k, k * sp[i] if k == k else np.nan))

    out = pd.DataFrame(rows, columns=["frame", "t", "ped_id", "x", "y",
                                      "speed", "vor_area", "k", "q"])
    if max_area:
        bad = out["vor_area"] > max_area
        out.loc[bad, ["vor_area", "k", "q"]] = np.nan
    return out


def local_density_knn(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """用第 k 近邻距离估算局部密度（Voronoi 的稳健替代，抗边界和退化）。

    密度 ~ k / (π r_k²)，其中 r_k 是到第 k 近邻的距离。
    这个方法没有 Voronoi 的数值退化问题，且对孤立个体给出很低的密度，
    更适合稀疏场景。本仓库主要用它来画基本图。

    返回列：frame, t, ped_id, x, y, speed, r_k, k_density, q
    """
    rows = []
    for frame, g in df.groupby("frame", sort=True):
        g = g.dropna(subset=["speed"])
        n = len(g)
        if n < 2:
            continue
        ids = g["ped_id"].to_numpy()
        xy = g[["x", "y"]].to_numpy()
        sp = g["speed"].to_numpy()
        t_now = float(g["t"].iloc[0])

        diff = xy[:, None, :] - xy[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)

        kk = min(k, n - 1)
        r_k = np.sort(d, axis=1)[:, kk - 1]        # 到第 kk 近邻的距离
        area = np.pi * r_k ** 2
        dens = kk / np.maximum(area, 1e-9)
        for i in range(n):
            rows.append((frame, t_now, int(ids[i]), xy[i, 0], xy[i, 1],
                         sp[i], r_k[i], dens[i], dens[i] * sp[i]))
    return pd.DataFrame(rows, columns=["frame", "t", "ped_id", "x", "y",
                                       "speed", "r_k", "k_density", "q"])


# ---------------------------------------------------------------- 绘图辅助

def setup_matplotlib():
    """统一中文与矢量输出设置。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.axisbelow"] = True
    return plt


def savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"  [saved] {os.path.relpath(path, ROOT)}")
