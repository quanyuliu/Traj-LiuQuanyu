# Traj-LiuQuanyu —— 行人轨迹数据分析

用公开的 **ETH / UCY** 行人轨迹数据集，完成的数据分析。

> 课程作业仓库。**当前进度：Week 01（已完成 A1–A3）**

---

## 仓库导航

| 位置 | 内容 |
|------|------|
| **[`week01/`](week01/README.md)** | **本周全部作业内容（A1–A5 + 跨场景对比 + 总结）** |
| `common/trajlib.py` | 五个场景共用的分析工具库 |
| `data/raw/` | 原始轨迹数据（5 个场景的 txt） |


---

## 五个动作

| 编号 | 分析内容 | 对应目录 | 产出 |
|------|----------|---------|------|
| **A1** | 轨迹可视化 | `week01/a1/` | 每个个体一条线的轨迹图 |
| **A2** | 个体运动学 | `week01/a2/` | 速度 / 方向 / 加速度分布图 |
| **A3** | 对子交互 | `week01/a3/` | 最近邻间距、TTC 碰撞时间分布 |
| **A4** | 基本图 | `week01/a4/` | 密度 – 流率 – 速度（q-k-v 图） |
| **A5** | 行为分析 | `week01/a5/` | 成行、避让、震荡、瓶颈 |

---

## 数据说明

### 五个场景

| 代号 | 场景 | 来源 | 个体数 | 时长 | 特点 |
|------|------|------|--------|------|------|
| `seq_eth` | ETH 校园路口 | ETH | 330 | 348.8 s | 露天路口，行人过马路，速度最快 |
| `seq_hotel` | Hotel 酒店大堂 | ETH | 318 | 458.8 s | 室内大堂，人**静止站立**为主 |
| `zara01` | Zara 商业街 01 | UCY | 148 | 348.8 s | 商业街人行道，双向流 |
| `zara02` | Zara 商业街 02 | UCY | 203 | 420.8 s | 商业街，含**出入口瓶颈** |
| `students03` | UCY 校园学生 | UCY | 427 | 216.4 s | 校园广场，人群最密 |

### 数据格式

每行 4 列，空白分隔：

```
frame_id    ped_id    x(m)    y(m)
```



## 目录结构

```
.
├── README.md              本文件
├── common/
│   └── trajlib.py         五场景共用分析库
├── data/raw/              原始轨迹数据（*.txt）
├── week01/                ★ 本周全部内容
│   ├── README.md          本周总说明与结论
│   ├── a1/  a2/  a3/ 
│   ├── compare/           跨场景对比
│   └── summary/           总结总览图
└── week02/ … week08/      待完成
```

每个动作目录下统一放：

```
aN/
├── README.md      该动作的结论
├── aN_xxx.py      生成脚本
├── figures/       图（每组挑 1 张代表图）
└── data/          该动作产出的中间数据（csv）
```

---

## 复现

```bash
pip install numpy pandas matplotlib scipy opencv-python

cd week01
python a1/a1_trajectories.py           # A1 轨迹图
python a2/a2_kinematics.py             # A2 运动学分布
python a3/a3_pairwise.py               # A3 最近邻 + TTC
```

数据在 `data/raw/`，工具库在 `common/trajlib.py`。
**全部脚本已端到端验证可重跑**（`opencv-python` 仅在需要看原始视频时用到，
纯轨迹分析可以不装）。

---

## 数据来源

- ETH：Pellegrini et al., *"You'll Never Walk Alone: Modeling Social Behavior for
  Multi-Target Tracking"*, ICCV 2009（seq_eth、seq_hotel）
- UCY：Lerner et al., *"Crowds by Example"*, Eurographics 2007
  （zara01、zara02、students03）

轨迹文件从 `InhwanBae/ETH-UCY-Trajectory-Visualizer` 仓库取得
（通过 GitHub Contents API 下载，`raw.githubusercontent.com` 在部分网络下不可达）。
