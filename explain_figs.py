"""
動画の図解スライド用の図（runs/figs/ に保存）。

- planner_steps.png：経路計画の各段階（間取り → 格子地図 → ロボットの幅だけ太らせる → A* の経路 → 経由地の間引き）
- robot_drawing.png：大型版の犬型ロボットの図面（横から・上から、寸法つき）
- mjcf_code.png：ロボットを記述した MJCF（XML）の一部
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import home  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "figs")
INK, YELLOW, LIGHT, SLATE, RED, GRAY = "#141414", "#ffd966", "#fff2cc", "#44546a", "#c00000", "#a6a6a6"
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]


def planner_steps(start="dock", goal="kitchen"):
    _, lv, a = home.PLACES[start]
    _, _, b = home.PLACES[goal]
    a, b = np.array(a), np.array(b)
    ext = [home.X0 - 1, home.X1 + 1, home.Y0 - 1, home.Y1 + 1]
    raw = np.zeros_like(home._occupancy(1))
    # 太らせる前の地図（壁・家具そのもの）
    inflate = home.INFLATE
    home.INFLATE = 0.0
    raw = home._occupancy(1)
    home.INFLATE = inflate
    occ = home._occupancy(1)
    home._OCC.pop(1, None)
    # A* の経路（間引く前の格子の点列）
    import heapq

    s, g = home._cell(a), home._cell(b)
    o = occ.copy()
    o[s] = o[g] = False
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    openq, cost, prev = [(0.0, s)], {s: 0.0}, {}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == g:
            break
        for dr, dc, w in moves:
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < o.shape[0] and 0 <= nb[1] < o.shape[1] and not o[nb]:
                nc = cost[cur] + w
                if nc < cost.get(nb, 1e18):
                    cost[nb], prev[nb] = nc, cur
                    heapq.heappush(openq, (nc + np.hypot(nb[0] - g[0], nb[1] - g[1]), nb))
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    grid_path = np.array([home._point(c) for c in reversed(cells)])
    wps = np.array([a] + [w for w in home._astar(1, a, b)])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    titles = ["① 間取り（壁・家具）", "② 10 cm の格子地図", "③ ロボットの幅だけ太らせる", "④ A* で最短経路 → 経由地に間引く"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(home.X0 - 0.3, home.X1 + 0.3)
        ax.set_ylim(home.Y0 - 0.3, home.Y1 + 0.3)
        ax.set_aspect("equal")
    ax = axes[0]
    ax.add_patch(patches.Rectangle((home.X0, home.Y0), home.X1 - home.X0, home.Y1 - home.Y0, color=LIGHT))
    for _, box in home._furniture():
        if box is not None and box[4] < 1.5:
            ax.add_patch(patches.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], color="#bfae96"))
    for w in home._walls():
        if w[4] < 1.5:
            ax.add_patch(patches.Rectangle((w[0], w[2]), w[1] - w[0], w[3] - w[2], color=INK))
    ax.add_patch(patches.Rectangle((home.SX0, home.SY0), home.SX1 - home.SX0, home.SY1 - home.SY0, color=GRAY))
    ax.text((home.SX0 + home.SX1) / 2, (home.SY0 + home.SY1) / 2, "階段", ha="center", va="center", fontsize=10, rotation=90)
    cmap = matplotlib.colors.ListedColormap(["white", INK])
    axes[1].imshow(raw, origin="lower", extent=ext, cmap=cmap, interpolation="nearest")
    axes[2].imshow(raw.astype(int) + occ.astype(int), origin="lower", extent=ext,
                   cmap=matplotlib.colors.ListedColormap(["white", GRAY, INK]), interpolation="nearest")
    axes[3].imshow(occ, origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap(["white", GRAY]), interpolation="nearest")
    axes[3].plot(grid_path[:, 0], grid_path[:, 1], color=SLATE, lw=1.5, label="A* の経路（格子）")
    axes[3].plot(wps[:, 0], wps[:, 1], "o-", color=RED, lw=2.5, ms=7, label="間引いた経由地")
    axes[3].legend(loc="lower right", fontsize=9)
    for ax in axes:
        ax.plot(*a, "o", color=SLATE, ms=10)
        ax.plot(*b, "*", color=RED, ms=16)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "planner_steps.png"), dpi=130)


def robot_drawing():
    from gpu.make_assets import SCALE

    s = SCALE
    L, W, H = 0.26 * s, 0.085 * s, 0.06 * s  # 胴体の半分の長さ・幅・高さ
    thigh = shin = 0.21 * s
    hip_x, hip_y = 0.21 * s, 0.065 * s + 0.06 * s
    stand = [0.75, -1.45]
    z = 0.46
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.set_title("横から見た図", fontsize=14)
    ax.add_patch(patches.Rectangle((-L, z - H), 2 * L, 2 * H, fc=LIGHT, ec=INK, lw=1.5))
    ax.add_patch(patches.Rectangle((L - 0.07 * s, z - 0.03), 0.13 * s, 0.11 * s, fc=LIGHT, ec=INK, lw=1.2))
    for x in (hip_x, -hip_x):
        k = np.array([x + thigh * np.sin(stand[0]), z - thigh * np.cos(stand[0])])
        f = k + shin * np.array([np.sin(stand[0] + stand[1]), -np.cos(stand[0] + stand[1])])
        ax.plot([x, k[0], f[0]], [z, k[1], f[1]], color=SLATE, lw=5, solid_capstyle="round")
        for p in ([x, z], k):
            ax.add_patch(patches.Circle(p, 0.03, fc=YELLOW, ec=INK, lw=1.2, zorder=5))
        ax.add_patch(patches.Circle(f, 0.036, fc=INK, zorder=5))
    ax.axhline(0.0, color=GRAY, lw=1)

    def dim(ax, p, q, text, off=(0, 0)):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, color=RED, ha="center", fontsize=12)

    dim(ax, (-L, z + H + 0.12), (L, z + H + 0.12), f"胴体 {2 * L:.2f} m", (0, 0.02))
    dim(ax, (L + 0.22, 0), (L + 0.22, z), f"胴体の高さ\n{z:.2f} m", (0.13, -0.05))
    ax.text(hip_x + 0.11, z - 0.13, f"太もも {thigh:.2f} m", fontsize=11)
    ax.text(hip_x + 0.13, z - 0.36, f"すね {shin:.2f} m", fontsize=11)
    ax.set_xlim(-0.55, 0.85)
    ax.set_ylim(-0.05, 0.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax = axes[1]
    ax.set_title("上から見た図", fontsize=14)
    ax.add_patch(patches.Rectangle((-L, -W), 2 * L, 2 * W, fc=LIGHT, ec=INK, lw=1.5))
    for x in (hip_x, -hip_x):
        for y in (hip_y, -hip_y):
            ax.add_patch(patches.Circle((x, y), 0.045, fc=YELLOW, ec=INK, lw=1.2))
    dim(ax, (-hip_x, -0.33), (hip_x, -0.33), f"前後の脚の間 {2 * hip_x:.2f} m", (0, -0.06))
    dim(ax, (L + 0.1, -hip_y), (L + 0.1, hip_y), f"左右 {2 * hip_y:.2f} m", (0.16, 0))
    ax.text(0, 0, "12 関節\n（各脚：外転・股・膝）\n34 kg", ha="center", va="center", fontsize=12)
    ax.set_xlim(-0.5, 0.75)
    ax.set_ylim(-0.45, 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "robot_drawing.png"), dpi=130)


def mjcf_code():
    """ロボットの脚 1 本分の MJCF を、コード画面風に描く。"""
    from PIL import Image, ImageDraw, ImageFont

    from goal_dog import leg_xml

    code = [ln for ln in leg_xml("FL", 0.21, 0.065).strip().splitlines() if 'class="deco"' not in ln][:12]
    code = [ln[6:] if ln.startswith("      ") else ln for ln in code]
    W, Hh = 1400, 60 + 34 * len(code)
    img = Image.new("RGB", (W, Hh), (30, 32, 36))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 22)
    d.text((20, 15), "goal_dog.py（脚 1 本分の MJCF を生成）", font=ImageFont.truetype("C:/Windows/Fonts/YuGothM.ttc", 22), fill=(170, 170, 170))
    for i, ln in enumerate(code):
        ln = ln if len(ln) < 112 else ln[:109] + "..."
        d.text((20, 55 + 34 * i), ln, font=f, fill=(220, 220, 210))
    img.save(os.path.join(OUT, "mjcf_code.png"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    planner_steps()
    robot_drawing()
    mjcf_code()
