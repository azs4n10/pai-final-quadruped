"""
説明資料と動画用の図（report/ に保存）。

  python make_figures.py
- robot.jpg：犬型ロボットの外観（斜め前と真横）
- curriculum.png：学習中の地形ごとの難しさ（カリキュラム）の推移
- terrain_eval.png：地形の種類×難しさごとの成功率
- house_map.png：家と庭の間取り図と、3 つの評価経路の計画結果
"""
import json
import os
import re

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report")
RUN = os.path.join(HERE, "runs", "terrain_dog")
# 講義資料の配色
INK, YELLOW, LIGHT, SLATE, RED, GRAY = "#141414", "#ffd966", "#fff2cc", "#44546a", "#c00000", "#a6a6a6"
TERRAIN_JA = {"flat": "平地", "rough": "凸凹", "hills": "丘（坂）", "stairs_up": "上り階段", "stairs_down": "下り階段"}
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]


def curriculum():
    steps, levels = [], []
    for line in open(os.path.join(HERE, "train_terrain.log"), encoding="utf-8"):
        m = re.search(r"step=(\d+) .*level (.*)$", line)
        if m:
            steps.append(int(m.group(1)) / 1e6)
            levels.append(dict(kv.split("=") for kv in m.group(2).split()))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    styles = {"flat": (GRAY, "-"), "rough": (SLATE, "--"), "hills": (SLATE, "-"), "stairs_up": (RED, "-"), "stairs_down": (RED, "--")}
    for k, (c, ls) in styles.items():
        ax.plot(steps, [float(lv[k]) for lv in levels], ls, color=c, lw=2, label=TERRAIN_JA[k])
    ax.axvline(31, color=GRAY, lw=1)
    ax.text(31.8, 4.6, "脚の動かせる範囲を広げた", color="#555555", fontsize=9)
    ax.set_xlabel("学習ステップ数 [百万]")
    ax.set_ylabel("難しさ（0〜1）")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("地形ごとの難しさの推移（直近 100 エピソードの平均）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curriculum.png"), dpi=150)


def terrain_eval():
    ev = json.load(open(os.path.join(RUN, "eval_final.json")))
    lvs = list(ev)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    kinds = list(TERRAIN_JA)
    colors = [GRAY, LIGHT, YELLOW, RED, SLATE]
    w = 0.16
    for i, k in enumerate(kinds):
        xs = np.arange(len(lvs)) + (i - 2) * w
        ax.bar(xs, [100 * ev[lv][k]["success"] for lv in lvs], w, color=colors[i], edgecolor=INK, lw=0.6, label=TERRAIN_JA[k])
    ax.set_xticks(np.arange(len(lvs)))
    ax.set_xticklabels([f"難しさ {lv}\n段差 {2 + 8 * float(lv):.0f} cm" for lv in lvs])
    ax.set_ylabel("成功率 [%]")
    ax.set_ylim(0, 128)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=5, fontsize=8, loc="upper center")
    ax.set_title("地形の種類×難しさごとの成功率（各 30 回）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "terrain_eval.png"), dpi=150)


def robot():
    from house import HouseDogEnv

    env = HouseDogEnv(render_mode="rgb_array", width=960, height=600, camera_id=-1)
    env.reset(seed=0)
    env.set_goal([30.0, 30.0])  # ゴール旗は写さない
    for _ in range(int(1.0 / env.dt)):
        env.step(np.zeros(env.action_space.shape))
    env.render()
    v = env.mujoco_renderer._get_viewer("rgb_array")
    tiles = []
    for az, el in [(130, -18), (180, -6)]:
        v.cam.lookat[:] = [*env.xy(), 0.22]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.4, az, el
        tiles.append(Image.fromarray(env.render()))
    img = Image.new("RGB", (960 * 2 + 20, 600), "white")
    for i, t in enumerate(tiles):
        img.paste(t, (980 * i, 0))
    img.save(os.path.join(OUT, "robot.jpg"), quality=90)


def house_map():
    from evaluate_terrain import HOUSE_ROUTES
    from house import HOME, PLACES, plan_path
    from house_demo import NAMES, house_map as draw

    img = draw(900, [], [], None, [])
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    x0, x1, y0, y1 = -7.0, 7.0, -7.2, 8.6
    w, h = img.size

    def px(x, y):
        return ((x - x0) / (x1 - x0) * w, (y1 - y) / (y1 - y0) * h)

    styles = [((192, 0, 0, 255), 0), ((68, 84, 106, 255), 1), ((120, 120, 120, 255), 2)]
    for route, (col, k) in zip(HOUSE_ROUTES, styles):
        cur = np.array(PLACES[HOME][1])
        pts = [cur]
        for name in route:
            pts += plan_path(cur, PLACES[name][1])
            cur = np.array(PLACES[name][1])
        off = (k - 1) * 4  # 重なりを見やすく少しずらす
        d.line([(px(*p)[0] + off, px(*p)[1] + off) for p in pts], fill=col, width=5)
    from overlay import font

    for key, (_, (x, y)) in PLACES.items():
        cx, cy = px(x, y)
        d.text((cx, cy + 22), NAMES[key], font=font(24, "bold"), fill=(20, 20, 20, 255), anchor="mt",
               stroke_width=3, stroke_fill=(255, 255, 255, 255))
    bg = Image.new("RGB", img.size, "white")
    bg.paste(img, mask=img.split()[3])
    bg.save(os.path.join(OUT, "house_map.png"))


if __name__ == "__main__":
    curriculum()
    robot()
    house_map()
    if os.path.exists(os.path.join(RUN, "eval_final.json")):
        terrain_eval()


def big_robot():
    """大型版の犬型ロボットを、斜め前と真横から写す（report/big_robot.jpg）。床だけの撮影用の場所で撮る。"""
    import re

    import mujoco

    from train_scene import render_xml

    src = open(render_xml(), encoding="utf-8").read()
    src = re.sub(r'<geom name="floor" type="hfield"[^>]*/>',
                 '<geom name="floor" type="plane" size="20 20 1" material="ground" contype="1" conaffinity="1"/>', src)
    src = re.sub(r'<hfield[^>]*/>', '', src)
    m = mujoco.MjModel.from_xml_string(src)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    m.vis.map.shadowclip = 0.15  # 影を細かく（床の縞模様を防ぐ）
    d.qpos[0:3] = [0, 0, 0.47]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 600, 960)
    cam = mujoco.MjvCamera()
    tiles = []
    for az, el in [(150, -15), (90, -3)]:
        cam.lookat[:] = [0, 0, 0.33]
        cam.distance, cam.azimuth, cam.elevation = 1.8, az, el
        r.update_scene(d, cam)
        tiles.append(Image.fromarray(r.render()))
    img = Image.new("RGB", (960 * 2 + 20, 600), "white")
    for i, t in enumerate(tiles):
        img.paste(t, (980 * i, 0))
    img.save(os.path.join(OUT, "big_robot.jpg"), quality=90)


def home_views():
    """家の 1 階・2 階を斜め上から写す（report/home_1f.jpg・home_2f.jpg）。1 階は 2 階を外して写す。"""
    import mujoco

    import home
    from home_demo import Scene

    sc = Scene(render=True, size=(1280, 820), policy_path=os.path.join(HERE, "runs", "big_stairs", "ckpt", "model_010000000.zip"))
    sc.m.vis.map.shadowclip = 0.3
    sc.d.mocap_pos[sc.goal] = sc.d.mocap_pos[sc.parcel] = [80, 80, -20]  # 目標の印と荷物は写さない
    mujoco.mj_forward(sc.m, sc.d)
    for lv, name in ((1, "home_1f.jpg"), (2, "home_2f.jpg")):
        sc.m.geom_rgba[:] = sc.rgba0
        sc.m.geom_rgba[sc.cargo, 3] = 0.0
        if lv == 1:
            sc.m.geom_rgba[sc.upper, 3] = 0.0
            sc.m.geom_rgba[sc.upper_doors, 3] = 0.0
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [(home.X0 + home.X1) / 2, (home.Y0 + home.Y1) / 2 - 0.3, home.door_base_z(lv)]
        cam.distance, cam.azimuth, cam.elevation = 14.5, 90, -62
        sc.renderer.update_scene(sc.d, cam, sc.opt)
        Image.fromarray(sc.renderer.render()).save(os.path.join(OUT, name), quality=90)


def learning():
    """大型版の学習の経過：練習場の難しさ（段差の高さ）と、家の階段を上り切れた割合（report/learning.png）。"""
    import json

    from gpu.make_assets import STEP_MAX, STEP_MIN

    # 前半（31M まで）は脚の動かせる範囲が狭い設定。後半はそこから続けて学習した（ステップ数は通算）
    steps, levels, offset = [], [], 0.0
    for log, until in (("train_big_stairs.log", 31.0), ("train_big_stairs2.log", None)):
        for line in open(os.path.join(HERE, log), encoding="utf-8", errors="ignore"):
            m = re.search(r"\[ckpt\] step=(\d+) .*level stairs=([\d.]+)", line)
            if m and (until is None or int(m.group(1)) / 1e6 <= until):
                steps.append(offset + int(m.group(1)) / 1e6)
                levels.append(float(m.group(2)))
        offset = 31.0
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(steps, [100 * (STEP_MIN + (STEP_MAX - STEP_MIN) * lv / 9) for lv in levels], color=SLATE, lw=2,
            label="練習場で挑戦中の段差の高さ（平均）")
    ax.axhline(15, color=GRAY, ls="--", lw=1.5)
    ax.text(steps[0] if steps else 0, 15.3, "家の階段（15 cm）", color="#555555", fontsize=10)
    ax.axvline(31, color=GRAY, lw=1)
    ax.text(31.8, 4.6, "脚の動かせる範囲を広げた", color="#555555", fontsize=9)
    ax.set_xlabel("学習ステップ数 [百万]")
    ax.set_ylabel("段差 [cm]")
    ax.set_ylim(4, 19)
    ax.grid(alpha=0.3)
    path = os.path.join(HERE, "runs", "big_stairs2", "climb.json")
    if os.path.exists(path):
        climb = json.load(open(path))
        xs = sorted(int(k) for k in climb)
        ax2 = ax.twinx()
        ax2.plot([31 + x / 1e6 for x in xs], [100 * climb[str(x)]["success"] for x in xs], "o-", color=RED, lw=2,
                 label="家の階段（14 段）を上り切れた割合")
        ax2.set_ylabel("成功率 [%]")
        ax2.set_ylim(-3, 103)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    else:
        ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning.png"), dpi=150)
