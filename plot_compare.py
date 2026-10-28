"""
説明資料用の図：学習曲線（report/eval.png）と犬型ロボットの外観（report/robot.jpg）。

  python plot_compare.py
"""
import json
import os

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from demo import ROBOTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report")


def curves():
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ev = json.load(open(os.path.join(HERE, "runs", "goal_dog", "eval.json")))
    xs = [int(k) / 1e6 for k in ev]
    ax.plot(xs, [100 * v["success"] for v in ev.values()], "o-", color="#f08c00", label="到達率")
    ax.plot(xs, [100 * v["fell"] for v in ev.values()], "s--", color="#888888", label="転倒率")
    ax.set_xlabel("学習ステップ数 [百万]")
    ax.set_ylabel("%")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    ax.set_title("ゴール到達（4〜6 m 先・30 秒以内・各 30 回）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eval.png"), dpi=150)


def robots():
    """説明資料用：犬型ロボットを斜め前と真横から写す（report/robot.jpg）。"""
    env = ROBOTS["dog"](render_mode="rgb_array", width=960, height=600, camera_id=-1)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1, 0, 0, 0]
    env.set_goal([30.0, 30.0])  # ゴール旗は写さない
    for _ in range(int(1.0 / env.dt)):
        env.step(np.zeros(env.action_space.shape))
    env.render()
    v = env.mujoco_renderer._get_viewer("rgb_array")
    tiles = []
    for az, el in [(140, -18), (90, -6)]:
        v.cam.lookat[:] = [*env.xy(), 0.22]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.4, az, el
        tiles.append(Image.fromarray(env.render()))
    img = Image.new("RGB", (960 * 2 + 20, 600), "white")
    for i, t in enumerate(tiles):
        img.paste(t, (980 * i, 0))
    img.save(os.path.join(OUT, "robot.jpg"), quality=90)


if __name__ == "__main__":
    curves()
    robots()
