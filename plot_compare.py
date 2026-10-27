"""
説明資料用の図：Ant と犬型の学習曲線の比較（report/eval.png）と、2 台の外観（report/robots.jpg）。

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
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for robot, label, color in [("ant", "初版：Ant（クモ型）", "#888888"), ("dog", "改良版：犬型", "#f08c00")]:
        ev = json.load(open(os.path.join(HERE, "runs", f"goal_{robot}", "eval.json")))
        xs = [int(k) / 1e6 for k in ev]
        ax.plot(xs, [100 * v["success"] for v in ev.values()], "o-", color=color, label=label)
    ax.set_xlabel("学習ステップ数 [百万]")
    ax.set_ylabel("ゴール到達率 [%]")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("ゴール到達率（4〜6 m 先・30 秒以内・各 30 回）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eval.png"), dpi=150)


def robots():
    tiles, keep = [], []
    for robot, dist, z in [("ant", 2.4, 0.4), ("dog", 1.7, 0.25)]:
        env = ROBOTS[robot](render_mode="rgb_array", width=960, height=600, camera_id=-1)
        env.reset(seed=0)
        for _ in range(int(1.0 / env.dt)):
            env.step(np.zeros(env.action_space.shape))
        env.render()
        v = env.mujoco_renderer._get_viewer("rgb_array")
        v.cam.lookat[:] = [*env.xy(), z]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = dist, 135, -22
        tiles.append(Image.fromarray(env.render()))
        keep.append(env)  # 同じプロセスで閉じると次の描画が真っ黒になる
    img = Image.new("RGB", (1940, 600), "white")
    img.paste(tiles[0], (0, 0))
    img.paste(tiles[1], (980, 0))
    img.save(os.path.join(OUT, "robots.jpg"), quality=90)


if __name__ == "__main__":
    curves()
    robots()
