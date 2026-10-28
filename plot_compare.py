"""
説明資料・動画用の図：Ant・犬型（旧版）・犬型（改良版）の学習曲線の比較（report/eval.png）と外観（report/robots.jpg）。

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
from goal_ant import yaw_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report")


def curves():
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for robot, label, color in [("ant", "① Ant（クモ型）", "#9aa0a6"), ("dog_v1", "② 犬型・旧版", "#4c8bf5"),
                                ("dog", "③ 犬型・改良版", "#f08c00")]:
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
    for robot, dist, z in [("ant", 2.4, 0.4), ("dog_v1", 1.5, 0.22), ("dog", 1.5, 0.22)]:
        env = ROBOTS[robot](render_mode="rgb_array", width=960, height=600, camera_id=-1)
        env.reset(seed=0)
        env.set_goal([30.0, 30.0])  # ゴール旗は写さない
        for _ in range(int(1.0 / env.dt)):
            env.step(np.zeros(env.action_space.shape))
        env.render()
        v = env.mujoco_renderer._get_viewer("rgb_array")
        v.cam.lookat[:] = [*env.xy(), z]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = dist, 140 + np.degrees(yaw_of(env.data.qpos[3:7])), -18
        tiles.append(Image.fromarray(env.render()))
        keep.append(env)  # 同じプロセスで閉じると次の描画が真っ黒になる
    img = Image.new("RGB", (960 * 3 + 40, 600), "white")
    for i, t in enumerate(tiles):
        img.paste(t, (980 * i, 0))
    img.save(os.path.join(OUT, "robots.jpg"), quality=90)


if __name__ == "__main__":
    curves()
    robots()
