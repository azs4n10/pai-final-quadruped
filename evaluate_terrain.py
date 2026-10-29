"""
坂・階段・凸凹での到達性能と、家・庭でのデモ経路の成功率を測る。

  python evaluate_terrain.py              # チェックポイントごと（500 万ステップおき）：runs/terrain_dog/eval.json
  python evaluate_terrain.py --final      # 採用モデルを地形ごと 50 回：runs/terrain_dog/eval_final.json
  python evaluate_terrain.py --house      # 家・庭のデモ経路を各 20 回：runs/terrain_dog/house_eval.json
評価方法：最も難しい設定（段差 10 cm・最大勾配 約 20°・凸凹 ±3.5 cm）で、3.5 m 先のゴールに 20 秒以内に着けば成功。
"""
import glob
import json
import os
import sys

import numpy as np

from demo import HERE, Policy, vecnorm_for
from terrain_dog import TERRAINS, TerrainDogEnv

RUN = os.path.join(HERE, "runs", "terrain_dog")
MODEL = os.path.join(HERE, "models", "terrain_model.zip")
HOUSE_ROUTES = [["kitchen", "bedroom"], ["stairs_top", "slope_top", "dock"], ["kids", "living", "dock"]]


def evaluate(model_path, n=20, level=1.0, limit=20.0, seed=500):
    policy = Policy(model_path, vecnorm_for(model_path), "terrain")
    env = TerrainDogEnv(curriculum=False)
    env.fixed_goal = True
    out = {}
    for kind in TERRAINS:
        env.fixed_terrain = (kind, level)
        rng = np.random.default_rng(seed)
        ok, fell, times = 0, 0, []
        for i in range(n):
            env.reset(seed=seed + i)
            a = rng.uniform(-np.pi, np.pi)
            env.set_goal(env.xy() + 3.5 * np.array([np.cos(a), np.sin(a)]))
            t = 0.0
            while t < limit:
                _, _, term, _, info = env.step(policy(env._get_obs()))
                t += env.dt
                if info["reached"]:
                    ok += 1
                    times.append(t)
                    break
                if term:
                    fell += 1
                    break
        out[kind] = {"success": ok / n, "fell": fell / n, "mean_time": float(np.mean(times)) if times else None}
    out["mean"] = float(np.mean([v["success"] for v in out.values()]))
    return out


def evaluate_house(model_path, n=20):
    from house_demo import run

    policy = Policy(model_path, vecnorm_for(model_path), "terrain")
    out = {}
    for route in HOUSE_ROUTES:
        logs = [run(policy, route, render=False, seed=s) for s in range(n)]
        ok = [lg for lg in logs if len(lg["visited"]) == len(route) and not lg["fell"]]
        out["→".join(route)] = {"success": len(ok) / n, "fell": sum(lg["fell"] for lg in logs) / n,
                                "mean_time": float(np.mean([lg["time"] for lg in ok])) if ok else None}
        print(route, out["→".join(route)], flush=True)
    return out


def main():
    if "--house" in sys.argv:
        with open(os.path.join(RUN, "house_eval.json"), "w", encoding="utf-8") as f:
            json.dump(evaluate_house(MODEL), f, indent=1, ensure_ascii=False)
        return
    if "--final" in sys.argv:
        r = evaluate(MODEL, n=50, seed=900)
        print(r)
        with open(os.path.join(RUN, "eval_final.json"), "w") as f:
            json.dump(r, f, indent=1)
        return
    path = os.path.join(RUN, "eval.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for m in sorted(glob.glob(os.path.join(RUN, "ckpt", "model_*.zip"))):
        step = int(os.path.basename(m)[6:15])
        if str(step) in out or not (step < 100 or step % 5_000_000 == 0):
            continue
        out[str(step)] = evaluate(m)
        print(step, out[str(step)], flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
