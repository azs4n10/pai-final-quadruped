"""
チェックポイントごとの到達性能を測り、学習曲線を描く。

  python evaluate.py            # runs/goal_ant/eval.json と eval.png を出力
  python evaluate.py --routes   # デモ経路ごとの成功率 runs/goal_ant/route_eval.json
  --robot dog を付けると犬型（runs/goal_dog/）
評価方法：ランダムな方向・距離 4〜6 m のゴールを 1 つ置き、30 秒以内に半径 0.6 m に入れば成功。
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from demo import HERE, ROBOTS, Policy, vecnorm_for  # noqa: E402


def evaluate(model_path, n=30, limit=30.0, seed=100, robot="ant"):
    policy = Policy(model_path, vecnorm_for(model_path), robot)
    env = ROBOTS[robot]()
    env.fixed_goal = True
    rng = np.random.default_rng(seed)
    ok, times, fell = 0, [], 0
    for i in range(n):
        env.reset(seed=seed + i)
        d, a = rng.uniform(4, 6), rng.uniform(-np.pi, np.pi)
        env.set_goal(env.xy() + d * np.array([np.cos(a), np.sin(a)]))
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
    return {"success": ok / n, "mean_time": float(np.mean(times)) if times else None, "fell": fell / n}


ROUTES = [["red", "blue", "home"], ["yellow", "home"], ["red", "blue", "green", "yellow", "home"]]


def evaluate_routes(model_path, n=20, robot="ant"):
    """デモと同じ経由地リストを n 回（初期姿勢の乱数違い）歩かせた成功率。"""
    from demo import run

    policy = Policy(model_path, vecnorm_for(model_path), robot)
    out = {}
    for route in ROUTES:
        logs = [run(policy, route, render=False, seed=s, robot=robot) for s in range(n)]
        ok = [lg for lg in logs if len(lg["visited"]) == len(route) and not lg["fell"]]
        out["→".join(route)] = {
            "success": len(ok) / n,
            "fell": sum(lg["fell"] for lg in logs) / n,
            "mean_time": float(np.mean([lg["time"] for lg in ok])) if ok else None,
        }
        print(route, out["→".join(route)], flush=True)
    return out


def main():
    robot = "dog" if "--robot" in sys.argv and sys.argv[sys.argv.index("--robot") + 1] == "dog" else "ant"
    RUN = os.path.join(HERE, "runs", f"goal_{robot}")
    if "--routes" in sys.argv:
        model = os.path.join(HERE, "models", "dog_model.zip" if robot == "dog" else "final_model.zip")
        with open(os.path.join(RUN, "route_eval.json"), "w", encoding="utf-8") as f:
            json.dump(evaluate_routes(model, robot=robot), f, indent=1, ensure_ascii=False)
        return
    models = sorted(glob.glob(os.path.join(RUN, "ckpt", "model_*.zip")))
    steps = [int(os.path.basename(m)[6:15]) for m in models]
    keep = [i for i, s in enumerate(steps) if s < 100 or s % 1_000_000 == 0]
    path = os.path.join(RUN, "eval.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    out = {}
    for i in keep:
        if str(steps[i]) in cache:
            out[steps[i]] = cache[str(steps[i])]
            continue
        r = evaluate(models[i], robot=robot)
        out[steps[i]] = r
        print(steps[i], r, flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)

    xs = [s / 1e6 for s in out]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [100 * v["success"] for v in out.values()], "o-", color="#f08c00", label="到達成功率")
    ax.plot(xs, [100 * v["fell"] for v in out.values()], "s--", color="#888", label="転倒率")
    ax.set_xlabel("学習ステップ数 [百万]")
    ax.set_ylabel("%")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("ゴール到達（4〜6 m 先・30 秒以内）の成功率")
    plt.rcParams["font.family"] = "Yu Gothic"
    for t in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_legend().get_texts():
        t.set_fontfamily(["Yu Gothic", "Meiryo", "sans-serif"])
    fig.tight_layout()
    fig.savefig(os.path.join(RUN, "eval.png"), dpi=150)


if __name__ == "__main__":
    main()
