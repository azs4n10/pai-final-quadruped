"""
大型版の犬型ロボットの評価。

  python evaluate_big.py --stairs MODEL      # 学習の練習場で、段の高さごとに上り・下り（各 20 回）
  python evaluate_big.py --home MODEL        # 家の中の配達の経路（各 20 回）
  python evaluate_big.py --ckpts             # チェックポイントごとに、家の階段（15 cm × 14 段）を上れるか（各 10 回）
"""
import glob
import json
import os
import sys

import mujoco
import numpy as np

import home
from big_dog import BigDog, follow_command, load_policy
from stairs_cpu import N_COL, TILE, BigStairsEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs", "big_stairs")
MODEL = os.path.join(HERE, "models", "big_dog_model.zip")
HOME_ROUTES = [  # (経路, 受け取る場所, 届ける場所)
    (["dock", "bedroom", "dock"], "dock", "bedroom"),
    (["kitchen", "washitsu"], "kitchen", "washitsu"),
    (["kids", "dock"], "kids", "dock"),
]


def stairs_eval(policy, n=20, limit=15.0):
    """練習場の上り階段（列 0・1）と下り階段（列 2・3）の各難しさで、中央から外へ 2.6 m 進めたら成功。"""
    env = BigStairsEnv()
    out = {}
    from gpu.make_assets import STEP_MAX, STEP_MIN

    for level in range(0, 10, 1):
        rise = STEP_MIN + (STEP_MAX - STEP_MIN) * level / 9
        res = {}
        for kind, cols in (("up", (0, 1)), ("down", (2, 3))):
            ok = fell = 0
            for i in range(n):
                env.reset(seed=1000 + i)
                col = cols[i % 2]
                env.d.qpos[0:2] = (np.array([col, level]) + 0.5) * TILE
                env.d.qpos[2] = env.terrain_height(*env.d.qpos[0:2]) + 0.49
                yaw = env.np_random.uniform(-np.pi, np.pi)
                env.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                env.d.qvel[:] = 0
                mujoco.mj_forward(env.m, env.d)
                env.spawn = env.d.qpos[0:2].copy()
                env.cmd = np.array([0.6, 0.0, 0.0])
                env.until_cmd = 10 ** 9
                obs = env._obs()
                t = 0.0
                while t < limit:
                    obs, _, term, _, _ = env.step(policy(obs))
                    t += 0.02
                    if term:
                        fell += 1
                        break
                    if np.max(np.abs(env.d.qpos[0:2] - env.spawn)) > 2.6:
                        ok += 1
                        break
            res[kind] = {"success": ok / n, "fell": fell / n}
        out[f"{rise * 100:.0f}"] = res
        print(f"段差 {rise * 100:.0f} cm", res, flush=True)
    return out


def climb_home(policy, n=10, limit=40.0):
    """家の階段（1 階の下 → 2 階の上）を上り切れるか。"""
    m = mujoco.MjModel.from_xml_path(home.build_xml())
    ok = fell = 0
    for i in range(n):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        rng = np.random.default_rng(i)
        d.qpos[0:2] = home.STAIR_BOTTOM + rng.uniform(-0.1, 0.1, 2)
        yaw = np.pi / 2 + rng.uniform(-0.2, 0.2)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        dog = BigDog(m, d, policy)
        t = 0.0
        while t < limit:
            dog.step(follow_command(d.qpos[:2], dog.yaw(), home.STAIR_TOP, v_max=0.6))
            t += 0.02
            if dog.fallen():
                fell += 1
                break
            if d.qpos[1] > home.STAIR_TOP[1] - 0.3 and d.qpos[2] > home.FH + 0.3:
                ok += 1
                break
    return {"success": ok / n, "fell": fell / n}


def home_eval(n=20):
    from home_demo import run

    out = {}
    for route, pick, drop in HOME_ROUTES:
        logs = [run(route, pickup=pick, dropoff=drop, render=False, seed=s) for s in range(n)]
        ok = [lg for lg in logs if len(lg["visited"]) == len(route) and not lg["fell"]]
        out["→".join(route)] = {"success": len(ok) / n, "fell": sum(lg["fell"] for lg in logs) / n,
                               "mean_time": float(np.mean([lg["time"] for lg in ok])) if ok else None}
        print(route, out["→".join(route)], flush=True)
    return out


def main():
    if "--ckpts" in sys.argv:
        path = os.path.join(RUN, "climb.json")
        out = json.load(open(path)) if os.path.exists(path) else {}
        for mp in sorted(glob.glob(os.path.join(RUN, "ckpt", "model_*.zip"))):
            step = int(os.path.basename(mp)[6:15])
            if str(step) in out or step % 10_000_000:
                continue
            out[str(step)] = climb_home(load_policy(mp))
            print(step, out[str(step)], flush=True)
            with open(path, "w") as f:
                json.dump(out, f, indent=1)
        return
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    if "--stairs" in sys.argv:
        r = stairs_eval(load_policy(model))
        with open(os.path.join(RUN, "stairs_eval.json"), "w") as f:
            json.dump(r, f, indent=1)
    if "--home" in sys.argv:
        r = home_eval()
        with open(os.path.join(RUN, "home_eval.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
