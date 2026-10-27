"""
日本語の指示 → LLM で経由地リスト → 学習済み歩行方策で順に歩く、を録画する。

  python demo.py --text "赤い柱に寄ってから青い柱へ行き、最後はホームに戻って" --out runs/demo1.mp4
  python demo.py --route red --model runs/goal_ant/ckpt/model_000000016.zip --label "学習前" --seconds 10
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from goal_ant import LANDMARK_JA, LANDMARKS, GoalAntEnv
from overlay import font, minimap, panel, route_text
from planner import plan

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs", "goal_ant")


class Policy:
    def __init__(self, model_path, vecnorm_path):
        self.model = PPO.load(model_path, device="cpu")
        self.vn = VecNormalize.load(vecnorm_path, DummyVecEnv([GoalAntEnv]))
        self.vn.training = False

    def __call__(self, obs):
        action, _ = self.model.predict(self.vn.normalize_obs(obs), deterministic=True)
        return action


def vecnorm_for(model_path):
    base = os.path.basename(model_path)
    if base.startswith("model_"):
        return os.path.join(os.path.dirname(model_path), base.replace("model_", "vecnorm_").replace(".zip", ".pkl"))
    return os.path.join(os.path.dirname(model_path), "final_vecnorm.pkl")


def run(policy, route, *, size=(1920, 1080), seconds_per_leg=40.0, max_seconds=None, seed=0,
        render=True, instruction="", reply="", label="", speed=1, on_frame=None):
    """route を順に歩かせる。返り値はログ（到達・時間など）。"""
    w, h = size
    env = GoalAntEnv(render_mode="rgb_array" if render else None, width=w, height=h, camera_id=-1)
    env.fixed_goal = True
    env.reset(seed=seed)
    viewer = None
    cam_xy = env.xy()

    trail, visited, events = [env.xy()], [], []
    t, fell, flash = 0.0, False, None
    leg_steps = int(seconds_per_leg / env.dt)
    for leg, name in enumerate(route):
        env.set_goal(LANDMARKS[name][0])
        for _ in range(leg_steps):
            obs, _, terminated, _, info = env.step(policy(env._get_obs()))
            t += env.dt
            trail.append(info["xy"])
            if info["reached"]:
                visited.append(name)
                events.append({"t": round(t, 2), "reached": name})
                flash = (f"{LANDMARK_JA[name]}に到着！", t)
            if terminated:
                fell = True
                events.append({"t": round(t, 2), "fell": True})
            if render and int(round(t / env.dt)) % speed == 0:
                if viewer is None:
                    env.render()
                    viewer = env.mujoco_renderer._get_viewer("rgb_array")
                cam_xy = 0.92 * cam_xy + 0.08 * info["xy"]
                viewer.cam.lookat[:] = [cam_xy[0], cam_xy[1], 0.4]
                viewer.cam.distance = 6.0
                viewer.cam.azimuth = 120.0
                viewer.cam.elevation = -32.0
                frame = env.render()
                on_frame(compose(frame, instruction, reply, route, leg, visited, trail, t, label, speed,
                                 flash if flash and t - flash[1] < 1.5 else None, fell))
            if info["reached"] or terminated or (max_seconds and t >= max_seconds):
                break
        if fell or (max_seconds and t >= max_seconds):
            break
    env.close()
    return {"route": route, "visited": visited, "fell": fell, "time": round(t, 2), "events": events,
            "trail": [list(map(float, p)) for p in trail[::5]]}


def compose(frame, instruction, reply, route, leg, visited, trail, t, label, speed, flash, fell):
    img = Image.fromarray(frame).convert("RGBA")
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # 文字はすべて画面下端の帯にまとめ、ロボットと柱が映る上〜中央を空けておく
    foot = f"経過 {t:5.1f} 秒" + (f"   ▶▶ {speed}倍速" if speed > 1 else "")
    if instruction:
        top = H - 200
        panel(d, (40, top, W - 40, H - 30), alpha=190)
        y1, y2 = top + 52, top + 128
        d.text((70, y1), "指示", font=font(28, "bold"), fill=(255, 175, 70), anchor="lm")
        d.text((150, y1), f"「{instruction}」", font=font(38), fill=(255, 255, 255), anchor="lm")
        d.text((70, y2), "LLM", font=font(28, "bold"), fill=(120, 200, 255), anchor="lm")
        x = 150
        for i, k in enumerate(route):
            txt = LANDMARK_JA[k]
            done = i < len(visited)
            cur = i == leg and not done
            fill = (255, 140, 0, 255) if cur else ((90, 200, 120, 255) if done else (70, 76, 88, 255))
            tw = d.textlength(txt, font=font(30, "bold"))
            d.rounded_rectangle((x, y2 - 26, x + tw + 30, y2 + 26), radius=12, fill=fill)
            d.text((x + 15, y2), txt, font=font(30, "bold"), fill=(255, 255, 255), anchor="lm")
            x += tw + 30
            if i < len(route) - 1:
                d.text((x + 10, y2), "→", font=font(30, "bold"), fill=(200, 200, 200), anchor="lm")
                x += 48
        d.text((W - 70, y1), foot, font=font(32, "bold"), fill=(255, 255, 255), anchor="rm")
        if reply:
            d.text((W - 70, y2), reply, font=font(28), fill=(200, 210, 222), anchor="rm")
    else:
        panel(d, (40, H - 120, W - 40, H - 30), alpha=190)
        d.text((70, H - 75), label, font=font(44, "bold"), fill=(255, 255, 255), anchor="lm")
        d.text((W - 70, H - 75), foot, font=font(32, "bold"), fill=(255, 255, 255), anchor="rm")

    mm = minimap(330, trail, route[min(leg, len(route) - 1)] if route else None, visited=visited)
    layer.alpha_composite(mm, (W - 370, 40))

    msg, color = (("転倒", (255, 90, 90)) if fell else (flash[0], (120, 230, 140)) if flash else (None, None))
    if msg:
        tw = d.textlength(msg, font=font(64, "bold"))
        panel(d, (W / 2 - tw / 2 - 40, 40, W / 2 + tw / 2 + 40, 150), alpha=200, radius=26)
        d.text((W / 2, 95), msg, font=font(64, "bold"), fill=color, anchor="mm")

    img.alpha_composite(layer)
    return np.asarray(img.convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="", help="日本語の指示（LLM で経由地に変換）")
    p.add_argument("--route", default="", help="経由地をカンマ区切りで直接指定（例: red,blue,home）")
    p.add_argument("--model", default=os.path.join(HERE, "models", "final_model.zip"))
    p.add_argument("--reply", default="", help="--text と --route を両方渡したときに表示する LLM の返答")
    p.add_argument("--label", default="")
    p.add_argument("--seconds", type=float, default=None, help="打ち切り秒数")
    p.add_argument("--speed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "demo.mp4"))
    args = p.parse_args()

    reply, source = args.reply, "manual"
    if args.text and args.route:  # 計画済み（LLM を呼ばずに再録画）
        route = [r for r in args.route.split(",") if r]
        source = "cached"
    elif args.text:
        res = plan(args.text)
        route, reply, source = res["route"], res["reply"], res["source"]
    else:
        route = [r for r in args.route.split(",") if r]
    print(f"[plan] {source}: {route_text(route)}  {reply}", flush=True)

    policy = Policy(args.model, vecnorm_for(args.model))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=20, quality=8, macro_block_size=8)
    log = run(policy, route, max_seconds=args.seconds, seed=args.seed, instruction=args.text, reply=reply,
              label=args.label, speed=args.speed, on_frame=writer.append_data)
    writer.close()
    log.update({"instruction": args.text, "reply": reply, "planner": source, "model": args.model})
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"[done] visited={log['visited']} fell={log['fell']} time={log['time']}s -> {args.out}")


if __name__ == "__main__":
    main()
