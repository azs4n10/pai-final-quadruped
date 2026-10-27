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

    y = 40
    if instruction:
        panel(d, (40, y, 1300, y + 170))
        d.text((70, y + 22), "指示（日本語）", font=font(30, "bold"), fill=(255, 175, 70))
        d.text((70, y + 70), f"「{instruction}」", font=font(40), fill=(255, 255, 255))
        y += 190
    if route and instruction:
        panel(d, (40, y, 1300, y + 160))
        d.text((70, y + 20), "LLM の計画（gemma4 / ローカル）", font=font(30, "bold"), fill=(120, 200, 255))
        x = 70
        for i, k in enumerate(route):
            txt = LANDMARK_JA[k]
            done = i < len(visited)
            cur = i == leg and not done
            fill = (255, 140, 0, 255) if cur else ((90, 200, 120, 255) if done else (70, 76, 88, 255))
            tw = d.textlength(txt, font=font(34, "bold"))
            d.rounded_rectangle((x, y + 72, x + tw + 36, y + 130), radius=14, fill=fill)
            d.text((x + 18, y + 101), txt, font=font(34, "bold"), fill=(255, 255, 255), anchor="lm")
            x += tw + 36
            if i < len(route) - 1:
                d.text((x + 12, y + 101), "→", font=font(34, "bold"), fill=(200, 200, 200), anchor="lm")
                x += 56
    if label:
        panel(d, (40, y, 40 + d.textlength(label, font=font(56, "bold")) + 60, y + 100))
        d.text((70, y + 50), label, font=font(56, "bold"), fill=(255, 255, 255), anchor="lm")

    mm = minimap(420, trail, route[min(leg, len(route) - 1)] if route else None, visited=visited)
    layer.alpha_composite(mm, (W - 460, 40))

    foot = f"経過 {t:5.1f} 秒" + (f"   ▶▶ {speed}倍速" if speed > 1 else "")
    panel(d, (40, H - 110, 40 + d.textlength(foot, font=font(40, "bold")) + 60, H - 30))
    d.text((70, H - 70), foot, font=font(40, "bold"), fill=(255, 255, 255), anchor="lm")

    if reply and not flash:
        tw = d.textlength(reply, font=font(36))
        panel(d, (W - 80 - tw, H - 110, W - 40, H - 30))
        d.text((W - 60 - tw, H - 70), reply, font=font(36), fill=(220, 230, 240), anchor="lm")
    if flash:
        msg = flash[0]
        tw = d.textlength(msg, font=font(80, "bold"))
        panel(d, (W / 2 - tw / 2 - 50, H / 2 - 90, W / 2 + tw / 2 + 50, H / 2 + 70), alpha=200, radius=30)
        d.text((W / 2, H / 2 - 10), msg, font=font(80, "bold"), fill=(120, 230, 140), anchor="mm")
    if fell:
        d.text((W / 2, H / 2), "転倒", font=font(90, "bold"), fill=(255, 90, 90), anchor="mm")

    img.alpha_composite(layer)
    return np.asarray(img.convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="", help="日本語の指示（LLM で経由地に変換）")
    p.add_argument("--route", default="", help="経由地をカンマ区切りで直接指定（例: red,blue,home）")
    p.add_argument("--model", default=os.path.join(HERE, "models", "final_model.zip"))
    p.add_argument("--label", default="")
    p.add_argument("--seconds", type=float, default=None, help="打ち切り秒数")
    p.add_argument("--speed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "demo.mp4"))
    args = p.parse_args()

    reply, source = "", "manual"
    if args.text:
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
