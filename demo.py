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

from goal_ant import LANDMARK_JA, LANDMARKS, GoalAntEnv, yaw_of
from goal_dog import GoalDogEnv
from goal_dog_v1 import GoalDogEnv as GoalDogV1Env
from terrain_dog import TerrainDogEnv
from overlay import INK, LIGHT, RED, SLATE, YELLOW, font, minimap, panel, route_text, tag
from planner import plan

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs", "goal_ant")
ROBOTS = {"ant": GoalAntEnv, "dog": GoalDogEnv, "dog_v1": GoalDogV1Env, "terrain": TerrainDogEnv}
# 追従カメラ（距離, 注視点の高さ, 追従の速さ, 仰角, ロボットの向きに対する方位 or None=固定）
# 犬型は斜め前から顔と脚の運びが見えるように、向きに合わせて回り込む
CAMERA = {"ant": (6.0, 0.4, 0.08, -32.0, None), "dog": (2.4, 0.22, 0.3, -16.0, 140.0),
          "dog_v1": (2.4, 0.22, 0.3, -16.0, 140.0), "terrain": (2.4, 0.22, 0.3, -16.0, 140.0)}


class Policy:
    def __init__(self, model_path, vecnorm_path, robot="ant"):
        self.model = PPO.load(model_path, device="cpu")
        self.vn = VecNormalize.load(vecnorm_path, DummyVecEnv([ROBOTS[robot]]))
        self.vn.training = False

    def __call__(self, obs):
        action, _ = self.model.predict(self.vn.normalize_obs(obs), deterministic=True)
        return action


def vecnorm_for(model_path):
    base = os.path.basename(model_path)
    if base.startswith("model_"):
        return os.path.join(os.path.dirname(model_path), base.replace("model_", "vecnorm_").replace(".zip", ".pkl"))
    return os.path.join(os.path.dirname(model_path), base.replace("_model.zip", "_vecnorm.pkl"))


def run(policy, route, *, size=(1920, 1080), seconds_per_leg=40.0, max_seconds=None, seed=0,
        render=True, instruction="", reply="", label="", speed=1, on_frame=None, robot="ant", fps=20):
    """route を順に歩かせる。返り値はログ（到達・時間など）。"""
    w, h = size
    env = ROBOTS[robot](render_mode="rgb_array" if render else None, width=w, height=h, camera_id=-1)
    cam_dist, cam_z, cam_gain, cam_el, cam_rel = CAMERA[robot]
    next_frame = 0.0
    env.fixed_goal = True
    env.reset(seed=seed)
    viewer = None
    cam_xy = env.xy()
    cam_az = None
    pillars = [env.model.geom(f"lm_{k}").id for k in LANDMARKS if k != "home"]
    pillar_rgba = env.model.geom_rgba[pillars].copy()

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
            if render and t + 1e-9 >= next_frame:
                next_frame += speed / fps
                if viewer is None:
                    env.render()
                    viewer = env.mujoco_renderer._get_viewer("rgb_array")
                cam_xy = (1 - cam_gain) * cam_xy + cam_gain * info["xy"]
                if cam_rel is None:
                    cam_az = 120.0
                else:
                    target = np.degrees(yaw_of(env.data.qpos[3:7])) + cam_rel
                    cam_az = target if cam_az is None else cam_az + 0.06 * ((target - cam_az + 180) % 360 - 180)
                viewer.cam.lookat[:] = [cam_xy[0], cam_xy[1], cam_z]
                viewer.cam.distance = cam_dist
                viewer.cam.azimuth = cam_az
                viewer.cam.elevation = cam_el
                # カメラとロボットの間に入った柱は半透明にする
                a = np.radians(cam_az)
                eye = cam_xy - cam_dist * np.cos(np.radians(cam_el)) * np.array([np.cos(a), np.sin(a)])
                for gid, rgba in zip(pillars, pillar_rgba):
                    c = env.model.geom_pos[gid][:2]
                    seg = cam_xy - eye
                    u = np.clip(np.dot(c - eye, seg) / np.dot(seg, seg), 0, 1)
                    near = np.linalg.norm(eye + u * seg - c) < 0.45 and u < 0.97
                    env.model.geom_rgba[gid] = [*rgba[:3], 0.18 if near else rgba[3]]
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


def compose(frame, instruction, reply, route, leg, visited, trail, t, label, speed, flash, fell, names=LANDMARK_JA, mm=None):
    img = Image.fromarray(frame).convert("RGBA")
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # 文字はすべて画面下端の帯にまとめ、ロボットと柱が映る上〜中央を空けておく
    foot = f"経過 {t:5.1f} 秒" + (f"   ▶▶ {speed}倍速" if speed > 1 else "")
    if instruction:
        top = H - 200
        panel(d, (40, top, W - 40, H - 30), alpha=235, fill=(255, 255, 255))
        y1, y2 = top + 52, top + 128
        tag(d, (70, y1), "指示")
        room = W - 70 - d.textlength(foot, font=font(32, "bold")) - 40 - 170
        size = 38
        while size > 24 and d.textlength(f"「{instruction}」", font=font(size)) > room:  # 長い指示は文字を小さくして収める
            size -= 2
        d.text((170, y1), f"「{instruction}」", font=font(size), fill=INK, anchor="lm")
        tag(d, (70, y2), "LLM")
        x = 170
        for i, k in enumerate(route):
            txt = names[k]
            done = i < len(visited)
            cur = i == leg and not done
            # 今向かっている地点＝黄、到着済み＝紺灰、これから＝薄い黄（講義資料の配色）
            fill, ink = (YELLOW, INK) if cur else ((SLATE, (255, 255, 255)) if done else (LIGHT, INK))
            txt = ("✓ " if done else "") + txt
            tw = d.textlength(txt, font=font(30, "bold"))
            d.rectangle((x, y2 - 26, x + tw + 30, y2 + 26), fill=fill, outline=INK, width=1)
            d.text((x + 15, y2), txt, font=font(30, "bold"), fill=ink, anchor="lm")
            x += tw + 30
            if i < len(route) - 1:
                d.text((x + 10, y2), "→", font=font(30, "bold"), fill=INK, anchor="lm")
                x += 48
        d.text((W - 70, y1), foot, font=font(32, "bold"), fill=INK, anchor="rm")
        size = 28
        while reply and size > 20 and d.textlength(reply, font=font(size)) > W - 70 - x - 40:  # 経路の札に重ねない
            size -= 2
        if reply and d.textlength(reply, font=font(size)) <= W - 70 - x - 40:
            d.text((W - 70, y2), reply, font=font(size), fill=(90, 90, 90), anchor="rm")
    else:
        panel(d, (40, H - 120, W - 40, H - 30), alpha=235, fill=(255, 255, 255))
        d.text((70, H - 75), label, font=font(44, "bold"), fill=INK, anchor="lm")
        d.text((W - 70, H - 75), foot, font=font(32, "bold"), fill=INK, anchor="rm")

    if mm is None:
        mm = minimap(330, trail, route[min(leg, len(route) - 1)] if route else None, visited=visited)
    layer.alpha_composite(mm, (W - 40 - mm.width, 40))

    msg, color = (("転倒", RED) if fell else (flash[0], INK) if flash else (None, None))
    if msg:
        tw = d.textlength(msg, font=font(64, "bold"))
        panel(d, (W / 2 - tw / 2 - 40, 40, W / 2 + tw / 2 + 40, 150), alpha=235, radius=0, fill=(255, 255, 255))
        d.text((W / 2, 95), msg, font=font(64, "bold"), fill=color, anchor="mm")

    img.alpha_composite(layer)
    return np.asarray(img.convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="", help="日本語の指示（LLM で経由地に変換）")
    p.add_argument("--route", default="", help="経由地をカンマ区切りで直接指定（例: red,blue,home）")
    p.add_argument("--robot", default="ant", choices=list(ROBOTS))
    p.add_argument("--model", default=None, help="既定は models/final_model.zip（ant）/ models/dog_model.zip（dog）")
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

    if args.model is None:
        args.model = os.path.join(HERE, "models", {"ant": "final_model.zip"}.get(args.robot, f"{args.robot}_model.zip"))
    policy = Policy(args.model, vecnorm_for(args.model), args.robot)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=20, quality=8, macro_block_size=8)
    log = run(policy, route, max_seconds=args.seconds, seed=args.seed, instruction=args.text, reply=reply,
              label=args.label, speed=args.speed, on_frame=writer.append_data, robot=args.robot)
    writer.close()
    log.update({"instruction": args.text, "reply": reply, "planner": source, "model": args.model})
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"[done] visited={log['visited']} fell={log['fell']} time={log['time']}s -> {args.out}")


if __name__ == "__main__":
    main()
