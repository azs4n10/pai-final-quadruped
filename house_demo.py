"""
家と庭で、日本語の指示 → LLM で部屋の順番 → 経路計画で経由地 → 歩行方策 で動かして録画する。

  python house_demo.py --text "キッチンで荷物を受け取って、寝室に届けて" --out runs/house1.mp4
  python house_demo.py --route stairs_top,dock --model runs/terrain_dog/ckpt/model_010000000.zip --label "学習 1000万ステップ"
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from demo import Policy, compose, vecnorm_for
from goal_ant import REACH_RADIUS, yaw_of
from house import FURNITURE, HOME, PLACES, SLOPE, STAIRS, HouseDogEnv, _walls, plan_path
from planner import plan

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"dock": "充電ステーション", "genkan": "玄関", "living": "リビング", "kitchen": "キッチン", "bedroom": "寝室",
         "kids": "子ども部屋", "stairs_top": "階段の高台", "slope_top": "坂の高台"}
WAYPOINT_RADIUS = 0.45  # 途中の経由地はこの距離で次へ切り替える
CAM = dict(distance=2.6, z=0.22, gain=0.25, elevation=-20.0, rel=140.0)


def house_map(size, trail, path, target, visited):
    """家の間取り図（上から見た図）に、計画した経路・軌跡・地点を重ねる。"""
    x0, x1, y0, y1 = -7.0, 7.0, -7.2, 8.6
    w = size
    h = int(size * (y1 - y0) / (x1 - x0))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=20, fill=(15, 18, 24, 200))

    def px(x, y):
        return ((x - x0) / (x1 - x0) * w, (y1 - y) / (y1 - y0) * h)

    def rect(a, b, c, e, **kw):
        p, q = px(a, e), px(b, c)
        d.rectangle((p[0], p[1], q[0], q[1]), **kw)

    rect(-6, 6, 0, 8, fill=(70, 62, 52, 255))
    for s in (STAIRS, SLOPE):
        rect(s["x0"], s["x1"], s["y0"], s["y1"], fill=(110, 116, 124, 255))
    rect(STAIRS["x1"], STAIRS["x1"] + 1.2, STAIRS["y0"], STAIRS["y1"], fill=(90, 96, 104, 255))
    rect(SLOPE["x0"] - SLOPE["run"], SLOPE["x0"], SLOPE["y0"], SLOPE["y1"], fill=(90, 96, 104, 255))
    for f in FURNITURE:
        rect(*f[:4], fill=(120, 100, 80, 255))
    for wl in _walls():
        rect(*wl[:4], fill=(235, 235, 230, 255))
    for k, (_, (x, y)) in PLACES.items():
        cx, cy = px(x, y)
        r = 9
        col = (90, 200, 120, 255) if k in visited else ((255, 140, 0, 255) if k == target else (200, 205, 215, 200))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
    if path:
        pts = [px(*p) for p in path]
        d.line(pts, fill=(120, 200, 255, 200), width=3)
    if len(trail) >= 2:
        d.line([px(*p) for p in trail], fill=(255, 170, 40, 255), width=4, joint="curve")
    if trail:
        cx, cy = px(*trail[-1])
        d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(255, 255, 255, 255), outline=(255, 140, 0, 255), width=3)
    return img


def _fade_obstacles(env, boxes, rgba0, eye, target):
    """カメラ（eye）とロボット（target）の間に入った壁・家具を半透明にする。"""
    seg = target - eye
    for (gid, lo, hi), rgba in zip(boxes, rgba0):
        t0, t1, hit = 0.0, 1.0, True
        for k in range(2):  # 線分と箱（xy 平面）の交差判定
            if abs(seg[k]) < 1e-9:
                if eye[k] < lo[k] or eye[k] > hi[k]:
                    hit = False
                continue
            a, b = (lo[k] - eye[k]) / seg[k], (hi[k] - eye[k]) / seg[k]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        hit = hit and t0 <= t1
        env.model.geom_rgba[gid] = [*rgba[:3], 0.15 if hit else rgba[3]]


def run(policy, route, *, size=(1920, 1080), seed=0, render=True, instruction="", reply="", label="", speed=1,
        on_frame=None, fps=20, max_seconds=None, leg_seconds=90.0):
    w, h = size
    env = HouseDogEnv(render_mode="rgb_array" if render else None, width=w, height=h, camera_id=-1)
    env.reset(seed=seed)
    viewer, cam_xy, cam_az, next_frame = None, env.xy(), None, 0.0
    obstacles = [i for i in range(env.model.ngeom) if env.model.geom_group[i] == 1]
    boxes = [(i, env.model.geom_pos[i][:2] - env.model.geom_size[i][:2], env.model.geom_pos[i][:2] + env.model.geom_size[i][:2])
             for i in obstacles]
    rgba0 = env.model.geom_rgba[obstacles].copy()

    trail, visited, events = [env.xy()], [], []
    t, fell, flash, full_path = 0.0, False, None, []
    for leg, name in enumerate(route):
        wps = plan_path(env.xy(), PLACES[name][1])
        full_path = [env.xy()] + wps
        leg_t = 0.0
        for wi, wp in enumerate(wps):
            last = wi == len(wps) - 1
            env.set_goal(wp)
            while True:
                _, _, terminated, _, info = env.step(policy(env._get_obs()))
                t += env.dt
                leg_t += env.dt
                trail.append(info["xy"])
                dist = np.linalg.norm(env.goal - env.xy())
                arrived = dist < (REACH_RADIUS if last else WAYPOINT_RADIUS)
                if arrived and last:
                    visited.append(name)
                    events.append({"t": round(t, 2), "reached": name})
                    flash = (f"{NAMES[name]}に到着！", t)
                if terminated:
                    fell = True
                    events.append({"t": round(t, 2), "fell": True})
                if render and t + 1e-9 >= next_frame:
                    next_frame += speed / fps
                    if viewer is None:
                        env.render()
                        viewer = env.mujoco_renderer._get_viewer("rgb_array")
                    cam_xy = (1 - CAM["gain"]) * cam_xy + CAM["gain"] * info["xy"]
                    target_az = np.degrees(yaw_of(env.data.qpos[3:7])) + CAM["rel"]
                    cam_az = target_az if cam_az is None else cam_az + 0.06 * ((target_az - cam_az + 180) % 360 - 180)
                    ground = env.data.qpos[2] - env.body_height()
                    viewer.cam.lookat[:] = [cam_xy[0], cam_xy[1], ground + CAM["z"]]
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = CAM["distance"], cam_az, CAM["elevation"]
                    a = np.radians(cam_az)
                    eye = cam_xy - CAM["distance"] * np.cos(np.radians(CAM["elevation"])) * np.array([np.cos(a), np.sin(a)])
                    _fade_obstacles(env, boxes, rgba0, eye, env.xy())
                    frame = env.render()
                    mm = house_map(300, trail[::3], full_path, name, visited)
                    on_frame(compose(frame, instruction, reply, route, leg, visited, trail, t, label, speed,
                                     flash if flash and t - flash[1] < 1.5 else None, fell, names=NAMES, mm=mm))
                if arrived or terminated or leg_t > leg_seconds or (max_seconds and t >= max_seconds):
                    break
            if fell or leg_t > leg_seconds or (max_seconds and t >= max_seconds):
                break
        if fell or (max_seconds and t >= max_seconds) or name not in visited:
            break
    env.close()
    return {"route": route, "visited": visited, "fell": fell, "time": round(t, 2), "events": events,
            "trail": [list(map(float, p)) for p in trail[::5]]}


def llm_places():
    return {k: (desc, xy) for k, (desc, xy) in PLACES.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="")
    p.add_argument("--route", default="")
    p.add_argument("--reply", default="")
    p.add_argument("--model", default=os.path.join(HERE, "models", "terrain_model.zip"))
    p.add_argument("--label", default="")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--speed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "house.mp4"))
    args = p.parse_args()

    reply, source = args.reply, "manual"
    if args.text and args.route:
        route, source = [r for r in args.route.split(",") if r], "cached"
    elif args.text:
        res = plan(args.text, llm_places(), HOME)
        route, reply, source = res["route"], res["reply"], res["source"]
    else:
        route = [r for r in args.route.split(",") if r]
    print(f"[plan] {source}: {route} {reply}", flush=True)

    policy = Policy(args.model, vecnorm_for(args.model), "terrain")
    writer = imageio.get_writer(args.out, fps=20, quality=8, macro_block_size=8)
    log = run(policy, route, seed=args.seed, instruction=args.text, reply=reply, label=args.label, speed=args.speed,
              on_frame=writer.append_data, max_seconds=args.seconds)
    writer.close()
    log.update({"instruction": args.text, "reply": reply, "planner": source, "model": args.model})
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"[done] visited={log['visited']} fell={log['fell']} time={log['time']}s -> {args.out}")


if __name__ == "__main__":
    main()
