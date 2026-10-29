"""
2 階建ての家で、日本語の指示 → LLM で場所の順番 → 経路計画で経由地（階段を含む）→ 速度指令 → 歩行方策 で動かして録画する。

  python home_demo.py --text "玄関に届いた荷物を、2 階の寝室に運んで" --out runs/home1.mp4
  python home_demo.py --route dock,bedroom --pickup dock --dropoff bedroom
荷物は、受け取る場所で家の人が背中のカゴに載せ、届け先で降ろす想定（画面では箱の表示を切り替える）。
"""
import argparse
import json
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import home
from big_dog import BigDog, follow_command, load_policy
from demo import compose
from overlay import INK, LIGHT, RED, SLATE, YELLOW, font

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {k: v[0] for k, v in home.PLACES.items()}
WAYPOINT_RADIUS, GOAL_RADIUS = 0.5, 0.6
CAM = dict(distance=3.2, z=0.35, gain=0.2, elevation=-28.0, rel=35.0)  # 斜め後ろから、進む先が見える向き
DT = 0.02


def floor_maps(size, trail, path, target, visited, level):
    """1 階と 2 階の間取り図を並べ、今いる階を太枠で示す。"""
    w = size
    x0, x1, y0, y1 = home.X0 - 0.3, home.X1 + 0.3, home.Y0 - 0.3, home.Y1 + 0.3
    h = int(w * (y1 - y0) / (x1 - x0))
    img = Image.new("RGBA", (w, 2 * h + 70), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for k, lv in enumerate((2, 1)):  # 上に 2 階、下に 1 階
        oy = k * (h + 35) + 30
        d.rectangle((0, oy - 30, w - 1, oy + h - 1), fill=(255, 255, 255, 235), outline=(*INK, 255), width=4 if lv == level else 1)
        d.text((10, oy - 15), f"{lv} 階", font=font(22, "bold"), fill=(*INK, 255), anchor="lm")

        def px(x, y):
            return ((x - x0) / (x1 - x0) * w, oy + (y1 - y) / (y1 - y0) * h)

        def rect(a, b, c, e, **kw):
            p, q = px(a, e), px(b, c)
            d.rectangle((p[0], p[1], q[0], q[1]), **kw)

        rect(home.X0, home.X1, home.Y0, home.Y1, fill=(*LIGHT, 255))
        rect(home.SX0, home.SX1, home.SY0, home.SY1, fill=(200, 200, 200, 255))  # 階段
        zlo, zhi = (0.05, home.FH - 0.1) if lv == 1 else (home.FH + 0.05, home.FH + 3)
        for _, b in home._furniture():
            if b is not None and b[4] < zhi and b[5] > zlo:
                rect(*b[:4], fill=(190, 175, 150, 255))
        for b in home._walls() + home._rails():
            if b[4] < zhi and b[5] > zlo:
                rect(*b[:4], fill=(*INK, 255))
        for key, (_, plv, (x, y)) in home.PLACES.items():
            if plv != lv:
                continue
            cx, cy = px(x, y)
            col = (*SLATE, 255) if key in visited else ((*YELLOW, 255) if key == target else (255, 255, 255, 255))
            d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=col, outline=(*INK, 255), width=2)
        seg = [px(x, y) for x, y, l in path if l == lv]
        if len(seg) >= 2:
            d.line(seg, fill=(*SLATE, 200), width=3)
        tr = [px(x, y) for x, y, l in trail if l == lv]
        if len(tr) >= 2:
            d.line(tr, fill=(*RED, 255), width=4)
        if trail and trail[-1][2] == lv:
            cx, cy = px(*trail[-1][:2])
            d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(255, 255, 255, 255), outline=(*RED, 255), width=3)
    return img


class Scene:
    """家のモデル・扉・荷物・カメラ・描画をまとめて扱う。"""

    def __init__(self, render=True, size=(1920, 1080), policy_path=None):
        self.m = mujoco.MjModel.from_xml_path(home.build_xml())
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.dog = BigDog(self.m, self.d, load_policy(policy_path))
        self.doors = [self.m.body(f"door_{n}").mocapid[0] for n, *_ in home.DOORS]
        self.door_open = np.zeros(len(home.DOORS))
        self.parcel = self.m.body("parcel").mocapid[0]
        self.goal = self.m.body("goal").mocapid[0]
        self.cargo = self.m.geom("cargo").id
        self.renderer = mujoco.Renderer(self.m, size[1], size[0]) if render else None
        if render:
            self.opt = mujoco.MjvOption()
            self.cam = mujoco.MjvCamera()
            self.upper = [i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == 0 and self.m.geom_pos[i][2] > home.FH - home.SLAB - 0.01]
            self.upper_doors = [i for i in range(self.m.ngeom) if self.m.body_mocapid[self.m.geom_bodyid[i]] in
                                [self.doors[k] for k, dd in enumerate(home.DOORS) if dd[1] == 2]]
            self.walls = [i for i in range(self.m.ngeom) if self.m.geom_group[i] == 1 and self.m.geom_bodyid[i] == 0]
            self.rgba0 = self.m.geom_rgba.copy()
            self.cam_xy, self.cam_az = self.d.qpos[:2].copy(), None
        mujoco.mj_forward(self.m, self.d)
        self.update_doors(0.0)

    def level(self):
        return 2 if self.d.qpos[2] > home.FH * 0.6 else 1

    def update_doors(self, dt):
        """ロボットが 1.8 m 以内（同じ階）に来た扉を開け、離れたら閉める（毎秒 1 m で動く）。"""
        xy, lv = self.d.qpos[:2], self.level()
        for k, (name, dlv, x, y0, y1, _) in enumerate(home.DOORS):
            near = dlv == lv and np.linalg.norm(xy - [x, (y0 + y1) / 2]) < 1.8
            target = 1.0 if near else 0.0
            self.door_open[k] += np.clip(target - self.door_open[k], -dt / (y1 - y0), dt / (y1 - y0))
            self.d.mocap_pos[self.doors[k]] = [x, (y0 + y1) / 2 - self.door_open[k] * (y1 - y0), home.door_base_z(dlv) + home.DOOR_H / 2]

    def set_goal(self, x, y, lv):
        self.d.mocap_pos[self.goal] = [x, y, home.door_base_z(lv)]

    def show_parcel(self, where):
        """where：None（どこにも無い）／"robot"（カゴの中）／場所のキー（その場所の床に置いてある）。"""
        self.m.geom_rgba[self.cargo, 3] = 1.0 if where == "robot" else 0.0
        if where in home.PLACES:
            _, lv, (x, y) = home.PLACES[where]
            self.d.mocap_pos[self.parcel] = [x + 0.55, y - 0.35, home.door_base_z(lv) + 0.1]
        else:
            self.d.mocap_pos[self.parcel] = [0, 0, -5]

    def render(self):
        lv = self.level()
        self.m.geom_rgba[:] = self.rgba0
        self.m.geom_rgba[self.cargo, 3] = self._cargo_alpha
        if lv == 1:  # 1 階にいる間は 2 階を外して見せる
            self.m.geom_rgba[self.upper, 3] = 0.0
            self.m.geom_rgba[self.upper_doors, 3] = 0.0
        self.cam_xy = (1 - CAM["gain"]) * self.cam_xy + CAM["gain"] * self.d.qpos[:2]
        target_az = np.degrees(self.dog.yaw()) + CAM["rel"]
        self.cam_az = target_az if self.cam_az is None else self.cam_az + 0.06 * ((target_az - self.cam_az + 180) % 360 - 180)
        base = self.d.qpos[2] - self.dog.body_height()
        self.cam.lookat[:] = [*self.cam_xy, base + CAM["z"]]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = CAM["distance"], self.cam_az, CAM["elevation"]
        # カメラとロボットの間の壁・手すりは半透明にする
        a = np.radians(self.cam_az)
        eye = self.cam_xy - CAM["distance"] * np.cos(np.radians(CAM["elevation"])) * np.array([np.cos(a), np.sin(a)])
        seg = self.d.qpos[:2] - eye
        for i in self.walls:
            lo = self.m.geom_pos[i][:2] - self.m.geom_size[i][:2]
            hi = self.m.geom_pos[i][:2] + self.m.geom_size[i][:2]
            t0, t1 = 0.0, 1.0
            for k in range(2):
                if abs(seg[k]) < 1e-9:
                    if not lo[k] <= eye[k] <= hi[k]:
                        t0 = 2.0
                    continue
                u, v = (lo[k] - eye[k]) / seg[k], (hi[k] - eye[k]) / seg[k]
                t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
            if t0 <= t1 and self.m.geom_rgba[i, 3] > 0:
                self.m.geom_rgba[i, 3] = 0.08
        self.renderer.update_scene(self.d, self.cam, self.opt)
        return self.renderer.render()

    _cargo_alpha = 0.0


def run(route, *, pickup=None, dropoff=None, render=True, instruction="", reply="", label="", speed=2,
        on_frame=None, fps=20, max_seconds=None, leg_seconds=120.0, seed=0, size=(1920, 1080)):
    sc = Scene(render=render, size=size)
    rng = np.random.default_rng(seed)
    sc.d.qpos[0:2] += rng.uniform(-0.05, 0.05, 2)
    mujoco.mj_forward(sc.m, sc.d)
    if pickup:
        sc.show_parcel(pickup)
    lv = 1
    trail, visited, events = [(*sc.d.qpos[:2], 1)], [], []
    t, fell, flash, next_frame, full_path = 0.0, False, None, 0.0, []
    for leg, name in enumerate(route):
        _, glv, goal = home.PLACES[name]
        wps = home.plan_path(sc.d.qpos[:2].copy(), sc.level(), np.array(goal), glv)
        full_path = [(*sc.d.qpos[:2], sc.level())] + wps
        leg_t = 0.0
        for wi, (wx, wy, wl) in enumerate(wps):
            last = wi == len(wps) - 1
            sc.set_goal(wx, wy, wl)
            while True:
                cmd = follow_command(sc.d.qpos[:2], sc.dog.yaw(), (wx, wy), v_max=0.7)
                sc.dog.step(cmd)
                sc.update_doors(DT)
                t += DT
                leg_t += DT
                trail.append((*sc.d.qpos[:2], sc.level()))
                dist = np.hypot(wx - sc.d.qpos[0], wy - sc.d.qpos[1])
                arrived = dist < (GOAL_RADIUS if last else WAYPOINT_RADIUS) and (sc.level() == wl)
                if arrived and last:
                    visited.append(name)
                    events.append({"t": round(t, 2), "reached": name})
                    msg = f"{NAMES[name]}に到着"
                    if name == pickup:
                        sc.show_parcel("robot")
                        sc._cargo_alpha = 1.0
                        msg = f"{NAMES[name]}で荷物を受け取りました"
                    elif name == dropoff:
                        sc.show_parcel(name)
                        sc._cargo_alpha = 0.0
                        msg = f"{NAMES[name]}に荷物を届けました"
                    flash = (msg, t)
                if sc.dog.fallen():
                    fell = True
                    events.append({"t": round(t, 2), "fell": True})
                if render and t + 1e-9 >= next_frame:
                    next_frame += speed / fps
                    frame = sc.render()
                    mm = floor_maps(300, trail[::3], full_path, name, visited, sc.level())
                    on_frame(compose(frame, instruction, reply, route, leg, visited, [p[:2] for p in trail], t, label, speed,
                                     flash if flash and t - flash[1] < 1.8 else None, fell, names=NAMES, mm=mm))
                if arrived or fell or leg_t > leg_seconds or (max_seconds and t >= max_seconds):
                    break
            if fell or leg_t > leg_seconds or (max_seconds and t >= max_seconds):
                break
        if fell or (max_seconds and t >= max_seconds) or name not in visited:
            break
    return {"route": route, "visited": visited, "fell": fell, "time": round(t, 2), "events": events,
            "trail": [list(map(float, p)) for p in trail[::5]]}


def llm_places():
    return {k: (f"{desc}（{lv} 階）", xy) for k, (desc, lv, xy) in home.PLACES.items()}


def main():
    from planner import plan

    p = argparse.ArgumentParser()
    p.add_argument("--text", default="")
    p.add_argument("--route", default="")
    p.add_argument("--reply", default="")
    p.add_argument("--pickup", default=None)
    p.add_argument("--dropoff", default=None)
    p.add_argument("--label", default="")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--speed", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "home.mp4"))
    args = p.parse_args()

    reply, source = args.reply, "manual"
    pickup, dropoff = args.pickup, args.dropoff
    if args.text and not args.route:
        res = plan(args.text, llm_places(), home.HOME)
        route, reply, source = res["route"], res["reply"], res["source"]
        pickup = pickup or res.get("pickup")
        dropoff = dropoff or res.get("dropoff")
    else:
        route = [r for r in args.route.split(",") if r]
        source = "cached" if args.text else "manual"
    print(f"[plan] {source}: {route} pickup={pickup} dropoff={dropoff} {reply}", flush=True)
    writer = imageio.get_writer(args.out, fps=20, quality=8, macro_block_size=8)
    log = run(route, pickup=pickup, dropoff=dropoff, instruction=args.text, reply=reply, label=args.label,
              speed=args.speed, on_frame=writer.append_data, max_seconds=args.seconds, seed=args.seed)
    writer.close()
    log.update({"instruction": args.text, "reply": reply, "planner": source, "pickup": pickup, "dropoff": dropoff})
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"[done] visited={log['visited']} fell={log['fell']} time={log['time']}s -> {args.out}")


if __name__ == "__main__":
    main()
