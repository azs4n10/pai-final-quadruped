"""
「学習の様子」の映像：練習場（gpu/terrain.npy）の 6 種類の地形で、同じ方策が同時に歩く様子を 2×3 に並べる。

  python train_scene.py --model runs/big_stairs/ckpt/model_060000000.zip --level 6 --out runs/clips_home/train_scene.mp4
"""
import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from big_dog import load_policy
from gpu.make_assets import COLUMNS, STEP_MAX, STEP_MIN
from overlay import INK, font
from stairs_cpu import TILE, BigStairsEnv

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = {"stairs_up": "上り階段", "stairs_down": "下り階段", "slope": "坂", "rough": "凸凹"}


def render_xml():
    """学習用の XML に、空・光・地面の模様を足した表示用の XML。"""
    src = open(os.path.join(HERE, "gpu", "big_dog.xml"), encoding="utf-8").read()
    src = src.replace("<asset>", """<asset>
    <texture builtin="gradient" height="512" rgb1="0.45 0.66 0.90" rgb2="0.92 0.95 0.99" type="skybox" width="512"/>
    <texture builtin="checker" height="512" name="grid" rgb1="0.82 0.81 0.77" rgb2="0.74 0.73 0.69" type="2d" width="512"/>
    <material name="ground" texture="grid" texrepeat="60 100" reflectance="0"/>""", 1)
    src = src.replace('<material name="shell"/>', '<material name="shell" specular="0.5" shininess="0.6"/>')
    src = src.replace('<material name="metal"/>', '<material name="metal" specular="0.8" shininess="0.9"/>')
    src = src.replace('<geom name="floor" type="hfield"', '<geom name="floor" type="hfield" material="ground"', 1)
    src = src.replace("<worldbody>", """<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/></visual>
  <worldbody>
    <light directional="true" dir="-0.4 -0.3 -1" pos="0 0 10" diffuse="0.6 0.58 0.55" castshadow="true"/>""", 1)
    path = os.path.join(HERE, "assets", "big_dog_render.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return path


def place(env, level, col, seed=0):
    """地形のタイルの中央に、+x 向きで立たせる。"""
    env.reset(seed=col + 10 * seed)
    env.d.qpos[0:2] = (np.array([col, level]) + 0.5) * TILE
    env.d.qpos[2] = env.terrain_height(*env.d.qpos[0:2]) + 0.49
    env.d.qpos[3:7] = [1, 0, 0, 0]
    env.d.qvel[:] = 0
    mujoco.mj_forward(env.m, env.d)
    env.cmd, env.until_cmd = np.array([0.6, 0.0, 0.0]), 10 ** 9
    return env._obs()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "train_scene.mp4"))
    args = p.parse_args()
    policy = load_policy(args.model)
    xml = render_xml()
    sims = []
    for col, kind in enumerate(COLUMNS):
        env = BigStairsEnv(gait=len(policy.mean) == 100)
        env.m = mujoco.MjModel.from_xml_path(xml)
        env.m.hfield_data[:] = (env.H / env.m.hfield_size[0, 2]).ravel()
        env.d = mujoco.MjData(env.m)
        env.first = False
        env.fell, env.dist = False, 0.0
        env.level = args.level
        env.col = col
        sims.append((env, place(env, args.level, col), kind, mujoco.Renderer(env.m, 540, 640)))
    rise = STEP_MIN + (STEP_MAX - STEP_MIN) * args.level / 9
    writer = imageio.get_writer(args.out, fps=20, quality=8, macro_block_size=8)
    cam = mujoco.MjvCamera()
    for k in range(int(args.seconds / 0.05)):
        tiles = []
        for i, (env, obs, kind, r) in enumerate(sims):
            for _ in range(int(0.05 / 0.02 + 0.5)):
                obs, _, term, _, _ = env.step(policy(obs))
                center = (np.array([env.col, args.level]) + 0.5) * TILE
                if term or np.max(np.abs(env.d.qpos[0:2] - center)) > 2.5:  # タイルの端まで来たら、学習と同じく中央からやり直す
                    obs = place(env, args.level, env.col, seed=k)
                    break
            sims[i] = (env, obs, kind, r)
            cam.lookat[:] = [*env.d.qpos[:2], env.terrain_height(*env.d.qpos[:2]) + 0.3]
            cam.distance, cam.azimuth, cam.elevation = 3.0, 220, -22
            r.update_scene(env.d, cam)
            img = Image.fromarray(r.render())
            d = ImageDraw.Draw(img)
            text = LABEL[kind] + (f"（段差 {rise * 100:.0f} cm）" if kind.startswith("stairs") else "")
            d.rectangle((10, 10, 20 + d.textlength(text, font=font(26, "bold")), 52), fill=(255, 255, 255))
            d.text((15, 31), text, font=font(26, "bold"), fill=INK, anchor="lm")
            tiles.append(np.asarray(img))
        top, bottom = np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)
        writer.append_data(np.concatenate([top, bottom], 0))
    writer.close()
    print("->", args.out)


if __name__ == "__main__":
    main()
