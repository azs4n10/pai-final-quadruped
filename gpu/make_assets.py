"""
GPU（MJX）学習用の部品を作る：大型版の犬型ロボットと、階段の練習場（高さマップ）。

  python gpu/make_assets.py      # gpu/big_dog.xml・gpu/terrain.npy を書き出す
大型版は、自作の犬型ロボットの寸法を SCALE 倍したもの（質量は SCALE^3 倍、関節の強さは SCALE^4 倍）。
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from goal_dog import LEGS, STAND, robot_xml  # noqa: E402

SCALE = 1.4
# 練習場：難しさ 10 段階（行）× 地形 6 種類（列）のタイル。1 タイル 6 m 四方
TILE, N_LEVEL, RES = 6.0, 10, 0.03
COLUMNS = ["stairs_up", "stairs_up", "stairs_down", "stairs_down", "slope", "rough"]
STEP_MIN, STEP_MAX = 0.05, 0.18  # 段差 [m]（難しさ 0〜9）


def _scale_attr(m, s):
    vals = [float(v) * s for v in m.group(2).split()]
    return f'{m.group(1)}="{" ".join(f"{v:.5g}" for v in vals)}"'


def big_robot_xml(s=SCALE):
    """robot_xml() の長さ（pos・size・fromto）を s 倍、質量を s^3 倍にする。"""
    xml = robot_xml()
    xml = re.sub(r'\b(pos|size|fromto)="([^"]+)"', lambda m: _scale_attr(m, s), xml)
    xml = re.sub(r'\bmass="([^"]+)"', lambda m: f'mass="{float(m.group(1)) * s ** 3:.5g}"', xml)
    # 胴体の中心に IMU（角速度・速度のセンサー用）
    xml = xml.replace('<freejoint name="root"/>', '<freejoint name="root"/>\n      <site name="imu" pos="0 0 0"/>', 1)
    for n, _, _ in LEGS:  # 足先の位置（足が地面からどれだけ上がったかの計算用）
        xml = xml.replace(f'<geom name="{n}_foot"', f'<site name="{n}_foot_site" pos="0 0 {-0.21 * s:.5g}"/>\n            <geom name="{n}_foot"', 1)
    return xml


def stairs_profile(r, step, tread, up):
    """中心からの距離 r（チェビシェフ距離）での高さ。中心 1 m 四方は平らな踊り場。"""
    n = np.clip(np.floor((r - 0.5) / tread) + 1, 0, None)
    n = np.minimum(n, 9)  # 9 段で頭打ち（その先は平ら）
    return (1 if up else -1) * n * step


def make_terrain(seed=0):
    rng = np.random.default_rng(seed)
    nx = int(round(TILE * len(COLUMNS) / RES)) + 1
    ny = int(round(TILE * N_LEVEL / RES)) + 1
    xs = np.arange(nx) * RES
    ys = np.arange(ny) * RES
    X, Y = np.meshgrid(xs, ys)
    H = np.zeros_like(X)
    for row in range(N_LEVEL):
        d = row / (N_LEVEL - 1)
        for col, kind in enumerate(COLUMNS):
            m = (X >= col * TILE) & (X < (col + 1) * TILE) & (Y >= row * TILE) & (Y < (row + 1) * TILE)
            cx, cy = (col + 0.5) * TILE, (row + 0.5) * TILE
            r = np.maximum(np.abs(X - cx), np.abs(Y - cy))
            if kind.startswith("stairs"):
                step = STEP_MIN + (STEP_MAX - STEP_MIN) * d
                tread = 0.30 if col % 2 == 0 else 0.26
                # 上り：中心が低く外へ上がる（中心から出発して上る）。下り：中心が高く外へ下がる
                h = stairs_profile(r, step, tread, up=kind == "stairs_up")
            elif kind == "slope":
                h = np.clip(r - 0.5, 0, 2.5) * np.tan(np.radians(5 + 20 * d))
            else:
                h = rng.uniform(-1, 1, X.shape) * (0.01 + 0.05 * d)
            H[m] = h[m]
    H -= H.min()
    return H.astype(np.float32)


def scene_xml(H):
    ny, nx = H.shape
    sx, sy = (nx - 1) * RES / 2, (ny - 1) * RES / 2
    top = float(H.max()) + 1e-3
    acts = "\n".join(f'<position name="{n}_{j}" joint="{n}_{j}"/>' for n, _, _ in LEGS for j in ("abd", "hip", "knee"))
    kp, force = 60 * SCALE ** 4, 30 * SCALE ** 4
    stand = " ".join(f"{v:.4f}" for v in STAND)
    return f"""<mujoco model="big_dog_stairs">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" iterations="4" ls_iterations="8"/>
  <default>
    <joint damping="{0.8 * SCALE ** 4:.3g}" armature="{0.02 * SCALE ** 5:.3g}"/>
    <geom contype="0" conaffinity="0" condim="3" friction="1 0.5 0.5"/>
    <position kp="{kp:.4g}" forcerange="-{force:.4g} {force:.4g}" ctrlrange="-3 3"/>
    <default class="deco"><geom contype="0" conaffinity="0" mass="0" group="2"/></default>
  </default>
  <asset>
    <hfield name="terrain" nrow="{ny}" ncol="{nx}" size="{sx} {sy} {top:.4f} 0.1"/>
    <material name="shell"/>
    <material name="metal"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" pos="{sx} {sy} 0" contype="1" conaffinity="1"/>
    {big_robot_xml()}
  </worldbody>
  <actuator>
{acts}
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <velocimeter name="local_linvel" site="imu"/>
    <framezaxis name="upvector" objtype="site" objname="imu"/>
  </sensor>
  <keyframe>
    <key name="home" qpos="1 1 {0.36 * SCALE:.4f} 1 0 0 0 {stand}" ctrl="{stand}"/>
  </keyframe>
</mujoco>
"""


if __name__ == "__main__":
    H = make_terrain()
    np.save(os.path.join(HERE, "terrain.npy"), H)
    xml = scene_xml(H)
    # 接触判定：足（球）・すね（カプセル）・胴体だけを地面と当てる。見た目の部品・太もも・頭は当てない
    for n, _, _ in LEGS:
        xml = xml.replace(f'<geom name="{n}_foot" type="sphere"', f'<geom name="{n}_foot" type="sphere" contype="1" conaffinity="0"', 1)
    # 胴体の当たり判定は、箱より計算の軽いカプセル（胴体と同じ長さ・太さ）で代用する（見た目は箱のまま）
    xml = xml.replace('<site name="imu" pos="0 0 0"/>', '<site name="imu" pos="0 0 0"/>\n      <geom name="torso_col" type="capsule" '
                      f'fromto="{-0.26 * SCALE + 0.084:.4f} 0 0 {0.26 * SCALE - 0.084:.4f} 0 0" size="0.084" mass="0" '
                      'contype="1" conaffinity="0" group="5" rgba="0 0 0 0"/>', 1)
    xml = re.sub(r'(<body name="(\w\w)_calf"[^>]*>\s*<joint[^>]*/>\s*<geom[^>]*/>\s*)<geom type="capsule"',
                 r'\1<geom name="\2_shin" type="capsule" contype="1" conaffinity="0"', xml)
    with open(os.path.join(HERE, "big_dog.xml"), "w", encoding="utf-8") as f:
        f.write(xml)
    print(H.shape, float(H.max()), xml.count('contype="1"'))
