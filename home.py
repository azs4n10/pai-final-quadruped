"""
2 階建ての家（大型版の犬型ロボット用）と、階をまたぐ経路計画。

- 1 階：玄関ホール（充電ステーション・靴箱・2 階への階段）、LDK（リビング・ダイニング・キッチン）、書斎、和室
- 2 階：寝室、子ども部屋、ホール。階段の吹き抜けのまわりに手すり
- 階段：1 段 RISE・踏み面 TREAD・N_STEPS 段（玄関ホールの西側を北へ上る）
- 床・階段は箱（グループ 3：足元の高さセンサーのレイが当たる）。壁・家具・手すりはグループ 1（経路計画で避ける）
- 経路計画：階ごとに A*。階をまたぐときは、階段の下と上の通過点を必ず通して、階段をまっすぐ上り下りする
"""
import heapq
import os
import re

import numpy as np

import furniture as F

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "assets", "home.xml")
BIG_DOG = os.path.join(HERE, "gpu", "big_dog.xml")

RISE, TREAD, N_STEPS = 0.15, 0.30, 14
FH = RISE * N_STEPS  # 2 階の床の高さ
SLAB = 0.12  # 2 階の床の厚さ（上面が FH＝14 段目の高さ）
WALL_1F, WALL_2F, WALL_IN = FH - SLAB, 0.9, 1.0  # 2 階の壁と 1 階の内壁は中が見えるよう低め（屋根なし）
T = 0.05
X0, X1, Y0, Y1 = -6.0, 6.0, 0.0, 8.0
# 階段（玄関ホールの西側、y が増える向きに上る）
SX0, SX1, SY0 = -1.45, -0.35, 2.0
SY1 = SY0 + N_STEPS * TREAD  # 階段の上端（2 階の床の縁）

# 場所：キー -> (日本語名, 階, (x, y))
PLACES = {
    "dock": ("充電ステーション", 1, (0.35, 1.0)),
    "living": ("リビング", 1, (4.2, 3.3)),
    "kitchen": ("キッチン", 1, (4.6, 6.3)),
    "study": ("書斎", 1, (-3.6, 2.0)),
    "washitsu": ("和室", 1, (-3.6, 5.0)),
    "bedroom": ("寝室", 2, (3.1, 5.6)),
    "kids": ("子ども部屋", 2, (3.6, 2.0)),
    "hall2": ("2 階のホール", 2, (0.5, 7.2)),
    "library": ("書庫", 2, (-3.8, 1.2)),
}
HOME = "dock"


def floor_z(level):
    return 0.0 if level == 1 else FH


# --- 壁 ---
def _walls():
    """(x0, x1, y0, y1, z0, z1) のリスト。出入口は壁を分けて開ける。"""
    w = []

    def hwall(y, x0, x1, z0, z1):
        w.append((x0 - T, x1 + T, y - T, y + T, z0, z1))

    def vwall(x, y0, y1, z0, z1):
        w.append((x - T, x + T, y0 - T, y1 + T, z0, z1))

    a, b = 0.0, WALL_1F
    # 1 階：外壁は 2 階の床まで（玄関扉は閉じた状態で描く）。内壁は中が見えるよう低め
    hwall(Y0, X0, X1, a, b), hwall(Y1, X0, X1, a, b), vwall(X0, Y0, Y1, a, b), vwall(X1, Y0, Y1, a, b)
    b = WALL_IN
    # 玄関ホール（x∈[-1.5, 1.5]）と LDK の間：出入口 y∈[2.8, 4.2]
    vwall(1.5, Y0, 2.8, a, b), vwall(1.5, 4.2, Y1, a, b)
    # 玄関ホールと西の部屋の間（階段の脇）：書斎の出入口 y∈[0.2, 1.4]、和室の出入口 y∈[6.6, 7.8]
    vwall(-1.5, 1.4, 6.6, a, b)
    vwall(-1.5, 7.8, Y1, a, b)
    # 書斎と和室の間
    hwall(4.0, X0, -1.5, a, b)
    # LDK の中：リビングとダイニングの間に腰壁は置かない（ひと続き）
    c, d = FH, FH + WALL_2F
    # 2 階：外壁
    hwall(Y0, X0, X1, c, d), hwall(Y1, X0, X1, c, d), vwall(X0, Y0, Y1, c, d), vwall(X1, Y0, Y1, c, d)
    # 2 階の廊下（x∈[-0.3, 1.5]）と東の部屋：子ども部屋の出入口 y∈[2.4, 3.6]、寝室の出入口 y∈[6.2, 7.4]
    vwall(1.5, Y0, 2.4, c, d), vwall(1.5, 3.6, 6.2, c, d), vwall(1.5, 7.4, Y1, c, d)
    hwall(4.0, 1.5, X1, c, d)  # 子ども部屋と寝室の間
    # 2 階の西（書庫）：出入口 y∈[0.3, 1.5]
    vwall(-1.5, Y0, 0.3, c, d), vwall(-1.5, 1.5, SY0 - 0.1, c, d)
    vwall(-1.5, SY1 + 0.1, Y1, c, d)
    return w


def _rails():
    """2 階の吹き抜け（階段の上）のまわりの手すり。(x0, x1, y0, y1)。"""
    c = FH
    return [(SX1 + 0.02, SX1 + 0.08, SY0, SY1, c, c + 0.9),  # 廊下側
            (SX0 - 0.08, SX0 - 0.02, SY0, SY1, c, c + 0.9),  # 書庫側
            (SX0, SX1 + 0.08, SY0 - 0.06, SY0, c, c + 0.9)]  # 階段の下側の縁


# --- 家具 ---
def _furniture():
    """(見た目の XML, 当たり判定の箱 (x0, x1, y0, y1, z0, z1)) のリスト。"""
    out = []

    def add(res, z0=0.0):
        parts, ext = res
        out.append((parts, None if ext is None else (ext[0], ext[1], ext[2], ext[3], z0, z0 + ext[4])))

    z2 = FH
    # 1 階：玄関ホール
    add(F.shoe_rack(1.0, 0.3, 0.0, yaw=0))
    add(F.plant(1.2, 7.6, 0.0, 0.8))
    # リビング
    add(F.rug(4.2, 1.9, 0.0, 1.3, 1.0))
    add(F.sofa(3.0, 1.9, 0.0, yaw=-90, w=0.85))
    add(F.coffee_table(4.2, 1.9, 0.0, yaw=90))
    add(F.tv_stand(5.7, 1.9, 0.0, yaw=90))
    add(F.plant(5.6, 0.4, 0.0, 1.1))
    add(F.plant(2.0, 0.4, 0.0, 0.9))
    # ダイニング・キッチン
    add(F.dining_set(3.0, 5.6, 0.0))
    add(F.kitchen(4.0, 7.6, 0.0, yaw=180, w=1.1))
    add(F.fridge(5.6, 7.55, 0.0, yaw=180))
    # 書斎
    add(F.desk(-3.6, 0.45, 0.0, yaw=0))
    add(F.bookshelf(-5.8, 2.2, 0.0, yaw=90))
    add(F.bookshelf(-3.0, 3.8, 0.0, yaw=180))
    # 和室（畳の上に座卓）
    add(F.rug(-3.75, 6.0, 0.0, 2.1, 1.85, "MatTatami"))
    add(F.coffee_table(-3.6, 6.4, 0.0))
    # 2 階：寝室
    add(F.bed(4.85, 6.2, z2, yaw=90), z2)
    add(F.nightstand(5.6, 7.55, z2, yaw=180), z2)
    add(F.plant(2.0, 4.4, z2, 0.8), z2)
    # 子ども部屋
    add(F.rug(3.6, 2.0, z2, 1.2, 0.9), z2)
    add(F.bed(5.2, 1.6, z2, yaw=0, w=0.5, l=0.95), z2)
    add(F.desk(2.6, 0.45, z2, yaw=0), z2)
    add(F.toy_box(3.4, 3.5, z2, yaw=180), z2)
    # 書庫（2 階の西）
    add(F.rug(-3.8, 4.0, z2, 1.2, 1.4), z2)
    add(F.bookshelf(-5.8, 2.6, z2, yaw=90), z2)
    add(F.bookshelf(-5.8, 4.0, z2, yaw=90), z2)
    add(F.bookshelf(-5.8, 5.4, z2, yaw=90), z2)
    add(F.sofa(-3.4, 4.0, z2, yaw=-90, w=0.45), z2)
    add(F.plant(-5.6, 7.5, z2, 0.9), z2)
    return out


# --- 自動の引き戸（ロボットが近づくと開く） ---
# (名前, 階, 壁の位置 x, 開口の始まり y0, 開口の終わり y1, 材質)。扉は y0 側の壁の中へ引き込まれる
DOORS = [
    ("ldk", 1, 1.5, 2.8, 4.2, "MatWoodFurniture"),
    ("study", 1, -1.5, 0.0, 1.4, "MatWoodFurniture"),
    ("washitsu", 1, -1.5, 6.6, 7.8, "MatShoji"),
    ("kids", 2, 1.5, 2.4, 3.6, "MatWoodFurniture"),
    ("bedroom", 2, 1.5, 6.2, 7.4, "MatWoodFurniture"),
    ("library", 2, -1.5, 0.3, 1.5, "MatWoodFurniture"),
]
DOOR_H = 0.95  # 扉の高さ（中が見えるよう内壁と同じくらい）


def door_base_z(level):
    return floor_z(level)


def door_xml():
    out = []
    for name, level, x, y0, y1, mat in DOORS:
        z = door_base_z(level) + DOOR_H / 2
        out.append(f'<body name="door_{name}" mocap="true" pos="{x} {(y0 + y1) / 2} {z}">'
                   f'<geom type="box" size="0.025 {(y1 - y0) / 2 - 0.01} {DOOR_H / 2}" rgba="1 1 1 1" material="{mat}" class="obstacle"/>'
                   f'<geom type="box" size="0.03 0.02 0.02" pos="0 {(y1 - y0) / 2 - 0.12} 0.05" rgba="0.3 0.3 0.3 1" class="deco"/></body>')
    return out


# --- XML ---
def _box(name, x0, x1, y0, y1, z0, z1, rgba, cls, mat=""):
    m = f' material="{mat}"' if mat else ""
    return (f'<geom name="{name}" type="box" size="{(x1 - x0) / 2:.4f} {(y1 - y0) / 2:.4f} {(z1 - z0) / 2:.4f}" '
            f'pos="{(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {(z0 + z1) / 2:.4f}" rgba="{rgba}" class="{cls}"{m}/>')


def floors():
    """歩く面の箱 (名前, x0, x1, y0, y1, z0, z1, 材質)。"""
    z2 = FH
    b = [("floor1", X0, X1, Y0, Y1, -0.05, 0.0, "MatWood")]
    # 2 階の床（階段の吹き抜けを除く）
    b += [("floor2_w", X0, SX0, Y0, Y1, z2 - SLAB, z2, "MatWood"),
          ("floor2_e", SX1, X1, Y0, Y1, z2 - SLAB, z2, "MatWood"),
          ("floor2_s", SX0, SX1, Y0, SY0, z2 - SLAB, z2, "MatWood"),
          ("floor2_n", SX0, SX1, SY1, Y1, z2 - SLAB, z2, "MatWood")]
    for i in range(1, N_STEPS):  # 段（i 段目の上面は i*RISE）
        y0 = SY0 + (i - 1) * TREAD
        b.append((f"step{i}", SX0, SX1, y0, y0 + TREAD, 0.0, i * RISE, "MatStairs"))
    b.append((f"step{N_STEPS}", SX0, SX1, SY1 - TREAD, SY1, 0.0, z2, "MatStairs"))
    return b


HF_RES = 0.03  # 学習した地形（gpu/make_assets.py）と同じ間隔


def stairs_hfield():
    """階段の物理用の高さマップ：(x0, x1, y0, y1, 高さの 2 次元配列（行が y）)。"""
    x0, x1, y0, y1 = SX0, SX1, SY0 - 0.15, SY1 + 0.15
    xs = np.arange(x0, x1 + 1e-9, HF_RES)
    ys = np.arange(y0, y1 + 1e-9, HF_RES)
    X, Y = np.meshgrid(xs, ys)
    n = np.floor((Y - SY0) / TREAD) + 1  # i 段目（1〜N）
    h = np.where(Y < SY0, 0.0, np.minimum(n, N_STEPS) * RISE)
    return xs[0], xs[-1], ys[0], ys[-1], h


def build_xml():
    src = open(BIG_DOG, encoding="utf-8").read()
    robot = re.search(r'(<body name="torso".*?</body>\s*)</worldbody>', src, re.S).group(1)
    head, tail = src.split("<worldbody>")[0], src.split("</worldbody>")[1]
    head = re.sub(r'<hfield[^>]*/>', '', head)
    parts = []
    for name, x0, x1, y0, y1, z0, z1, mat in floors():  # 物理とレイ用（非表示）と、見た目用の 2 つ
        if not name.startswith("step"):  # 階段の物理は下の高さマップ（学習した地形と同じ 3 cm 間隔）
            parts.append(_box(name, x0, x1, y0, y1, z0, z1, "1 1 1 1", "ground", mat))
        parts.append(_box(name + "_vis", x0, x1, y0, y1, z0, z1, "1 1 1 1", "ground_vis", mat))
    hx0, hx1, hy0, hy1, elev = stairs_hfield()
    parts.append(f'<geom name="stairs" type="hfield" hfield="stairs" pos="{(hx0 + hx1) / 2:.4f} {(hy0 + hy1) / 2:.4f} 0" class="ground"/>')
    for i, (x0, x1, y0, y1, z0, z1) in enumerate(_walls()):
        parts.append(_box(f"wall{i}", x0, x1, y0, y1, z0, z1, "1 1 1 1", "obstacle", "MatWall"))
    for i, (x0, x1, y0, y1, z0, z1) in enumerate(_rails()):
        parts.append(_box(f"rail{i}", x0, x1, y0, y1, z0, z1, "0.42 0.32 0.24 1", "obstacle"))
    for i, (vis, box) in enumerate(_furniture()):
        parts += vis
        if box is not None:
            x0, x1, y0, y1, z0, z1 = box
            parts.append(_box(f"furn{i}", x0, x1, y0, y1, z0, z1, "0 0 0 0", "obstacle_hidden"))
    # 玄関扉（閉じた状態の見た目）
    parts.append('<geom type="box" pos="0 -0.02 1.0" size="0.5 0.03 1.0" rgba="0.45 0.32 0.22 1" class="deco" material="MatWoodFurniture"/>')
    # 充電ステーション
    x, y = PLACES["dock"][2]
    parts.append(f'<geom type="cylinder" pos="{x} {y} 0.005" size="0.45 0.005" rgba="0.32 0.35 0.40 1" class="deco"/>')
    parts.append(f'<geom type="cylinder" pos="{x} {y} 0.006" size="0.38 0.005" rgba="1 0.85 0.4 1" class="deco"/>')
    parts.append(f'<geom type="cylinder" pos="{x} {y} 0.007" size="0.34 0.005" rgba="0.32 0.35 0.40 1" class="deco"/>')

    from house import MATERIALS as HOUSE_MATERIALS, TEX_DIR, make_textures

    if not os.path.exists(os.path.join(TEX_DIR, "fabric.png")):
        make_textures()
    top = float(elev.max())
    hfield = (f'<hfield name="stairs" nrow="{elev.shape[0]}" ncol="{elev.shape[1]}" '
              f'size="{(hx1 - hx0) / 2:.4f} {(hy1 - hy0) / 2:.4f} {top:.4f} 0.05" '
              f'elevation="{" ".join(f"{v / top:.4f}" for v in elev[::-1].ravel())}"/>')  # XML では先頭の行が +y 側
    assets = (hfield + "\n    " + HOUSE_MATERIALS + "\n    " + F.MATERIALS + """
    <texture name="tex_tatami" type="2d" builtin="flat" rgb1="0.80 0.78 0.55" width="64" height="64"/>
    <material name="MatTatami" texture="tex_tatami" specular="0.05"/>
    <texture name="tex_shoji" type="2d" builtin="checker" rgb1="0.97 0.95 0.90" rgb2="0.97 0.95 0.90" mark="edge" markrgb="0.55 0.42 0.30" width="256" height="256"/>
    <material name="MatShoji" texture="tex_shoji" texrepeat="3 3" texuniform="true"/>
    <material name="MatStairs" texture="tex_wood" texrepeat="1 1" texuniform="true" rgba="0.85 0.75 0.62 1" specular="0.3"/>
    <material name="shell" specular="0.5" shininess="0.6"/>
    <material name="metal" specular="0.8" shininess="0.9"/>
    <texture builtin="gradient" height="512" rgb1="0.45 0.66 0.90" rgb2="0.92 0.95 0.99" type="skybox" width="512"/>""")
    head = head.replace("<asset>", "<asset>\n    " + assets).replace('<material name="shell"/>', "").replace('<material name="metal"/>', "")
    head = head.replace("<default>", """<default>
    <default class="ground"><geom contype="1" conaffinity="1" group="3"/></default>
    <default class="ground_vis"><geom contype="0" conaffinity="0" group="0" mass="0"/></default>
    <default class="obstacle"><geom contype="1" conaffinity="1" group="1"/></default>
    <default class="obstacle_hidden"><geom contype="1" conaffinity="1" group="4"/></default>""", 1)
    head = head.replace('<mujoco model="big_dog_stairs">', '<mujoco model="home">')
    head = head.replace("<compiler", '<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/>'
                        '<headlight ambient="0.45 0.45 0.45" diffuse="0.3 0.3 0.3" specular="0.05 0.05 0.05"/></visual>\n  <compiler', 1)
    head = head.replace('<compiler angle="radian"', f'<compiler texturedir="{os.path.join(HERE, "assets").replace(os.sep, "/")}" angle="radian"', 1)
    # 背中のカゴ（見た目だけ）と、載せた荷物（最初は透明）
    basket = ('<geom type="box" pos="-0.06 0 0.105" size="0.2 0.14 0.008" rgba="0.25 0.25 0.27 1" class="deco"/>'
              + "".join(f'<geom type="box" pos="{px} {py} 0.14" size="{sx} {sy} 0.035" rgba="0.25 0.25 0.27 1" class="deco"/>'
                        for px, py, sx, sy in [(-0.06, 0.14, 0.2, 0.006), (-0.06, -0.14, 0.2, 0.006), (0.14, 0, 0.006, 0.14), (-0.26, 0, 0.006, 0.14)])
              + '<geom name="cargo" type="box" pos="-0.06 0 0.21" size="0.17 0.13 0.1" rgba="0.78 0.62 0.42 0" class="deco"/>')
    robot = robot.replace('<site name="imu" pos="0 0 0"/>', '<site name="imu" pos="0 0 0"/>\n      ' + basket + '\n      <light name="lamp" mode="trackcom" pos="0.9 0.6 1.3" dir="-0.5 -0.35 -1" '
                          'cutoff="70" exponent="2" diffuse="0.5 0.48 0.45" specular="0.1 0.1 0.1" castshadow="true"/>', 1)
    parts += door_xml()
    # 荷物（受け取る場所に置いてある箱。受け取ると消えて、背中のカゴに載る）
    parts.append('<body name="parcel" mocap="true" pos="0 0 -5"><geom type="box" size="0.17 0.13 0.1" rgba="0.78 0.62 0.42 1" class="deco"/>'
                 '<geom type="box" size="0.171 0.02 0.101" rgba="0.9 0.85 0.75 1" class="deco"/></body>')
    body = "\n    ".join(parts)
    goal = """<body name="goal" mocap="true" pos="0 0 0">
      <geom type="cylinder" size="0.45 0.008" pos="0 0 0.01" rgba="0.75 0 0 0.3" class="deco"/>
      <geom type="cylinder" size="0.025 0.5" pos="0 0 0.5" rgba="0.75 0 0 1" class="deco"/>
      <geom type="box" size="0.16 0.01 0.1" pos="0.16 0 0.9" rgba="0.75 0 0 1" class="deco"/>
    </body>"""
    xml = f"{head}<worldbody>\n    {body}\n    {robot}\n    {goal}\n  </worldbody>{tail}"
    dx, dy = PLACES[HOME][2]  # 立ち姿勢のキーフレームを、充電ステーションの上に置き直す
    xml = re.sub(r'(<key name="home" qpos=")\S+ \S+ ', lambda mm: f"{mm.group(1)}{dx} {dy} ", xml)
    os.makedirs(os.path.dirname(XML_PATH), exist_ok=True)
    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)
    return XML_PATH


# --- 経路計画 ---
GRID, INFLATE = 0.1, 0.4


def _occupancy(level):
    nx, ny = int((X1 - X0 + 2) / GRID), int((Y1 - Y0 + 2) / GRID)
    cx = X0 - 1 + (np.arange(nx) + 0.5) * GRID
    cy = Y0 - 1 + (np.arange(ny) + 0.5) * GRID
    X, Y = np.meshgrid(cx, cy)
    occ = (X < X0 + 0.1) | (X > X1 - 0.1) | (Y < Y0 + 0.1) | (Y > Y1 - 0.1)
    zlo, zhi = (0.05, FH - SLAB - 0.05) if level == 1 else (FH + 0.05, FH + 3)
    boxes = [b for b in _walls() + _rails() if b[4] < zhi and b[5] > zlo]
    boxes += [b for _, b in _furniture() if b is not None and b[4] < zhi and b[5] > zlo]
    for x0, x1, y0, y1, _, _ in boxes:
        occ |= (X > x0 - INFLATE) & (X < x1 + INFLATE) & (Y > y0 - INFLATE) & (Y < y1 + INFLATE)
    # 階段そのものは階ごとの地図では通れない（階段は通過点で上り下りする）
    occ |= (X > SX0 - 0.05) & (X < SX1 + 0.05) & (Y > SY0 - 0.05) & (Y < SY1 + 0.05) if level == 1 else \
        (X > SX0 - 0.3) & (X < SX1 + 0.3) & (Y > SY0 - 0.3) & (Y < SY1 - 0.05)
    return occ


_OCC = {}


def _cell(p):
    return int((p[1] - (Y0 - 1)) / GRID), int((p[0] - (X0 - 1)) / GRID)


def _point(c):
    return np.array([X0 - 1 + (c[1] + 0.5) * GRID, Y0 - 1 + (c[0] + 0.5) * GRID])


def _free_line(occ, a, b):
    for t in np.linspace(0, 1, int(np.ceil(np.linalg.norm(b - a) / (GRID / 2))) + 1):
        if occ[_cell(a + t * (b - a))]:
            return False
    return True


def _astar(level, start, goal):
    if level not in _OCC:
        _OCC[level] = _occupancy(level)
    occ = _OCC[level].copy()
    s, g = _cell(start), _cell(goal)
    occ[s] = occ[g] = False
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    openq, cost, prev = [(0.0, s)], {s: 0.0}, {}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == g:
            break
        for dr, dc, w in moves:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < occ.shape[0] and 0 <= nb[1] < occ.shape[1]) or occ[nb]:
                continue
            nc = cost[cur] + w
            if nc < cost.get(nb, 1e18):
                cost[nb], prev[nb] = nc, cur
                heapq.heappush(openq, (nc + np.hypot(nb[0] - g[0], nb[1] - g[1]), nb))
    if g not in prev and g != s:
        return [np.asarray(goal, float)]
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    pts = [np.asarray(start, float)] + [_point(c) for c in reversed(cells[1:-1])] + [np.asarray(goal, float)]
    out, i = [], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _free_line(_OCC[level], pts[i], pts[j]):
            j -= 1
        if not out or np.linalg.norm(pts[j] - out[-1]) > 0.05:
            out.append(pts[j])
        i = j
    return out


STAIR_X = (SX0 + SX1) / 2
STAIR_BOTTOM = np.array([STAIR_X, SY0 - 0.7])  # 階段の下（1 階）
STAIR_TOP = np.array([STAIR_X, SY1 + 0.7])  # 階段の上（2 階）


def plan_path(start, level, goal, goal_level):
    """経由地 [(x, y, 階)]（最後が goal）。階が違えば、階段の下と上を通る。"""
    if level == goal_level:
        return [(*p, level) for p in _astar(level, start, goal)]
    if level == 1:
        return ([(*p, 1) for p in _astar(1, start, STAIR_BOTTOM)] + [(*STAIR_TOP, 2)]
                + [(*p, 2) for p in _astar(2, STAIR_TOP, goal)])
    return ([(*p, 2) for p in _astar(2, start, STAIR_TOP)] + [(*STAIR_BOTTOM, 1)]
            + [(*p, 1) for p in _astar(1, STAIR_BOTTOM, goal)])
