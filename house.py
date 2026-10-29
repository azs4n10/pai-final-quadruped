"""
家と庭（アスレチック）の環境と、部屋から部屋への経路計画。

- 家：玄関・廊下・リビング・キッチン・寝室・子ども部屋。床は地面より 18 cm 高く、玄関前に 3 段の階段。
  壁は中が見えるよう低め（床から 0.9 m）にし、出入口は幅 1 m の開口。家具は箱で表す。
- 庭：充電ステーション（出発地点）、6 cm × 4 段の階段で上る高台、約 11° の坂で上る高台。
- 経路計画：壁・家具・手すりを地図（10 cm 格子）に描き、ロボットの幅ぶん太らせてから A* で最短経路を探し、
  見通しのきく点を間引いて経由地にする。段差や坂は通れる場所として扱う（歩行方策が自分で越える）。
"""
import heapq
import os

import numpy as np

from terrain_dog import TerrainDogEnv, scene_xml

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "assets", "house_dog.xml")

FLOOR = 0.18  # 家の床の高さ [m]
PORCH_STEP = 0.06  # 玄関前の階段の 1 段の高さ
PORCH_DEPTH = 0.8  # 玄関の扉の前の踊り場の奥行き
WALL_H = 0.9  # 壁の高さ（床から）
T = 0.05  # 壁の厚さの半分

# 地点：キー -> (日本語名, (x, y))
PLACES = {
    "dock": ("充電ステーション", (0.0, -6.0)),
    "genkan": ("玄関", (0.0, 1.3)),
    "living": ("リビング", (4.3, 1.4)),
    "kitchen": ("キッチン", (3.0, 5.4)),
    "bedroom": ("寝室", (-2.6, 5.2)),
    "kids": ("子ども部屋", (-3.6, 2.0)),
    "stairs_top": ("庭の階段の上の高台", (-5.2, -3.8)),
    "slope_top": ("庭の坂の上の高台", (5.2, -3.8)),
}
HOME = "dock"


def _walls():
    """(x0, x1, y0, y1, z0, z1, 色) の箱のリスト。出入口は壁を分けて開ける。"""
    z0, z1 = FLOOR, FLOOR + WALL_H
    w = []

    def hwall(y, x0, x1):  # x 方向の壁
        w.append((x0 - T, x1 + T, y - T, y + T, z0, z1))

    def vwall(x, y0, y1):  # y 方向の壁
        w.append((x - T, x + T, y0 - T, y1 + T, z0, z1))

    # 外壁（南に玄関の出入口 x∈[-0.8, 0.8]）
    hwall(0.0, -6.0, -0.8), hwall(0.0, 0.8, 6.0), hwall(8.0, -6.0, 6.0)
    vwall(-6.0, 0.0, 8.0), vwall(6.0, 0.0, 8.0)
    # 東側：玄関・廊下とリビング・キッチンの間（x=1.2）。出入口 y∈[0.7,1.9]（リビング）、y∈[5.7,6.9]（キッチン）
    vwall(1.2, 0.0, 0.7), vwall(1.2, 1.9, 5.7), vwall(1.2, 6.9, 8.0)
    # リビングとキッチンの間（y=4.5）。出入口 x∈[3.4,4.6]
    hwall(4.5, 1.2, 3.4), hwall(4.5, 4.6, 6.0)
    # 西側：玄関・廊下と子ども部屋・寝室の間（x=-1.2）。出入口 y∈[0.9,2.1]（子ども部屋）、y∈[5.7,6.9]（寝室）
    vwall(-1.2, 0.0, 0.9), vwall(-1.2, 2.1, 5.7), vwall(-1.2, 6.9, 8.0)
    # 子ども部屋と寝室の間（y=4）
    hwall(4.0, -6.0, -1.2)
    return w


FURNITURE = [  # (x0, x1, y0, y1, 高さ, 色, 名前)
    (2.0, 4.0, 3.3, 4.0, 0.42, "0.35 0.42 0.55 1", "ソファ"),
    (2.5, 3.5, 2.4, 3.0, 0.35, "0.55 0.40 0.28 1", "テーブル"),
    (5.45, 5.9, 0.8, 3.2, 0.5, "0.25 0.25 0.28 1", "テレビ台"),
    (5.3, 5.9, 4.7, 7.9, 0.9, "0.85 0.85 0.82 1", "台所"),
    (2.2, 3.6, 6.3, 7.3, 0.72, "0.55 0.40 0.28 1", "食卓"),
    (-5.9, -3.9, 5.6, 7.9, 0.45, "0.75 0.80 0.90 1", "ベッド"),
    (-1.9, -1.3, 7.0, 7.9, 1.0, "0.55 0.40 0.28 1", "本棚"),
    (-5.9, -5.2, 0.2, 1.8, 0.72, "0.55 0.40 0.28 1", "机"),
    (-2.4, -1.5, 3.1, 3.9, 0.4, "0.9 0.6 0.2 1", "おもちゃ箱"),
]

# 庭の高台と手すり
STAIRS = dict(x0=-6.0, x1=-4.4, y0=-5.0, y1=-2.6, top=0.24, step=0.06, tread=0.4)  # 東から西へ上る
SLOPE = dict(x0=4.4, x1=6.0, y0=-5.0, y1=-2.6, top=0.4, run=2.0)  # 西から東へ上る（約 11°）


def _ground_blocks():
    """地面より高い部分（家の床・ポーチ・庭の高台と段）。(名前, x0, x1, y0, y1, 高さ, 色, 材質) のリスト。"""
    b = [("house_floor", -6.0, 6.0, 0.0, 8.0, FLOOR, "1 1 1 1", "MatWood"),
         ("porch", -1.2, 1.2, -PORCH_DEPTH, 0.0, FLOOR, "1 1 1 1", "MatStone")]
    n = int(round(FLOOR / PORCH_STEP))
    for i in range(1, n):  # 玄関前の階段（踊り場の手前に、低い方から）
        b.append((f"porch{i}", -1.2, 1.2, -PORCH_DEPTH - 0.4 * (n - i), -PORCH_DEPTH, i * FLOOR / n, "0.93 0.93 0.93 1", "MatStone"))
    s = STAIRS
    b.append(("stairs_top", s["x0"], s["x1"], s["y0"], s["y1"], s["top"], "1 1 1 1", "MatStone"))
    for i in range(1, int(round(s["top"] / s["step"]))):  # 高台の東側に段（高い方から）
        x1 = s["x1"] + i * s["tread"]
        b.append((f"stair{i}", x1 - s["tread"], x1, s["y0"], s["y1"], s["top"] - i * s["step"], "0.93 0.93 0.93 1", "MatStone"))
    p = SLOPE
    b.append(("slope_top", p["x0"], p["x1"], p["y0"], p["y1"], p["top"], "1 1 1 1", "MatStone"))
    return b


# 地面の高さマップ（歩く面はすべてこれ。見た目は同じ形の箱で描く）
GX0, GX1, GY0, GY1, GRES = -7.0, 7.0, -7.0, 9.0, 0.05
HF_TOP = 0.5  # 高さマップの最大の高さ [m]


def ground_heights():
    """高さマップの値 [m]（行が y、列が x）。段は箱の形、坂は直線の斜面。"""
    xs = np.arange(GX0, GX1 + 1e-9, GRES)
    ys = np.arange(GY0, GY1 + 1e-9, GRES)
    X, Y = np.meshgrid(xs, ys)
    h = np.zeros_like(X)
    for _, x0, x1, y0, y1, top, _, _ in _ground_blocks():
        h = np.where((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1), np.maximum(h, top), h)
    p = SLOPE
    inside = (X >= p["x0"] - p["run"]) & (X <= p["x0"]) & (Y >= p["y0"]) & (Y <= p["y1"])
    h = np.where(inside, np.maximum(h, p["top"] * (X - (p["x0"] - p["run"])) / p["run"]), h)
    return h


def _garden():
    """(見た目の geom の XML 文字列のリスト, 経路計画で通れない箱のリスト)。"""
    g, blocked = [], []
    s, p = STAIRS, SLOPE
    n = int(round(s["top"] / s["step"]))
    ang = np.arctan2(p["top"], p["run"])
    L = np.hypot(p["top"], p["run"])
    cx, cz = p["x0"] - p["run"] / 2, p["top"] / 2 - 0.05 * np.cos(ang)
    g.append(f'<geom name="ramp" type="box" size="{L / 2 + 0.03} {(p["y1"] - p["y0"]) / 2} 0.05" pos="{cx} {(p["y0"] + p["y1"]) / 2} {cz}" '
             f'euler="0 {-ang} 0" rgba="0.93 0.93 0.93 1" material="MatStone" class="deco"/>')
    # 手すり（高台と坂の、上り口以外の縁）。経路計画では通れない
    rails = [
        (s["x0"], s["x1"] + (n - 1) * s["tread"], s["y0"] - 0.04, s["y0"] + 0.04),
        (s["x0"], s["x1"] + (n - 1) * s["tread"], s["y1"] - 0.04, s["y1"] + 0.04),
        (s["x0"] - 0.04, s["x0"] + 0.04, s["y0"], s["y1"]),
        (p["x0"] - p["run"], p["x1"], p["y0"] - 0.04, p["y0"] + 0.04),
        (p["x0"] - p["run"], p["x1"], p["y1"] - 0.04, p["y1"] + 0.04),
        (p["x1"] - 0.04, p["x1"] + 0.04, p["y0"], p["y1"]),
    ]
    for i, (x0, x1, y0, y1) in enumerate(rails):
        g.append(_box(f"wall_rail{i}", x0, x1, y0, y1, 0, 0.5, "0.42 0.32 0.24 1", group=1))
        blocked.append((x0, x1, y0, y1))
    # 玄関前の階段の両脇の植え込み（階段に正面からまっすぐ入るように）
    y_bottom = -PORCH_DEPTH - 0.4 * (int(round(FLOOR / PORCH_STEP)) - 1)
    for i, (x0, x1) in enumerate([(-1.6, -1.25), (1.25, 1.6)]):
        g.append(_box(f"hedge{i}", x0, x1, y_bottom - 0.8, -0.05, 0, 0.4, "0.30 0.48 0.26 1", group=1))
        blocked.append((x0, x1, y_bottom - 0.8, -0.05))
    return g, blocked


def _box(name, x0, x1, y0, y1, z0, z1, rgba, group=0, material=None):
    """箱の geom。group=1（壁・家具・手すり）は足元の高さセンサーのレイに当てず、経路計画で避ける。"""
    cls = "world" if group == 0 else "obstacle"
    mat = f' material="{material}"' if material else ""
    return (f'<geom name="{name}" type="box" size="{(x1 - x0) / 2} {(y1 - y0) / 2} {(z1 - z0) / 2}" '
            f'pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {(z0 + z1) / 2}" rgba="{rgba}" class="{cls}"{mat}/>')


TEX_DIR = os.path.join(HERE, "assets", "tex")


def _noise(rng, n, scales, weights):
    """なめらかなノイズ（粗い乱数を拡大して重ねる。継ぎ目なく並べられる）。値はおよそ -1〜1。"""
    from PIL import Image

    out = np.zeros((n, n))
    for s, w in zip(scales, weights):
        a = np.tile(rng.uniform(-1, 1, (s, s)), (3, 3))
        img = Image.fromarray(((a + 1) * 127.5).astype(np.uint8)).resize((3 * n, 3 * n), Image.BICUBIC)
        out += w * (np.asarray(img, float)[n:2 * n, n:2 * n] / 127.5 - 1)
    return out


def make_textures():
    """芝生・木の床・石畳・壁紙の画像を作る（乱数の種を固定）。"""
    from PIL import Image

    os.makedirs(TEX_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 512
    y, x = np.mgrid[0:n, 0:n]

    def save(name, rgb):
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(os.path.join(TEX_DIR, name))

    # 芝生：緑のむら＋細かい草の粒
    v = _noise(rng, n, (8, 32, 128), (0.5, 0.3, 0.2)) + 0.6 * rng.uniform(-1, 1, (n, n))
    save("grass.png", np.stack([84 + 12 * v, 122 + 15 * v, 66 + 9 * v], -1))
    # 木の床：4 列の板、木目と板ごとの色の違い、継ぎ目
    plank = x // (n // 4)
    tone = rng.uniform(-1, 1, 4)[plank]
    grain = np.sin((x % (n // 4)) * 0.35 + 6 * _noise(rng, n, (4, 16), (0.7, 0.3))) * 0.5 + 0.3 * rng.uniform(-1, 1, (n, n))
    wood = np.stack([196 + 14 * tone + 10 * grain, 158 + 12 * tone + 9 * grain, 112 + 10 * tone + 7 * grain], -1)
    wood[(x % (n // 4) < 2) | ((y + plank * 197) % n < 2)] *= 0.72
    save("wood.png", wood)
    # 石畳：明るい灰色、目地
    v = _noise(rng, n, (8, 32, 128), (0.4, 0.3, 0.3)) + 0.4 * rng.uniform(-1, 1, (n, n))
    stone = np.stack([186 + 12 * v, 184 + 12 * v, 178 + 12 * v], -1)
    stone[(x % (n // 2) < 3) | (y % (n // 2) < 3)] *= 0.78
    save("stone.png", stone)
    # 壁紙：ごく淡い縦じま
    v = 3 * np.sin(x * 2 * np.pi / 32) + 2 * rng.uniform(-1, 1, (n, n))
    save("wall.png", np.stack([243 + v, 240 + v, 232 + v], -1))


MATERIALS = """<texture name="tex_grass" type="2d" file="tex/grass.png"/>
    <texture name="tex_wood" type="2d" file="tex/wood.png"/>
    <texture name="tex_stone" type="2d" file="tex/stone.png"/>
    <texture name="tex_wall" type="2d" file="tex/wall.png"/>
    <material name="MatGrass" texture="tex_grass" texrepeat="0.5 0.5" texuniform="true" specular="0.05" shininess="0.05"/>
    <material name="MatWood" texture="tex_wood" texrepeat="0.6 0.6" texuniform="true" specular="0.3" shininess="0.4"/>
    <material name="MatStone" texture="tex_stone" texrepeat="1 1" texuniform="true" specular="0.1" shininess="0.1"/>
    <material name="MatWall" texture="tex_wall" texrepeat="1 1" texuniform="true" specular="0.05" shininess="0.05"/>"""


def build_xml():
    # 歩く面：物理と足元のレイは高さマップ（グループ 3：画面には出さない）、見た目は同じ形の箱（グループ 2）
    ny, nx = ground_heights().shape
    parts = [f'<geom name="ground" type="hfield" hfield="ground" pos="{(GX0 + GX1) / 2} {(GY0 + GY1) / 2} 0" '
             f'contype="1" conaffinity="1" group="3"/>']
    for name, x0, x1, y0, y1, top, rgba, mat in _ground_blocks():
        parts.append(_box(name, x0, x1, y0, y1, 0, top, rgba, material=mat).replace('class="world"', 'class="deco"'))
    for i, (x0, x1, y0, y1, z0, z1) in enumerate(_walls()):
        parts.append(_box(f"wall_{i}", x0, x1, y0, y1, z0, z1, "1 1 1 1", group=1, material="MatWall"))
    for i, (x0, x1, y0, y1, h, rgba, _) in enumerate(FURNITURE):
        parts.append(_box(f"furn_{i}", x0, x1, y0, y1, FLOOR, FLOOR + h, rgba, group=1))
    garden, _ = _garden()
    parts += garden
    # 充電ステーション（見た目だけ）
    x, y = PLACES["dock"][1]
    parts.append(f'<geom type="cylinder" pos="{x} {y} 0.005" size="0.6 0.005" rgba="0.32 0.35 0.40 1" class="deco"/>')
    parts.append(f'<geom type="cylinder" pos="{x} {y} 0.006" size="0.5 0.005" rgba="1 0.85 0.4 1" class="deco"/>')
    parts.append(f'<geom type="cylinder" pos="{x} {y} 0.007" size="0.45 0.005" rgba="0.32 0.35 0.40 1" class="deco"/>')
    floor = '<geom name="floor" type="plane" size="40 40 1" material="MatGrass" class="deco"/>'
    xml = scene_xml(floor, "\n    ".join(parts), model="house_dog")
    xml = xml.replace('<material name="MatTerrain"', f'<hfield name="ground" nrow="{ny}" ncol="{nx}" '
                      f'size="{(GX1 - GX0) / 2} {(GY1 - GY0) / 2} {HF_TOP} 0.1"/>\n    <material name="MatTerrain"')
    xml = xml.replace('<default class="world">',
                      '<default class="obstacle"><geom contype="1" conaffinity="1" group="1"/></default>\n    <default class="world">')
    xml = xml.replace('<material name="MatTerrain"', MATERIALS + '\n    <material name="MatTerrain"')
    # 影：太陽光をロボットに追従させ、ロボットのまわりだけ細かい影を描く
    xml = xml.replace('castshadow="true"/>', 'castshadow="false"/>', 1)
    xml = xml.replace('<freejoint name="root"/>', '<freejoint name="root"/>\n      <light name="sun" mode="trackcom" directional="true" '
                      'dir="-0.4 -0.3 -1" pos="1.6 1.2 4" diffuse="0.55 0.53 0.5" specular="0.1 0.1 0.1" castshadow="true"/>', 1)
    xml = xml.replace('<map haze=', '<map shadowclip="0.25" haze=', 1)
    if not os.path.exists(os.path.join(TEX_DIR, "wood.png")):
        make_textures()
    os.makedirs(os.path.dirname(XML_PATH), exist_ok=True)
    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)


class HouseDogEnv(TerrainDogEnv):
    """家と庭で歩かせる環境（地形は固定、充電ステーションから出発）。"""

    def __init__(self, **kwargs):
        if not os.path.exists(XML_PATH):
            build_xml()
        kwargs.setdefault("curriculum", False)
        super().__init__(xml_path=XML_PATH, **kwargs)
        hf = self.model.hfield("ground")
        self.model.hfield_data[hf.adr[0]:hf.adr[0] + hf.nrow[0] * hf.ncol[0]] = (ground_heights() / HF_TOP).ravel()
        self.fixed_goal = True
        self.start_pos = np.array(PLACES[HOME][1])
        self.start_yaw = np.pi / 2  # 家の方を向いて出発

    def start_xy(self):
        return self.start_pos

    def reset_model(self):
        obs = super().reset_model()
        qpos = self.data.qpos.copy()
        qpos[3:7] = [np.cos(self.start_yaw / 2), 0, 0, np.sin(self.start_yaw / 2)]
        self.set_state(qpos, self.data.qvel)
        return self._get_obs()


# --- 経路計画 ---
GRID = 0.1
EXTENT = (-8.0, 8.0, -8.0, 9.0)  # x0, x1, y0, y1
INFLATE = 0.35  # ロボットの胴幅の半分＋余裕


def occupancy():
    x0, x1, y0, y1 = EXTENT
    nx, ny = int((x1 - x0) / GRID), int((y1 - y0) / GRID)
    occ = np.zeros((ny, nx), bool)
    boxes = [b[:4] for b in _walls()] + [f[:4] for f in FURNITURE] + _garden()[1]
    # 高台の縁から落ちないよう、高台の上り口以外の側面も通れなくする（手すりで表現済み）
    cx = x0 + (np.arange(nx) + 0.5) * GRID
    cy = y0 + (np.arange(ny) + 0.5) * GRID
    X, Y = np.meshgrid(cx, cy)
    for bx0, bx1, by0, by1 in boxes:
        occ |= (X > bx0 - INFLATE) & (X < bx1 + INFLATE) & (Y > by0 - INFLATE) & (Y < by1 + INFLATE)
    occ |= (X < x0 + 0.5) | (X > x1 - 0.5) | (Y < y0 + 0.5) | (Y > y1 - 0.5)
    return occ


_OCC = None


def _cell(p):
    return int((p[1] - EXTENT[2]) / GRID), int((p[0] - EXTENT[0]) / GRID)


def _point(c):
    return np.array([EXTENT[0] + (c[1] + 0.5) * GRID, EXTENT[2] + (c[0] + 0.5) * GRID])


def _free_line(occ, a, b):
    n = int(np.ceil(np.linalg.norm(b - a) / (GRID / 2))) + 1
    for t in np.linspace(0, 1, n):
        r, c = _cell(a + t * (b - a))
        if occ[r, c]:
            return False
    return True


def _porch_gates():
    """玄関の階段の下（外）と、扉を入ったところ（中）の通過点。"""
    y_bottom = -PORCH_DEPTH - 0.4 * (int(round(FLOOR / PORCH_STEP)) - 1)
    return np.array([0.0, y_bottom - 0.6]), np.array([0.0, 0.7])


def plan_path(start, goal):
    """start から goal までの経由地（最後が goal）。家の出入りは、玄関の階段を正面からまっすぐ上り下りする。"""
    out_gate, in_gate = _porch_gates()
    inside = lambda p: p[1] > -PORCH_DEPTH  # noqa: E731
    if inside(start) != inside(goal):
        a, b = (out_gate, in_gate) if inside(goal) else (in_gate, out_gate)
        return _plan(start, a) + [b] + _plan(b, goal)
    return _plan(start, goal)


def _plan(start, goal):
    """A* で start から goal までの経由地（最後が goal）。通れる経路が無ければ goal だけを返す。"""
    global _OCC
    if _OCC is None:
        _OCC = occupancy()
    occ = _OCC.copy()
    s, g = _cell(start), _cell(goal)
    occ[s] = occ[g] = False
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    openq = [(0.0, s)]
    cost, prev = {s: 0.0}, {}
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
    # 見通しのきく点まで飛ばして経由地を間引く
    out, i = [], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _free_line(_OCC, pts[i], pts[j]):
            j -= 1
        if not out or np.linalg.norm(pts[j] - out[-1]) > 0.05:
            out.append(pts[j])
        i = j
    return out
