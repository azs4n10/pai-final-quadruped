"""
家具の見た目（MuJoCo の基本形状を組み合わせる）。

各関数は (見た目の geom の XML 文字列のリスト, 当たり判定と経路計画に使う外形の箱 (x0, x1, y0, y1, 高さ)) を返す。
見た目の部品は質量 0・接触なし（class="deco"）。家具は床の高さ z0 の上に置く。
yaw は 0・90・180・270 度（家具の正面の向き。0 は +y 向き）。
"""
import numpy as np

WOOD, WOOD_D, WHITE, GRAY, DARK = "0.62 0.45 0.30 1", "0.42 0.29 0.18 1", "0.93 0.93 0.91 1", "0.55 0.56 0.58 1", "0.12 0.12 0.13 1"
FABRIC, CUSHION, BEIGE = "0.36 0.44 0.56 1", "0.84 0.80 0.70 1", "0.86 0.80 0.70 1"


class Piece:
    """家具 1 つ分の部品を、家具の中心からの相対位置で積み上げ、最後に置き場所と向きで変換する。"""

    def __init__(self, cx, cy, z0, yaw=0):
        self.cx, self.cy, self.z0, self.yaw = cx, cy, z0, np.radians(yaw)
        self.parts = []
        self.ext = [np.inf, -np.inf, np.inf, -np.inf, 0.0]

    def _place(self, x, y):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return self.cx + c * x - s * y, self.cy + s * x + c * y

    def box(self, x, y, z, sx, sy, sz, rgba, mat=""):
        """中心 (x, y, z)（z は床から）、半分の大きさ (sx, sy, sz)。"""
        px, py = self._place(x, y)
        m = f' material="{mat}"' if mat else ""
        self.parts.append(f'<geom type="box" pos="{px:.4f} {py:.4f} {self.z0 + z:.4f}" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
                          f'euler="0 0 {self.yaw:.4f}" rgba="{rgba}" class="deco"{m}/>')
        for dx in (-sx, sx):
            for dy in (-sy, sy):
                qx, qy = self._place(x + dx, y + dy)
                self.ext = [min(self.ext[0], qx), max(self.ext[1], qx), min(self.ext[2], qy), max(self.ext[3], qy), max(self.ext[4], z + sz)]

    def cyl(self, x, y, z, r, h, rgba, mat=""):
        px, py = self._place(x, y)
        m = f' material="{mat}"' if mat else ""
        self.parts.append(f'<geom type="cylinder" pos="{px:.4f} {py:.4f} {self.z0 + z:.4f}" size="{r:.4f} {h:.4f}" rgba="{rgba}" class="deco"{m}/>')
        self.ext = [min(self.ext[0], px - r), max(self.ext[1], px + r), min(self.ext[2], py - r), max(self.ext[3], py + r), max(self.ext[4], z + h)]

    def sphere(self, x, y, z, r, rgba):
        px, py = self._place(x, y)
        self.parts.append(f'<geom type="sphere" pos="{px:.4f} {py:.4f} {self.z0 + z:.4f}" size="{r:.4f}" rgba="{rgba}" class="deco"/>')
        self.ext = [min(self.ext[0], px - r), max(self.ext[1], px + r), min(self.ext[2], py - r), max(self.ext[3], py + r), max(self.ext[4], z + r)]

    def done(self):
        return self.parts, tuple(self.ext)


def legs(p, w, d, h, r=0.025, rgba=WOOD_D):
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box(sx * (w - r), sy * (d - r), h / 2, r, r, h / 2, rgba)


def sofa(cx, cy, z0, yaw=0, w=1.0):
    """3 人掛けソファ（正面が yaw の向き）。w は幅の半分。"""
    p = Piece(cx, cy, z0, yaw)
    d = 0.45
    p.box(0, 0, 0.12, w, d, 0.08, FABRIC, "MatFabric")  # 台座
    for i, x in enumerate(np.linspace(-w + w / 3, w - w / 3, 3)):  # 座面クッション
        p.box(x, 0.05, 0.27, w / 3 - 0.01, d - 0.08, 0.07, FABRIC, "MatFabric")
    p.box(0, -d + 0.1, 0.45, w, 0.1, 0.25, FABRIC, "MatFabric")  # 背もたれ
    for sx in (-1, 1):
        p.box(sx * (w - 0.08), 0, 0.33, 0.08, d, 0.13, FABRIC, "MatFabric")  # 肘掛け
        p.box(sx * (w - 0.35), -d + 0.25, 0.48, 0.16, 0.06, 0.14, CUSHION, "MatFabric")  # 置きクッション
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.cyl(sx * (w - 0.06), sy * (d - 0.06), 0.02, 0.02, 0.02, DARK)
    return p.done()


def coffee_table(cx, cy, z0, yaw=0):
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.36, 0.5, 0.3, 0.025, WOOD, "MatWoodFurniture")
    p.box(0, 0, 0.12, 0.46, 0.26, 0.01, WOOD_D)
    legs(p, 0.48, 0.28, 0.35)
    p.box(0.25, 0.05, 0.4, 0.06, 0.04, 0.015, "0.9 0.9 0.92 1")  # 本
    p.cyl(-0.2, -0.05, 0.42, 0.04, 0.035, "0.95 0.95 0.95 1")  # マグカップ
    return p.done()


def tv_stand(cx, cy, z0, yaw=0):
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.22, 0.8, 0.22, 0.22, WOOD_D, "MatWoodFurniture")
    for x in (-0.4, 0.4):
        p.box(x, 0.221, 0.22, 0.36, 0.002, 0.17, WOOD)  # 引き出しの面
    p.box(0, -0.02, 0.47, 0.12, 0.08, 0.02, DARK)  # 台
    p.box(0, -0.02, 0.53, 0.03, 0.02, 0.06, DARK)
    p.box(0, -0.02, 0.92, 0.7, 0.025, 0.4, DARK)  # 画面の枠
    p.box(0, 0.004, 0.92, 0.67, 0.002, 0.37, "0.10 0.16 0.26 1", "MatScreen")  # 画面
    return p.done()


def rug(cx, cy, z0, sx, sy, mat="MatRug"):
    p = Piece(cx, cy, z0)
    p.box(0, 0, 0.004, sx, sy, 0.004, "1 1 1 1", mat)
    parts, _ = p.done()
    return parts, None  # 敷物は通れる


def dining_set(cx, cy, z0, yaw=0):
    """4 人掛けの食卓と椅子。"""
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.72, 0.75, 0.42, 0.025, WOOD, "MatWoodFurniture")
    legs(p, 0.7, 0.38, 0.7, 0.03)
    for x in (-0.35, 0.35):
        for sy in (-1, 1):
            y = sy * 0.62
            p.box(x, y, 0.45, 0.2, 0.2, 0.02, WOOD, "MatWoodFurniture")  # 座面
            for lx in (-0.17, 0.17):
                for ly in (-0.17, 0.17):
                    p.box(x + lx, y + ly, 0.22, 0.018, 0.018, 0.22, WOOD_D)  # 椅子の脚
            p.box(x, y + sy * 0.19, 0.75, 0.2, 0.02, 0.28, WOOD_D)  # 背もたれ
    for x in (-0.25, 0.25):
        p.cyl(x, 0, 0.76, 0.12, 0.008, "0.97 0.97 0.97 1")  # 皿
    p.cyl(0, 0.05, 0.82, 0.05, 0.08, "0.5 0.75 0.55 1")  # 花瓶
    return p.done()


def kitchen(cx, cy, z0, yaw=0, w=1.4):
    """壁付けのキッチン（シンク・コンロ・吊り戸棚）。正面が yaw の向き。"""
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.42, w, 0.3, 0.42, WHITE, "MatCabinet")  # 下の戸棚
    for i, x in enumerate(np.linspace(-w + w / 4, w - w / 4, 4)):
        p.box(x, 0.301, 0.42, w / 4 - 0.02, 0.002, 0.36, "0.88 0.88 0.86 1")
        p.box(x, 0.31, 0.72, 0.08, 0.01, 0.01, GRAY)  # 取っ手
    p.box(0, 0, 0.86, w + 0.02, 0.32, 0.02, "0.25 0.26 0.28 1", "MatCounter")  # 天板
    p.box(-w / 2, 0.02, 0.87, 0.3, 0.2, 0.012, "0.78 0.80 0.82 1")  # シンク
    p.box(-w / 2, -0.24, 0.98, 0.015, 0.015, 0.12, GRAY)  # 蛇口
    p.box(-w / 2, -0.16, 1.09, 0.015, 0.09, 0.015, GRAY)
    for dx in (-0.12, 0.12):  # コンロ
        p.cyl(w / 2 + dx, 0.02, 0.885, 0.08, 0.004, DARK)
        p.cyl(w / 2 + dx, 0.02, 0.89, 0.05, 0.003, "0.35 0.35 0.37 1")
    p.cyl(w / 2 + 0.12, 0.02, 0.95, 0.09, 0.06, GRAY)  # 鍋
    p.box(0, -0.18, 1.7, w, 0.17, 0.3, WHITE, "MatCabinet")  # 吊り戸棚
    return p.done()


def fridge(cx, cy, z0, yaw=0):
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.9, 0.33, 0.33, 0.9, "0.90 0.91 0.92 1")
    p.box(0, 0.331, 1.25, 0.32, 0.002, 0.52, "0.85 0.86 0.88 1")
    p.box(0, 0.331, 0.38, 0.32, 0.002, 0.33, "0.85 0.86 0.88 1")
    p.box(0.27, 0.35, 1.0, 0.012, 0.015, 0.2, GRAY)
    return p.done()


def bed(cx, cy, z0, yaw=0, w=0.75, l=1.05):
    """ベッド（頭側が -y、yaw で回す）。"""
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.17, w, l, 0.12, WOOD_D, "MatWoodFurniture")  # 台
    p.box(0, 0, 0.36, w - 0.03, l - 0.03, 0.08, "0.96 0.96 0.95 1")  # マットレス
    p.box(0, 0.25, 0.46, w - 0.01, l - 0.28, 0.035, "0.55 0.68 0.82 1", "MatFabric")  # 掛け布団
    p.box(0, -l + 0.25, 0.48, w * 0.6, 0.14, 0.05, "0.98 0.98 0.98 1")  # 枕
    p.box(0, -l - 0.03, 0.5, w + 0.02, 0.03, 0.45, WOOD_D, "MatWoodFurniture")  # ヘッドボード
    return p.done()


def nightstand(cx, cy, z0, yaw=0):
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.25, 0.22, 0.2, 0.25, WOOD, "MatWoodFurniture")
    p.box(0, 0.201, 0.32, 0.18, 0.002, 0.08, WOOD_D)
    p.cyl(0, 0, 0.55, 0.07, 0.05, "0.95 0.93 0.85 1")  # ランプの台
    p.cyl(0, 0, 0.7, 0.11, 0.09, "1 0.96 0.85 1")  # ランプのかさ
    return p.done()


def bookshelf(cx, cy, z0, yaw=0, w=0.45):
    p = Piece(cx, cy, z0, yaw)
    h, d = 0.9, 0.16
    for sx in (-1, 1):
        p.box(sx * w, 0, h, 0.015, d, h, WOOD, "MatWoodFurniture")
    p.box(0, -d + 0.005, h, w, 0.005, h, WOOD_D)
    colors = ["0.70 0.25 0.22 1", "0.25 0.42 0.62 1", "0.88 0.78 0.45 1", "0.30 0.52 0.36 1", "0.92 0.92 0.88 1", "0.45 0.35 0.55 1"]
    rng = np.random.default_rng(abs(int(cx * 100 + cy * 10)))
    for k, z in enumerate(np.linspace(0.02, 2 * h - 0.02, 5)):
        p.box(0, 0, z, w, d, 0.012, WOOD, "MatWoodFurniture")
        if k < 4:  # 本を並べる
            x = -w + 0.04
            while x < w - 0.08:
                t = rng.uniform(0.018, 0.035)
                hb = rng.uniform(0.13, 0.19)
                p.box(x + t, 0.01, z + hb + 0.012, t, d - 0.03, hb, colors[rng.integers(len(colors))])
                x += 2 * t + 0.004
    return p.done()


def desk(cx, cy, z0, yaw=0):
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.7, 0.55, 0.3, 0.02, WOOD, "MatWoodFurniture")
    for sx in (-1, 1):
        p.box(sx * 0.52, 0, 0.35, 0.02, 0.28, 0.35, WOOD_D)
    p.box(0.1, -0.1, 0.83, 0.2, 0.012, 0.12, DARK)  # モニター
    p.box(0.1, -0.08, 0.75, 0.03, 0.03, 0.04, DARK)
    p.box(-0.35, 0.05, 0.73, 0.1, 0.07, 0.012, "0.85 0.25 0.25 1")  # ノート
    p.box(0, 0.5, 0.45, 0.2, 0.2, 0.02, "0.25 0.25 0.27 1")  # 椅子
    p.box(0, 0.68, 0.7, 0.2, 0.02, 0.22, "0.25 0.25 0.27 1")
    p.cyl(0, 0.5, 0.22, 0.025, 0.22, GRAY)
    return p.done()


def toy_box(cx, cy, z0, yaw=0):
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.2, 0.35, 0.25, 0.2, "0.95 0.75 0.30 1", "MatWoodFurniture")
    p.box(-0.1, 0.0, 0.44, 0.06, 0.06, 0.06, "0.85 0.25 0.25 1")  # 積み木
    p.box(0.06, 0.05, 0.42, 0.05, 0.05, 0.04, "0.25 0.45 0.80 1")
    p.sphere(0.2, -0.08, 0.47, 0.08, "0.95 0.45 0.55 1")  # ボール
    p.sphere(-0.45, 0.25, 0.1, 0.1, "0.30 0.65 0.40 1")
    return p.done()


def plant(cx, cy, z0, h=0.9):
    p = Piece(cx, cy, z0)
    p.cyl(0, 0, 0.15, 0.14, 0.15, "0.85 0.82 0.78 1")  # 鉢
    p.cyl(0, 0, h * 0.5, 0.015, h * 0.35, "0.35 0.28 0.18 1")
    for dx, dy, dz, r in [(0, 0, h, 0.22), (0.12, 0.05, h - 0.15, 0.16), (-0.1, -0.08, h - 0.12, 0.17), (0.02, -0.12, h + 0.12, 0.13)]:
        p.sphere(dx, dy, dz, r, "0.30 0.55 0.30 1")
    return p.done()


def shoe_rack(cx, cy, z0, yaw=0):
    p = Piece(cx, cy, z0, yaw)
    p.box(0, 0, 0.45, 0.5, 0.17, 0.45, WOOD, "MatWoodFurniture")
    p.box(0, 0.171, 0.45, 0.47, 0.002, 0.42, WOOD_D)
    p.box(0.2, 0, 0.95, 0.12, 0.08, 0.05, "0.9 0.9 0.9 1")
    return p.done()


def ceiling_light(cx, cy, z):
    return [f'<geom type="cylinder" pos="{cx} {cy} {z}" size="0.22 0.03" rgba="1 0.98 0.9 1" class="deco"/>']


MATERIALS = """<texture name="tex_fabric" type="2d" file="tex/fabric.png"/>
    <texture name="tex_rug" type="2d" file="tex/rug.png"/>
    <texture name="tex_furn" type="2d" file="tex/wood_furniture.png"/>
    <material name="MatFabric" texture="tex_fabric" texrepeat="4 4" texuniform="true" specular="0.05" shininess="0.05"/>
    <material name="MatRug" texture="tex_rug" texrepeat="1 1" specular="0" shininess="0"/>
    <material name="MatWoodFurniture" texture="tex_furn" texrepeat="2 2" texuniform="true" specular="0.3" shininess="0.4"/>
    <material name="MatCabinet" specular="0.3" shininess="0.5"/>
    <material name="MatCounter" specular="0.6" shininess="0.8"/>
    <material name="MatScreen" specular="0.9" shininess="0.95" reflectance="0.1"/>"""
