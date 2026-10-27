"""動画用の文字・ミニマップ描画（PIL）。"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from goal_ant import LANDMARK_JA, LANDMARKS

_FONT_CANDIDATES = {
    "regular": ["C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"],
    "bold": ["C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/meiryob.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"],
}
_cache = {}


def font(size, weight="regular"):
    key = (size, weight)
    if key not in _cache:
        for path in _FONT_CANDIDATES[weight]:
            if os.path.exists(path):
                _cache[key] = ImageFont.truetype(path, size)
                break
        else:
            _cache[key] = ImageFont.load_default(size)
    return _cache[key]


def rgba255(rgba, alpha=255):
    r, g, b, _ = rgba
    return (int(r * 255), int(g * 255), int(b * 255), alpha)


def panel(draw, box, alpha=205, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=(15, 18, 24, alpha))


def minimap(size, trail, goal_name, extent=9.0, visited=()):
    """上から見た地図。trail は (x, y) 列。"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=20, fill=(15, 18, 24, 190))

    def px(x, y):
        return ((x + extent) / (2 * extent) * size, (extent - y) / (2 * extent) * size)

    # グリッド
    for v in range(-8, 9, 2):
        a, b = px(v, -extent), px(v, extent)
        d.line([a, b], fill=(255, 255, 255, 28))
        a, b = px(-extent, v), px(extent, v)
        d.line([a, b], fill=(255, 255, 255, 28))
    # ランドマーク
    for name, ((x, y), rgba) in LANDMARKS.items():
        cx, cy = px(x, y)
        r = 16 if name == "home" else 11
        outline = (255, 140, 0, 255) if name == goal_name else (255, 255, 255, 120)
        width = 4 if name == goal_name else 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=rgba255(rgba, 230), outline=outline, width=width)
        if name in visited:
            d.text((cx, cy), "✓", font=font(18, "bold"), fill=(20, 20, 20, 255), anchor="mm")
    # 軌跡
    if len(trail) >= 2:
        pts = [px(x, y) for x, y in trail]
        d.line(pts, fill=(255, 170, 40, 255), width=4, joint="curve")
    if trail:
        cx, cy = px(*trail[-1])
        d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(255, 255, 255, 255), outline=(255, 140, 0, 255), width=3)
    return img


def route_text(route):
    return " → ".join(LANDMARK_JA[k] for k in route)


def title_card(size, title, subtitle="", lines=(), bg=(14, 17, 22)):
    """全面の文字カード。"""
    w, h = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 14, h), fill=(255, 140, 0))
    y = h * 0.30 if lines else h * 0.40
    d.text((120, y), title, font=font(84, "bold"), fill=(255, 255, 255))
    y += 120
    if subtitle:
        d.text((124, y), subtitle, font=font(40), fill=(255, 175, 70))
        y += 90
    for line in lines:
        d.text((124, y), line, font=font(38), fill=(220, 225, 232))
        y += 64
    return np.asarray(img)
