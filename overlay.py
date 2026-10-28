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


INK, SUB, KHAKI, ACCENT = (28, 30, 34), (96, 92, 70), (179, 176, 145), (240, 140, 0)


def slide_base(size):
    """講座テンプレートに合わせた白地＋右端のカーキの斜め帯。"""
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.polygon([(w - 110, 0), (w, 0), (w, h), (w - 330, h)], fill=KHAKI)
    d.polygon([(w - 140, 0), (w - 126, 0), (w - 346, h), (w - 360, h)], fill=KHAKI)
    return img, d


def section_card(size, title, subtitle=""):
    """章の区切り（大きな見出し＋一言）。"""
    img, d = slide_base(size)
    d.rectangle((120, 470, 132, 610), fill=ACCENT)
    d.text((170, 462), title, font=font(84, "bold"), fill=INK)
    if subtitle:
        d.text((174, 572), subtitle, font=font(38), fill=SUB)
    return np.asarray(img)


def system_card(size, robot_img):
    """しくみ：指示 → LLM → 経由地 → 歩行方策 → ロボット の流れ図と、観測・報酬。"""
    img, d = slide_base(size)
    d.text((120, 90), "しくみ", font=font(72, "bold"), fill=INK)
    boxes = [("日本語の指示", "「黄色い柱に荷物を\n届けたら帰ってきて」"), ("ローカル LLM", "gemma4（Ollama）\nPC 内で動作"),
             ("経由地リスト", "黄色い柱 → ホーム"), ("歩行方策", "強化学習（PPO）\nゴールまで歩く")]
    x, y, bw, bh = 120, 300, 270, 210
    for i, (head, body) in enumerate(boxes):
        d.rounded_rectangle((x, y, x + bw, y + bh), radius=22, fill=(255, 247, 235), outline=ACCENT, width=4)
        d.text((x + bw / 2, y + 48), head, font=font(36, "bold"), fill=INK, anchor="mm")
        d.multiline_text((x + bw / 2, y + 128), body, font=font(26), fill=SUB, anchor="mm", align="center", spacing=8)
        x += bw
        d.text((x + 30, y + bh / 2), "→", font=font(52, "bold"), fill=ACCENT, anchor="mm")
        x += 60
    rob = robot_img.resize((330, int(robot_img.height * 330 / robot_img.width)))
    img.paste(rob, (x, y + bh // 2 - rob.height // 2))
    d.text((x + 165, y + bh / 2 + rob.height / 2 + 30), "自作の犬型ロボット（12関節）", font=font(26, "bold"), fill=INK, anchor="mm")
    rows = [("観測", "胴体の姿勢・速度、12 関節の角度と速度、ゴールの方向と距離、歩行のリズム"),
            ("報酬", "目標の速さ（0.8 m/秒）でゴールへ ＋ 到着ボーナス ＋ トロットのリズム"),
            ("", "− 胴体の傾き・横歩き・脚の開き・急な動き"),
            ("環境", "物理シミュレータ MuJoCo")]
    y = 680
    for head, body in rows:
        if head:
            d.rounded_rectangle((120, y - 4, 230, y + 46), radius=12, fill=KHAKI)
            d.text((175, y + 21), head, font=font(28, "bold"), fill=(255, 255, 255), anchor="mm")
        d.text((260, y + 21), body, font=font(32), fill=INK, anchor="lm")
        y += 72
    return np.asarray(img)


def summary_card(size, stats, footer=(), credit=""):
    """まとめ：数字のタイル＋一言。stats は (大きな数字, 説明) の列。"""
    img, d = slide_base(size)
    d.text((120, 90), "まとめ", font=font(72, "bold"), fill=INK)
    tw, th, gap, top = 350, 340, 36, 280
    for i, (big, label) in enumerate(stats):
        x = 120 + i * (tw + gap)
        d.rounded_rectangle((x, top, x + tw, top + th), radius=26, fill=(255, 247, 235), outline=ACCENT, width=4)
        d.text((x + tw / 2, top + 135), big, font=font(92, "bold"), fill=ACCENT, anchor="mm")
        d.multiline_text((x + tw / 2, top + 255), label, font=font(30), fill=INK, anchor="mm", align="center", spacing=8)
    y = 730
    for line in footer:
        d.text((120, y), line, font=font(34), fill=INK)
        y += 62
    if credit:
        d.text((120, size[1] - 70), credit, font=font(24), fill=SUB)
    return np.asarray(img)


def slide_title(slide_path, title, subtitle):
    """講座の「最終課題発表」スライドの空き（ロゴの上）に作品タイトルを書き込む。"""
    img = Image.open(slide_path).convert("RGB").resize((1920, 1080))
    d = ImageDraw.Draw(img)
    d.text((88, 700), title, font=font(60, "bold"), fill=(20, 20, 20))
    d.text((92, 790), subtitle, font=font(29), fill=(110, 104, 80))
    return np.asarray(img)
