"""動画用の文字・ミニマップ・スライド描画（PIL）。配色と見出しの様式は講座の講義資料に合わせる。"""
import os
import re

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

# 講義資料の配色：黒の文字と罫線、黄（図の主役）、薄い黄（図の脇役）、紺灰（番号）、赤（強調）、灰（帯）
INK, YELLOW, LIGHT, SLATE, RED, GRAY = (20, 20, 20), (255, 217, 102), (255, 242, 204), (68, 84, 106), (192, 0, 0), (217, 217, 217)


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


def panel(draw, box, alpha=205, radius=0, fill=(15, 18, 24)):
    draw.rounded_rectangle(box, radius=radius, fill=(*fill, alpha), outline=(*INK, alpha) if fill == (255, 255, 255) else None)


def tag(draw, xy, text, size=26):
    """紺灰の四角に白抜き文字（講義資料の番号タグの様式）。xy は左端の中央。"""
    x, y = xy
    w = draw.textlength(text, font=font(size, "bold")) + 24
    draw.rectangle((x, y - size * 0.8, x + w, y + size * 0.8), fill=SLATE)
    draw.text((x + w / 2, y), text, font=font(size, "bold"), fill=(255, 255, 255), anchor="mm")
    return x + w


def minimap(size, trail, goal_name, extent=9.0, visited=()):
    """上から見た地図。trail は (x, y) 列。"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, size - 1, size - 1), fill=(255, 255, 255, 230), outline=(*INK, 255))

    def px(x, y):
        return ((x + extent) / (2 * extent) * size, (extent - y) / (2 * extent) * size)

    for v in range(-8, 9, 2):
        d.line([px(v, -extent), px(v, extent)], fill=(0, 0, 0, 25))
        d.line([px(-extent, v), px(extent, v)], fill=(0, 0, 0, 25))
    for name, ((x, y), rgba) in LANDMARKS.items():
        cx, cy = px(x, y)
        r = 16 if name == "home" else 11
        width = 4 if name == goal_name else 1
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=rgba255(rgba, 230), outline=(*INK, 255), width=width)
        if name in visited:
            d.text((cx, cy), "✓", font=font(18, "bold"), fill=(*INK, 255), anchor="mm")
    if len(trail) >= 2:
        d.line([px(x, y) for x, y in trail], fill=(*RED, 255), width=4, joint="curve")
    if trail:
        cx, cy = px(*trail[-1])
        d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(255, 255, 255, 255), outline=(*RED, 255), width=3)
    return img


def route_text(route):
    return " → ".join(LANDMARK_JA[k] for k in route)


# --- スライド（講義資料の様式：白地、左上に太字の見出し＋下に黒い罫線） ---
def slide_base(size, title="", page=None):
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    if title:
        d.text((90, 62), title, font=font(58, "bold"), fill=INK)
        d.line((90, 150, w - 90, 150), fill=INK, width=4)
    if page is not None:  # スライド番号（講義資料と同じく右下）
        d.text((w - 60, h - 45), str(page), font=font(28), fill=INK, anchor="rm")
    return img, d


def _wrap(d, text, fnt, width):
    """全角の文章を幅 width に収まるよう折り返す。"""
    lines = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            if d.textlength(cur + ch, font=fnt) > width:
                lines.append(cur)
                cur = ""
            cur += ch
        lines.append(cur)
    return lines


def section_card(size, title, subtitle="", page=None):
    """章の区切り（中央に大きな見出し＋一言）。"""
    img, d = slide_base(size, page=page)
    w, h = size
    d.text((w / 2, h / 2 - 40), title, font=font(84, "bold"), fill=INK, anchor="mm")
    d.line((w / 2 - 500, h / 2 + 30, w / 2 + 500, h / 2 + 30), fill=INK, width=4)
    if subtitle:
        d.text((w / 2, h / 2 + 90), subtitle, font=font(38), fill=INK, anchor="mm")
    return np.asarray(img)


def pairs_card(size, title, lead, pairs, heads=("課題", "選んだ方法"), page=None):
    """左に課題（薄い黄）→ 右に対応（黄）の組を縦に並べる。"""
    img, d = slide_base(size, title, page=page)
    w, _ = size
    if lead:
        d.text((90, 190), lead, font=font(44, "bold"), fill=INK)
    lx0, lx1, rx0, rx1 = 160, 900, 1010, w - 90
    y = 300
    d.text(((lx0 + lx1) / 2, y), heads[0], font=font(32, "bold"), fill=INK, anchor="mm")
    d.text(((rx0 + rx1) / 2, y), heads[1], font=font(32, "bold"), fill=INK, anchor="mm")
    y += 40
    bh, gap = 150, 34
    for i, (left, right) in enumerate(pairs):
        top = y + i * (bh + gap)
        d.rectangle((90, top, 140, top + bh), fill=SLATE)
        d.text((115, top + bh / 2), str(i + 1), font=font(34, "bold"), fill=(255, 255, 255), anchor="mm")
        d.rectangle((lx0, top, lx1, top + bh), fill=LIGHT, outline=INK, width=2)
        d.multiline_text(((lx0 + lx1) / 2, top + bh / 2), left, font=font(36), fill=INK, anchor="mm", align="center", spacing=10)
        d.text(((lx1 + rx0) / 2, top + bh / 2), "→", font=font(56, "bold"), fill=INK, anchor="mm")
        d.rectangle((rx0, top, rx1, top + bh), fill=YELLOW, outline=INK, width=2)
        d.multiline_text(((rx0 + rx1) / 2, top + bh / 2), right, font=font(38, "bold"), fill=INK, anchor="mm", align="center", spacing=10)
    return np.asarray(img)


def system_card(size, robot_img, boxes, rows, page=None):
    """全体構成：左から右への流れ図（boxes＝(見出し, 説明)）＋ロボットの写真、下に層状の帯（rows＝(見出し, 説明)）。"""
    img, d = slide_base(size, "全体構成", page=page)
    w, _ = size
    rw, gap = 300, 64
    bw = (w - 180 - rw - gap * len(boxes)) // len(boxes)
    x, y, bh = 90, 220, 300
    for head, body in boxes:
        d.rectangle((x, y, x + bw, y + bh), fill=YELLOW, outline=INK, width=2)
        d.text((x + bw / 2, y + 60), head, font=font(38, "bold"), fill=INK, anchor="mm")
        d.multiline_text((x + bw / 2, y + 185), body, font=font(29), fill=INK, anchor="mm", align="center", spacing=10)
        x += bw
        d.line((x + 8, y + bh / 2, x + gap - 14, y + bh / 2), fill=INK, width=4)
        d.polygon([(x + gap - 8, y + bh / 2), (x + gap - 24, y + bh / 2 - 10), (x + gap - 24, y + bh / 2 + 10)], fill=INK)
        x += gap
    rob = robot_img.resize((rw, int(robot_img.height * rw / robot_img.width)))
    img.paste(rob, (x, y + bh // 2 - rob.height // 2))
    d.rectangle((x, y + bh // 2 - rob.height // 2, x + rw, y + bh // 2 + rob.height // 2), outline=INK, width=2)
    y = 600
    for head, body in rows:
        d.rectangle((90, y, w - 90, y + 76), fill=(255, 255, 255), outline=INK, width=2)
        d.rectangle((90, y, 300, y + 76), fill=GRAY, outline=INK, width=2)
        d.text((195, y + 38), head, font=font(32, "bold"), fill=INK, anchor="mm")
        d.text((330, y + 38), body, font=font(32), fill=INK, anchor="lm")
        y += 96
    return np.asarray(img)


def points_card(size, title, lead, items, page=None):
    """番号タグ＋太字の見出し＋説明 の箇条書き（items＝(見出し, 説明)）。説明中の【…】は赤で強調。"""
    img, d = slide_base(size, title, page=page)
    w, _ = size
    y = 190
    if lead:
        d.text((90, y), lead, font=font(44, "bold"), fill=INK)
        y += 100
    step = (1000 - y) / max(len(items), 1)
    for i, (head, body) in enumerate(items):
        d.rectangle((90, y, 150, y + 60), fill=SLATE)
        d.text((120, y + 30), str(i + 1), font=font(34, "bold"), fill=(255, 255, 255), anchor="mm")
        d.text((180, y + 30), head, font=font(40, "bold"), fill=INK, anchor="lm")
        _rich(d, (180, y + 82), body, font(32), w - 270)
        y += step
    return np.asarray(img)


_LATIN = "C:/Windows/Fonts/segoeui.ttf"
_HALF = "°"  # 和文フォントでは全角幅になり後ろが空くので、欧文フォントで描く文字


def _runs(text, fnt):
    """text を (文字列, フォント) の並びに分ける。"""
    out, cur = [], ""
    for ch in text:
        if ch in _HALF and os.path.exists(_LATIN):
            if cur:
                out.append((cur, fnt))
                cur = ""
            out.append((ch, ImageFont.truetype(_LATIN, fnt.size)))
        else:
            cur += ch
    if cur:
        out.append((cur, fnt))
    return out


def _textlength(d, text, fnt):
    return sum(d.textlength(s, font=f) for s, f in _runs(text, fnt))


def _draw(d, xy, text, fnt, fill):
    """text を描いて、描き終わりの x を返す。"""
    x, y = xy
    base = y + fnt.getmetrics()[0]  # フォントが混ざっても高さがずれないよう、ベースラインで揃える
    for s, f in _runs(text, fnt):
        d.text((x, base), s, font=f, fill=fill, anchor="ls")
        x += d.textlength(s, font=f)
    return x


def _rich(d, xy, text, fnt, width):
    """【】で囲んだ部分を赤字にして、幅 width で折り返して描く。字間が崩れないよう、同じ色の続きはまとめて描く。"""
    x0, y = xy
    for line in text.split("\n"):
        x = x0
        for i, seg in enumerate(re.split("[【】]", line)):
            color = RED if i % 2 else INK
            while seg:
                n = len(seg)  # 行の残りに収まる最長の部分
                while n > 1 and x + _textlength(d, seg[:n], fnt) > x0 + width:
                    n -= 1
                if x > x0 and x + _textlength(d, seg[:n], fnt) > x0 + width:
                    x, y = x0, y + fnt.size * 1.5
                    continue
                x = _draw(d, (x, y), seg[:n], fnt, color)
                seg = seg[n:]
                if seg:
                    x, y = x0, y + fnt.size * 1.5
        y += fnt.size * 1.5


def results_card(size, title, chart_title, bars, table_title, table, note="", page=None):
    """左：棒グラフ（bars＝(名前, 0〜1 の値)）、右：表（table＝[見出し行, 行...]）と注記。"""
    img, d = slide_base(size, title, page=page)
    w, h = size
    # 棒グラフ
    gx0, gy0, gx1, gy1 = 170, 330, 930, 880
    d.text((90, 220), chart_title, font=font(36, "bold"), fill=INK)
    for v in (0, 50, 100):
        yy = gy1 - (gy1 - gy0) * v / 100
        d.line((gx0, yy, gx1, yy), fill=GRAY if v else INK, width=2)
        d.text((gx0 - 12, yy), f"{v}%", font=font(24), fill=INK, anchor="rm")
    d.line((gx0, gy0, gx0, gy1), fill=INK, width=2)
    bwid = (gx1 - gx0) / len(bars)
    for i, (name, v) in enumerate(bars):
        x0 = gx0 + i * bwid + bwid * 0.2
        x1 = gx0 + (i + 1) * bwid - bwid * 0.2
        yy = gy1 - (gy1 - gy0) * v
        d.rectangle((x0, yy, x1, gy1), fill=YELLOW, outline=INK, width=2)
        d.text(((x0 + x1) / 2, yy - 26), f"{v * 100:.0f}%", font=font(28, "bold"), fill=INK, anchor="mm")
        d.multiline_text(((x0 + x1) / 2, gy1 + 44), name, font=font(26), fill=INK, anchor="mm", align="center")
    # 表
    tx0, tx1, ty = 1010, w - 90, 330
    d.text((tx0, 220), table_title, font=font(36, "bold"), fill=INK)
    cols = len(table[0])
    widths = [0.52] + [(0.48 / (cols - 1))] * (cols - 1)
    rh = 70
    for r, row in enumerate(table):
        x = tx0
        for c, cell in enumerate(row):
            cw = (tx1 - tx0) * widths[c]
            d.rectangle((x, ty, x + cw, ty + rh), fill=SLATE if r == 0 else (255, 255, 255), outline=INK, width=2)
            d.text((x + (16 if c == 0 else cw / 2), ty + rh / 2), cell, font=font(28, "bold" if r == 0 else "regular"),
                   fill=(255, 255, 255) if r == 0 else INK, anchor="lm" if c == 0 else "mm")
            x += cw
        ty += rh
    if note:
        _rich(d, (tx0, ty + 40), note, font(30), tx1 - tx0)
    return np.asarray(img)


def _code_box(d, box, title, lines, size=24):
    """見出し付きのコード（等幅）の枠。"""
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=(248, 248, 248), outline=INK, width=2)
    d.rectangle((x0, y0, x1, y0 + 46), fill=GRAY, outline=INK, width=2)
    d.text((x0 + 16, y0 + 23), title, font=font(26, "bold"), fill=INK, anchor="lm")
    mono = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size) if os.path.exists("C:/Windows/Fonts/consola.ttf") else font(size)
    y = y0 + 62
    for ln in lines:
        f = font(size) if any(ord(c) > 127 for c in ln) else mono
        d.text((x0 + 16, y), ln, font=f, fill=INK)
        y += size * 1.45


def llm_card(size, instruction, prompt_lines, schema_lines, output_lines, notes, page=None):
    """ローカル LLM の構成：指示 → プロンプト＋JSON Schema → LLM → 出力 JSON。"""
    img, d = slide_base(size, "ローカル LLM の構成", page=page)
    w, _ = size
    d.rectangle((90, 190, 520, 330), fill=LIGHT, outline=INK, width=2)
    d.text((110, 210), "入力（日本語の指示）", font=font(26, "bold"), fill=INK)
    _rich(d, (110, 255), f"「{instruction}」", font(28), 390)
    d.rectangle((90, 380, 520, 560), fill=YELLOW, outline=INK, width=2)
    d.text((305, 430), "gemma4（4B）", font=font(36, "bold"), fill=INK, anchor="mm")
    d.text((305, 490), "Ollama で PC 内で動作", font=font(28), fill=INK, anchor="mm")
    d.line((305, 330, 305, 372), fill=INK, width=4)
    d.polygon([(305, 380), (295, 362), (315, 362)], fill=INK)
    _code_box(d, (560, 190, 1260, 560), "システムプロンプト（抜粋）", prompt_lines, 22)
    _code_box(d, (560, 590, 1260, 950), "JSON Schema（出力の形を固定）", schema_lines, 22)
    _code_box(d, (1300, 190, w - 90, 700), "出力（実際の結果）", output_lines, 22)
    for i, (head, body) in enumerate(notes):
        y = 610 + i * 120
        d.rectangle((90, y, 150, y + 56), fill=SLATE)
        d.text((120, y + 28), str(i + 1), font=font(30, "bold"), fill=(255, 255, 255), anchor="mm")
        d.text((170, y + 2), head, font=font(30, "bold"), fill=INK)
        _rich(d, (170, y + 48), body, font(24), 370)
    d.line((520, 470, 552, 470), fill=INK, width=3)
    d.line((1260, 445, 1292, 445), fill=INK, width=3)
    d.polygon([(1300, 445), (1284, 435), (1284, 455)], fill=INK)
    return np.asarray(img)


def image_card(size, title, image, caption="", page=None, top=180):
    """図を 1 枚大きく載せるスライド（image は PIL 画像）。"""
    img, d = slide_base(size, title, page=page)
    w, h = size
    bottom = h - (130 if caption else 70)
    im = image.copy()
    im.thumbnail((w - 180, bottom - top))
    img.paste(im, ((w - im.width) // 2, top + (bottom - top - im.height) // 2))
    if caption:
        _rich(d, (90, h - 110), caption, font(30), w - 180)
    return np.asarray(img)


def credit_card(size, lines, page=None):
    """最後のクレジット（音声など）。"""
    img, d = slide_base(size, "クレジット", page=page)
    y = 220
    for line in lines:
        d.text((90, y), line, font=font(36), fill=INK)
        y += 70
    return np.asarray(img)


def slide_title(slide_path, title, subtitle):
    """講座の「最終課題発表」スライドの空き（ロゴの上）に作品タイトルを書き込む。"""
    img = Image.open(slide_path).convert("RGB").resize((1920, 1080))
    d = ImageDraw.Draw(img)
    d.text((88, 700), title, font=font(60, "bold"), fill=(20, 20, 20))
    d.text((92, 790), subtitle, font=font(29), fill=(110, 104, 80))
    return np.asarray(img)
