"""
説明資料（report/report.html → PDF）を、評価の結果（runs/big_stairs2/*.json）から作る。

  python make_report.py
"""
import json
import os
import shutil
import subprocess

import imageio.v2 as imageio
from PIL import Image

from home_demo import NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "report")
RUN = os.path.join(HERE, "runs", "big_stairs2")
CLIPS = os.path.join(HERE, "runs", "clips_home")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def pct(v):
    return f"{v * 100:.0f}%"


def frame(clip, sec, out):
    """デモ映像の sec 秒目のコマを、レポート用の画像にする。"""
    r = imageio.get_reader(os.path.join(CLIPS, clip))
    fps = r.get_meta_data()["fps"]
    for i, f in enumerate(r):
        if i >= sec * fps:
            Image.fromarray(f).resize((960, 540)).save(os.path.join(REPORT, out), quality=88)
            break
    r.close()


def stairs_table(st):
    rows = ["<table>", "<tr><th>段差</th><th>上り：成功</th><th>上り：転倒</th><th>下り：成功</th><th>下り：転倒</th></tr>"]
    for k, v in st.items():
        mark = ' style="background:#fff2cc"' if k == "15" else ""
        rows.append(f"<tr{mark}><td>{k} cm{'（家の階段）' if k == '15' else ''}</td><td class=\"n\">{pct(v['up']['success'])}</td>"
                    f"<td class=\"n\">{pct(v['up']['fell'])}</td><td class=\"n\">{pct(v['down']['success'])}</td>"
                    f"<td class=\"n\">{pct(v['down']['fell'])}</td></tr>")
    return "\n".join(rows + ["</table>"])


def home_table(hv):
    rows = ["<table>", "<tr><th>経路</th><th>成功</th><th>転倒</th><th>成功時の平均所要時間</th></tr>"]
    for k, v in hv.items():
        route = " → ".join(NAMES[n] for n in k.split("→"))
        t = f"{v['mean_time']:.0f} 秒" if v["mean_time"] else "―"
        rows.append(f"<tr><td>{route}</td><td class=\"n\">{pct(v['success'])}</td><td class=\"n\">{pct(v['fell'])}</td><td class=\"n\">{t}</td></tr>")
    return "\n".join(rows + ["</table>"])


def main():
    st = json.load(open(os.path.join(RUN, "stairs_eval.json")))
    hv = json.load(open(os.path.join(RUN, "home_eval.json"), encoding="utf-8"))
    climb = json.load(open(os.path.join(RUN, "climb.json")))
    frame("demo1.mp4", 11.5, "demo1.jpg")
    frame("demo1.mp4", 7.0, "demo2.jpg")
    final = climb[max(climb, key=int)]
    html = open(os.path.join(REPORT, "report_template.html"), encoding="utf-8").read()
    fill = {
        "{{STEPS}}": "約 1.2 億",
        "{{HOURS}}": "13",
        "{{STAIRS_TABLE}}": stairs_table(st),
        "{{HOME_TABLE}}": home_table(hv),
        "{{HOME_TEXT}}": (f"<p>家の階段（15 cm × 14 段）だけを取り出した試験では、最終の方策で 10 回中 {final['success'] * 10:.0f} 回上り切った"
                          "（学習の経過は下のグラフ）。</p>"),
        "{{DISCUSSION}}": open(os.path.join(REPORT, "discussion.html"), encoding="utf-8").read(),
    }
    for k, v in fill.items():
        html = html.replace(k, v)
    out = os.path.join(REPORT, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    pdf = os.path.join(REPORT, "report.pdf")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out], check=True)
    shutil.copy(pdf, os.path.join(REPORT, "説明資料_PAI最終課題_maichans.pdf"))
    print("->", pdf)


if __name__ == "__main__":
    main()
