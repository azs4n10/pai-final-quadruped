"""
提出動画を組み立てる：OP → タイトル → 仕組み → 学習の成長 → 指示デモ×3 → まとめ → ED

  python make_video.py --op path/to/OP.mp4 --ed path/to/ED.mp4
"""
import argparse
import json
import os
import subprocess
import sys

import imageio.v2 as imageio
import imageio_ffmpeg

from overlay import title_card

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs", "goal_ant")
CLIPS = os.path.join(HERE, "runs", "clips")
SIZE = (1920, 1080)
FPS = 30

# 学習の成長（同じゴール＝赤い柱、各 6 秒）
GROWTH = [
    ("000000016", "学習前（ランダムな動き）"),
    ("001000000", "学習 100万ステップ"),
    ("002000000", "学習 200万ステップ"),
    (None, "学習 700万ステップ（採用モデル）"),
]
DEMOS = [
    ("demo1", "赤い柱に寄ってから青い柱へ行き、最後はホームに戻って", 1),
    ("demo2", "黄色い柱に荷物を届けたら帰ってきて", 1),
    ("demo3", "全部の柱を一周してから戻ってきて", 1),
]


def model_path(tag):
    return os.path.join(HERE, "models", "final_model.zip") if tag is None else os.path.join(RUN, "ckpt", f"model_{tag}.zip")


def render_clips(force=False):
    os.makedirs(CLIPS, exist_ok=True)
    py = sys.executable
    for i, (tag, label) in enumerate(GROWTH):
        out = os.path.join(CLIPS, f"growth{i}.mp4")
        if force or not os.path.exists(out):
            subprocess.run([py, os.path.join(HERE, "demo.py"), "--route", "red,home", "--model", model_path(tag),
                            "--label", label, "--seconds", "6", "--out", out], check=True)
    for name, text, speed in DEMOS:
        out = os.path.join(CLIPS, f"{name}.mp4")
        if force or not os.path.exists(out):
            subprocess.run([py, os.path.join(HERE, "demo.py"), "--text", text, "--speed", str(speed),
                            "--seed", "1", "--out", out], check=True)


def card(name, seconds, *a, **kw):
    out = os.path.join(CLIPS, f"card_{name}.mp4")
    frame = title_card(SIZE, *a, **kw)
    w = imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=8)
    for _ in range(int(seconds * FPS)):
        w.append_data(frame)
    w.close()
    return out


def summary_lines():
    lines = []
    ev_path = os.path.join(RUN, "eval.json")
    if os.path.exists(ev_path):
        ev = json.load(open(ev_path))
        last = ev["7000000"]  # 採用モデル
        first = ev[min(ev, key=int)]
        lines.append(f"ゴール到達率（4〜6 m 先・30 秒以内）：学習前 {first['success']*100:.0f}% → 学習後（700万ステップ） {last['success']*100:.0f}%")
    rt_path = os.path.join(RUN, "route_eval.json")
    if os.path.exists(rt_path):
        rt = json.load(open(rt_path, encoding="utf-8"))
        rates = " / ".join(f"{v['success']*100:.0f}%" for v in rt.values())
        lines.append(f"デモ3経路の成功率（各20回）：{rates}")
    lines += ["失敗は歩行中の転倒と、途中で進めなくなる停止。回復動作の追加が今後の課題",
              "LLM はローカル実行（gemma4 / Ollama）。外部 API なしで動作",
              "今後：障害物回避、実機（四足ロボット）への sim-to-real"]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True)
    p.add_argument("--ed", required=True)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "final_video.mp4"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    render_clips(args.force)
    parts = [args.op]
    parts.append(card("title", 4, "ことばで動く四足配送ロボット",
                      "ローカルLLM × 強化学習（PPO）で「指示どおりに歩く」"))
    parts.append(card("system", 8, "しくみ", "", [
        "① 日本語の指示を、ローカルLLM（gemma4）が経由地のリストに変換",
        "② 強化学習で身につけた「ゴールまで歩く」方策が、経由地を1つずつ目指す",
        "③ 物理シミュレータ MuJoCo 上の4脚ロボット（Ant）で実行",
        "   観測：関節の角度・速度 ＋ ゴールの方向と距離（胴体から見た向き）",
        "   報酬：ゴールに近づいた速さ ＋ 到着ボーナス − 転倒・無駄な力",
    ]))
    parts.append(card("growth", 3, "学習による成長", "同じゴール（赤い柱 → ホーム）を目指す様子を、学習段階ごとに6秒ずつ比較"))
    parts += [os.path.join(CLIPS, f"growth{i}.mp4") for i in range(len(GROWTH))]
    parts.append(card("demo", 3, "日本語で指示してみる", "指示 → LLM が経由地を計画 → 歩いて順に回る（等速）"))
    parts += [os.path.join(CLIPS, f"{name}.mp4") for name, _, _ in DEMOS]
    parts.append(card("summary", 8, "まとめ", "", summary_lines()))
    parts.append(args.ed)

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for prt in parts:
        cmd += ["-i", prt]
    norm = "".join(
        f"[{i}:v]scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease,"
        f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p,setsar=1[v{i}];"
        for i in range(len(parts))
    )
    concat = "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[out]"
    cmd += ["-filter_complex", norm + concat, "-map", "[out]", "-c:v", "libx264", "-crf", "20",
            "-preset", "medium", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    probe = subprocess.run([ff, "-i", args.out], capture_output=True, text=True).stderr
    print([ln.strip() for ln in probe.splitlines() if "Duration" in ln], "->", args.out)


if __name__ == "__main__":
    main()
