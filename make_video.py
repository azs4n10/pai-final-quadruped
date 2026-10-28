"""
提出動画を組み立てる：OP → タイトル → しくみ → 学習の成長 → 指示デモ×3 → まとめ → ED

  python make_video.py --op path/to/OP.mp4 --ed path/to/ED.mp4 --slide path/to/slide.jpg
"""
import argparse
import json
import os
import subprocess
import sys

import imageio.v2 as imageio
import imageio_ffmpeg

from overlay import slide_title, title_card

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = "dog"  # 提出動画に使うロボット（ant / dog）
RUN = os.path.join(HERE, "runs", f"goal_{ROBOT}")
CLIPS = os.path.join(HERE, "runs", "clips" if ROBOT == "ant" else f"clips_{ROBOT}")
SIZE = (1920, 1080)
FPS = 30

# 学習の成長（同じゴール＝赤い柱、各 6 秒）
ADOPTED = {"ant": 7_000_000, "dog": 29_000_000}[ROBOT]  # 採用したチェックポイントのステップ数
GROWTH = {
    "ant": [("000000016", "学習前（ランダムな動き）"), ("001000000", "学習 100万ステップ"),
            ("002000000", "学習 200万ステップ"), (None, "学習 700万ステップ（採用モデル）")],
    "dog": [("000000016", "学習前（ランダムな動き）"), ("001000000", "学習 100万ステップ"),
            ("005000000", "学習 500万ステップ"), (None, "")],
}[ROBOT]
DEMO_SEED = 1
DEMOS = [
    ("demo1", "赤い柱に寄ってから青い柱へ行き、最後はホームに戻って", 1),
    ("demo2", "黄色い柱に荷物を届けたら帰ってきて", 1),
    ("demo3", "全部の柱を一周してから戻ってきて", 2),
]


def model_path(tag):
    if tag is None:
        return os.path.join(HERE, "models", "final_model.zip" if ROBOT == "ant" else f"{ROBOT}_model.zip")
    return os.path.join(RUN, "ckpt", f"model_{tag}.zip")


def render_clips(force=False):
    os.makedirs(CLIPS, exist_ok=True)
    py = sys.executable
    for i, (tag, label) in enumerate(GROWTH):
        out = os.path.join(CLIPS, f"growth{i}.mp4")
        if tag is None:
            label = f"学習 {ADOPTED // 10000}万ステップ（採用モデル）"
        if force or not os.path.exists(out):
            subprocess.run([py, os.path.join(HERE, "demo.py"), "--robot", ROBOT, "--route", "red,home", "--model",
                            model_path(tag), "--label", label, "--seconds", "6", "--out", out], check=True)
    for name, text, speed in DEMOS:
        out = os.path.join(CLIPS, f"{name}.mp4")
        if force or not os.path.exists(out):
            cmd = [py, os.path.join(HERE, "demo.py"), "--robot", ROBOT, "--text", text, "--speed", str(speed),
                   "--seed", str(DEMO_SEED), "--out", out]
            log_path = os.path.splitext(out)[0] + ".json"
            if not os.path.exists(log_path):  # LLM の計画は ant 版の録画時の結果を使う
                log_path = os.path.join(HERE, "runs", "clips", f"{name}.json")
            prev = json.load(open(log_path, encoding="utf-8")) if os.path.exists(log_path) else None
            if prev:  # LLM の計画は前回の結果を使い回す（温度 0 なので同じ出力）
                cmd += ["--route", ",".join(prev["route"]), "--reply", prev["reply"]]
            subprocess.run(cmd, check=True)
            if prev:
                log_path = os.path.splitext(out)[0] + ".json"
                log = json.load(open(log_path, encoding="utf-8"))
                log["planner"] = prev["planner"]
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, ensure_ascii=False, indent=1)


def card(name, seconds, *a, frame=None, **kw):
    out = os.path.join(CLIPS, f"card_{name}.mp4")
    if frame is None:
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
        last = ev[str(ADOPTED)]  # 採用モデル
        first = ev[min(ev, key=int)]
        e100 = os.path.join(RUN, "eval100.json")  # 採用モデルは 100 回で測り直した値を使う
        if os.path.exists(e100):
            last = json.load(open(e100))[f"100trials_{ADOPTED // 1_000_000}M"]
        lines.append(f"ゴール到達率（4〜6 m 先・30 秒以内）：学習前 {first['success']*100:.0f}% → "
                     f"学習後（{ADOPTED // 10000}万ステップ） {last['success']*100:.0f}%")
    rt_path = os.path.join(RUN, "route_eval.json")
    if os.path.exists(rt_path):
        rt = json.load(open(rt_path, encoding="utf-8"))
        rates = " / ".join(f"{v['success']*100:.0f}%" for v in rt.values())
        lines.append(f"デモ3経路の成功率（各20回）：{rates}")
    gait = os.path.join(RUN, "gait_compare.json")
    if os.path.exists(gait):
        g = json.load(open(gait))
        lines.append(f"歩き方：対角の脚をそろえて出すトロット（平均 {g['dog']['speed']:.2f} m/秒、"
                     f"対角の脚の接地がそろう割合 {g['dog']['diag']*100:.0f}%）")
    lines += [
              "LLM はローカル実行（gemma4 / Ollama）。外部 API なしで動作",
              "今後：障害物回避、転倒からの起き上がり、実機の四足ロボットへの sim-to-real"]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True)
    p.add_argument("--ed", required=True)
    p.add_argument("--slide", default=None, help="講座の最終課題発表スライド（タイトル画面に使う）")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "final_video.mp4"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    render_clips(args.force)
    parts = [args.op]
    title, sub = "日本語で指示できる四足歩行ロボット", "ローカルLLM × 強化学習（PPO）で「指示どおりに歩く」"
    if args.slide:
        parts.append(card("title", 4, frame=slide_title(args.slide, title, sub)))
    else:
        parts.append(card("title", 4, title, sub))
    parts.append(card("system", 8, "しくみ", "", [
        "① 日本語の指示を、ローカルLLM（gemma4）が経由地のリストに変換",
        "② 強化学習で身につけた「ゴールまで歩く」方策が、経由地を1つずつ目指す",
        "③ 物理シミュレータ MuJoCo 上に自作した犬型ロボット（12関節）で実行",
        "   観測：姿勢・関節の角度と速度 ＋ ゴールの方向と距離（胴体から見た向き）",
        "   報酬：目標の速さ（0.8 m/秒）でゴールへ ＋ 到着ボーナス ＋ トロットのリズム",
        "           − 胴体の傾き・横歩き・脚の開き・急な動き",
    ]))
    parts.append(card("growth", 3, "学習による成長", "同じゴール（赤い柱 → ホーム）を目指す様子を、学習段階ごとに6秒ずつ比較"))
    parts += [os.path.join(CLIPS, f"growth{i}.mp4") for i in range(len(GROWTH))]
    parts.append(card("demo", 3, "日本語で指示してみる", "指示 → LLM が経由地を計画 → 歩いて順に回る（3本目は2倍速）"))
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
