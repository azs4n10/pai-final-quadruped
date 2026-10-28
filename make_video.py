"""
提出動画を組み立てる：OP → タイトル → しくみ → 学習の成長 → 指示デモ×3 → まとめ → ED（ナレーション付き）

  python make_video.py --op path/to/OP.mp4 --ed path/to/ED.mp4 --slide path/to/slide.jpg
"""
import argparse
import json
import os
import subprocess
import sys

import imageio.v2 as imageio
import imageio_ffmpeg

from overlay import slide_title

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


# ナレーション（VOICEVOX:ずんだもん）。部品ごとに (開始秒, 文)
VOICE = {
    "title": [(0.4, "日本語で指示できる、四足歩行ロボットを紹介します。")],
    "system": [(0.5, "日本語の指示を、パソコンの中で動くLLMが、立ち寄る場所のリストに変換します。"),
               (7.3, "そのリストに沿って、強化学習で歩き方を身につけたロボットが、ひとつずつゴールを目指します。")],
    "growth": [(0.3, "まず、学習によって歩き方が上達していく様子をご覧ください。")],
    "growth0": [(0.3, "学習前は、脚を動かしても前に進めません。")],
    "growth1": [(0.3, "百万ステップでは前に進めるようになりましたが、まだ転倒します。")],
    "growth2": [(0.3, "五百万ステップで、対角の脚をそろえて歩けるようになりました。")],
    "growth3": [(0.3, "こちらが採用したモデルです。安定してゴールへ向かいます。")],
    "demo": [(0.3, "それでは、日本語で指示を出してみます。")],
    "demo1": [(0.6, "赤い柱に寄ってから青い柱、最後にホーム。LLMが、三つの場所の順番に変換しました。"),
              (14.0, "場所に着くたびに、次のゴールへ切り替えて歩いていきます。")],
    "demo2": [(0.6, "荷物を届けて帰ってきて、という言い方でも、黄色い柱とホームのことだと理解できています。")],
    "demo3": [(0.6, "全部の柱を一周、という指示も、五つの場所の順番に展開されました。ここからは2倍速です。"),
              (14.0, "転倒することなく、順番どおりに柱を回っていきます。")],
    "summary": [(0.5, "ゴールへの到達は百回中百回、指示どおりの経路も、六十回すべて成功しました。"),
                (7.6, "LLMはパソコンの中だけで動作するため、外部のサービスにデータを送る必要がありません。")],
}


def card(name, seconds, frame):
    """静止画の部品。ナレーションが収まる長さに伸ばす。"""
    from narration import seconds as voice_seconds

    need = max((t + voice_seconds(text) + 0.8 for t, text in VOICE.get(name, [])), default=0)
    out = os.path.join(CLIPS, f"card_{name}.mp4")
    w = imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=8)
    for _ in range(int(max(seconds, need) * FPS)):
        w.append_data(frame)
    w.close()
    return out


def summary_stats():
    e100 = json.load(open(os.path.join(RUN, "eval100.json")))[f"100trials_{ADOPTED // 1_000_000}M"]
    rt = json.load(open(os.path.join(RUN, "route_eval.json"), encoding="utf-8"))
    ok = sum(round(v["success"] * 20) for v in rt.values())
    g = json.load(open(os.path.join(RUN, "gait_compare.json")))["dog"]
    return [(f"{e100['success'] * 100:.0f}%", "ゴール到達\n（4〜6 m 先・100回）"),
            (f"{ok}/{len(rt) * 20}", "指示どおりの経路\n（3経路×20回）"),
            (f"{g['speed']:.2f}", f"m/秒のトロット歩行\n（対角の脚がそろう {g['diag'] * 100:.0f}%）"),
            ("0 件", "外部 API の利用\n（LLM は PC 内で動作）")]


def duration(path):
    return imageio_ffmpeg.count_frames_and_secs(path)[1]


def main():
    from PIL import Image

    from narration import write_track
    from overlay import section_card, summary_card, system_card

    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True)
    p.add_argument("--ed", required=True)
    p.add_argument("--slide", default=None, help="講座の最終課題発表スライド（タイトル画面に使う）")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "final_video.mp4"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    render_clips(args.force)
    title, sub = "日本語で指示できる四足歩行ロボット", "ローカルLLM × 強化学習（PPO）で「指示どおりに歩く」"
    robot = Image.open(os.path.join(HERE, "report", "robot.jpg")).crop((120, 60, 860, 560))
    parts = [("op", args.op),
             ("title", card("title", 4, slide_title(args.slide, title, sub) if args.slide else section_card(SIZE, title, sub))),
             ("system", card("system", 8, system_card(SIZE, robot))),
             ("growth", card("growth", 3, section_card(SIZE, "学習による成長", "同じゴール（赤い柱 → ホーム）を、学習段階ごとに 6 秒ずつ")))]
    parts += [(f"growth{i}", os.path.join(CLIPS, f"growth{i}.mp4")) for i in range(len(GROWTH))]
    parts.append(("demo", card("demo", 3, section_card(SIZE, "日本語で指示してみる", "指示 → LLM が経由地を計画 → 順に歩いて回る（3本目は2倍速）"))))
    parts += [(name, os.path.join(CLIPS, f"{name}.mp4")) for name, _, _ in DEMOS]
    parts.append(("summary", card("summary", 8, summary_card(SIZE, summary_stats(), [
        "今後：障害物の回避、転倒からの起き上がり、実機の四足ロボットへの sim-to-real"], "音声：VOICEVOX:ずんだもん"))))
    parts.append(("ed", args.ed))

    # ナレーションを部品の開始時刻に合わせて 1 本の音声トラックにする
    from narration import seconds as voice_seconds

    events, t, pads = [], 0.0, []
    for name, path in parts:
        d = duration(path)
        pad = max(6.0 - d, 0.0) if name.startswith("growth") and name != "growth" else 0.0  # 転倒で早く終わった成長記録は最後の画面で止める
        pads.append(pad)
        d += pad
        for off, text in VOICE.get(name, []):
            if off + voice_seconds(text) > d:
                print(f"[warn] {name}: ナレーションが映像より長い（{off + voice_seconds(text):.1f}s > {d:.1f}s）")
            events.append((t + off, text))
        t += d
    voice = os.path.join(CLIPS, "voice.wav")
    write_track(events, t, voice)

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for _, prt in parts:
        cmd += ["-i", prt]
    cmd += ["-i", voice]
    n = len(parts)
    norm = "".join(
        f"[{i}:v]scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease,"
        f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p,setsar=1"
        + (f",tpad=stop_mode=clone:stop_duration={pads[i]:.3f}" if pads[i] else "") + f"[v{i}];"
        for i in range(n)
    )
    concat = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", norm + concat, "-map", "[out]", "-map", f"{n}:a", "-c:v", "libx264", "-crf", "20",
            "-preset", "medium", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    probe = subprocess.run([ff, "-i", args.out], capture_output=True, text=True).stderr
    print([ln.strip() for ln in probe.splitlines() if "Duration" in ln], "->", args.out)


if __name__ == "__main__":
    main()
