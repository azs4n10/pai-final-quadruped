"""
提出動画を組み立てる：OP → タイトル → しくみ → 学習の成長 → 家と庭での指示デモ×3 → まとめ → ED（ナレーション付き）

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
RUN = os.path.join(HERE, "runs", "terrain_dog")
CLIPS = os.path.join(HERE, "runs", "clips_house")
MODEL = os.path.join(HERE, "models", "terrain_model.zip")
SIZE = (1920, 1080)
FPS = 30

ADOPTED = 50_000_000  # 採用モデル（学習の最終時点）のステップ数
GROWTH_SECONDS = 10
# 学習の成長：充電ステーションから庭の階段の上の高台へ
GROWTH = [("000000016", "学習前"), ("010000000", "学習 1000万ステップ"), ("025000000", "学習 2500万ステップ"),
          (None, "学習 5000万ステップ（採用モデル）")]
DEMO_SEED = 1
DEMOS = [
    ("demo1", "キッチンで荷物を受け取って、寝室に届けて", 1),
    ("demo2", "庭の階段を上って高台に行き、そのあと坂の上にも寄ってから充電ステーションに戻って", 2),
    ("demo3", "子ども部屋のおもちゃ箱を見てきて、リビングを通って帰ってきて", 2),
]


def model_path(tag):
    return MODEL if tag is None else os.path.join(RUN, "ckpt", f"model_{tag}.zip")


def render_clips(force=False):
    os.makedirs(CLIPS, exist_ok=True)
    py, demo = sys.executable, os.path.join(HERE, "house_demo.py")
    for i, (tag, label) in enumerate(GROWTH):
        out = os.path.join(CLIPS, f"growth{i}.mp4")
        if force or not os.path.exists(out):
            subprocess.run([py, demo, "--route", "stairs_top", "--model", model_path(tag), "--label", label,
                            "--seconds", str(GROWTH_SECONDS), "--seed", str(DEMO_SEED), "--out", out], check=True)
    for name, text, speed in DEMOS:
        out = os.path.join(CLIPS, f"{name}.mp4")
        if force or not os.path.exists(out):
            cmd = [py, demo, "--text", text, "--speed", str(speed), "--seed", str(DEMO_SEED), "--out", out]
            log_path = os.path.splitext(out)[0] + ".json"
            prev = json.load(open(log_path, encoding="utf-8")) if os.path.exists(log_path) else None
            if prev:  # LLM の計画は前回の結果を使い回す（温度 0 なので同じ出力）
                cmd += ["--route", ",".join(prev["route"]), "--reply", prev["reply"]]
            subprocess.run(cmd, check=True)
            if prev:
                log = json.load(open(log_path, encoding="utf-8"))
                log["planner"] = prev["planner"]
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, ensure_ascii=False, indent=1)


# ナレーション（VOICEVOX:ずんだもん）。部品ごとに (開始秒, 文)
VOICE = {
    "title": [(0.4, "日本語で指示できる、四足歩行ロボットを紹介します。")],
    "system": [(0.5, "日本語の指示を、パソコンの中で動くLLMが、立ち寄る部屋の順番に変換します。"),
               (7.4, "間取り図から壁を避ける経路を計算し、強化学習で歩き方を身につけたロボットが、段差や坂も越えて進みます。")],
    "growth": [(0.3, "まず、学習によって階段を上れるようになる様子をご覧ください。")],
    "growth0": [(0.3, "学習前は、脚を動かしても前に進めません。")],
    "growth1": [(0.3, "一千万ステップでは歩けるようになりましたが、段差ではまだ不安定です。")],
    "growth2": [(0.3, "二千五百万ステップでは、階段を上れるようになってきました。")],
    "growth3": [(0.3, "こちらが採用したモデルです。足元の高さを感じ取りながら、階段を上り切ります。")],
    "demo": [(0.3, "それでは、家と庭で、日本語の指示を出してみます。")],
    "demo1": [(0.6, "キッチンで荷物を受け取って、寝室に届けて。LLMが、キッチンと寝室の順番に変換しました。"),
              (12.0, "玄関の階段を上り、ドアを通って、壁を避けながら進みます。")],
    "demo2": [(0.6, "庭の階段と坂を上る指示です。ここからは2倍速です。"),
              (9.0, "階段の高台、坂の上、充電ステーションの順に回ります。")],
    "demo3": [(0.6, "子ども部屋を見てから、リビングを通って帰る指示です。こちらも2倍速です。")],
    "summary": [(0.5, "最も難しい地形でも、ゴールへの到達率は高く、家と庭での指示も成功しました。"),
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
    ev = json.load(open(os.path.join(RUN, "eval_final.json")))
    hv = json.load(open(os.path.join(RUN, "house_eval.json"), encoding="utf-8"))
    ok = sum(round(v["success"] * 20) for v in hv.values())
    return [(f"{ev['mean'] * 100:.0f}%", "ゴール到達（最難の地形）\n段差 10 cm・坂・凸凹"),
            (f"{ok}/{len(hv) * 20}", "家と庭での指示\n（3経路×20回）"),
            ("10 cm", "上れる段差\n（体高の約 3 割）"),
            ("0 件", "外部 API の利用\n（LLM は PC 内で動作）")]


def duration(path):
    return imageio_ffmpeg.count_frames_and_secs(path)[1]


def main():
    from PIL import Image

    from narration import seconds as voice_seconds
    from narration import write_srt, write_track
    from overlay import section_card, summary_card, system_card

    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True)
    p.add_argument("--ed", required=True)
    p.add_argument("--slide", default=None, help="講座の最終課題発表スライド（タイトル画面に使う）")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "final_video.mp4"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    render_clips(args.force)
    title, sub = "日本語で指示できる四足歩行ロボット", "ローカルLLM × 経路計画 × 強化学習で、家の中も段差も歩く"
    robot = Image.open(os.path.join(HERE, "report", "robot.jpg")).crop((120, 60, 860, 560))
    system = system_card(SIZE, robot, [
        ("日本語の指示", "「キッチンで荷物を\n受け取って…」"),
        ("ローカル LLM", "gemma4（Ollama）\n部屋の順番に変換"),
        ("経路計画", "間取り図から A* で\n壁を避ける経由地"),
        ("歩行方策", "強化学習（PPO）\n段差・坂も歩く"),
    ], [
        ("観測", "胴体の姿勢・速度、関節の角度と速度、ゴールの方向と距離、足元 35 点の地面の高さ"),
        ("報酬", "目標の速さ（0.8 m/秒）でゴールへ ＋ 到着 ＋ トロットのリズム − 傾き・横歩き・急な動き"),
        ("学習", "坂・階段・凸凹の地形を、上達に合わせて難しくしていく（カリキュラム学習）"),
        ("環境", "物理シミュレータ MuJoCo 上に、ロボット・家・庭をすべて自作"),
    ])
    parts = [("op", args.op),
             ("title", card("title", 4, slide_title(args.slide, title, sub) if args.slide else section_card(SIZE, title, sub))),
             ("system", card("system", 8, system)),
             ("growth", card("growth", 3, section_card(SIZE, "学習による成長", "充電ステーションから庭の階段の上の高台へ（学習段階ごとに 10 秒）")))]
    parts += [(f"growth{i}", os.path.join(CLIPS, f"growth{i}.mp4")) for i in range(len(GROWTH))]
    parts.append(("demo", card("demo", 3, section_card(SIZE, "日本語で指示してみる", "指示 → LLM が部屋の順番を計画 → 経路計画 → 歩いて回る（2・3本目は2倍速）"))))
    parts += [(name, os.path.join(CLIPS, f"{name}.mp4")) for name, _, _ in DEMOS]
    parts.append(("summary", card("summary", 8, summary_card(SIZE, summary_stats(), [
        "今後：扉の開け閉め（アームの追加）、カメラ画像からの部屋の認識、実機の四足ロボットへの sim-to-real"],
        "音声：VOICEVOX:ずんだもん"))))
    parts.append(("ed", args.ed))

    # ナレーションを部品の開始時刻に合わせて 1 本の音声トラックにする
    events, t, pads = [], 0.0, []
    for name, path in parts:
        d = duration(path)
        # 転倒などで早く終わった成長記録は、最後の画面で止めて長さをそろえる
        pad = max(GROWTH_SECONDS - d, 0.0) if name.startswith("growth") and name != "growth" else 0.0
        pads.append(pad)
        d += pad
        for off, text in VOICE.get(name, []):
            if off + voice_seconds(text) > d:
                print(f"[warn] {name}: ナレーションが映像より長い（{off + voice_seconds(text):.1f}s > {d:.1f}s）")
            events.append((t + off, text))
        t += d
    voice = os.path.join(CLIPS, "voice.wav")
    write_track(events, t, voice)
    write_srt(events, os.path.splitext(args.out)[0] + ".srt")  # 字幕（YouTube で ON/OFF できる）

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
