"""
提出動画を組み立てる：OP → タイトル → 動機 → 全体構成 → 手法と工夫 → 評価結果 → 学習の成長 → デモ×3 → 考察 → ED
（ナレーション付き。字幕ファイル .srt も書き出す）

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

GROWTH_SECONDS = 10
GROWTH_SEED = 3
# 学習の成長：充電ステーションから庭の階段の上の高台へ（4 段階を 2×2 の画面で同時に見せる）
GROWTH = [("000000016", "学習前"), ("002000000", "200万ステップ"), ("005000000", "500万ステップ"),
          (None, "5000万ステップ（採用）")]
DEMO_SEED = 1
DEMOS = [  # (名前, 指示, 倍速)
    ("demo1", "キッチンで荷物を受け取って、寝室に届けて", 2),
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
                            "--seconds", str(GROWTH_SECONDS), "--seed", str(GROWTH_SEED), "--out", out], check=True)
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


def growth_grid():
    """4 段階の成長記録を 2×2 に並べた 1 本の動画にする（短いものは最後の画面で止める）。"""
    out = os.path.join(CLIPS, "growth_grid.mp4")
    ins = [os.path.join(CLIPS, f"growth{i}.mp4") for i in range(len(GROWTH))]
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for p in ins:
        cmd += ["-i", p]
    w, h = SIZE[0] // 2, SIZE[1] // 2
    f = "".join(f"[{i}:v]scale={w}:{h},tpad=stop_mode=clone:stop_duration={GROWTH_SECONDS},"
                f"trim=duration={GROWTH_SECONDS},setpts=PTS-STARTPTS[g{i}];" for i in range(4))
    f += "[g0][g1][g2][g3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
    cmd += ["-filter_complex", f, "-map", "[out]", "-r", str(FPS), "-c:v", "libx264", "-crf", "18", out]
    subprocess.run(cmd, check=True)
    return out


def results():
    ev = json.load(open(os.path.join(RUN, "eval_final.json")))
    hv = json.load(open(os.path.join(RUN, "house_eval.json"), encoding="utf-8"))
    return ev, hv


def pct(v):
    return f"{v * 100:.0f}%"


# ナレーション（VOICEVOX:ずんだもん）。部品ごとに (開始秒, 文)。結果の数字は評価結果から入れる
def voice(ev, hv):
    up = {lv: ev[lv]["stairs_up"]["success"] for lv in ev}
    house_ok = sum(v["success"] for v in hv.values()) / len(hv)
    return {
        "title": [(0.4, "日本語で指示できる、四足歩行ロボットを紹介します。")],
        "motive": [(0.5, "家の中で、荷物を運んだり見回りをしたりするロボットを、日本語で動かしたいと考えました。"),
                   (None, "家には玄関の段差や階段があるため、車輪ではなく四足歩行にしました。"),
                   (None, "また、間取りや会話を外部に送らないよう、LLMはパソコンの中で動かします。")],
        "system": [(0.5, "日本語の指示を、ローカルLLMが、立ち寄る場所の順番に変換します。"),
                   (None, "間取り図から壁や家具を避ける経路を計算し、強化学習で身につけた歩き方で、段差や坂を越えて進みます。")],
        "method": [(0.5, "工夫は四つです。"),
                   (None, "一つ目に、足元三十五点の地面の高さを観測に入れました。"),
                   (None, "二つ目に、地形の種類ごとに、上達に合わせて難しくするカリキュラム学習を行いました。"),
                   (None, "三つ目に、LLMの出力を、登録した場所の名前だけに制限しました。"),
                   (None, "四つ目に、壁は経路計画で避け、段差は歩行方策が越えるように、役割を分けました。")],
        "results": [(0.5, "評価の結果です。地形の種類と難しさを変えて、各三十回、ゴールまで歩かせました。"),
                    (None, f"上り階段は、段差四センチで{up['0.25'] * 100:.0f}パーセント、六センチで{up['0.5'] * 100:.0f}パーセント成功し、"
                           "八センチ以上は上れませんでした。"),
                    (None, f"家と庭の三つの経路は、平均{house_ok * 100:.0f}パーセントで、最後の地点まで着きました。")],
        "growth": [(0.3, "同じ階段でも、学習が進むにつれて、転ばずに上れるようになりました。")],
        "demo1": [(0.6, "それでは、日本語で指示を出してみます。ここからは二倍速です。"),
                  (None, "玄関の階段を上り、壁を避けて、キッチンから寝室へ向かいます。")],
        "demo2": [(0.6, "次は、庭の階段と坂です。"),
                  (None, "階段の高台、坂の上、充電ステーションの順に回ります。")],
        "demo3": [(0.6, "最後は、子ども部屋からリビングを通って帰る指示です。")],
        "discussion": [(0.5, "考察です。上れる段差は六センチ前後までで、家と庭での失敗も、すべて段差や坂のある場所で起きました。"),
                       (None, "また、角が鋭い段では失敗が増えたため、段の形を変えながら学習させる必要があります。"),
                       (None, "今後は、扉の開け閉めや、カメラ画像からの部屋の認識、実機での検証に取り組みたいと考えています。")],
    }


def card(name, seconds, frame, lines):
    """静止画の部品。ナレーションが収まる長さに伸ばす。"""
    need = max((t + d + 0.8 for t, d in lines), default=0)
    out = os.path.join(CLIPS, f"card_{name}.mp4")
    w = imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=8)
    for _ in range(int(max(seconds, need) * FPS)):
        w.append_data(frame)
    w.close()
    return out


def duration(path):
    return imageio_ffmpeg.count_frames_and_secs(path)[1]


def main():
    from PIL import Image

    from narration import seconds as voice_seconds
    from narration import write_srt, write_track
    from overlay import pairs_card, points_card, results_card, section_card, system_card

    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True)
    p.add_argument("--ed", required=True)
    p.add_argument("--slide", default=None, help="講座の最終課題発表スライド（タイトル画面に使う）")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "final_video.mp4"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    render_clips(args.force)
    ev, hv = results()
    VOICE = voice(ev, hv)
    # 文の開始時刻：None は前の文の直後（0.3 秒あけて）
    timed = {}
    for name, items in VOICE.items():
        t, out = 0.0, []
        for off, text in items:
            start = off if off is not None else t + 0.3
            d = voice_seconds(text)
            out.append((start, text, d))
            t = start + d
        timed[name] = out

    def lines(name):
        return [(s, d) for s, _, d in timed.get(name, [])]

    title, sub = "日本語で指示できる四足歩行ロボット", "ローカルLLM × 経路計画 × 強化学習で、家の中も段差も歩く"
    robot = Image.open(os.path.join(HERE, "report", "robot.jpg")).crop((120, 60, 860, 560))
    motive = pairs_card(SIZE, "動機", "家の中で荷物運びや見回りをするロボットを、日本語で動かしたい", [
        ("家の中には段差・階段・坂がある", "車輪ではなく四足歩行"),
        ("指示は普段の言葉で出したい", "LLM で指示を解釈"),
        ("間取りや会話を外部に送りたくない", "PC 内で動くローカル LLM"),
    ])
    system = system_card(SIZE, robot, [
        ("日本語の指示", "「キッチンで荷物を\n受け取って…」"),
        ("ローカル LLM", "gemma4（Ollama）\n立ち寄る場所の順番"),
        ("経路計画", "間取り図から A* で\n壁・家具を避ける"),
        ("歩行方策", "強化学習（PPO）\n段差・坂を越える"),
    ], [
        ("シミュレータ", "MuJoCo 上に、犬型ロボット（12 関節）・家・庭をすべて自作"),
        ("観測", "胴体の姿勢・速度、関節の角度・速度、ゴールの方向と距離、足元 35 点の地面の高さ"),
        ("報酬", "目標の速さでゴールへ向かう＋到着＋トロットのリズム − 傾き・横歩き・急な動き"),
        ("学習", "PPO・16 環境並列・5000 万ステップ（CPU のみ、約 5 時間）"),
    ])
    method = points_card(SIZE, "手法と工夫", "", [
        ("足元の地形を測る", "胴体のまわり 35 点の地面の高さを観測に入れ、段差や坂を見ながら足を運ぶ"),
        ("地形ごとのカリキュラム学習", "平地・凸凹・丘・上り階段・下り階段のそれぞれで、うまく歩けたら難しく、転んだら易しくする"),
        ("LLM の出力を制限", "JSON Schema で出力を【登録した場所の名前】に限定し、存在しない部屋や形式の崩れを防ぐ"),
        ("経路計画と歩行の分担", "壁・家具は地図と A* で避け、段差・坂は歩行方策が自分で越える（家の出入りは階段を正面から）"),
    ])
    lv_cm = {"0.25": "4 cm", "0.5": "6 cm", "0.75": "8 cm", "1.0": "10 cm"}
    route_ja = {"kitchen→bedroom": "キッチン → 寝室", "stairs_top→slope_top→dock": "階段の高台 → 坂の高台 → 充電",
                "kids→living→dock": "子ども部屋 → リビング → 充電"}
    table = [["経路", "成功", "転倒"]] + [
        [route_ja.get(k, k), pct(v["success"]), pct(v["fell"])] for k, v in hv.items()]
    res = results_card(SIZE, "評価結果", "上り階段の成功率（段の高さ別、各 30 回）",
                       [(f"段差\n{lv_cm[lv]}", ev[lv]["stairs_up"]["success"]) for lv in ev],
                       "家と庭での指示（各 20 回）", table,
                       "成功：3.5 m 先のゴールに 20 秒以内に転ばずに着く\n"
                       f"平地・丘（勾配 14° まで）：【{pct(min(ev[lv][k]['success'] for lv in ev for k in ('flat', 'hills')))} 以上】\n"
                       f"凸凹：±2.8 cm まで【{pct(min(ev[lv]['rough']['success'] for lv in ('0.25', '0.5', '0.75')))} 以上】、"
                       f"±3.5 cm で {pct(ev['1.0']['rough']['success'])}")
    discussion = points_card(SIZE, "考察と今後", "", [
        ("上れる段差は 6 cm 前後まで", "8 cm 以上は上れない。家と庭での失敗も【すべて段差や坂のある場所】。学習中の階段の難しさも 0.3（段差 約 4.5 cm）付近で頭打ち"),
        ("段の形の違いに弱い", "学習した高さマップの段では上れても、【角の鋭い箱の段】では失敗が増えた → 段の形を変えながら学習する必要"),
        ("今後", "扉の開け閉め（アームの追加）、カメラ画像からの部屋の認識、実機の四足ロボットでの検証（sim-to-real）"),
    ])

    parts = [("op", args.op),
             ("title", card("title", 4, slide_title(args.slide, title, sub) if args.slide else section_card(SIZE, title, sub), lines("title"))),
             ("motive", card("motive", 6, motive, lines("motive"))),
             ("system", card("system", 6, system, lines("system"))),
             ("method", card("method", 6, method, lines("method"))),
             ("results", card("results", 6, res, lines("results"))),
             ("growth", growth_grid())]
    parts += [(name, os.path.join(CLIPS, f"{name}.mp4")) for name, _, _ in DEMOS]
    parts.append(("discussion", card("discussion", 6, discussion, lines("discussion"))))
    parts.append(("ed", args.ed))

    # ナレーションを部品の開始時刻に合わせて 1 本の音声トラックにする
    events, t = [], 0.0
    for name, path in parts:
        d = duration(path)
        for start, text, dur in timed.get(name, []):
            if start + dur > d:
                print(f"[warn] {name}: ナレーションが映像より長い（{start + dur:.1f}s > {d:.1f}s）")
            events.append((t + start, text))
        t += d
    print(f"[info] 全体 {t:.1f} 秒")
    voice_wav = os.path.join(CLIPS, "voice.wav")
    write_track(events, t, voice_wav)
    write_srt(events, os.path.splitext(args.out)[0] + ".srt")  # 字幕（YouTube で ON/OFF できる）

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for _, prt in parts:
        cmd += ["-i", prt]
    cmd += ["-i", voice_wav]
    n = len(parts)
    norm = "".join(
        f"[{i}:v]scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease,"
        f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p,setsar=1[v{i}];"
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
