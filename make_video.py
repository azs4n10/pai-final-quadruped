"""
提出動画を組み立てる：OP → タイトル → 動機 → 全体構成 → ロボットの設計と自作 → ローカル LLM → 経路計画 → 学習 → 結果 → デモ → 考察 → ED
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
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs", "big_stairs")
CLIPS = os.path.join(HERE, "runs", "clips_home")
FIGS = os.path.join(HERE, "runs", "figs")
MODEL = os.path.join(HERE, "models", "big_dog_model.zip")
SIZE = (1920, 1080)
FPS = 30
DEMO_SEED = 0
DEMOS = [  # (名前, 指示, 倍速)
    ("demo1", "玄関に届いた荷物を、2階の寝室まで運んで。終わったら充電ステーションに戻って", 2),
    ("demo2", "キッチンでお茶を受け取って、和室のおばあちゃんに届けて", 2),
]


def render_clips(force=False):
    os.makedirs(CLIPS, exist_ok=True)
    py = sys.executable
    for name, text, speed in DEMOS:
        out = os.path.join(CLIPS, f"{name}.mp4")
        if force or not os.path.exists(out):
            cmd = [py, os.path.join(HERE, "home_demo.py"), "--text", text, "--speed", str(speed), "--seed", str(DEMO_SEED), "--out", out]
            log_path = os.path.splitext(out)[0] + ".json"
            if os.path.exists(log_path):  # LLM の計画は前回の結果を使い回す（温度 0 なので同じ出力）
                prev = json.load(open(log_path, encoding="utf-8"))
                cmd += ["--route", ",".join(prev["route"]), "--reply", prev["reply"]]
                cmd += (["--pickup", prev["pickup"]] if prev.get("pickup") else []) + (["--dropoff", prev["dropoff"]] if prev.get("dropoff") else [])
            subprocess.run(cmd, check=True)
    scene = os.path.join(CLIPS, "train_scene.mp4")
    if force or not os.path.exists(scene):
        subprocess.run([py, os.path.join(HERE, "train_scene.py"), "--model", MODEL, "--level", "7", "--seconds", "9", "--out", scene], check=True)


def pct(v):
    return f"{v * 100:.0f}%"


def voice(st, hv):
    up15 = st["15"]["up"]["success"]
    home_ok = sum(v["success"] for v in hv.values()) / len(hv)
    return {
        "title": [(0.4, "日本語で指示できる、四足歩行ロボットを紹介します。")],
        "motive": [(0.5, "家の中で、荷物を部屋から部屋へ運ぶロボットを、日本語で動かしたいと考えました。"),
                   (None, "家には二階への階段があるため、車輪ではなく四足歩行にしました。"),
                   (None, "また、間取りや会話を外部に送らないよう、LLMはパソコンの中で動かします。")],
        "system": [(0.5, "日本語の指示を、ローカルLLMが、立ち寄る場所の順番と、荷物の受け取り先と届け先に変換します。"),
                   (None, "間取り図から経路を計算し、強化学習で身につけた歩き方で、階段も上り下りします。")],
        "design": [(0.5, "ロボットは、物理シミュレータのMuJoCoの上に、十二個の関節を持つ犬型を、コードで記述して自作しました。"),
                   (None, "体重は三十四キロ、胴体の高さは四十六センチです。")],
        "llm": [(0.5, "LLMには、場所の一覧とルールを与え、出力の形をJSON Schemaで固定しました。"),
                (None, "存在しない部屋は出力できないため、そのまま安全に実行できます。")],
        "planner": [(0.5, "経路計画では、間取りを十センチの格子にし、ロボットの幅だけ障害物を太らせてから、A*で最短経路を探します。"),
                    (None, "階をまたぐときは、階段の下と上を必ず通ります。")],
        "train": [(0.5, "歩き方は、強化学習のPPOで身につけました。"),
                  (None, "段差五センチから十八センチの階段や坂を、上達に合わせて難しくしながら、一億回以上練習しました。")],
        "results": [(0.5, f"評価の結果、段差十五センチの上り階段は、{up15 * 100:.0f}パーセント上り切りました。"),
                    (None, f"家の中の配達の三つの経路は、平均{home_ok * 100:.0f}パーセントで、最後まで届けられました。")],
        "demo1": [(0.6, "それでは、日本語で指示を出してみます。ここからは二倍速です。"),
                  (None, "荷物をカゴに載せ、十四段の階段を上って、二階の寝室に届けます。")],
        "demo2": [(0.6, "次は一階で、キッチンから和室へお茶を届けます。扉は近づくと自動で開きます。")],
        "discussion": [(0.5, "考察です。階段は上れるようになりましたが、失敗の多くは、階段の上り始めと下り始めで起きました。"),
                       (None, "今後は、腕を付けて扉の開け閉めや物の受け渡しを自分で行うこと、カメラ画像から部屋を認識することに取り組みたいと考えています。")],
    }


def card(name, seconds, frame, lines):
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
    from narration import seconds as voice_seconds
    from narration import write_srt, write_track
    from overlay import image_card, llm_card, pairs_card, points_card, results_card, section_card, slide_title, system_card

    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True)
    p.add_argument("--ed", required=True)
    p.add_argument("--slide", default=None)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "final_video.mp4"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    render_clips(args.force)
    st = json.load(open(os.path.join(RUN, "stairs_eval.json")))
    hv = json.load(open(os.path.join(RUN, "home_eval.json"), encoding="utf-8"))
    plan1 = json.load(open(os.path.join(CLIPS, "demo1.json"), encoding="utf-8"))
    V = voice(st, hv)
    timed = {}
    for name, items in V.items():
        t, out = 0.0, []
        for off, text in items:
            start = off if off is not None else t + 0.3
            d = voice_seconds(text)
            out.append((start, text, d))
            t = start + d
        timed[name] = out

    def lines(name):
        return [(s, d) for s, _, d in timed.get(name, [])]

    title, sub = "日本語で指示できる四足歩行ロボット", "ローカル LLM × 経路計画 × 強化学習で、2 階まで荷物を運ぶ"
    robot = Image.open(os.path.join(HERE, "report", "big_robot.jpg"))
    cards = {}
    cards["motive"] = pairs_card(SIZE, "動機", "家の中で荷物を部屋から部屋へ運ぶロボットを、日本語で動かしたい", [
        ("家には 2 階への階段や段差がある", "車輪ではなく四足歩行"),
        ("指示は普段の言葉で出したい", "LLM で指示を解釈"),
        ("間取りや会話を外部に送りたくない", "PC 内で動くローカル LLM"),
    ], page=2)
    cards["system"] = system_card(SIZE, robot.crop((0, 0, robot.width // 2, robot.height)), [
        ("日本語の指示", "「玄関の荷物を\n2 階の寝室へ」"),
        ("ローカル LLM", "gemma4（Ollama）\n場所の順番・受け渡し先"),
        ("経路計画", "階ごとの A*\n階段は上下の通過点"),
        ("歩行方策", "強化学習（PPO）\n速度指令で歩く"),
    ], [
        ("シミュレータ", "MuJoCo 上に、犬型ロボット・2 階建ての家・家具・自動ドアをすべて自作"),
        ("観測", "胴体の姿勢・速度、関節の角度・速度、速度指令、足元 50 点の地面の高さ（98 次元）"),
        ("報酬", "指令した速度で歩く＋足を段の上面から上げる − 傾き・滑り・急な動き・転倒"),
        ("学習", "PPO・32 環境並列・1.2 億ステップ（約 14 時間）。階段の難しさを上達に合わせて上げる"),
    ], page=3)
    design = Image.new("RGB", (1800, 900), "white")
    dr = Image.open(os.path.join(FIGS, "robot_drawing.png"))
    dr.thumbnail((1100, 520))
    design.paste(dr, (0, 0))
    code = Image.open(os.path.join(FIGS, "mjcf_code.png"))
    code.thumbnail((1000, 380))
    design.paste(code, (0, 520))
    rb = robot.crop((robot.width // 2, 0, robot.width, robot.height))
    rb.thumbnail((760, 520))
    design.paste(rb, (1040, 560))
    cards["design"] = image_card(SIZE, "ロボットの設計と自作", design,
                                 "寸法・関節・質量をコード（MJCF）で記述し、MuJoCo で動かす。見た目の部品は質量 0・接触なし", page=4)
    cards["llm"] = llm_card(SIZE, plan1["instruction"], [
        "あなたは四足歩行ロボットの経路プランナーです。", "使える地点（キー: 説明, 座標）:", "- dock: 充電ステーション（1 階）",
        "- bedroom: 寝室（2 階）", "  …（全 10 か所）", "ルール:", "- route にはキーだけを訪れる順に並べる",
        "- 荷物を運ぶなら pickup と dropoff を入れる"],
        ['"route":   [enum: dock, bedroom, …]', '"pickup":  enum + "none"', '"dropoff": enum + "none"', '"reply":   string'],
        ["{", f'  "route": {json.dumps(plan1["route"])},', f'  "pickup": "{plan1["pickup"]}",', f'  "dropoff": "{plan1["dropoff"]}",',
         f'  "reply": "{plan1["reply"]}"', "}"],
        [("外部に送らない", "間取りや指示を【PC の外に出さない】"), ("出力を型で縛る", "存在しない部屋や崩れた JSON を【出せない】"),
         ("役割を限定", "LLM は「どこへ・どの順で」だけ決める")], page=5)
    cards["planner"] = image_card(SIZE, "経路計画", Image.open(os.path.join(FIGS, "planner_steps.png")),
                                  "階をまたぐときは【階段の下と上の通過点】を必ず通り、階段を正面からまっすぐ上り下りする", page=6)
    cards["results"] = results_card(
        SIZE, "評価結果", "練習場の上り階段（段の高さ別、各 20 回）",
        [(f"{k} cm", v["up"]["success"]) for k, v in st.items() if int(k) in (5, 9, 12, 15, 18)],
        "家の中の配達（各 20 回）", [["経路", "成功", "転倒"]] + [[k.replace("→", " → "), pct(v["success"]), pct(v["fell"])] for k, v in hv.items()],
        "成功：中央から 2.6 m 先まで 15 秒以内に転ばずに進む", page=8)
    cards["discussion"] = points_card(SIZE, "考察と今後", "", [
        ("階段は上り始め・下り始めが難しい", "失敗の多くは段に足をかける最初の数歩。足元の高さの見える範囲を前に広げる・速度を落とすなどの改善が考えられる"),
        ("学習の速さ", "この PC の GPU では物理計算が速くならず、CPU 並列で 1.2 億ステップに約 14 時間かかった"),
        ("今後", "腕を付けて扉の開け閉めや物の受け渡しを自分で行う、カメラ画像からの部屋の認識、実機での検証（sim-to-real）"),
    ], page=9)
    train_card = section_card(SIZE, "強化学習による歩行の練習", "6 種類の地形で同時に練習（段差・坂・凸凹は上達に合わせて難しくする）")

    parts = [("op", args.op),
             ("title", card("title", 4, slide_title(args.slide, title, sub) if args.slide else section_card(SIZE, title, sub), lines("title")))]
    for name in ("motive", "system", "design", "llm", "planner"):
        parts.append((name, card(name, 6, cards[name], lines(name))))
    parts.append(("train_title", card("train_title", 2, train_card, [])))
    parts.append(("train", os.path.join(CLIPS, "train_scene.mp4")))
    parts.append(("results", card("results", 6, cards["results"], lines("results"))))
    parts += [(name, os.path.join(CLIPS, f"{name}.mp4")) for name, _, _ in DEMOS]
    parts.append(("discussion", card("discussion", 6, cards["discussion"], lines("discussion"))))
    parts.append(("ed", args.ed))

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
    write_srt(events, os.path.splitext(args.out)[0] + ".srt")

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for _, prt in parts:
        cmd += ["-i", prt]
    cmd += ["-i", voice_wav]
    n = len(parts)
    norm = "".join(f"[{i}:v]scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease,"
                   f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p,setsar=1[v{i}];" for i in range(n))
    concat = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", norm + concat, "-map", "[out]", "-map", f"{n}:a", "-c:v", "libx264", "-crf", "20",
            "-preset", "medium", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    probe = subprocess.run([ff, "-i", args.out], capture_output=True, text=True).stderr
    print([ln.strip() for ln in probe.splitlines() if "Duration" in ln], "->", args.out)


if __name__ == "__main__":
    main()
