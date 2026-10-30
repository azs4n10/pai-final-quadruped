# 日本語で指示できる四足歩行ロボット

Physical AI 応用1 最終課題。「玄関に届いた荷物を、2 階の寝室まで運んで」のような日本語の指示を出すと、自作の犬型四足歩行ロボット（34 kg・12 関節）が 2 階建ての家の中を歩き、14 段の階段を上り下りして荷物を届ける（MuJoCo 上）。

```
日本語の指示 → ローカルLLM（gemma4 / Ollama）→ 立ち寄る場所の順番・荷物の受け取り先と届け先
            → 経路計画（階ごとの A*）→ 経由地 → 歩行方策（PPO）→ 犬型ロボット
```

- **歩行方策**：足元 50 点の地面の高さを観測に入れ、上り/下り階段（段差 5〜18 cm）・坂・凸凹を、難しさ 10 段階のタイルのカリキュラムで学習（PPO・通算約 1.6 億ステップ・CPU）。歩くリズムの位相（周期 0.5 秒）も観測に入れ、対角の 2 本ずつ交互に足を上げ下ろす速歩を報酬で教える
- **LLM**：場所の名前の列挙型（JSON Schema）に出力を制限して、指示を場所の順番と荷物の受け取り先・届け先に変換
- **経路計画**：階ごとに間取りを 10 cm の格子地図にして A* で壁・家具を避ける経由地を出す。階をまたぐときは階段の下と上を通り、階段では中央線に沿う向きと 0.8 m/秒の速度を指令する

## ファイル
| ファイル | 内容 |
|---|---|
| `gpu/make_assets.py` | 大型版の犬型ロボット（MJCF）と練習場の高さマップ |
| `stairs_cpu.py` | 練習場を歩く学習環境（BigStairs：速度指令に従って歩く、BigTrot：さらに速歩のリズムに合わせる） |
| `train_big.py` | PPO による学習（CPU 並列。`--init` で続きから学習） |
| `widen_policy.py` | 学習済みの方策の入力に歩くリズムの位相を足す（増えた入力の重みは 0） |
| `big_dog.py` | 学習した方策でロボットを動かす部品（速度指令の計算を含む） |
| `home.py` | 2 階建ての家（間取り・家具・階段・自動ドア）と経路計画（A*） |
| `planner.py` | 日本語の指示 → 場所の順番（Ollama の構造化出力） |
| `home_demo.py` | 家の中で指示を実行して録画（指示・LLM の計画・1 階と 2 階の間取り図を重ねる） |
| `evaluate_big.py` | 練習場の階段（段の高さ別）・家の配達経路・チェックポイントごとの家の階段の評価 |
| `train_scene.py` | 練習場の 6 種類の地形を同時に歩く映像 |
| `make_video.py` / `narration.py` / `overlay.py` | 提出動画の組み立て、ナレーション（VOICEVOX ENGINE）、スライドと画面の描画 |
| `make_figures.py` / `explain_figs.py` / `make_report.py` | 説明資料の図と PDF |
| `models/big_dog_*` | 採用モデルと観測正規化の統計。ほかは以前の小型版・平地版のモデル |

## セットアップ
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
ollama pull gemma4:e4b      # LLM を使う場合
```

## 使い方
```
# 日本語の指示を実行（runs/home.mp4 に録画）
python home_demo.py --text "玄関に届いた荷物を、2階の寝室まで運んで"

# 場所を直接指定
python home_demo.py --route dock,bedroom --pickup dock --dropoff bedroom

# 学習（CPU 並列）
python train_big.py --env BigStairs-v0 --steps 90000000 --lr-decay --n-envs 64

# 評価
python evaluate_big.py --stairs     # 練習場の階段、段の高さ別に上り・下り各 20 回
python evaluate_big.py --home       # 家の中の配達の 3 経路を各 20 回
python evaluate_big.py --ckpts      # チェックポイントごとに家の階段を 10 回
```

場所：`dock`（充電ステーション）、`living`（リビング）、`kitchen`（キッチン）、`study`（書斎）、`washitsu`（和室）、`bedroom`（寝室・2 階）、`kids`（子ども部屋・2 階）、`hall2`（2 階のホール）、`library`（書庫・2 階）。

## 結果（採用モデル）
練習場の階段：中央から 2.6 m 先まで 15 秒以内に転ばずに進めた割合（各 20 回）

| 段差 | 5 cm | 9 cm | 12 cm | 14 cm | 15 cm | 17 cm | 18 cm |
|---|---|---|---|---|---|---|---|
| 上り | 90% | 95% | 95% | 100% | 60% | 20% | 0% |
| 下り | 100% | 100% | 95% | 100% | 90% | 80% | 35% |

家の中の配達（各 20 回）

| 指示 | LLM の出力 | 成功 | 転倒 |
|---|---|---|---|
| 玄関に届いた荷物を、2階の寝室まで運んで。終わったら充電ステーションに戻って | 充電ステーション → 寝室 → 充電ステーション | 85% | 15% |
| キッチンでお茶を受け取って、和室のおばあちゃんに届けて | キッチン → 和室 | 100% | 0% |
| （経路を直接指定） | 子ども部屋 → 充電ステーション | 90% | 5% |

家の階段（15 cm × 14 段）だけを上らせる試験では、10 回中 7 回上り切った。

## 使用ツール
MuJoCo 3.12 / Gymnasium 1.3 / Stable-Baselines3 2.9（PPO）/ PyTorch / Ollama（gemma4:e4b）/ Python 3.12 / VOICEVOX ENGINE（動画のナレーション。音声：VOICEVOX:ずんだもん）
