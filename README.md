# 日本語で指示できる四足歩行ロボット

Physical AI 応用1 最終課題。日本語の指示をローカル LLM が経由地のリストに変換し、強化学習（PPO）で学習した「ゴールまで歩く」方策で、自作の犬型 4 脚ロボット（MuJoCo）がトロットで順に歩いて回る。

```
日本語の指示 → ローカルLLM（gemma4 / Ollama） → 経由地リスト → ゴール到達方策（PPO） → 犬型4脚ロボット（MuJoCo）
```

## ファイル
| ファイル | 内容 |
|---|---|
| `goal_dog.py` | 犬型ロボット（12 関節・位置制御、MJCF を生成）とゴール到達環境 GoalDog。目標速度追従＋歩行クロック（トロット）の報酬 |
| `goal_ant.py` | ランドマーク（地点）の定義と、比較用のゴール到達環境 GoalAnt（Ant-v5 ベース） |
| `goal_dog_v1.py` | 比較用：近づく速さを最大化する報酬の犬型環境 |
| `train_goal.py` | PPO による学習（16 並列、CPU） |
| `planner.py` | 日本語の指示 → 経由地リスト（Ollama の JSON 構造化出力） |
| `demo.py` | 指示を実行して録画（字幕・LLM の計画・俯瞰マップを重ねる） |
| `evaluate.py` | チェックポイントごとの到達率・デモ経路の成功率の評価 |
| `make_video.py` | 提出動画の組み立て |
| `plot_compare.py` | 説明資料の図（学習曲線・外観） |
| `overlay.py` | 動画用の文字・ミニマップ描画 |
| `narration.py` | 動画のナレーション音声（VOICEVOX ENGINE で合成） |
| `models/` | 採用モデルと観測正規化の統計（`dog_*`：犬型 2,900 万ステップ。`dog_v1_*`・`final_*` は比較用） |

## セットアップ
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
ollama pull gemma4:e4b      # LLM を使う場合（無い場合は色名を拾う簡易変換で動く）
```

## 使い方
```
# 学習済みモデルで指示を実行（runs/demo.mp4 に録画）
python demo.py --robot dog --text "黄色い柱に荷物を届けたら帰ってきて"

# 経由地を直接指定
python demo.py --robot dog --route red,blue,home

# 学習（犬型：3,000 万ステップで約 2.3 時間 / 16 コア CPU）
python train_goal.py --env GoalDog-v0 --steps 30000000
python train_goal.py --env GoalAnt-v0 --steps 15000000   # 比較用

# 評価
python evaluate.py --robot dog            # 学習曲線（runs/goal_dog/eval.png）
python evaluate.py --robot dog --routes   # デモ経路ごとの成功率
python evaluate.py --robot dog --final 29000000   # 採用モデルを 100 回で測り直す
```

地点：`home`（出発地点、原点）、`red`（6, 3）、`blue`（-4, 6）、`green`（-6, -3）、`yellow`（4, -6）。

## 結果（採用モデル：2,900 万ステップ）
| 評価 | 結果 |
|---|---|
| 4〜6 m 先のゴールに 30 秒以内で到達（100 回） | 100%・平均 6.08 秒・転倒 0% |
| 赤→青→ホーム（20 回） | 100%・平均 28.2 秒 |
| 黄→ホーム（20 回） | 100%・平均 16.9 秒 |
| 全部の柱を一周→ホーム（20 回） | 100%・平均 51.9 秒 |
| 歩き方（30 秒 × 10 回の平均） | 0.75 m/秒、対角の脚の接地がそろう割合 73% |

## 使用ツール
MuJoCo 3.12 / Gymnasium 1.3 / Stable-Baselines3 2.9（PPO）/ Ollama（gemma4:e4b）/ Python 3.12 / VOICEVOX ENGINE（動画のナレーション。音声：VOICEVOX:ずんだもん）
