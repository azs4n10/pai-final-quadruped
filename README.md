# ことばで動く四足配送ロボット

Physical AI 応用1 最終課題。日本語の指示をローカル LLM が経由地のリストに変換し、強化学習（PPO）で学習した「ゴールまで歩く」方策で、自作の犬型 4 脚ロボット（MuJoCo）が順に歩いて回る。初版は Gymnasium 標準の Ant（クモ型）で作り、犬型に作り直したうえで、歩き方が虫のように見えたため報酬を設計し直してトロット歩行にした（3 台すべてのコードを含む）。

```
日本語の指示 → ローカルLLM（gemma4 / Ollama） → 経由地リスト → ゴール到達方策（PPO） → 犬型4脚ロボット（MuJoCo）
```

## ファイル
| ファイル | 内容 |
|---|---|
| `goal_dog.py` | 犬型ロボット・改良版（12 関節・位置制御、MJCF を生成）とゴール到達環境 GoalDog。目標速度追従＋歩行クロック（トロット）の報酬 |
| `goal_dog_v1.py` | 犬型・旧版（近づく速さを最大化する報酬。速いが脚を広げた小走りになった） |
| `goal_ant.py` | 初版のゴール到達環境 GoalAnt（Ant-v5 の観測・報酬を作り替え、ランドマークとゴール旗を追加） |
| `train_goal.py` | PPO による学習（16 並列、CPU） |
| `planner.py` | 日本語の指示 → 経由地リスト（Ollama の JSON 構造化出力） |
| `demo.py` | 指示を実行して録画（字幕・LLM の計画・俯瞰マップを重ねる） |
| `evaluate.py` | チェックポイントごとの到達率・デモ経路の成功率の評価 |
| `make_video.py` | 提出動画の組み立て |
| `plot_compare.py` | 説明資料の図（3 台の学習曲線・外観） |
| `overlay.py` | 動画用の文字・ミニマップ描画 |
| `models/` | 採用モデルと観測正規化の統計（`dog_*`：犬型・改良版 2,900 万ステップ、`dog_v1_*`：旧版 200 万ステップ、`final_*`：Ant 700 万ステップ） |

## セットアップ
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
ollama pull gemma4:e4b      # LLM を使う場合（無い場合は色名を拾う簡易変換で動く）
```

## 使い方
```
# 学習済みモデルで指示を実行（runs/demo.mp4 に録画）。--robot ant で初版
python demo.py --robot dog --text "黄色い柱に荷物を届けたら帰ってきて"

# 経由地を直接指定
python demo.py --robot dog --route red,blue,home

# 学習（犬型：3,000 万ステップで約 2.3 時間 / 16 コア CPU）
python train_goal.py --env GoalDog-v0 --steps 30000000
python train_goal.py --env GoalAnt-v0 --steps 15000000   # 初版

# 評価
python evaluate.py --robot dog            # 学習曲線（runs/goal_dog/eval.png）
python evaluate.py --robot dog --routes   # デモ経路ごとの成功率
python evaluate.py --robot dog --final 29000000   # 採用モデルを 100 回で測り直す
```

地点：`home`（出発地点、原点）、`red`（6, 3）、`blue`（-4, 6）、`green`（-6, -3）、`yellow`（4, -6）。

## 結果
| | 犬型・改良版（2,900 万ステップ・採用） | 犬型・旧版（200 万ステップ） | 初版 Ant（700 万ステップ） |
|---|---|---|---|
| 4〜6 m 先のゴールに 30 秒以内で到達（100 回） | 100%・平均 6.08 秒・転倒 0% | 98%・平均 2.76 秒・転倒 1% | 99%・平均 2.87 秒・転倒 0% |
| 赤→青→ホーム（20 回） | 100% | 95% | 80% |
| 黄→ホーム（20 回） | 100% | 100% | 95% |
| 全部の柱を一周→ホーム（20 回） | 100% | 100% | 85% |

歩き方（30 秒 × 10 回の平均、旧版 → 改良版）：速さ 1.76 → 0.75 m/秒、足の接地時間 31% → 48%、対角の脚の接地がそろう割合 34% → 73%、横方向の速さ 0.24 → 0.07 m/秒。改良版は目標速度を 0.8 m/秒に抑えているため所要時間は長い。

## 使用ツール
MuJoCo 3.12 / Gymnasium 1.3 / Stable-Baselines3 2.9（PPO）/ Ollama（gemma4:e4b）/ Python 3.12
