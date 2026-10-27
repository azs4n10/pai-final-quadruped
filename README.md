# ことばで動く四足配送ロボット

Physical AI 応用1 最終課題。日本語の指示をローカル LLM が経由地のリストに変換し、強化学習（PPO）で学習した「ゴールまで歩く」方策で、4 脚ロボット（MuJoCo / Ant）が順に歩いて回る。

```
日本語の指示 → ローカルLLM（gemma4 / Ollama） → 経由地リスト → ゴール到達方策（PPO） → 4脚ロボット（MuJoCo）
```

## ファイル
| ファイル | 内容 |
|---|---|
| `goal_ant.py` | ゴール到達環境 GoalAnt（Ant-v5 の観測・報酬を作り替え、ランドマークとゴール旗を追加） |
| `train_goal.py` | PPO による学習（16 並列、CPU） |
| `planner.py` | 日本語の指示 → 経由地リスト（Ollama の JSON 構造化出力） |
| `demo.py` | 指示を実行して録画（字幕・LLM の計画・俯瞰マップを重ねる） |
| `evaluate.py` | チェックポイントごとの到達率・デモ経路の成功率の評価 |
| `make_video.py` | 提出動画の組み立て |
| `overlay.py` | 動画用の文字・ミニマップ描画 |
| `models/` | 採用モデル（700 万ステップ時点）と観測正規化の統計 |

## セットアップ
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
ollama pull gemma4:e4b      # LLM を使う場合（無い場合は色名を拾う簡易変換で動く）
```

## 使い方
```
# 学習済みモデルで指示を実行（runs/demo.mp4 に録画）
python demo.py --text "黄色い柱に荷物を届けたら帰ってきて"

# 経由地を直接指定
python demo.py --route red,blue,home

# 学習（約 2 時間 / 16 コア CPU）
python train_goal.py --steps 15000000

# 評価
python evaluate.py            # 学習曲線（runs/goal_ant/eval.png）
python evaluate.py --routes   # デモ経路ごとの成功率
```

地点：`home`（出発地点、原点）、`red`（6, 3）、`blue`（-4, 6）、`green`（-6, -3）、`yellow`（4, -6）。

## 結果（採用モデル）
- 4〜6 m 先のゴールに 30 秒以内で到達：99%（100 回）、平均 2.87 秒、転倒 0%
- デモ 3 経路の成功率（各 20 回）：赤→青→ホーム 80% / 黄→ホーム 95% / 全部の柱を一周→ホーム 85%

## 使用ツール
MuJoCo 3.12 / Gymnasium 1.3（Ant-v5）/ Stable-Baselines3 2.9（PPO）/ Ollama（gemma4:e4b）/ Python 3.12
