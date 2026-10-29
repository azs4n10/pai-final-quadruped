"""
大型版の犬型ロボットの階段の学習を起動する。

  python train_big.py --env BigStairs-v0 --steps 150000000 --lr-decay --n-envs 32
子プロセス（物理シミュレーション）に PyTorch を読み込ませないよう、重い import は main の中で行う。
"""
if __name__ == "__main__":
    import train_goal

    train_goal.main()
