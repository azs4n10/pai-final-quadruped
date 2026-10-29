"""
GoalAnt を PPO で学習する（CPU 並列）。

  python train_goal.py --steps 20000000
  python train_goal.py --env GoalDog-v0 --steps 30000000   # 犬型（平地）
  python train_goal.py --env TerrainDog-v0 --steps 50000000   # 犬型（坂・階段・凸凹）
  python train_goal.py --env TerrainDog-v0 --steps 30000000 --init runs/terrain_dog/final_model.zip --out runs/terrain_dog2
                                                              # 学習済みモデルから続けて学習（追加学習）
途中経過のチェックポイントは runs/goal_ant/ckpt/ に保存される（動画の「成長記録」に使う）。
"""
import argparse
import os
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import goal_ant  # noqa: F401  (GoalAnt-v0 を登録)
import goal_dog  # noqa: F401  (GoalDog-v0 を登録)
import house  # noqa: F401  (HouseDog-v0 を登録)
import stairs_cpu  # noqa: F401  (BigStairs-v0 を登録)
import terrain_dog  # noqa: F401  (TerrainDog-v0 を登録)

HERE = os.path.dirname(os.path.abspath(__file__))


class CheckpointCallback(BaseCallback):
    """一定ステップごとにモデルと正規化統計を保存し、到達数を表示する。"""

    def __init__(self, every, out_dir):
        super().__init__()
        self.every, self.out_dir = every, out_dir
        self.next_save = 0
        self.t0 = time.time()
        self.reached = []
        self.levels = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.reached.append(info["n_reached"])
                if "levels" in info:
                    self.levels.append(info["levels"])
        if self.num_timesteps >= self.next_save:
            tag = f"{self.num_timesteps:09d}"
            self.model.save(os.path.join(self.out_dir, f"model_{tag}"))
            self.training_env.save(os.path.join(self.out_dir, f"vecnorm_{tag}.pkl"))
            if self.reached:
                avg = sum(self.reached[-100:]) / len(self.reached[-100:])
                self.logger.record("goal/reached_per_ep", avg)
            fps = self.num_timesteps / max(time.time() - self.t0, 1e-6)
            lv = ""
            if self.levels:  # 地形の種類ごとの難しさ（直近 100 エピソードの平均）
                last = self.levels[-100:]
                lv = " level " + " ".join(f"{k}={sum(x[k] for x in last) / len(last):.2f}" for k in last[0])
            print(f"[ckpt] step={self.num_timesteps} fps={fps:.0f} reached/ep(last100)="
                  f"{(sum(self.reached[-100:]) / max(len(self.reached[-100:]), 1)):.2f}{lv}", flush=True)
            self.next_save += self.every
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="GoalAnt-v0", choices=["GoalAnt-v0", "GoalDog-v0", "TerrainDog-v0", "TerrainDog-v1", "BigStairs-v0"])
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--save-every", type=int, default=1_000_000)
    p.add_argument("--lr-decay", action="store_true", help="学習率を最後に 0 まで直線的に下げる（最終モデルを安定させる）")
    p.add_argument("--init", default=None, help="続きから学習するモデル（同じ場所の *_vecnorm.pkl も読む）")
    p.add_argument("--house-envs", type=int, default=0, help="並列環境のうち家と庭（HouseDog-v0）で学習する数")
    p.add_argument("--resume", default=None, help="中断した学習をチェックポイント（ckpt/model_*.zip）から再開する")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default=None, help="既定は runs/goal_ant・runs/goal_dog・runs/terrain_dog")
    args = p.parse_args()

    if args.out is None:
        args.out = os.path.join(HERE, "runs", {"GoalAnt-v0": "goal_ant", "GoalDog-v0": "goal_dog", "TerrainDog-v0": "terrain_dog", "TerrainDog-v1": "terrain_dog_v1", "BigStairs-v0": "big_stairs"}[args.env])
    torch.set_num_threads(4)
    ckpt = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt, exist_ok=True)

    if args.env == "BigStairs-v0":  # 1 つの子プロセスで複数の環境（train_big.py から起動する）
        from vec_workers import BatchedSubprocVecEnv

        venv = BatchedSubprocVecEnv("BigStairs-v0", args.n_envs, min(16, args.n_envs), "stairs_cpu")
    elif args.house_envs:  # 家と庭と、元の地形を混ぜて学習する（元の地形での歩き方を忘れないように）
        from stable_baselines3.common.monitor import Monitor
        import gymnasium as gym

        def maker(env_id, rank):
            def f():
                e = Monitor(gym.make(env_id))
                e.reset(seed=rank)
                return e
            return f
        ids = ["HouseDog-v0"] * args.house_envs + [args.env] * (args.n_envs - args.house_envs)
        venv = SubprocVecEnv([maker(e, i) for i, e in enumerate(ids)])
    else:
        venv = make_vec_env(args.env, n_envs=args.n_envs, seed=0, vec_env_cls=SubprocVecEnv)
    start = 0
    if args.resume:  # 中断前と同じ学習率の予定・ステップ数のまま続ける
        args.init = args.resume
        start = int(os.path.basename(args.resume)[6:15])
    if args.init:
        from demo import vecnorm_for

        venv = VecNormalize.load(vecnorm_for(args.init), venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    lr = (lambda progress_remaining: args.lr * progress_remaining) if args.lr_decay else args.lr

    big = args.env == "BigStairs-v0"
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=4096 if big else 2048,
        n_epochs=5 if big else 10,
        learning_rate=lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        policy_kwargs=(dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]), log_std_init=-0.5)
                       if args.env == "BigStairs-v0" else
                       dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5 if args.env == "GoalAnt-v0" else -0.7)),
        device="cuda" if big and torch.cuda.is_available() else "cpu",  # 大型版はネットが大きいので更新を GPU で
        verbose=0,
        seed=0,
    )
    if args.init:
        model.set_parameters(args.init)
    model.set_logger(configure(args.out, ["csv"]))
    cb = CheckpointCallback(args.save_every, ckpt)
    if start:
        model.num_timesteps = start
        cb.next_save = start + args.save_every
    model.learn(total_timesteps=args.steps - start, callback=cb, reset_num_timesteps=not start)
    model.save(os.path.join(args.out, "final_model"))
    venv.save(os.path.join(args.out, "final_vecnorm.pkl"))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
