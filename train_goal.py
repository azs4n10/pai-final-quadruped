"""
GoalAnt を PPO で学習する（CPU 並列）。

  python train_goal.py --steps 20000000
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

HERE = os.path.dirname(os.path.abspath(__file__))


class CheckpointCallback(BaseCallback):
    """一定ステップごとにモデルと正規化統計を保存し、到達数を表示する。"""

    def __init__(self, every, out_dir):
        super().__init__()
        self.every, self.out_dir = every, out_dir
        self.next_save = 0
        self.t0 = time.time()
        self.reached = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.reached.append(info["n_reached"])
        if self.num_timesteps >= self.next_save:
            tag = f"{self.num_timesteps:09d}"
            self.model.save(os.path.join(self.out_dir, f"model_{tag}"))
            self.training_env.save(os.path.join(self.out_dir, f"vecnorm_{tag}.pkl"))
            if self.reached:
                avg = sum(self.reached[-100:]) / len(self.reached[-100:])
                self.logger.record("goal/reached_per_ep", avg)
            fps = self.num_timesteps / max(time.time() - self.t0, 1e-6)
            print(f"[ckpt] step={self.num_timesteps} fps={fps:.0f} reached/ep(last100)="
                  f"{(sum(self.reached[-100:]) / max(len(self.reached[-100:]), 1)):.2f}", flush=True)
            self.next_save += self.every
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--save-every", type=int, default=1_000_000)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "goal_ant"))
    args = p.parse_args()

    torch.set_num_threads(4)
    ckpt = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt, exist_ok=True)

    venv = make_vec_env("GoalAnt-v0", n_envs=args.n_envs, seed=0, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=2048,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5),
        device="cpu",
        verbose=0,
        seed=0,
    )
    model.set_logger(configure(args.out, ["csv"]))
    model.learn(total_timesteps=args.steps, callback=CheckpointCallback(args.save_every, ckpt))
    model.save(os.path.join(args.out, "final_model"))
    venv.save(os.path.join(args.out, "final_vecnorm.pkl"))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
