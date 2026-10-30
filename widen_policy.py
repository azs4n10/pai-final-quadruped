"""
位相なしで学習した方策（観測 98 次元）を、歩くリズムの位相（sin・cos）を足した観測 100 次元の方策に広げる。
増えた入力の重みは 0 にするので、広げた直後の行動は元の方策と同じ。行動のばらつきは探索のため少し大きくする。

  python widen_policy.py runs/big_stairs2/ckpt/model_090000000.zip runs/big_trot/init_model.zip
"""
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import stairs_cpu  # noqa: F401  （BigTrot-v0 を登録する）
from demo import vecnorm_for

MIN_STD = 0.25


def main(src, dst):
    old = PPO.load(src, device="cpu")
    env = gym.make("BigTrot-v0")
    new = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])), device="cpu")
    sd = old.policy.state_dict()
    for k in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        w = sd[k]
        sd[k] = torch.cat([w, torch.zeros(w.shape[0], 2)], dim=1)
    sd["log_std"] = torch.clamp(sd["log_std"], min=float(np.log(MIN_STD)))
    new.policy.load_state_dict(sd)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    new.save(dst)

    vn = VecNormalize.load(vecnorm_for(src), DummyVecEnv([lambda: gym.make("BigStairs-v0")]))
    vn.obs_rms.mean = np.concatenate([vn.obs_rms.mean, [0.0, 0.0]])
    vn.obs_rms.var = np.concatenate([vn.obs_rms.var, [0.5, 0.5]])  # sin・cos の分散
    vn.observation_space = env.observation_space
    vn.save(vecnorm_for(dst))
    print("->", dst, vecnorm_for(dst))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
