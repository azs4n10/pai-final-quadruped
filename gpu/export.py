"""
GPU で学習したチェックポイント（Brax の PPO）を、numpy だけで読める .npz に変換する（WSL で実行）。

  ~/mjx/bin/python gpu/export.py gpu/ckpt/params_0300000000.pkl models/big_dog_policy.npz
中身：観測の正規化（mean・std）と、方策の全結合層の重み（hidden_0 … の kernel・bias）。
"""
import pickle
import sys

import numpy as np


def main(src, dst):
    with open(src, "rb") as f:
        params = pickle.load(f)
    norm, policy = params[0], params[1]
    out = {"obs_mean": np.asarray(norm.mean["state"]), "obs_std": np.asarray(norm.std["state"])}
    layers = policy["params"]
    for i, name in enumerate(sorted(layers, key=lambda k: int(k.split("_")[-1]))):
        out[f"w{i}"] = np.asarray(layers[name]["kernel"])
        out[f"b{i}"] = np.asarray(layers[name]["bias"])
    np.savez(dst, **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
