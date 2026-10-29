"""
大型版の犬型ロボットの「速度指令で歩く」方策を、GPU（MJX）上の PPO（Brax）で学習する。WSL の Ubuntu で実行する。

  XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 ~/mjx/bin/python gpu/train.py --steps 300000000
  ~/mjx/bin/python gpu/train.py --smoke     # 動作確認だけ
チェックポイントは gpu/ckpt/params_<ステップ数>.pkl（正規化の統計・方策・価値の重み）。
"""
import argparse
import functools
import os
import pickle
import sys
import time

import jax
import jax.numpy as jp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stairs_env import BigDogStairs  # noqa: E402


def smoke():
    env = BigDogStairs()
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    st = reset(jax.random.PRNGKey(0))
    t0 = time.time()
    for i in range(200):
        st = step(st, jp.zeros(env.action_size))
    jax.block_until_ready(st.obs)
    print("obs", {k: v.shape for k, v in st.obs.items()}, "height", float(env.body_height(st.data)),
          "done", float(st.done), "reward", float(st.reward), "time", round(time.time() - t0, 1))


def main():
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000_000)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--evals", type=int, default=10)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        smoke()
        return

    env = BigDogStairs()
    out = os.path.join(HERE, "ckpt")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    def progress(step, m):
        print(f"[ckpt] step={step} t={time.time() - t0:.0f}s reward={float(m.get('eval/episode_reward', 0)):.2f} "
              f"level={float(m.get('eval/episode_level', 0)) / 1000:.2f} len={float(m.get('eval/avg_episode_length', 0)):.0f}",
              flush=True)

    def save(step, make_policy, params):
        with open(os.path.join(out, f"params_{step:010d}.pkl"), "wb") as f:
            pickle.dump(jax.device_get(params), f)

    net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(512, 256, 128),
                            value_hidden_layer_sizes=(512, 256, 128), policy_obs_key="state", value_obs_key="privileged_state")
    ppo.train(
        environment=env, wrap_env_fn=wrapper.wrap_for_brax_training, network_factory=net,
        num_timesteps=args.steps, num_evals=args.evals, num_eval_envs=128, episode_length=1000,
        normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=32, num_updates_per_batch=4,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=args.envs, batch_size=256,
        max_grad_norm=1.0, clipping_epsilon=0.2, num_resets_per_eval=1, seed=0,
        progress_fn=progress, policy_params_fn=save,
    )
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
