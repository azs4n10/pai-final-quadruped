"""
1 つの子プロセスで複数の環境をまとめて動かす並列化（Stable-Baselines3 の VecEnv の形）。

SubprocVecEnv は環境 1 つにつき子プロセス 1 つで、子プロセスごとに PyTorch まで読み込む。
ここでは子プロセスを少なくし（各プロセスが K 個の環境を順に進める）、子プロセスは物理シミュレーションだけを読み込む。
"""
import multiprocessing as mp

import numpy as np


def _worker(remote, env_id, seeds, module):
    import importlib

    import gymnasium as gym

    importlib.import_module(module)  # 環境を登録するモジュール（PyTorch は読み込まない）
    import time

    envs = [gym.make(env_id) for _ in seeds]
    for e, s in zip(envs, seeds):
        e.reset(seed=s)
    ret, length, t0 = [0.0] * len(envs), [0] * len(envs), time.time()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            out = []
            for k, (e, a) in enumerate(zip(envs, data)):
                obs, r, term, trunc, info = e.step(a)
                done = term or trunc
                ret[k] += r
                length[k] += 1
                if done:  # SB3 の決まり：終わった環境は自動でリセットし、最後の観測は info に入れる
                    info["episode"] = {"r": ret[k], "l": length[k], "t": time.time() - t0}  # Monitor と同じ形
                    ret[k], length[k] = 0.0, 0
                    info["terminal_observation"] = obs
                    info["TimeLimit.truncated"] = trunc and not term
                    obs, _ = e.reset()
                out.append((obs, r, done, info))
            remote.send(out)
        elif cmd == "reset":
            remote.send([e.reset()[0] for e in envs])
        elif cmd == "close":
            remote.close()
            return


class BatchedSubprocVecEnv:
    """n_envs 個の環境を n_procs 個の子プロセスに分けて動かす。"""

    def __new__(cls, *args, **kwargs):
        from stable_baselines3.common.vec_env.base_vec_env import VecEnv

        class _Impl(VecEnv):
            def __init__(self, env_id, n_envs, n_procs, module, seed=0):
                import importlib

                import gymnasium as gym

                importlib.import_module(module)

                ctx = mp.get_context("spawn")
                per = [list(range(i, n_envs, n_procs)) for i in range(n_procs)]
                self.remotes, self.procs, self.per = [], [], per
                for idx in per:
                    a, b = ctx.Pipe()
                    p = ctx.Process(target=_worker, args=(b, env_id, [seed + i for i in idx], module), daemon=True)
                    p.start()
                    b.close()
                    self.remotes.append(a)
                    self.procs.append(p)
                proto = gym.make(env_id)
                super().__init__(n_envs, proto.observation_space, proto.action_space)
                proto.close()
                self._actions = None

            def reset(self):
                for r in self.remotes:
                    r.send(("reset", None))
                obs = np.zeros((self.num_envs, *self.observation_space.shape), np.float32)
                for r, idx in zip(self.remotes, self.per):
                    obs[idx] = np.stack(r.recv())
                return obs

            def step_async(self, actions):
                for r, idx in zip(self.remotes, self.per):
                    r.send(("step", actions[idx]))

            def step_wait(self):
                obs = np.zeros((self.num_envs, *self.observation_space.shape), np.float32)
                rews = np.zeros(self.num_envs, np.float32)
                dones = np.zeros(self.num_envs, bool)
                infos = [None] * self.num_envs
                for r, idx in zip(self.remotes, self.per):
                    for i, (o, rw, d, info) in zip(idx, r.recv()):
                        obs[i], rews[i], dones[i], infos[i] = o, rw, d, info
                return obs, rews, dones, infos

            def close(self):
                for r in self.remotes:
                    r.send(("close", None))
                for p in self.procs:
                    p.join(timeout=5)

            def get_attr(self, attr_name, indices=None):
                return [None] * len(self._get_indices(indices))

            def set_attr(self, attr_name, value, indices=None):
                pass

            def env_method(self, method_name, *args, indices=None, **kwargs):
                return [None] * len(self._get_indices(indices))

            def env_is_wrapped(self, wrapper_class, indices=None):
                return [False] * len(self._get_indices(indices))

            def seed(self, seed=None):
                return [None] * self.num_envs

        return _Impl(*args, **kwargs)
