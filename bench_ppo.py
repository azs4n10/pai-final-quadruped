"""PPO の 1 回分の時間を、データ集め（collect_rollouts）と更新（train）に分けて測る。"""
if __name__ == "__main__":
    import sys, time
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from vec_workers import BatchedSubprocVecEnv
    dev, N = sys.argv[1], int(sys.argv[2])
    v = VecNormalize(BatchedSubprocVecEnv("BigStairs-v0", N, 16, "stairs_cpu"), clip_obs=10.0)
    m = PPO("MlpPolicy", v, n_steps=512, batch_size=8192, n_epochs=5, device=dev, verbose=0,
            policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])))
    T = {"collect": 0.0, "train": 0.0}
    oc, ot = m.collect_rollouts, m.train
    def c(*a, **k):
        t = time.time(); r = oc(*a, **k); T["collect"] += time.time() - t; return r
    def tr(*a, **k):
        t = time.time(); r = ot(*a, **k); T["train"] += time.time() - t; return r
    m.collect_rollouts, m.train = c, tr
    m.learn(total_timesteps=3 * 512 * N)
    print(dev, {k: round(v, 1) for k, v in T.items()}, N, "sps", round(3 * 512 * N / sum(T.values())), flush=True)
    v.close()
