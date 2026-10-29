"""並列環境の速さを測る（方策なし／方策の計算あり）。"""
if __name__ == "__main__":
    import time
    import numpy as np
    from vec_workers import BatchedSubprocVecEnv
    for n_envs, n_procs in [(32, 8), (32, 16), (64, 16)]:
        v = BatchedSubprocVecEnv("BigStairs-v0", n_envs, n_procs, "stairs_cpu")
        v.reset()
        t = time.time()
        for _ in range(200):
            v.step(np.random.uniform(-1, 1, (n_envs, 12)))
        print(n_envs, n_procs, "env only", round(200 * n_envs / (time.time() - t)), "sps", flush=True)
        v.close()
