"""環境の 1 ステップの速さを測る（並列数 N）。"""
import sys, time, os
import jax, jax.numpy as jp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stairs_env import BigDogStairs
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
env = BigDogStairs()
reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
st = reset(jax.random.split(jax.random.PRNGKey(0), N))
a = jp.zeros((N, env.action_size))
st = step(st, a); jax.block_until_ready(st.obs)
t0 = time.time()
for _ in range(50):
    st = step(st, a)
jax.block_until_ready(st.obs)
dt = time.time() - t0
print(f"N={N} steps/s={50 * N / dt:.0f}  ({dt / 50 * 1000:.1f} ms/step)", flush=True)
