import sys, time
import jax, jax.numpy as jp
from mujoco_playground import registry
name = sys.argv[1]; N = int(sys.argv[2])
cfg = registry.get_default_config(name); cfg.impl = sys.argv[3] if len(sys.argv) > 3 else "warp"
env = registry.load(name, config=cfg)
reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
st = reset(jax.random.split(jax.random.PRNGKey(0), N)); a = jp.zeros((N, env.action_size))
st = step(st, a); jax.block_until_ready(st.obs)
t0 = time.time()
for _ in range(50): st = step(st, a)
jax.block_until_ready(st.obs); dt = time.time() - t0
print(f"{name} {cfg.impl} N={N} steps/s={50 * N / dt:.0f} ({dt / 50 * 1000:.1f} ms/step)", flush=True)
