"""GPU（MJX）での学習速度を、Playground の Go1 の例で測る。"""
import functools
import time

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

name = "Go1JoystickFlatTerrain"
env_cfg = registry.get_default_config(name)
env_cfg.impl = "jax"
env = registry.load(name, config=env_cfg)
p = locomotion_params.brax_ppo_config(name, "jax")
p.num_timesteps = 3_000_000
p.num_evals = 2
p.num_envs = 512
p.batch_size = 256
t0 = time.time()
kw = dict(p)
net = kw.pop("network_factory", None)
nf = functools.partial(ppo_networks.make_ppo_networks, **p.network_factory) if "network_factory" in p else ppo_networks.make_ppo_networks
kw.pop("network_factory", None)
times = []
def progress(step, m):
    times.append((step, time.time() - t0))
    print(step, round(time.time() - t0, 1), float(m.get("eval/episode_reward", 0)), flush=True)
make_inference, params, _ = ppo.train(environment=env, network_factory=nf, progress_fn=progress,
                                      wrap_env_fn=wrapper.wrap_for_brax_training, **kw)
print("devices", jax.devices(), "total", time.time() - t0)
