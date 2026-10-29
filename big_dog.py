"""
大型版の犬型ロボットを、GPU で学習した「速度指令で歩く」方策で動かす（MuJoCo・CPU）。

- 方策：models/big_dog_policy.npz（gpu/export.py で変換）を numpy で計算する（Brax の MLP：swish、出力は tanh）
- 観測：GPU 学習（gpu/stairs_env.py）と同じ並び。足元 50 点の地面の高さは、下向きのレイで測る
- 経由地へは、向きのずれから旋回の速度、距離から前進の速度を決めて指令する
"""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = os.path.join(HERE, "models", "big_dog_policy.npz")
BASE_H = 0.46
ACTION_SCALE = 0.5
SCAN_X = np.linspace(-0.45, 0.9, 10)
SCAN_Y = np.linspace(-0.4, 0.4, 5)
SCAN = np.array([(x, y) for x in SCAN_X for y in SCAN_Y])
RAY_GROUP = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)  # 床・段（グループ 3）だけに当てる


class Policy:
    def __init__(self, path=POLICY):
        p = np.load(path)
        self.mean, self.std = p["obs_mean"], p["obs_std"]
        n = len([k for k in p.files if k.startswith("w")])
        self.layers = [(p[f"w{i}"], p[f"b{i}"]) for i in range(n)]

    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = x / (1 + np.exp(-x))  # swish
        return np.tanh(x[: len(x) // 2])  # 平均（loc）を tanh に通す＝決定的な行動


class SB3Policy:
    """CPU で学習した方策（Stable-Baselines3 の PPO と、観測の正規化の統計）。"""

    def __init__(self, model_path, vecnorm_path):
        import pickle

        from stable_baselines3 import PPO

        self.model = PPO.load(model_path, device="cpu")
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        self.mean, self.var, self.clip, self.eps = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon

    def __call__(self, obs):
        x = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        a, _ = self.model.predict(x.astype(np.float32), deterministic=True)
        return np.clip(a, -1, 1)


def load_policy(path=None):
    """models/big_dog_model.zip（CPU で学習）があればそれを、なければ GPU 版の .npz を使う。"""
    path = path or os.path.join(HERE, "models", "big_dog_model.zip")
    if path.endswith(".zip"):
        from demo import vecnorm_for

        return SB3Policy(path, vecnorm_for(path))
    return Policy(path)


class BigDog:
    """MjModel / MjData を受け取り、観測を作って方策で 1 ステップ（0.02 秒）進める。"""

    def __init__(self, model, data, policy=None):
        self.m, self.d = model, data
        self.policy = policy or load_policy()
        self.default = model.key_qpos[0][7:].copy() if model.nkey else None
        self.imu = model.site("imu").id
        self.gyro = model.sensor_adr[model.sensor("gyro").id]
        self.vel = model.sensor_adr[model.sensor("local_linvel").id]
        self.last_act = np.zeros(model.nu)
        self.n_sub = int(round(0.02 / model.opt.timestep))
        self._gid = np.zeros(1, dtype=np.int32)

    def ground_z(self, x, y, top):
        dist = mujoco.mj_ray(self.m, self.d, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]), RAY_GROUP, 1, -1, self._gid)
        return top - dist if dist >= 0 else 0.0

    def yaw(self):
        q = self.d.qpos[3:7]
        return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def scan(self):
        c, s = np.cos(self.yaw()), np.sin(self.yaw())
        x, y, z = self.d.qpos[:3]
        top = z + 0.6  # 2 階の床に当たらないよう、胴体のすぐ上から測る
        h = [self.ground_z(x + c * px - s * py, y + s * px + c * py, top) for px, py in SCAN]
        return np.clip(z - np.array(h) - BASE_H, -1.0, 1.0)

    def body_height(self):
        x, y, z = self.d.qpos[:3]
        return z - self.ground_z(x, y, z + 0.6)

    def obs(self, cmd):
        g = self.d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([
            self.d.sensordata[self.vel:self.vel + 3], self.d.sensordata[self.gyro:self.gyro + 3], g,
            self.d.qpos[7:] - self.default, self.d.qvel[6:], self.last_act, cmd, self.scan(),
        ])

    def step(self, cmd):
        a = self.policy(self.obs(np.asarray(cmd, dtype=float)))
        self.d.ctrl[:] = self.default + ACTION_SCALE * a
        mujoco.mj_step(self.m, self.d, nstep=self.n_sub)
        self.last_act = a
        return a

    def fallen(self):
        up = self.d.site_xmat[self.imu].reshape(3, 3)[:, 2]
        return up[2] < 0.3 or self.body_height() < 0.22


def follow_command(xy, yaw, target, v_max=0.8):
    """経由地 target へ向かう速度指令 (vx, vy, wz)。向きがずれているほど前進を弱めて旋回する。"""
    d = np.asarray(target) - xy
    err = (np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
    wz = float(np.clip(1.5 * err, -0.8, 0.8))
    vx = float(np.clip(np.linalg.norm(d), 0.0, v_max) * max(np.cos(err), 0.0) ** 2)
    return np.array([vx, 0.0, wz])
