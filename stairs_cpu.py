"""
GPU 版（gpu/stairs_env.py）と同じ、大型版の犬型ロボットの「速度指令で歩く」環境を、CPU の MuJoCo で動かす版。
観測・報酬・地形・カリキュラムは GPU 版と同じ。SB3 の PPO で並列学習する（train_goal.py --env BigStairs-v0）。
"""
import os

import gymnasium as gym
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "gpu", "big_dog.xml")
TERRAIN = os.path.join(HERE, "gpu", "terrain.npy")
TILE, N_LEVEL, N_COL, RES = 6.0, 10, 6, 0.03
BASE_H, FOOT_R, ACTION_SCALE = 0.46, 0.026 * 1.4, 0.8  # 0.5 では膝が -1.95 rad までしか曲がらず、足を 14 cm 以上の段に上げられない
SCAN = np.array([(x, y) for x in np.linspace(-0.45, 0.9, 10) for y in np.linspace(-0.4, 0.4, 5)])
SCALES = dict(tracking_lin_vel=1.5, tracking_ang_vel=0.75, lin_vel_z=-0.2, ang_vel_xy=-0.05, orientation=-2.0, dof_pos_limits=-1.0,
              pose=0.2, termination=-1.0, stand_still=-0.5, torques=-0.00005, action_rate=-0.01, energy=-0.0003,
              feet_clearance=-1.0, feet_slip=-0.1, feet_air_time=0.2, body_height=-5.0)
MAX_FOOT_H, SIGMA, DT, EP_LEN = 0.15, 0.25, 0.02, 1000
COL_P = np.array([0.25, 0.25, 0.1, 0.1, 0.15, 0.15])  # 地形の列の選ばれやすさ（上り階段を多めに練習する）


class BigStairsEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.H = np.load(TERRAIN)
        self.m.hfield_data[:] = (self.H / self.m.hfield_size[0, 2]).ravel()
        self.d = mujoco.MjData(self.m)
        self.n_sub = int(round(DT / self.m.opt.timestep))
        self.default = self.m.key_qpos[0][7:].copy()
        lo, hi = self.m.jnt_range[1:].T
        self.lo, self.hi = lo * 0.95, hi * 0.95
        self.imu = self.m.site("imu").id
        self.feet = np.array([self.m.site(f"{n}_foot_site").id for n in ("FL", "FR", "HL", "HR")])
        self.gyro_adr = self.m.sensor_adr[self.m.sensor("gyro").id]
        self.vel_adr = self.m.sensor_adr[self.m.sensor("local_linvel").id]
        self.up_adr = self.m.sensor_adr[self.m.sensor("upvector").id]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (98,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.level = 0
        self.first = True

    # 地形
    def terrain_height(self, x, y):
        fx, fy = np.asarray(x) / RES, np.asarray(y) / RES
        ny, nx = self.H.shape
        x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
        H = self.H
        return (H[y0, x0] * (1 - tx) + H[y0, x0 + 1] * tx) * (1 - ty) + (H[y0 + 1, x0] * (1 - tx) + H[y0 + 1, x0 + 1] * tx) * ty

    def yaw(self):
        q = self.d.qpos[3:7]
        return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def scan(self):
        c, s = np.cos(self.yaw()), np.sin(self.yaw())
        x, y, z = self.d.qpos[:3]
        return np.clip(z - self.terrain_height(x + c * SCAN[:, 0] - s * SCAN[:, 1], y + s * SCAN[:, 0] + c * SCAN[:, 1]) - BASE_H, -1, 1)

    def body_height(self):
        return self.d.qpos[2] - self.terrain_height(self.d.qpos[0], self.d.qpos[1])

    def feet_height(self):
        p = self.d.site_xpos[self.feet]
        return p[:, 2] - FOOT_R - self.terrain_height(p[:, 0], p[:, 1])

    def sample_command(self):
        if self.np_random.uniform() < 0.1:
            return np.zeros(3)
        return np.array([self.np_random.uniform(0, 1.0), self.np_random.uniform(-0.3, 0.3), self.np_random.uniform(-0.8, 0.8)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.first:
            self.level = int(self.np_random.integers(0, 3))
            self.first = False
        else:  # カリキュラム：前のエピソードの結果で難しさを上下させる
            up = (not self.fell) and self.dist > 0.4 * TILE
            down = self.fell or self.dist < 1.0
            if up and self.level == N_LEVEL - 1:
                self.level = int(self.np_random.integers(0, N_LEVEL))
            else:
                self.level = int(np.clip(self.level + up - down, 0, N_LEVEL - 1))
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        col = self.np_random.choice(N_COL, p=COL_P)
        xy = (np.array([col, self.level]) + 0.5) * TILE + self.np_random.uniform(-0.3, 0.3, 2)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.d.qpos[0:2] = xy
        self.d.qpos[2] = self.terrain_height(*xy) + BASE_H + 0.03
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.qvel[0:6] = self.np_random.uniform(-0.3, 0.3, 6)
        self.d.ctrl[:] = self.default
        mujoco.mj_forward(self.m, self.d)
        self.spawn = xy.copy()
        self.cmd = self.sample_command()
        self.until_cmd = 250
        self.last_act = np.zeros(12)
        self.last_last_act = np.zeros(12)
        self.air = np.zeros(4)
        self.last_contact = np.zeros(4, bool)
        self.last_feet_xy = self.d.site_xpos[self.feet][:, :2].copy()
        self.t, self.fell, self.dist = 0, False, 0.0
        return self._obs(), {}

    def _obs(self):
        g = self.d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([self.d.sensordata[self.vel_adr:self.vel_adr + 3], self.d.sensordata[self.gyro_adr:self.gyro_adr + 3], g,
                               self.d.qpos[7:] - self.default, self.d.qvel[6:], self.last_act, self.cmd, self.scan()]).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.d.ctrl[:] = self.default + ACTION_SCALE * a
        mujoco.mj_step(self.m, self.d, nstep=self.n_sub)
        self.t += 1
        fh = self.feet_height()
        contact = fh < 0.02
        first_contact = (self.air > 0) & (contact | self.last_contact)
        self.air += DT
        v = self.d.sensordata[self.vel_adr:self.vel_adr + 3]
        gyro = self.d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        up = self.d.sensordata[self.up_adr:self.up_adr + 3]
        done = bool(up[2] < 0.3 or self.body_height() < 0.22)
        cmd_on = np.linalg.norm(self.cmd) > 0.01
        feet_xy = self.d.site_xpos[self.feet][:, :2]
        speed = np.linalg.norm((feet_xy - self.last_feet_xy) / DT, axis=-1)
        q, f = self.d.qpos[7:], self.d.actuator_force
        r = dict(
            tracking_lin_vel=np.exp(-np.sum((self.cmd[:2] - v[:2]) ** 2) / SIGMA),
            tracking_ang_vel=np.exp(-(self.cmd[2] - gyro[2]) ** 2 / SIGMA),
            lin_vel_z=self.d.qvel[2] ** 2, ang_vel_xy=np.sum(self.d.qvel[3:5] ** 2), orientation=np.sum(up[:2] ** 2),
            dof_pos_limits=np.sum(-np.clip(q - self.lo, None, 0) + np.clip(q - self.hi, 0, None)),
            pose=np.exp(-np.sum((q - self.default) ** 2 * np.array([1.0, 1.0, 0.1] * 4))),
            termination=float(done), stand_still=np.sum(np.abs(q - self.default)) * (not cmd_on),
            torques=np.sqrt(np.sum(f ** 2)) + np.sum(np.abs(f)), action_rate=np.sum((a - self.last_act) ** 2),
            energy=np.sum(np.abs(self.d.qvel[6:]) * np.abs(f)),
            feet_clearance=np.sum(np.abs(fh - MAX_FOOT_H) * np.sqrt(speed + 1e-6)),
            feet_slip=np.sum(speed ** 2 * contact) * cmd_on, feet_air_time=np.sum((self.air - 0.1) * first_contact) * cmd_on,
            body_height=(self.body_height() - BASE_H) ** 2,
        )
        reward = float(np.clip(sum(SCALES[k] * v for k, v in r.items()) * DT, 0, 1e4))
        self.last_last_act, self.last_act = self.last_act, a
        self.air *= ~contact
        self.last_contact = contact
        self.last_feet_xy = feet_xy.copy()
        self.until_cmd -= 1
        if self.until_cmd <= 0:
            self.cmd, self.until_cmd = self.sample_command(), 250
        self.fell = self.fell or done
        self.dist = float(np.linalg.norm(self.d.qpos[:2] - self.spawn))
        info = {"level": self.level, "levels": {"stairs": float(self.level)}, "n_reached": 0}
        return self._obs(), reward, done, self.t >= EP_LEN, info


gym.register(id="BigStairs-v0", entry_point=BigStairsEnv)
