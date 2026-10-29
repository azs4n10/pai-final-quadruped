"""
GPU（MJX）で学習する、大型版の犬型ロボットの「速度指令で歩く」環境（階段・坂・凸凹の練習場）。

MuJoCo Playground の Go1 Joystick 環境（Apache-2.0）を参考に、次を変えた。
- ロボット：自作の犬型ロボットの大型版（gpu/make_assets.py）
- 地形：難しさ 10 段階×6 種類のタイル（段差 5〜18 cm の上り・下り階段、坂、凸凹）
- 観測：足元 50 点の地面の高さ（高さマップから直接読む）を加える
- カリキュラム：エピソードの終わりに、2.4 m 以上進めたら 1 段難しいタイル、転ぶか 1 m 未満なら 1 段易しいタイルへ移す
"""
import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env

HERE = os.path.dirname(os.path.abspath(__file__))
TILE, N_LEVEL, N_COL, RES = 6.0, 10, 6, 0.03  # gpu/make_assets.py と同じ
LEGS = ["FL", "FR", "HL", "HR"]
BASE_H = 0.46  # 立ち姿勢での胴体の高さ（地面から）
FOOT_R = 0.026 * 1.4
SCAN_X = np.linspace(-0.45, 0.9, 10)
SCAN_Y = np.linspace(-0.4, 0.4, 5)
SCAN = np.array([(x, y) for x in SCAN_X for y in SCAN_Y], dtype=np.float32)  # 50 点（胴体座標）


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05, linvel=0.1, scan=0.02),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=1.5,
                tracking_ang_vel=0.75,
                lin_vel_z=-0.2,
                ang_vel_xy=-0.05,
                orientation=-2.0,
                dof_pos_limits=-1.0,
                pose=0.2,
                termination=-1.0,
                stand_still=-0.5,
                torques=-0.00005,
                action_rate=-0.01,
                energy=-0.0003,
                feet_clearance=-1.0,
                feet_slip=-0.1,
                feet_air_time=0.2,
                body_height=-5.0,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.15,
        ),
        command_config=config_dict.create(vx=[0.0, 1.0], vy=0.3, wz=0.8, zero_prob=0.1),
        impl="warp",
        naconmax=16 * 8192,
        njmax=80,
    )


class BigDogStairs(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides)
        self._xml_path = os.path.join(HERE, "big_dog.xml")
        m = mujoco.MjModel.from_xml_path(self._xml_path)
        H = np.load(os.path.join(HERE, "terrain.npy"))
        m.hfield_data[:] = (H / m.hfield_size[0, 2]).ravel()
        m.opt.timestep = self._config.sim_dt
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        self._H = jp.array(H)
        self._scan = jp.array(SCAN)
        self._init_q = jp.array(m.keyframe("home").qpos)
        self._default_pose = jp.array(m.keyframe("home").qpos[7:])
        lo, hi = m.jnt_range[1:].T
        f = self._config.soft_joint_pos_limit_factor
        self._soft_lo, self._soft_hi = jp.array(lo * f), jp.array(hi * f)
        self._imu = m.site("imu").id
        self._torso = m.body("torso").id
        self._feet_site = np.array([m.site(f"{n}_foot_site").id for n in LEGS])
        self._gyro_adr = m.sensor_adr[m.sensor("gyro").id]
        self._linvel_adr = m.sensor_adr[m.sensor("local_linvel").id]
        self._up_adr = m.sensor_adr[m.sensor("upvector").id]

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # --- 地形 ---
    def terrain_height(self, x, y):
        """高さマップの高さ（双線形補間）。x, y は同じ形の配列。"""
        fx, fy = x / RES, y / RES
        ny, nx = self._H.shape
        x0 = jp.clip(jp.floor(fx).astype(jp.int32), 0, nx - 2)
        y0 = jp.clip(jp.floor(fy).astype(jp.int32), 0, ny - 2)
        tx, ty = jp.clip(fx - x0, 0, 1), jp.clip(fy - y0, 0, 1)
        h00, h10 = self._H[y0, x0], self._H[y0, x0 + 1]
        h01, h11 = self._H[y0 + 1, x0], self._H[y0 + 1, x0 + 1]
        return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty

    def height_scan(self, data):
        q = data.qpos[3:7]
        yaw = jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        c, s = jp.cos(yaw), jp.sin(yaw)
        px = data.qpos[0] + c * self._scan[:, 0] - s * self._scan[:, 1]
        py = data.qpos[1] + s * self._scan[:, 0] + c * self._scan[:, 1]
        return jp.clip(data.qpos[2] - self.terrain_height(px, py) - BASE_H, -1.0, 1.0)

    def body_height(self, data):
        return data.qpos[2] - self.terrain_height(data.qpos[0], data.qpos[1])

    def feet_height(self, data):
        p = data.site_xpos[self._feet_site]
        return p[:, 2] - FOOT_R - self.terrain_height(p[:, 0], p[:, 1])

    # --- センサー ---
    def gyro(self, data):
        return data.sensordata[self._gyro_adr:self._gyro_adr + 3]

    def local_linvel(self, data):
        return data.sensordata[self._linvel_adr:self._linvel_adr + 3]

    def upvector(self, data):
        return data.sensordata[self._up_adr:self._up_adr + 3]

    def gravity(self, data):
        return data.site_xmat[self._imu].T @ jp.array([0.0, 0.0, -1.0])

    # --- 出発地点とカリキュラム ---
    def _spawn(self, rng, level):
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        col = jax.random.randint(k1, (), 0, N_COL)
        xy = (jp.array([col, level]) + 0.5) * TILE + jax.random.uniform(k2, (2,), minval=-0.3, maxval=0.3)
        yaw = jax.random.uniform(k3, (1,), minval=-jp.pi, maxval=jp.pi)
        qpos = self._init_q
        qpos = qpos.at[0:2].set(xy)
        qpos = qpos.at[2].set(self.terrain_height(xy[0], xy[1]) + BASE_H + 0.03)
        qpos = qpos.at[3:7].set(math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw))
        qvel = jp.zeros(self._mjx_model.nv).at[0:6].set(jax.random.uniform(k4, (6,), minval=-0.3, maxval=0.3))
        return qpos, qvel

    def _sample_command(self, rng):
        cc = self._config.command_config
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        cmd = jp.array([
            jax.random.uniform(k1, minval=cc.vx[0], maxval=cc.vx[1]),
            jax.random.uniform(k2, minval=-cc.vy, maxval=cc.vy),
            jax.random.uniform(k3, minval=-cc.wz, maxval=cc.wz),
        ])
        return jp.where(jax.random.uniform(k4) < cc.zero_prob, jp.zeros(3), cmd)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        level = jax.random.randint(k1, (), 0, 3)
        qpos, qvel = self._spawn(k2, level)
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:], impl=self.mjx_model.impl.value,
                                 naconmax=self._config.naconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        info = {
            "rng": rng,
            "command": self._sample_command(k3),
            "steps_until_next_cmd": jp.array(250, dtype=jp.int32),
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "last_feet_xy": data.site_xpos[self._feet_site][:, 0:2],
            "level": level,
            "spawn_xy": qpos[0:2],
            "fell": jp.array(False),
            "dist": jp.array(0.0),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["level"] = level.astype(jp.float32)
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _respawn(self, state):
        """自動リセットで最初の状態に戻った直後に、カリキュラムで決めたタイルへ移す。"""
        info = state.info
        just_reset = info["steps"] == 0 if "steps" in info else jp.array(False)
        up = (~info["fell"]) & (info["dist"] > 0.4 * TILE)
        down = info["fell"] | (info["dist"] < 1.0)
        level = jp.clip(info["level"] + up.astype(jp.int32) - down.astype(jp.int32), 0, N_LEVEL - 1)
        info["rng"], k1, k2, k3 = jax.random.split(info["rng"], 4)
        # 最も難しいタイルを越えたら、ランダムな難しさから練習し直す（忘れないように）
        level = jp.where(up & (info["level"] == N_LEVEL - 1), jax.random.randint(k3, (), 0, N_LEVEL), level)
        qpos, qvel = self._spawn(k1, level)
        data = state.data.replace(qpos=jp.where(just_reset, qpos, state.data.qpos),
                                  qvel=jp.where(just_reset, qvel, state.data.qvel),
                                  ctrl=jp.where(just_reset, qpos[7:], state.data.ctrl))
        info["level"] = jp.where(just_reset, level, info["level"])
        info["spawn_xy"] = jp.where(just_reset, qpos[0:2], info["spawn_xy"])
        info["fell"] = jp.where(just_reset, False, info["fell"])
        info["dist"] = jp.where(just_reset, 0.0, info["dist"])
        info["command"] = jp.where(just_reset, self._sample_command(k2), info["command"])
        for k in ("last_act", "last_last_act", "feet_air_time"):
            info[k] = jp.where(just_reset, jp.zeros_like(info[k]), info[k])
        data = jax.lax.cond(jp.any(just_reset), lambda d: mjx.forward(self.mjx_model, d), lambda d: d, data)
        info["last_feet_xy"] = jp.where(just_reset, data.site_xpos[self._feet_site][:, 0:2], info["last_feet_xy"])
        return state.replace(data=data, info=info), just_reset

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state, _ = self._respawn(state)
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        fh = self.feet_height(data)
        contact = fh < 0.02
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)
        rewards = self._get_reward(data, action, state.info, done, first_contact, contact, fh)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], k1 = jax.random.split(state.info["rng"])
        resample = state.info["steps_until_next_cmd"] <= 0
        state.info["command"] = jp.where(resample, self._sample_command(k1), state.info["command"])
        state.info["steps_until_next_cmd"] = jp.where(resample, 250, state.info["steps_until_next_cmd"])
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["last_feet_xy"] = data.site_xpos[self._feet_site][:, 0:2]
        state.info["fell"] = state.info["fell"] | done
        state.info["dist"] = jp.linalg.norm(data.qpos[0:2] - state.info["spawn_xy"])
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["level"] = state.info["level"].astype(jp.float32)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _get_termination(self, data):
        return (self.upvector(data)[2] < 0.3) | (self.body_height(data) < 0.22)

    def _get_obs(self, data, info) -> Dict[str, jax.Array]:
        ns = self._config.noise_config

        def noisy(x, scale):
            info["rng"], k = jax.random.split(info["rng"])
            return x + (2 * jax.random.uniform(k, x.shape) - 1) * ns.level * scale

        scan = self.height_scan(data)
        state = jp.hstack([
            noisy(self.local_linvel(data), ns.scales.linvel),
            noisy(self.gyro(data), ns.scales.gyro),
            noisy(self.gravity(data), ns.scales.gravity),
            noisy(data.qpos[7:] - self._default_pose, ns.scales.joint_pos),
            noisy(data.qvel[6:], ns.scales.joint_vel),
            info["last_act"],
            info["command"],
            noisy(scan, ns.scales.scan),
        ])
        privileged = jp.hstack([
            state, self.local_linvel(data), self.gyro(data), self.gravity(data), scan,
            self.feet_height(data), data.actuator_force, info["feet_air_time"], self.body_height(data),
        ])
        return {"state": state, "privileged_state": privileged}

    def _get_reward(self, data, action, info, done, first_contact, contact, fh):
        cmd = info["command"]
        v = self.local_linvel(data)
        rc = self._config.reward_config
        cmd_on = jp.linalg.norm(cmd) > 0.01
        feet_vel = (data.site_xpos[self._feet_site][:, 0:2] - info["last_feet_xy"]) / self.dt  # 足の水平速度
        speed = jp.linalg.norm(feet_vel, axis=-1)
        return {
            "tracking_lin_vel": jp.exp(-jp.sum(jp.square(cmd[:2] - v[:2])) / rc.tracking_sigma),
            "tracking_ang_vel": jp.exp(-jp.square(cmd[2] - self.gyro(data)[2]) / rc.tracking_sigma),
            "lin_vel_z": jp.square(data.qvel[2]),
            "ang_vel_xy": jp.sum(jp.square(data.qvel[3:5])),
            "orientation": jp.sum(jp.square(self.upvector(data)[:2])),
            "dof_pos_limits": jp.sum(-jp.clip(data.qpos[7:] - self._soft_lo, None, 0.0) + jp.clip(data.qpos[7:] - self._soft_hi, 0.0, None)),
            "pose": jp.exp(-jp.sum(jp.square(data.qpos[7:] - self._default_pose) * jp.array([1.0, 1.0, 0.1] * 4))),
            "termination": done,
            "stand_still": jp.sum(jp.abs(data.qpos[7:] - self._default_pose)) * (~cmd_on),
            "torques": jp.sqrt(jp.sum(jp.square(data.actuator_force))) + jp.sum(jp.abs(data.actuator_force)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
            # 振り出す足を、地面（段の上面）から max_foot_height まで上げる
            # 前へ振り出している足ほど、地面（段の上面）から max_foot_height の高さにあるように
            "feet_clearance": jp.sum(jp.abs(fh - rc.max_foot_height) * jp.sqrt(speed + 1e-6)),
            "feet_slip": jp.sum(jp.square(speed) * contact) * cmd_on,
            "feet_air_time": jp.sum((info["feet_air_time"] - 0.1) * first_contact) * cmd_on,
            "body_height": jp.square(self.body_height(data) - BASE_H),
        }
