"""
犬型 4 脚ロボットの目標到達環境（GoalDog）

GoalAnt（Ant ベース）と同じゴール・ランドマーク・報酬の考え方で、体の形だけ犬型にしたもの。
- 胴体の下に脚が伸びる犬型（各脚 3 関節：外転・股・膝、計 12 関節）
- 関節は位置制御（PD）。行動は「立ち姿勢からの関節角のずれ」
- 頭のある向き（胴体の x 軸）を前とし、ゴールの方を向いて歩くと少し得をする
"""
import os

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from goal_ant import LANDMARKS, REACH_RADIUS

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "assets", "goal_dog.xml")

LEGS = [("FL", 0.22, 0.09), ("FR", 0.22, -0.09), ("HL", -0.22, 0.09), ("HR", -0.22, -0.09)]
STAND = np.array([0.0, 0.8, -1.5] * 4)  # 外転・股・膝（立ち姿勢）
ACTION_SCALE = 0.4
WHITE, ORANGE, METAL, DARK = "0.93 0.94 0.96 1", "1 0.5 0.05 1", "0.3 0.33 0.37 1", "0.12 0.13 0.15 1"


def leg_xml(name, x, y):
    side = 1 if y > 0 else -1
    return f"""
      <body name="{name}_hip" pos="{x} {y} 0">
        <joint name="{name}_abd" axis="1 0 0" range="-0.6 0.6"/>
        <geom type="cylinder" size="0.045 0.025" euler="1.5708 0 0" pos="0 {side * 0.03} 0" mass="0.5" material="metal" rgba="{DARK}"/>
        <body name="{name}_thigh" pos="0 {side * 0.07} 0">
          <joint name="{name}_hip" axis="0 1 0" range="-1.2 2.4"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.032" mass="0.6" material="metal" rgba="{METAL}"/>
          <body name="{name}_calf" pos="0 0 -0.2">
            <joint name="{name}_knee" axis="0 1 0" range="-2.6 -0.4"/>
            <geom type="sphere" size="0.04" class="deco" material="metal" rgba="{DARK}"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.022" mass="0.25" material="metal" rgba="{WHITE}"/>
            <geom name="{name}_foot" type="sphere" pos="0 0 -0.2" size="0.032" mass="0.05" friction="1.2 0.02 0.01" material="metal" rgba="{DARK}"/>
          </body>
        </body>
      </body>"""


def build_xml():
    marks = []
    for name, ((x, y), (r, g, b, a)) in LANDMARKS.items():
        if name == "home":
            marks.append(f'<geom name="lm_{name}" type="cylinder" pos="{x} {y} 0.005" size="0.7 0.005" rgba="{r} {g} {b} 0.8" class="deco"/>')
        else:
            marks.append(f'<geom name="lm_{name}" type="cylinder" pos="{x} {y} 0.75" size="0.18 0.75" rgba="{r} {g} {b} {a}" class="deco"/>')
    actuators = "\n".join(f'<position name="{n}_{j}" joint="{n}_{j}"/>' for n, _, _ in LEGS for j in ("abd", "hip", "knee"))
    xml = f"""<mujoco model="goal_dog">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>
  <default>
    <joint damping="0.8" armature="0.02"/>
    <geom contype="1" conaffinity="0" condim="3" friction="1 0.5 0.5"/>
    <position kp="60" forcerange="-30 30" ctrlrange="-3 3"/>
    <default class="deco"><geom contype="0" conaffinity="0" mass="0"/></default>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0.30 0.33 0.37" rgb2="0.40 0.43 0.47" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.15" shininess="0.3" specular="0.3" texrepeat="40 40" texture="texplane"/>
    <material name="shell" specular="0.6" shininess="0.7"/>
    <material name="metal" specular="0.8" shininess="0.9"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom name="floor" type="plane" size="40 40 40" material="MatPlane" contype="1" conaffinity="1" condim="3"/>
    <body name="torso" pos="0 0 0.36">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.27 0.1 0.06" mass="6" material="shell" rgba="{WHITE}"/>
      <geom type="box" size="0.272 0.102 0.012" pos="0 0 -0.01" class="deco" material="shell" rgba="{ORANGE}"/>
      <geom type="box" size="0.2 0.08 0.012" pos="-0.02 0 0.065" class="deco" material="metal" rgba="{DARK}"/>
      <geom type="cylinder" size="0.045 0.025" pos="0.12 0 0.09" class="deco" material="metal" rgba="{DARK}"/>
      <geom type="cylinder" size="0.047 0.008" pos="0.12 0 0.1" class="deco" rgba="0.2 0.85 1 1"/>
      <geom name="head" type="box" size="0.07 0.085 0.06" pos="0.31 0 0.03" mass="0.5" material="shell" rgba="{WHITE}"/>
      <geom type="box" size="0.012 0.07 0.03" pos="0.375 0 0.035" class="deco" material="metal" rgba="0.05 0.05 0.07 1"/>
      <geom type="sphere" size="0.014" pos="0.385 0.035 0.04" class="deco" rgba="0.2 0.85 1 1"/>
      <geom type="sphere" size="0.014" pos="0.385 -0.035 0.04" class="deco" rgba="0.2 0.85 1 1"/>
      {"".join(leg_xml(*leg) for leg in LEGS)}
    </body>
    {chr(10).join(marks)}
    <body name="goal" mocap="true" pos="0 0 0">
      <geom type="cylinder" size="0.6 0.01" pos="0 0 0.01" rgba="1 0.5 0 0.35" class="deco"/>
      <geom type="cylinder" size="0.03 0.6" pos="0 0 0.6" rgba="1 0.5 0 1" class="deco"/>
      <geom type="box" size="0.2 0.01 0.12" pos="0.2 0 1.08" rgba="1 0.5 0 1" class="deco"/>
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""
    os.makedirs(os.path.dirname(XML_PATH), exist_ok=True)
    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)


def quat_rotate_inv(q, v):
    """世界座標のベクトル v を胴体座標系へ（q = w, x, y, z）。"""
    w, x, y, z = q
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return r.T @ v


class GoalDogEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"], "render_fps": 50}

    def __init__(self, goal_dist=(1.5, 5.0), reach_bonus=10.0, **kwargs):
        if not os.path.exists(XML_PATH):
            build_xml()
        self.goal_dist = goal_dist
        self.reach_bonus = reach_bonus
        self.goal = np.zeros(2)
        self.fixed_goal = False
        self.n_reached = 0
        self.prev_action = np.zeros(12)
        obs_space = gym.spaces.Box(-np.inf, np.inf, (51,), np.float64)
        kwargs.setdefault("default_camera_config", {"distance": 4.0, "elevation": -30.0})
        MujocoEnv.__init__(self, XML_PATH, 5, observation_space=obs_space, **kwargs)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)

    # --- ゴール関連（GoalAntEnv と同じインターフェース） ---
    def xy(self):
        return self.data.qpos[:2].copy()

    def set_goal(self, goal):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.data.mocap_pos[0] = [self.goal[0], self.goal[1], 0.0]

    def _sample_goal(self):
        d = self.np_random.uniform(*self.goal_dist)
        a = self.np_random.uniform(-np.pi, np.pi)
        self.set_goal(self.xy() + d * np.array([np.cos(a), np.sin(a)]))

    def _goal_features(self):
        v = self.goal - self.xy()
        d = np.linalg.norm(v) + 1e-8
        u = v / d
        ub = quat_rotate_inv(self.data.qpos[3:7], np.array([u[0], u[1], 0.0]))[:2]
        ub = ub / (np.linalg.norm(ub) + 1e-8)
        return np.array([ub[0], ub[1], u[0], u[1], min(d, 5.0) / 5.0])

    def _foot_contact(self):
        if not hasattr(self, "_foot_ids"):
            self._foot_ids = [self.model.geom(f"{n}_foot").id for n, _, _ in LEGS]
        touching = set()
        for c in self.data.contact[: self.data.ncon]:
            touching.update((c.geom1, c.geom2))
        return np.array([i in touching for i in self._foot_ids])

    def _gravity_body(self):
        return quat_rotate_inv(self.data.qpos[3:7], np.array([0.0, 0.0, -1.0]))

    def _get_obs(self):
        q = self.data.qpos[3:7]
        return np.concatenate([
            [self.data.qpos[2]],
            self._gravity_body(),
            quat_rotate_inv(q, self.data.qvel[:3]),
            self.data.qvel[3:6],  # freejoint の角速度はもともと胴体座標系
            self.data.qpos[7:] - STAND,
            self.data.qvel[6:],
            self.prev_action,
            self._goal_features(),
        ])

    def reset_model(self):
        qpos = self.init_qpos.copy()
        yaw = self.np_random.uniform(-np.pi, np.pi)
        qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        qpos[7:] = STAND + self.np_random.uniform(-0.1, 0.1, 12)
        qpos[2] = 0.34
        qvel = self.np_random.normal(0, 0.05, self.model.nv)
        self.set_state(qpos, qvel)
        self.data.ctrl[:] = qpos[7:]
        self.prev_action = np.zeros(12)
        self.air_time = np.zeros(4)
        self.n_reached = 0
        if not self.fixed_goal:
            self._sample_goal()
        else:
            self.set_goal(self.goal)
        return self._get_obs()

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        d0 = np.linalg.norm(self.goal - self.xy())
        self.do_simulation(STAND + ACTION_SCALE * a, self.frame_skip)
        d1 = np.linalg.norm(self.goal - self.xy())

        g = self._gravity_body()
        feat = self._goal_features()
        progress = (d0 - d1) / self.dt
        # 足を上げて踏み出す歩き方を促す：着地した瞬間に、それまで浮いていた時間に応じて加点
        contact = self._foot_contact()
        self.air_time += self.dt
        touchdown = contact & (self.air_time > self.dt * 1.5)
        air_reward = float(np.sum((self.air_time - 0.2) * touchdown)) if d1 > REACH_RADIUS else 0.0
        self.air_time[contact] = 0.0
        reward = (
            1.0 * progress
            + 0.3 * feat[0]  # ゴールの方を向いている
            + 0.1  # 生存
            + 1.0 * air_reward
            - 1.0 * (g[0] ** 2 + g[1] ** 2)  # 胴体の傾き
            - 10.0 * max(0.28 - self.data.qpos[2], 0.0)  # 腹ばいで進まない
            - 0.02 * float(np.sum(np.square(a - self.prev_action)))  # 動きのなめらかさ
            - 2e-5 * float(np.sum(np.square(self.data.actuator_force)))
        )
        self.prev_action = a.copy()

        reached = d1 < REACH_RADIUS
        if reached:
            reward += self.reach_bonus
            self.n_reached += 1
            if not self.fixed_goal:
                self._sample_goal()

        terminated = bool(self.data.qpos[2] < 0.2 or g[2] > -0.5)
        info = {"dist": d1, "reached": reached, "n_reached": self.n_reached, "xy": self.xy()}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info


gym.register(id="GoalDog-v0", entry_point=GoalDogEnv, max_episode_steps=1500)
