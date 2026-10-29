"""
坂・階段・凸凹のある地形を歩く犬型ロボットの目標到達環境（TerrainDog）

GoalDog と同じロボット・報酬の考え方に、次を加えたもの。
- 地形：高さマップ（heightfield）で、平地・凸凹・丘（坂）・上り階段・下り階段を毎回ランダムに作る
- カリキュラム：地形の種類ごとに難しさを持ち、その地形でうまく歩けたら難しく、転んだら易しくする
  （段差の高さ・坂の勾配・凸凹の大きさ）
- 観測：胴体のまわり 35 点の地面の高さ（下向きのレイで測る）と、地面からの胴体の高さ
  レイは地形などのグループ 0・3 の形状だけに当たり、ロボット自身や目印（見た目だけの部品）には当たらない
"""
import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from goal_ant import REACH_RADIUS
from goal_dog import (ACTION_SCALE, BASE_HEIGHT, DIAGONAL, GAIT_PERIOD, LEGS, STAND, WALK_SPEED, quat_rotate_inv,
                      robot_xml)

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "assets", "terrain_dog.xml")

HF_N, HF_HALF, HF_DEPTH = 241, 6.0, 0.5  # 高さマップ：241×241 点（5 cm 間隔）・12 m 四方・基準面の下 0.5 m まで
SCAN_X = np.arange(-0.3, 0.61, 0.15)  # 胴体の前後（前 0.6 m まで）
SCAN_Y = np.arange(-0.3, 0.31, 0.15)  # 左右
SCAN = np.array([(x, y) for x in SCAN_X for y in SCAN_Y])  # 35 点
TERRAINS = ["flat", "rough", "hills", "stairs_up", "stairs_down"]
TERRAIN_P = [0.15, 0.15, 0.2, 0.3, 0.2]  # v1：地形の出やすさ（階段を多めに練習する）
CLEARANCE = 0.08  # 足を振り出すときに地面から上げる高さ [m]


def scene_xml(floor, extra="", model="terrain_dog", hf_n=HF_N):
    """ロボット＋床（floor）＋追加の形状（extra）の MJCF。床や家具はグループ 0、ロボットと見た目の部品はグループ 2。"""
    actuators = "\n".join(f'<position name="{n}_{j}" joint="{n}_{j}"/>' for n, _, _ in LEGS for j in ("abd", "hip", "knee"))
    return f"""<mujoco model="{model}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/><map haze="0.25" fogstart="12" fogend="45"/><rgba haze="0.88 0.93 0.98 1" fog="0.88 0.93 0.98 1"/></visual>
  <default>
    <joint damping="0.8" armature="0.02"/>
    <geom contype="1" conaffinity="0" condim="3" friction="1 0.5 0.5"/>
    <position kp="60" forcerange="-30 30" ctrlrange="-3 3"/>
    <default class="deco"><geom contype="0" conaffinity="0" mass="0" group="2"/></default>
    <default class="robot"><geom group="2"/></default>
    <default class="world"><geom contype="1" conaffinity="1" group="0"/></default>
  </default>
  <asset>
    <texture builtin="gradient" height="512" rgb1="0.45 0.66 0.90" rgb2="0.92 0.95 0.99" type="skybox" width="512"/>
    <texture builtin="checker" height="512" name="texplane" rgb1="0.80 0.79 0.75" rgb2="0.75 0.74 0.70" mark="edge" markrgb="0.62 0.62 0.60" type="2d" width="512"/>
    <material name="MatPlane" reflectance="0" shininess="0.2" specular="0.2" texrepeat="40 40" texture="texplane"/>
    <material name="MatTerrain" reflectance="0" shininess="0.2" specular="0.2" texrepeat="12 12" texture="texplane"/>
    <material name="shell" specular="0.5" shininess="0.6"/>
    <material name="metal" specular="0.8" shininess="0.9"/>
    <hfield name="terrain" nrow="{hf_n}" ncol="{hf_n}" size="{HF_HALF} {HF_HALF} 1 {HF_DEPTH}"/>
  </asset>
  <worldbody>
    <light diffuse="0.75 0.73 0.70" dir="-0.4 -0.3 -1" directional="true" pos="0 0 5" specular=".15 .15 .15" castshadow="true"/>
    {floor}
    {extra}
    {robot_xml("robot")}
    <body name="goal" mocap="true" pos="0 0 0">
      <geom type="cylinder" size="0.6 0.01" pos="0 0 0.01" rgba="0.75 0 0 0.3" class="deco"/>
      <geom type="cylinder" size="0.03 0.6" pos="0 0 0.6" rgba="0.75 0 0 1" class="deco"/>
      <geom type="box" size="0.2 0.01 0.12" pos="0.2 0 1.08" rgba="0.75 0 0 1" class="deco"/>
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def xml_path_for(hf_n):
    return XML_PATH if hf_n == HF_N else XML_PATH.replace(".xml", f"_{hf_n}.xml")


def build_xml(hf_n=HF_N):
    floor = (f'<geom name="floor" type="hfield" hfield="terrain" pos="0 0 -{HF_DEPTH}" material="MatTerrain" class="world"/>'
             f'<geom type="plane" size="40 40 1" pos="0 0 -{HF_DEPTH + 0.01}" material="MatPlane" class="world"/>')
    os.makedirs(os.path.dirname(XML_PATH), exist_ok=True)
    with open(xml_path_for(hf_n), "w", encoding="utf-8") as f:
        f.write(scene_xml(floor, hf_n=hf_n))


def make_terrain(kind, level, rng, n=HF_N):
    """高さ [m] の 2 次元配列（n×n、中心が出発地点）。level は 0〜1 の難しさ。"""
    c = (np.arange(n) - n // 2) * (2 * HF_HALF / (n - 1))
    x, y = np.meshgrid(c, c)  # 行が y、列が x
    if kind == "flat":
        h = np.zeros_like(x)
    elif kind == "rough":
        h = rng.uniform(-1, 1, x.shape) * (0.005 + 0.03 * level)
    elif kind == "hills":  # なだらかな丘：最大勾配 ≒ 振幅 × 波数
        h = np.zeros_like(x)
        for _ in range(4):
            k = rng.uniform(0.6, 1.4)
            a = rng.uniform(-np.pi, np.pi)
            h += np.sin(k * (np.cos(a) * x + np.sin(a) * y) + rng.uniform(0, 2 * np.pi))
        h *= (0.02 + 0.13 * level) / 2  # 急な所（上位 1%）の勾配は 約 8°（level 0.25）〜14°（level 1）
        h += rng.uniform(-1, 1, x.shape) * 0.005
    else:  # 出発地点を中心とした四角い階段（上り＝外に向かって上がる、下り＝外に向かって下がる）
        step = 0.02 + 0.08 * level  # 段差 2〜10 cm
        width = rng.uniform(0.3, 0.45)
        r = np.maximum(np.abs(x), np.abs(y))
        n = np.clip(np.floor((r - 0.5) / width) + 1, 0, None)
        h = n * step * (1 if kind == "stairs_up" else -1)
        h = np.clip(h, -HF_DEPTH + 0.05, 1 - HF_DEPTH - 0.05)
    return h


class TerrainDogEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"], "render_fps": 50}

    def __init__(self, goal_dist=(1.5, 4.0), reach_bonus=10.0, xml_path=None, curriculum=True,
                 hf_n=HF_N, lift_weight=0.0, terrain_p=None, **kwargs):
        default_xml = xml_path_for(hf_n)
        if xml_path is None:
            xml_path = default_xml
            if not os.path.exists(xml_path):
                build_xml(hf_n)
        self.hf_n, self.lift_weight, self.terrain_p = hf_n, lift_weight, terrain_p
        self.goal_dist = goal_dist
        self.reach_bonus = reach_bonus
        self.curriculum = curriculum
        self.goal = np.zeros(2)
        self.fixed_goal = False
        self.n_reached = 0
        self.prev_action = np.zeros(12)
        self.phase = 0.0
        self.level = 0.0
        self.levels = {k: 0.0 for k in TERRAINS}  # 地形の種類ごとの難しさ
        self.terrain = "flat"
        self.fixed_terrain = None  # 評価用：(種類, 難しさ) を固定する
        self.steps = 0
        self.fell = False
        obs_space = gym.spaces.Box(-np.inf, np.inf, (53 + len(SCAN),), np.float64)
        kwargs.setdefault("default_camera_config", {"distance": 4.0, "elevation": -30.0})
        MujocoEnv.__init__(self, xml_path, 5, observation_space=obs_space, **kwargs)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self._hf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        self._ray_group = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)  # グループ 0 と 3（家の高さマップ）
        self._geomid = np.zeros(1, dtype=np.int32)
        self._foot_ids = [self.model.geom(f"{n}_foot").id for n, _, _ in LEGS]
        self.floor_mode = self._hf >= 0 and xml_path == default_xml  # 学習用：地形を毎回作り直す

    # --- 地面の高さ ---
    def ground_z(self, px, py):
        """(px, py) の真上から下向きにレイを飛ばし、最初に当たったグループ 0 の形状の高さを返す。"""
        top = 3.0
        d = mujoco.mj_ray(self.model, self.data, np.array([px, py, top]), np.array([0.0, 0.0, -1.0]),
                          self._ray_group, 1, -1, self._geomid)
        return top - d if d >= 0 else -HF_DEPTH

    def height_scan(self):
        yaw = np.arctan2(2 * (self.data.qpos[3] * self.data.qpos[6] + self.data.qpos[4] * self.data.qpos[5]),
                         1 - 2 * (self.data.qpos[5] ** 2 + self.data.qpos[6] ** 2))
        c, s = np.cos(yaw), np.sin(yaw)
        bx, by, bz = self.data.qpos[:3]
        pts = np.stack([bx + c * SCAN[:, 0] - s * SCAN[:, 1], by + s * SCAN[:, 0] + c * SCAN[:, 1]], 1)
        z = np.array([self.ground_z(px, py) for px, py in pts])
        return np.clip(bz - z - BASE_HEIGHT, -0.5, 0.5)  # 平地で立っていれば 0

    def body_height(self):
        return self.data.qpos[2] - self.ground_z(*self.data.qpos[:2])

    # --- ゴール関連（GoalDogEnv と同じインターフェース） ---
    def xy(self):
        return self.data.qpos[:2].copy()

    def set_goal(self, goal):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.data.mocap_pos[0] = [self.goal[0], self.goal[1], self.ground_z(*self.goal) if self.goal.any() else 0.0]

    def _sample_goal(self):
        for _ in range(20):  # 地形の外に出ないゴール
            d = self.np_random.uniform(*self.goal_dist)
            a = self.np_random.uniform(-np.pi, np.pi)
            g = self.xy() + d * np.array([np.cos(a), np.sin(a)])
            if np.all(np.abs(g) < HF_HALF - 1.0):
                break
        self.set_goal(g)

    def _goal_features(self):
        v = self.goal - self.xy()
        d = np.linalg.norm(v) + 1e-8
        u = v / d
        ub = quat_rotate_inv(self.data.qpos[3:7], np.array([u[0], u[1], 0.0]))[:2]
        ub = ub / (np.linalg.norm(ub) + 1e-8)
        return np.array([ub[0], ub[1], u[0], u[1], min(d, 5.0) / 5.0])

    def _foot_contact(self):
        touching = set()
        for c in self.data.contact[: self.data.ncon]:
            touching.update((c.geom1, c.geom2))
        return np.array([i in touching for i in self._foot_ids])

    def _foot_clearance(self):
        """各足の、真下の地面からの高さ。"""
        r = self.model.geom_size[self._foot_ids[0]][0]
        return np.array([p[2] - r - self.ground_z(p[0], p[1]) for p in self.data.geom_xpos[self._foot_ids]])

    def _gravity_body(self):
        return quat_rotate_inv(self.data.qpos[3:7], np.array([0.0, 0.0, -1.0]))

    def _desired_stance(self):
        first = np.sin(2 * np.pi * self.phase) >= 0
        return np.where(DIAGONAL == 0, first, not first)

    def _get_obs(self):
        q = self.data.qpos[3:7]
        return np.concatenate([
            [self.body_height()],
            self._gravity_body(),
            quat_rotate_inv(q, self.data.qvel[:3]),
            self.data.qvel[3:6],
            self.data.qpos[7:] - STAND,
            self.data.qvel[6:],
            self.prev_action,
            self._goal_features(),
            [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
            self.height_scan(),
        ])

    # --- gym API ---
    def _new_terrain(self):
        if self.fixed_terrain is not None:
            self.terrain, self.level = self.fixed_terrain
        else:
            if self.curriculum and self.steps > 0:  # 前のエピソードの地形での出来で、その地形の難しさを上下させる
                lv = self.levels[self.terrain]
                if not self.fell and self.n_reached >= 4:
                    lv += 0.05
                elif self.fell or self.n_reached < 2:
                    lv -= 0.05
                self.levels[self.terrain] = float(np.clip(lv, 0.0, 1.0))
            self.terrain = TERRAINS[self.np_random.choice(len(TERRAINS), p=self.terrain_p)]
            self.level = self.levels[self.terrain]
        h = make_terrain(self.terrain, self.level, self.np_random, self.hf_n)
        self.model.hfield_data[:] = ((h + HF_DEPTH) / 1.0).ravel()

    def reset_model(self):
        if self.floor_mode:
            self._new_terrain()
        self.steps = 0
        self.fell = False
        qpos = self.init_qpos.copy()
        yaw = self.np_random.uniform(-np.pi, np.pi)
        qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        qpos[7:] = STAND + self.np_random.uniform(-0.1, 0.1, 12)
        qpos[:2] = self.start_xy() if hasattr(self, "start_xy") else 0.0
        self.set_state(qpos, np.zeros(self.model.nv))
        qpos[2] = self.ground_z(*qpos[:2]) + 0.35
        qvel = self.np_random.normal(0, 0.05, self.model.nv)
        self.set_state(qpos, qvel)
        self.data.ctrl[:] = qpos[7:]
        self.prev_action = np.zeros(12)
        self.phase = self.np_random.uniform()
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
        self.phase = (self.phase + self.dt / GAIT_PERIOD) % 1.0
        self.steps += 1

        g = self._gravity_body()
        feat = self._goal_features()
        v_body = quat_rotate_inv(self.data.qpos[3:7], self.data.qvel[:3])
        v_goal = (d0 - d1) / self.dt
        q = self.data.qpos[7:]
        height = self.body_height()
        stance = self._desired_stance()
        gait = float(np.mean(self._foot_contact() == stance))
        lift = float(np.mean(np.clip(self._foot_clearance()[~stance] / CLEARANCE, 0.0, 1.0)))
        reward = (
            1.5 * np.exp(-((v_goal - WALK_SPEED) ** 2) / 0.25)  # 目標速度でゴールへ向かう
            + 0.5 * feat[0]  # 胴体の前をゴールの方へ向ける
            + 0.5 * gait  # トロットのリズムに合わせて足をつく
            + self.lift_weight * lift  # v1：振り出す足を高く上げる（段差につまずかない）
            + 0.2  # 生存
            - 1.0 * v_body[1] ** 2  # 横歩き（カニ歩き）をしない
            - 1.0 * (g[0] ** 2 + g[1] ** 2)  # 胴体の傾き（坂では多少傾くので平地版より弱く）
            - 20.0 * (height - BASE_HEIGHT) ** 2  # 地面からの胴体の高さ
            - 0.5 * float(np.sum(np.square(q[0::3])))  # 脚を横に開かない
            - 0.03 * float(np.sum(np.square(a - self.prev_action)))  # 動きのなめらかさ
            - 2e-4 * float(np.sum(np.square(self.data.qvel[6:])))  # 関節を急に動かさない
            - 2e-5 * float(np.sum(np.square(self.data.actuator_force)))
        )
        self.prev_action = a.copy()

        reached = d1 < REACH_RADIUS
        if reached:
            reward += self.reach_bonus
            self.n_reached += 1
            if not self.fixed_goal:
                self._sample_goal()

        terminated = bool(height < 0.18 or g[2] > -0.5)
        self.fell = self.fell or terminated
        info = {"dist": d1, "reached": reached, "n_reached": self.n_reached, "xy": self.xy(),
                "terrain": self.terrain, "level": self.level, "levels": dict(self.levels)}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info


gym.register(id="TerrainDog-v0", entry_point=TerrainDogEnv, max_episode_steps=1500)
# v1：高さマップを 2.5 cm 間隔にして段の角を鋭くし、足上げの報酬と階段を多めに出す設定を加えたもの
gym.register(id="TerrainDog-v1", entry_point=TerrainDogEnv, max_episode_steps=1500,
             kwargs=dict(hf_n=481, lift_weight=0.3, terrain_p=TERRAIN_P))
