"""
目標地点へ歩いていく 4 脚ロボット環境（GoalAnt）

Gymnasium の Ant-v5 をベースに、
- 観測に「ゴールの方向と距離（胴体座標系）」を追加
- 報酬を「前に進む速さ」から「ゴールに近づく速さ」に置き換え
- ゴールに着いたら次のゴールを出し直す（1 エピソードで何度も到達を練習）
- 床にランドマーク（色付きの柱）とゴール旗を描画（見た目だけで物理干渉なし）
"""
import os
import re

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "assets", "goal_ant.xml")

# 名前 -> (x, y, 表示色RGBA)。LLM プランナーもこの表を参照する
LANDMARKS = {
    "home":   ((0.0, 0.0),  (0.95, 0.95, 0.95, 1)),
    "red":    ((6.0, 3.0),  (0.90, 0.15, 0.15, 1)),
    "blue":   ((-4.0, 6.0), (0.15, 0.35, 0.95, 1)),
    "green":  ((-6.0, -3.0), (0.15, 0.75, 0.25, 1)),
    "yellow": ((4.0, -6.0), (0.95, 0.80, 0.10, 1)),
}
LANDMARK_JA = {"home": "ホーム", "red": "赤い柱", "blue": "青い柱", "green": "緑の柱", "yellow": "黄色い柱"}

REACH_RADIUS = 0.6


def build_xml():
    """Ant の XML にランドマークとゴール旗（mocap）を追加して書き出す。"""
    src = os.path.join(os.path.dirname(gym.__file__), "envs", "mujoco", "assets", "ant.xml")
    with open(src, encoding="utf-8") as f:
        xml = f.read()
    extra = []
    for name, ((x, y), rgba) in LANDMARKS.items():
        r, g, b, a = rgba
        if name == "home":
            extra.append(
                f'<geom name="lm_{name}" type="cylinder" pos="{x} {y} 0.005" size="0.7 0.005" '
                f'rgba="{r} {g} {b} 0.8" contype="0" conaffinity="0"/>'
            )
        else:
            extra.append(
                f'<geom name="lm_{name}" type="cylinder" pos="{x} {y} 0.75" size="0.18 0.75" '
                f'rgba="{r} {g} {b} {a}" contype="0" conaffinity="0"/>'
            )
    # ゴール旗：torso より後ろに置くので body index 1 = torso は変わらない
    extra.append(
        '<body name="goal" mocap="true" pos="0 0 0">'
        '<geom type="cylinder" size="0.6 0.01" pos="0 0 0.01" rgba="1 0.5 0 0.35" contype="0" conaffinity="0"/>'
        '<geom type="cylinder" size="0.03 0.6" pos="0 0 0.6" rgba="1 0.5 0 1" contype="0" conaffinity="0"/>'
        '<geom type="box" size="0.2 0.01 0.12" pos="0.2 0 1.08" rgba="1 0.5 0 1" contype="0" conaffinity="0"/>'
        "</body>"
    )
    xml = xml.replace("</worldbody>", "\n".join(extra) + "\n  </worldbody>")
    # 床を落ち着いた色に（見た目だけ）
    xml = xml.replace('rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d"', 'rgb1="0.30 0.33 0.37" rgb2="0.40 0.43 0.47" type="2d"')
    xml = xml.replace('reflectance="0.5" shininess="1" specular="1" texrepeat="60 60"',
                      'reflectance="0.15" shininess="0.3" specular="0.3" texrepeat="40 40"')
    xml = xml.replace('rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"', 'rgba="1 1 1 1" size="40 40 40" type="plane"')
    xml = robot_look(xml)
    # 1080p でオフスクリーン描画できるようにする（動画用）
    xml = xml.replace(
        "<custom>",
        '<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>\n  <custom>',
    )
    os.makedirs(os.path.dirname(XML_PATH), exist_ok=True)
    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)


def robot_look(xml):
    """見た目だけ機械らしくする。足す部品は質量 0・接触なしなので、物理は元の Ant と同じ。"""
    deco = 'contype="0" conaffinity="0" mass="0"'
    xml = xml.replace(
        '<material name="geom" texture="texgeom" texuniform="true"/>',
        '<material name="geom" texture="texgeom" texuniform="true"/>\n'
        '    <material name="shell" specular="0.6" shininess="0.7"/>\n'
        '    <material name="metal" specular="0.8" shininess="0.9"/>',
    )
    # 胴体：白い外殻＋オレンジの帯＋上面のセンサー
    xml = xml.replace(
        '<geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>',
        '<geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere" material="shell" rgba="0.93 0.94 0.96 1"/>'
        f'<geom type="cylinder" size="0.257 0.035" material="shell" rgba="1 0.5 0.05 1" {deco}/>'
        f'<geom type="cylinder" pos="0 0 0.25" size="0.1 0.05" material="metal" rgba="0.18 0.2 0.23 1" {deco}/>'
        f'<geom type="cylinder" pos="0 0 0.26" size="0.104 0.012" rgba="0.2 0.85 1 1" {deco}/>',
    )
    # 脚：付け根は白、脚はガンメタル、関節と足先に黒い球
    for i in range(1, 5):
        xml = xml.replace(f'name="aux_{i}_geom" size="0.08"', f'name="aux_{i}_geom" size="0.08" material="shell" rgba="0.93 0.94 0.96 1"')
    for leg in ["left_leg", "right_leg", "back_leg", "rightback_leg"]:
        xml = xml.replace(
            f'name="{leg}_geom" size="0.08" type="capsule"/>',
            f'name="{leg}_geom" size="0.08" type="capsule" material="metal" rgba="0.3 0.33 0.37 1"/>'
            f'<geom type="sphere" size="0.105" material="metal" rgba="0.12 0.13 0.15 1" {deco}/>',
        )
    for ankle in ["left_ankle", "right_ankle", "third_ankle", "fourth_ankle"]:
        xml = re.sub(
            rf'<geom fromto="0\.0 0\.0 0\.0 (\S+) (\S+) 0\.0" name="{ankle}_geom" size="0\.08" type="capsule"/>',
            lambda m: m.group(0)[:-2] + ' material="metal" rgba="0.3 0.33 0.37 1"/>'
            + f'<geom type="sphere" pos="{m.group(1)} {m.group(2)} 0" size="0.1" material="metal" rgba="0.12 0.13 0.15 1" {deco}/>'
            + f'<geom type="sphere" size="0.1" material="metal" rgba="0.12 0.13 0.15 1" {deco}/>',
            xml,
        )
    return xml


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class GoalAntEnv(AntEnv):
    def __init__(
        self,
        goal_dist=(1.5, 5.0),
        progress_weight=1.0,
        ctrl_weight=0.1,
        healthy_weight=0.25,
        reach_bonus=5.0,
        **kwargs,
    ):
        if not os.path.exists(XML_PATH):
            build_xml()
        self.goal_dist = goal_dist
        self.progress_weight = progress_weight
        self.ctrl_weight = ctrl_weight
        self.healthy_weight = healthy_weight
        self.reach_bonus = reach_bonus
        self.goal = np.zeros(2)
        self.fixed_goal = False  # デモ時は外部からゴールを与える
        self.n_reached = 0
        kwargs.setdefault("default_camera_config", {"distance": 7.0, "elevation": -35.0})
        super().__init__(
            xml_file=XML_PATH,
            include_cfrc_ext_in_observation=False,
            **kwargs,
        )
        obs_dim = self.data.qpos.size - 2 + self.data.qvel.size + 5
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)

    # --- ゴール関連 ---
    def xy(self):
        return self.data.body(self._main_body).xpos[:2].copy()

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
        yaw = yaw_of(self.data.qpos[3:7])
        c, s = np.cos(-yaw), np.sin(-yaw)
        ub = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])
        return np.array([ub[0], ub[1], u[0], u[1], min(d, 5.0) / 5.0])

    def _get_obs(self):
        base = np.concatenate([self.data.qpos.flatten()[2:], self.data.qvel.flatten()])
        return np.concatenate([base, self._goal_features()])

    # --- gym API ---
    def reset_model(self):
        obs = super().reset_model()
        self.n_reached = 0
        if not self.fixed_goal:
            self._sample_goal()
        return self._get_obs()

    def step(self, action):
        d0 = np.linalg.norm(self.goal - self.xy())
        self.do_simulation(action, self.frame_skip)
        d1 = np.linalg.norm(self.goal - self.xy())

        progress = (d0 - d1) / self.dt
        ctrl_cost = self.ctrl_weight * float(np.sum(np.square(action)))
        healthy = self.is_healthy
        reward = self.progress_weight * progress + self.healthy_weight * healthy - ctrl_cost

        reached = d1 < REACH_RADIUS
        if reached:
            reward += self.reach_bonus
            self.n_reached += 1
            if not self.fixed_goal:
                self._sample_goal()

        terminated = (not healthy) and self._terminate_when_unhealthy
        info = {"dist": d1, "reached": reached, "n_reached": self.n_reached, "xy": self.xy()}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info


gym.register(id="GoalAnt-v0", entry_point=GoalAntEnv, max_episode_steps=1000)
