import os
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pyb
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback


class StepMilestoneCallback(BaseCallback):
    """Custom callback to save checkpoints formatted precisely as name + step count in Millions"""

    def __init__(self, save_freq_total, save_path="trained_RL", verbose=0):
        super(StepMilestoneCallback, self).__init__(verbose)
        self.save_freq = save_freq_total  # Global steps interval (e.g., 1,000,000)
        self.save_path = save_path
        self.last_save_boundary = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if combined global timesteps crossed the next milestone boundary
        if self.num_timesteps - self.last_save_boundary >= self.save_freq:
            self.last_save_boundary = (self.num_timesteps // self.save_freq) * self.save_freq
            total_m_steps = self.num_timesteps / 1_000_000.0

            # FIXED: Added explicit .zip extension inside the string format definition
            checkpoint_name = f"ppo_robodog_walking-{total_m_steps:.2f}M.zip"
            full_save_path = os.path.join(self.save_path, checkpoint_name)

            self.model.save(full_save_path)
            if self.verbose > 0:
                print(f"\n[Checkpoint Saved]: Milestone hit. Archived state to '{full_save_path}'")
        return True

class QuadrupedPrecisionWalkingEnv(gym.Env):
    """Custom Environment for Stage 3: High-Precision 7cm Corridor Alignment"""

    def __init__(self, render_mode="direct"):
        super(QuadrupedPrecisionWalkingEnv, self).__init__()

        self.render_mode = render_mode
        self.client = pyb.connect(pyb.GUI if render_mode == "human" else pyb.DIRECT)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)

        self.robot_id = None
        self.step_count = 0

        self.prev_action = np.zeros(12, dtype=np.float32)
        self.smoothed_action = np.zeros(12, dtype=np.float32)

        self.neutral_stance = [
            0.0, -0.3, 0.6,
            0.0, -0.3, 0.6,
            0.0, -0.3, 0.6,
            0.0, -0.3, 0.6
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.prev_action = np.zeros(12, dtype=np.float32)
        self.smoothed_action = np.zeros(12, dtype=np.float32)

        pyb.resetSimulation(physicsClientId=self.client)
        pyb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pyb.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        self.plane_id = pyb.loadURDF("plane.urdf", physicsClientId=self.client)

        start_pos = [0, 0, 0.19]
        start_ori = pyb.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = pyb.loadURDF("robot.urdf", start_pos, start_ori, useFixedBase=False,
                                     physicsClientId=self.client)

        for joint in range(12):
            pyb.changeDynamics(self.robot_id, joint, lateralFriction=1.6, physicsClientId=self.client)
        pyb.changeDynamics(self.plane_id, -1, lateralFriction=1.6, physicsClientId=self.client)

        for i, angle in enumerate(self.neutral_stance):
            pyb.resetJointState(self.robot_id, i, angle, physicsClientId=self.client)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        self.smoothed_action = 0.8 * self.smoothed_action + 0.2 * action
        action_scale = 0.4
        target_angles = np.array(self.neutral_stance) + (self.smoothed_action * action_scale)

        for i in range(12):
            pyb.setJointMotorControl2(
                self.robot_id, i, pyb.POSITION_CONTROL,
                targetPosition=target_angles[i],
                force=5.88,
                physicsClientId=self.client
            )

        for _ in range(4):
            pyb.stepSimulation(physicsClientId=self.client)

        obs = self._get_obs()

        base_pos, base_quat = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_vel, base_ang_vel = pyb.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, yaw = pyb.getEulerFromQuaternion(base_quat)

        joint_states = pyb.getJointStates(self.robot_id, range(12), physicsClientId=self.client)
        joint_vels = np.array([state[1] for state in joint_states])

        global_forward_vel = -base_vel[0]
        global_lateral_vel = base_vel[1]

        foot_indices = [2, 5, 8, 11]
        contacts = []
        for leg_idx in foot_indices:
            cp = pyb.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=leg_idx,
                                      physicsClientId=self.client)
            contacts.append(1.0 if len(cp) > 0 else 0.0)

        # --------------------------------------------------
        # BASE LOCOMOTION REWARDS
        # --------------------------------------------------
        if global_forward_vel > 0.02:
            reward_vel = global_forward_vel * 40.0
        else:
            reward_vel = -5.0

        swing_activity = np.sum(np.abs(joint_vels))
        reward_gait_motion = 0.0
        if 0.5 < sum(contacts) < 3.5:
            reward_gait_motion = swing_activity * 0.10

        # --- REWARD ARCHITECTURE OVERHAUL ---
        penalty_yaw_deviation = (yaw ** 2) * 260.0
        penalty_lateral_drift = (base_pos[1] ** 2) * 300.0
        penalty_lateral_speed = (global_lateral_vel ** 2) * 40.0
        penalty_spinning = (base_ang_vel[2] ** 2) * 30.0

        # INTEGRATION 1: Target-Heading "Magnet" Reward Vector
        reward_magnet = 0.0
        if abs(base_pos[1]) > 0.01:
            if np.sign(base_pos[1]) != np.sign(global_lateral_vel):
                reward_magnet = abs(global_lateral_vel) * 25.0

        # INTEGRATION 2: Facing-Away Steering Penalty
        penalty_facing_away = 0.0
        if abs(base_pos[1]) > 0.01 and abs(yaw) > 0.02:
            if np.sign(base_pos[1]) == np.sign(yaw):
                penalty_facing_away = abs(yaw) * 200.0

        reward_height = 4.0 if (0.16 <= base_pos[2] <= 0.22) else -1.0
        penalty_orientation = (abs(roll) + abs(pitch)) * 4.0
        penalty_rocking = (np.square(base_ang_vel[0]) + np.square(base_ang_vel[1])) * 0.4
        penalty_flicker = np.sum(np.abs(action - self.prev_action)) * 0.05

        reward = (reward_vel + reward_gait_motion + reward_magnet + reward_height -
                  penalty_yaw_deviation - penalty_lateral_drift - penalty_lateral_speed -
                  penalty_spinning - penalty_facing_away - penalty_orientation -
                  penalty_rocking - penalty_flicker)

        self.prev_action = np.copy(action)

        # --- ULTRA STRICT TERMINATION ---
        terminated = bool(
            base_pos[2] < 0.12 or
            abs(roll) > 0.45 or
            abs(pitch) > 0.45 or
            abs(yaw) > 0.10 or
            abs(base_pos[1]) > 0.07 or  # ENFORCED CRITICAL CHANGE: 7cm tracking bounds boundary
            global_forward_vel < -0.05
        )

        truncated = bool(self.step_count >= 6000)

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        base_pos, base_quat = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        _, base_ang_vel = pyb.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, yaw = pyb.getEulerFromQuaternion(base_quat)

        joint_states = pyb.getJointStates(self.robot_id, range(12), physicsClientId=self.client)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]

        foot_indices = [2, 5, 8, 11]
        contacts = []
        for leg_idx in foot_indices:
            cp = pyb.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=leg_idx,
                                      physicsClientId=self.client)
            contacts.append(1.0 if len(cp) > 0 else 0.0)

        return np.concatenate([
            [roll, pitch, yaw],
            base_ang_vel,
            joint_pos,
            joint_vel,
            contacts
        ]).astype(np.float32)

    def close(self):
        pyb.disconnect(self.client)


def make_env(rank, seed=0):
    def _init():
        env = QuadrupedPrecisionWalkingEnv(render_mode="direct")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    num_cpu = 12
    print(f"Deploying {num_cpu} Instances for Magnet-Guided Stage 3 Loops...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    standing_checkpoint = "ppo_robodog_standing.zip"
    walking_checkpoint = "ppo_robodog_walking.zip"

    if os.path.exists(walking_checkpoint):
        print(f"Found active walking progression model '{walking_checkpoint}'. Resuming fine-tuning...")
        model = PPO.load(walking_checkpoint, env=vec_env, learning_rate=2e-4)
    elif os.path.exists(standing_checkpoint):
        print(f"\n[FOUND BASELINE]: Splicing balance weights from '{standing_checkpoint}'...")
        print("Seeding clean 34-observation walking layer matrices from standing metrics...")
        model = PPO.load(standing_checkpoint, env=vec_env, learning_rate=2e-4)
    else:
        print(f"\nCRITICAL ERROR: Could not find '{standing_checkpoint}' or '{walking_checkpoint}'!")
        exit()

    # Pass the absolute total target interval directly (1,000,000 steps globally)
    checkpoint_callback = StepMilestoneCallback(save_freq_total=1_000_000, save_path="trained_RL", verbose=1)

    print("\nLaunching Scratch-Walking Gait Optimization Engine (20M step limit)...")
    try:
        model.learn(total_timesteps=20_000_000, callback=checkpoint_callback, reset_num_timesteps=False)
        model.save("ppo_robodog_walking")
        print("Training Session Finished Successfully! Final 'ppo_robodog_walking.zip' compiled.")
    except KeyboardInterrupt:
        print("\nTraining manually intercepted. Archiving current weight state map...")
        model.save("ppo_robodog_walking")
        print("Progress safely logged.")
