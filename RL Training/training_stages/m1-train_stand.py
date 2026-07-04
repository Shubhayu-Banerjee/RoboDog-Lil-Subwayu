import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pyb
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


class QuadrupedEnv(gym.Env):
    """Custom Environment optimized for 1.8kg body and 60kg-cm (~5.88 Nm) actuators"""

    def __init__(self, render_mode="direct"):
        super(QuadrupedEnv, self).__init__()

        self.render_mode = render_mode
        self.client = pyb.connect(pyb.GUI if render_mode == "human" else pyb.DIRECT)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

        self.robot_id = None
        self.step_count = 0

        self.prev_action = np.zeros(12, dtype=np.float32)
        self.smoothed_action = np.zeros(12, dtype=np.float32)

        # Restored a natural, stable crouched stance utilizing the 60kg-cm torque headroom
        self.neutral_stance = [
            0.0, -0.3, 0.6,  # Leg 1 (Front Left)
            0.0, -0.3, 0.6,  # Leg 2 (Front Right)
            0.0, -0.3, 0.6,  # Leg 3 (Back Left)
            0.0, -0.3, 0.6  # Leg 4 (Back Right)
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

        # Spawn height adjusted down to 0.19m to align seamlessly with the crouched neutral profile
        start_pos = [0, 0, 0.19]
        start_ori = pyb.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = pyb.loadURDF("robot.urdf", start_pos, start_ori, useFixedBase=False,
                                     physicsClientId=self.client)

        for joint in range(12):
            pyb.changeDynamics(self.robot_id, joint, lateralFriction=1.5, physicsClientId=self.client)
        pyb.changeDynamics(self.plane_id, -1, lateralFriction=1.5, physicsClientId=self.client)

        for i, angle in enumerate(self.neutral_stance):
            pyb.resetJointState(self.robot_id, i, angle, physicsClientId=self.client)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        self.smoothed_action = 0.8 * self.smoothed_action + 0.2 * action
        # Widened action scale gives the network ample authority to enforce posture corrections
        action_scale = 0.4
        target_angles = np.array(self.neutral_stance) + (self.smoothed_action * action_scale)

        for i in range(12):
            pyb.setJointMotorControl2(
                self.robot_id, i, pyb.POSITION_CONTROL,
                targetPosition=target_angles[i],
                force=5.88,  # <--- Updated to match 60 kg-cm Max Stall Torque precisely
                physicsClientId=self.client
            )

        for _ in range(4):
            pyb.stepSimulation(physicsClientId=self.client)

        obs = self._get_obs()

        # Physics State
        base_pos, base_quat = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        _, base_ang_vel = pyb.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, _ = pyb.getEulerFromQuaternion(base_quat)

        joint_states = pyb.getJointStates(self.robot_id, range(12), physicsClientId=self.client)
        joint_vels = np.array([state[1] for state in joint_states])

        # --------------------------------------------------
        # STABILITY REWARDS (THE STAND-UP STANDARDS)
        # --------------------------------------------------
        # Reward window updated to expect the natural crouch height profile
        reward_height = 4.0 if (0.16 <= base_pos[2] <= 0.22) else 0.0
        penalty_drift = (abs(base_pos[0]) + abs(base_pos[1])) * 5.0
        penalty_orientation = (abs(roll) + abs(pitch)) * 2.0
        penalty_rocking = np.sum(np.square(base_ang_vel)) * 0.4
        penalty_joint_speed = np.sum(np.square(joint_vels)) * 0.05
        penalty_flicker = np.sum(np.abs(action - self.prev_action)) * 0.1

        reward = reward_height - penalty_drift - penalty_orientation - penalty_rocking - penalty_joint_speed - penalty_flicker

        self.prev_action = np.copy(action)

        # Balance limits protecting the model from breaking posture stability limits
        terminated = bool(
            base_pos[2] < 0.13 or  # Sagging beyond the recovery envelope
            abs(roll) > 0.40 or
            abs(pitch) > 0.40 or
            abs(base_pos[0]) > 0.25 or
            abs(base_pos[1]) > 0.25
        )
        truncated = bool(self.step_count >= 1000)

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        base_pos, base_quat = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        roll, pitch, _ = pyb.getEulerFromQuaternion(base_quat)

        joint_states = pyb.getJointStates(self.robot_id, range(12), physicsClientId=self.client)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]

        return np.concatenate([[roll, pitch, base_pos[2]], joint_pos, joint_vel]).astype(np.float32)

    def close(self):
        pyb.disconnect(self.client)


def make_env(rank, seed=0):
    def _init():
        env = QuadrupedEnv(render_mode="direct")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    num_cpu = 6
    print(f"Setting up {num_cpu} Parallel CPU Environments for 60kg Standing Task...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    checkpoint_name = "ppo_robodog_standing.zip"

    # CRITICAL: Delete any old standing zip files to clear the previous 40kg weight memory out!
    if os.path.exists(checkpoint_name):
        print(f"Found existing standing checkpoint '{checkpoint_name}'. Resuming progress...")
        model = PPO.load(checkpoint_name, env=vec_env, learning_rate=3e-4)
    else:
        print("Initializing brand new weights for 60kg/1.8kg Standing Milestone...")
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, batch_size=256)

    print("Training Standing Stability Network...")
    try:
        model.learn(total_timesteps=500_000, reset_num_timesteps=False)
        model.save("ppo_robodog_standing")
        print("Milestone 1 Complete! Saved as 'ppo_robodog_standing.zip'")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress logged to 'ppo_robodog_standing.zip'.")
        model.save("ppo_robodog_standing")