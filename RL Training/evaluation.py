import time
import gymnasium as gym
import numpy as np
import pybullet as pyb
from stable_baselines3 import PPO
from train_dog import QuadrupedEnv


def main():
    print("Loading interactive PyBullet GUI environment...")
    env = QuadrupedEnv(render_mode="human")

    print("Loading trained PPO brain...")
    try:
        model = PPO.load("ppo_robodog_standing", env=env)
        print("Successfully loaded 'ppo_robodog_standing.zip'")
    except Exception:
        model = PPO.load("ppo_robodog", env=env)
        print("Successfully loaded 'ppo_robodog_standing.zip'")

    print("Running policy execution loop. Close the window to exit.")
    obs, _ = env.reset()

    # Configure the tracking camera in human visualization mode
    pyb.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=-45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.15],
        physicsClientId=env.client
    )

    while True:
        # Pass observations into model to extract deterministic actions
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # Dynamically move camera target to follow the robot dog as it travels
        try:
            base_pos, _ = pyb.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client)
            pyb.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=-45,
                cameraPitch=-30,
                cameraTargetPosition=[base_pos[0], base_pos[1], 0.15],
                physicsClientId=env.client
            )
        except Exception:
            pass

        if terminated or truncated:
            print(f"Episode complete. Reason -> Terminated: {terminated}, Truncated: {truncated}. Resetting...")
            obs, _ = env.reset()

        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()