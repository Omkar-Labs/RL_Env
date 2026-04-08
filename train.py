"""
train.py — PPO training for the Cognitive Tutor using stable-baselines3.

Wraps TutorEnv (OpenEnv) into a standard Gymnasium env so SB3 can train on it.
Trains for 300,000 timesteps with PPO, then evaluates the trained model.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env.tutor_env import TutorEnv
from env.student_simulator import (
    NUM_SUBTOPICS,
    NUM_DIFFICULTY,
    TASK_CONFIGS,
    ALL_SUBTOPICS,
)
from models import TutorAction


# ======================================================================
# Gymnasium Wrapper — bridges OpenEnv TutorEnv → Gymnasium API
# ======================================================================
class TutorGymEnv(gym.Env):
    """
    Wraps the OpenEnv TutorEnv into a standard Gymnasium environment
    so stable-baselines3 PPO can train on it.

    Action space (Discrete):
        Flattened: subtopic × difficulty × hint = NUM_SUBTOPICS * 3 * 2

    Observation space (Box):
        [N knowledge levels, N mastery flags, steps_remaining_norm, frustration]
    """

    metadata = {"render_modes": []}

    def __init__(self, task_name: str = "medium"):
        super().__init__()
        self.task_name = task_name
        self.cfg = TASK_CONFIGS[task_name]
        self.max_steps = self.cfg["max_steps"]
        self.env = TutorEnv(task_name=task_name)

        # --- Action space ---
        self.num_actions = NUM_SUBTOPICS * NUM_DIFFICULTY * 2
        self.action_space = spaces.Discrete(self.num_actions)

        # --- Observation space ---
        # N knowledge levels + N mastery flags + 1 steps_remaining (norm) + 1 frustration
        obs_size = (NUM_SUBTOPICS * 2) + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

    def _decode_action(self, action_idx: int) -> TutorAction:
        """Convert flat action index → TutorAction(subtopic, difficulty, hint)."""
        hint = action_idx % 2
        remaining = action_idx // 2
        difficulty = remaining % NUM_DIFFICULTY
        subtopic_idx = remaining // NUM_DIFFICULTY
        return TutorAction(
            subtopic_index=subtopic_idx,
            difficulty_level=difficulty,
            hint_given=bool(hint),
        )

    def _make_obs(self, openenv_obs, frustration: float = 0.0) -> np.ndarray:
        """Convert OpenEnv observation → numpy array for SB3."""
        knowledge = np.array(openenv_obs.knowledge_levels, dtype=np.float32)
        mastery = np.array(openenv_obs.has_ever_mastered, dtype=np.float32)
        steps_norm = np.array(
            [openenv_obs.steps_remaining / self.max_steps], dtype=np.float32
        )
        frust = np.array([frustration], dtype=np.float32)
        return np.concatenate([knowledge, mastery, steps_norm, frust])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        return self._make_obs(obs, frustration=0.0), info

    def step(self, action):
        tutor_action = self._decode_action(int(action))
        obs, reward, done, info = self.env.step(tutor_action)

        frustration = info.get("frustration_level", 0.0)
        np_obs = self._make_obs(obs, frustration)

        # Gymnasium expects (obs, reward, terminated, truncated, info)
        terminated = info.get("all_mastered", False)
        truncated = done and not terminated

        return np_obs, float(reward), terminated, truncated, info


# ======================================================================
# Training callback — logs progress during training
# ======================================================================
class TrainingLogger(BaseCallback):
    """Logs training metrics every `log_interval` steps."""
    def __init__(self, log_interval: int = 5000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.mastery_count = 0
        self.episode_count = 0
        self.current_ep_reward = 0.0
        self.current_ep_length = 0

    def _on_step(self) -> bool:
        self.current_ep_reward += self.locals["rewards"][0]
        self.current_ep_length += 1

        dones = self.locals.get("dones", self.locals.get("done", [False]))
        if dones[0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            self.episode_count += 1

            infos = self.locals.get("infos", [{}])
            if infos[0].get("all_mastered", False):
                self.mastery_count += 1

            self.current_ep_reward = 0.0
            self.current_ep_length = 0

        if self.num_timesteps % self.log_interval == 0 and len(self.episode_rewards) > 0:
            recent = self.episode_rewards[-100:]
            avg_r = sum(recent) / len(recent)
            recent_len = self.episode_lengths[-100:]
            avg_len = sum(recent_len) / len(recent_len)

            mastery_rate = (
                self.mastery_count / self.episode_count * 100
                if self.episode_count > 0
                else 0
            )

            print(
                f"Step {self.num_timesteps:>7,} | "
                f"Episodes: {self.episode_count:>5} | "
                f"Avg Reward (100ep): {avg_r:>7.2f} | "
                f"Avg Length: {avg_len:>5.1f} | "
                f"Mastery%: {mastery_rate:>5.1f}%"
            )

        return True


def train(total_timesteps: int = 300_000, task_name: str = "medium"):
    env = TutorGymEnv(task_name=task_name)

    print("=" * 60)
    print(f"  TRAINING COGNITIVE TUTOR WITH PPO — {total_timesteps:,} timesteps")
    print(f"  Task difficulty:   {task_name}")
    print(f"  Max steps/episode: {env.max_steps}")
    print(f"  Action space:      {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print("=" * 60)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,        # moderate exploration
        verbose=0,
    )

    logger = TrainingLogger(log_interval=5000)
    model.learn(total_timesteps=total_timesteps, callback=logger)

    # Save model
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ppo_model")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}.zip")


if __name__ == "__main__":
    train(total_timesteps=300_000, task_name="medium")
