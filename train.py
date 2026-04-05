"""
train.py — PPO training for the Cognitive Tutor using stable-baselines3.

Wraps TutorEnv (OpenEnv) into a standard Gymnasium env so SB3 can train on it.
Trains for 100,000 timesteps with PPO, then evaluates the trained model.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from my_environment import (
    TutorEnv,
    NUM_SUBTOPICS,
    NUM_DIFFICULTY,
    MAX_STEPS,
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

    Action space (MultiDiscrete):
        [subtopic_index (0-8), difficulty (0-2), hint (0-1)]
        Flattened as a single Discrete(54) for PPO compatibility.

    Observation space (Box):
        [9 knowledge levels (0-1), steps_remaining (0-30), frustration (0-1)]
        = 11 floats total
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.env = TutorEnv()
        # FRONTEND INPUT: If TOPICS becomes dynamic (user-provided), NUM_SUBTOPICS
        # changes → action_space and obs_space sizes change automatically here.
        # No manual change needed in this wrapper if you update the globals in my_environment.py

        # --- Action space ---
        # 9 subtopics × 3 difficulties × 2 hint options = 54 discrete actions
        self.num_actions = NUM_SUBTOPICS * NUM_DIFFICULTY * 2
        self.action_space = spaces.Discrete(self.num_actions)

        # --- Observation space ---
        # 4 knowledge levels + 4 mastery flags + 1 steps_remaining (normalized) + 1 frustration
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
            [openenv_obs.steps_remaining / MAX_STEPS], dtype=np.float32
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

        # Get frustration from info for observation
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
        # Accumulate rewards
        self.current_ep_reward += self.locals["rewards"][0]
        self.current_ep_length += 1

        # Check if episode ended
        dones = self.locals.get("dones", self.locals.get("done", [False]))
        if dones[0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            self.episode_count += 1

            # Check mastery from info
            infos = self.locals.get("infos", [{}])
            if infos[0].get("all_mastered", False):
                self.mastery_count += 1

            self.current_ep_reward = 0.0
            self.current_ep_length = 0

        # Log periodically
        if self.num_timesteps % self.log_interval == 0 and len(self.episode_rewards) > 0:
            recent = self.episode_rewards[-100:]  # last 100 episodes
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


# ======================================================================
# Evaluation — test the trained model
# ======================================================================
def evaluate(model, num_episodes: int = 100):
    """Run the trained model for `num_episodes` and print results."""
    env = TutorGymEnv()
    total_rewards = []
    mastery_hits = 0
    total_frustrations = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if info.get("frustrated", False):
                total_frustrations += 1

        total_rewards.append(ep_reward)
        if info.get("all_mastered", False):
            mastery_hits += 1

    avg_reward = sum(total_rewards) / len(total_rewards)
    mastery_pct = mastery_hits / num_episodes * 100
    best_reward = max(total_rewards)
    worst_reward = min(total_rewards)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:             {num_episodes}")
    print(f"  Avg Reward:           {avg_reward:.3f}")
    print(f"  Best Reward:          {best_reward:.3f}")
    print(f"  Worst Reward:         {worst_reward:.3f}")
    print(f"  Mastery Rate:         {mastery_pct:.1f}%")
    print(f"  Frustration Events:   {total_frustrations}")
    print("=" * 60)

    return avg_reward, mastery_pct, best_reward, worst_reward, total_frustrations


# ======================================================================
# Main
# ======================================================================
# FRONTEND INPUT: Accept total_timesteps, learning_rate, etc. from frontend.
# e.g. user picks "training intensity = high" → total_timesteps=200_000
# Create an API route like POST /train that calls train(total_timesteps=user_value)
def train(total_timesteps: int = 100_000):
    env = TutorGymEnv()

    print("=" * 60)
    print(f"  TRAINING COGNITIVE TUTOR WITH PPO — {total_timesteps:,} timesteps")
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
        ent_coef=0.01,        # entropy bonus for exploration
        verbose=0,
    )

    logger = TrainingLogger(log_interval=5000)
    model.learn(total_timesteps=total_timesteps, callback=logger)

    # Save model
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "ppo_cognitive_tutor")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}.zip")

    # Evaluate
    avg_reward, mastery_pct, best_reward, worst_reward, total_frustrations = evaluate(model, num_episodes=200)

    return {
        "avg_reward": round(avg_reward, 3),
        "mastery_pct": round(mastery_pct, 1),
        "best_reward": round(best_reward, 3),
        "worst_reward": round(worst_reward, 3),
        "total_frustrations": total_frustrations,
        "episodes": 200,
    }


if __name__ == "__main__":
    
    train(total_timesteps=100_000)
    
