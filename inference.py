import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Environment specific imports
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
    Wraps the OpenEnv TutorEnv into a standard Gymnasium environment.
    Self-contained for the inference/benchmark script.
    """
    def __init__(self, task_name: str = "medium"):
        super().__init__()
        self.task_name = task_name
        self.cfg = TASK_CONFIGS[task_name]
        self.max_steps = self.cfg["max_steps"]
        self.env = TutorEnv(task_name=task_name)

        # Action space: subtopic × difficulty × hint
        self.num_actions = NUM_SUBTOPICS * NUM_DIFFICULTY * 2
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space: N knowledge + N mastery + steps_remaining + frustration
        obs_size = (NUM_SUBTOPICS * 2) + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

    def _decode_action(self, action_idx: int) -> TutorAction:
        """Convert flat action index → TutorAction."""
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
        """Convert OpenEnv observation → numpy array."""
        knowledge = np.array(openenv_obs.knowledge_levels, dtype=np.float32)
        mastery = np.array(openenv_obs.has_ever_mastered, dtype=np.float32)
        steps_norm = np.array([openenv_obs.steps_remaining / self.max_steps], dtype=np.float32)
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
        
        terminated = info.get("all_mastered", False)
        truncated = done and not terminated
        return np_obs, float(reward), terminated, truncated, info

def main():
    task_name = "curriculum-mastery"
    env_benchmark = "cognitive-tutor-v1"
    model_impl = "ppo"
    
    # 1. Initialize Environment and Model
    env = TutorGymEnv(task_name="medium")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        ent_coef=0.05,
        verbose=0  # SILENT training to ensure STDOUT format compliance
    )
    
    # 2. Train the model from scratch (Un-trained requirement)
    # Using 100k steps as a balance between training quality and script runtime
    model.learn(total_timesteps=100_000)
    
    # 3. Reset for Inference Episode
    obs, info = env.reset()
    
    # [START] line (As per rule)
    print(f"[START] task={task_name} env={env_benchmark} model={model_impl}")
    
    step_n = 0
    rewards_list = []
    done = False
    last_info = info
    last_error = "null"
    
    try:
        while not done:
            step_n += 1
            
            # Get PPO prediction
            action, _ = model.predict(obs, deterministic=True)
            tutor_action = env._decode_action(int(action))
            
            # Format action string for log
            action_str = f"teach(topic={tutor_action.subtopic_index}, diff={tutor_action.difficulty_level}, hint={str(tutor_action.hint_given).lower()})"
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards_list.append(reward)
            last_info = info
            
            # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
            # reward formatted to 2 decimal places
            r_val = f"{reward:.2f}"
            d_val = "true" if done else "false"
            print(f"[STEP] step={step_n} action={action_str} reward={r_val} done={d_val} error={last_error}")

    except Exception as e:
        last_error = str(e).replace("\n", " ").replace(",", ";")
    finally:
        # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
        success = "true" if last_info.get("all_mastered", False) else "false"
        
        # Calculate score in [0, 1] based on number of topics mastered
        # Indices for mastery flags in obs array: knowledge(4) + mastery(4) ...
        mastery_flags = obs[NUM_SUBTOPICS : 2*NUM_SUBTOPICS]
        score = sum(mastery_flags) / NUM_SUBTOPICS if NUM_SUBTOPICS > 0 else 0.0
        
        score_val = f"{score:.2f}"
        rewards_seq = ",".join([f"{r:.2f}" for r in rewards_list])
        
        print(f"[END] success={success} steps={step_n} score={score_val} rewards={rewards_seq}")

if __name__ == "__main__":
    main()
