import random
from openenv.core.env_server import Environment
from models import TutorAction, TutorObservation, TutorState
from .student_simulator import (
    StudentSimulator, ALL_SUBTOPICS, NUM_SUBTOPICS,
    NUM_DIFFICULTY, TASK_CONFIGS,
)


class TutorEnv(Environment):
    """
    Cognitive Tutor RL Environment with 3 difficulty tiers.

    Reward design (all values 0.0–1.0 per task):
      - Per-step task score based on correctness × difficulty tier
      - ZPD alignment bonus for choosing appropriate difficulty
      - Learning progress signal (partial credit)
      - Mastery milestones (one-time per subtopic)
      - Completion score at episode end = mastered_fraction
      
    Tasks: easy, medium, hard — each with different student parameters.
    """

    def __init__(self, task_name: str = "medium"):
        super().__init__()
        self.task_name = task_name
        self.cfg = TASK_CONFIGS[task_name]
        self.max_steps = self.cfg["max_steps"]
        self.student = StudentSimulator(task_difficulty=task_name)
        self.steps_taken: int = 0
        self.done: bool = False
        self.mastered_this_episode: set = set()
        self.cumulative_reward: float = 0.0

    def reset(self, seed=None, **kwargs):
        # Support task switching via kwargs (OpenEnv passes data dict)
        task = kwargs.get("task", self.task_name)
        if task != self.task_name:
            self.task_name = task
            self.cfg = TASK_CONFIGS[task]
            self.max_steps = self.cfg["max_steps"]
            self.student = StudentSimulator(task_difficulty=task)

        if seed is not None:
            random.seed(seed)

        initial_knowledge = {
            sub: round(random.uniform(*self.cfg["initial_knowledge_range"]), 2)
            for sub in ALL_SUBTOPICS
        }
        self.student.reset(initial_knowledge)
        self.steps_taken = 0
        self.done = False
        self.mastered_this_episode = set()
        self.cumulative_reward = 0.0

        observation = self._build_observation()
        info = {"message": "New tutoring session started", "task": self.task_name}
        return observation, info

    def step(self, action: TutorAction):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self.steps_taken += 1

        # 1. Decode the action
        subtopic_index = action.subtopic_index % NUM_SUBTOPICS
        difficulty = action.difficulty_level % NUM_DIFFICULTY
        hint_given = getattr(action, "hint_given", False)
        subtopic = ALL_SUBTOPICS[subtopic_index]

        # 2. Student responds
        result = self.student.respond(subtopic, difficulty, hint_given)

        # ════════════════════════════════════════════════════
        # REWARD — learning-gain-focused, mastery-dominant
        # ════════════════════════════════════════════════════
        knowledge_before = result["knowledge_before"]
        learning_gain = result["learning_gain"]
        knowledge_after = result["knowledge_after"]
        difficulty_norm = (difficulty + 1) / NUM_DIFFICULTY
        gap = difficulty_norm - knowledge_before

        cfg = self.cfg
        reward = 0.0

        # ── (1) Learning progress signal ──
        # Reward ONLY new learning peaks. Re-learning forgotten stuff gives 0.
        progress_reward = result.get("highest_knowledge_delta", 0.0)
        if progress_reward > 0.001:
            # Scale by difficulty: harder tasks give more reward for same learning
            diff_multiplier = [0.8, 1.0, 1.3][difficulty]
            reward += progress_reward * 10.0 * diff_multiplier  # Increased weight for true progress

        # ── (2) ZPD alignment bonus — reward choosing appropriate difficulty ──
        if result["correct"] and cfg["zpd_low"] <= gap <= cfg["zpd_high"]:
            reward += 0.05

        # ── (3) Proximity-to-mastery shaping — breadcrumbs toward mastery ──
        if learning_gain > 0 and not self.student.has_ever_mastered[subtopic]:
            mastery_thresh = cfg["mastery_threshold"]
            thresholds = [
                (mastery_thresh * 0.4, 0.15),
                (mastery_thresh * 0.65, 0.25),
                (mastery_thresh * 0.85, 0.35),
            ]
            for threshold, bonus in thresholds:
                if knowledge_before < threshold <= knowledge_after:
                    reward += bonus

        # ── (4) Mastery milestone — BIG one-time reward per subtopic ──
        # This must dominate per-step farming to incentivize mastery pursuit
        if result["mastered"]:
            self.mastered_this_episode.add(subtopic)
            reward += 1.0

        # ── Penalties ──

        # Anti-farming: too-easy tasks give no learning → penalize
        if gap < -0.2:
            reward -= 0.08

        # Push difficulty up as student progresses
        if difficulty == 0 and knowledge_before > 0.30:
            reward -= 0.16

        # Hint cost — hints make things easy but reduce learning signal
        if hint_given:
            reward -= 0.03

        # Frustration penalty
        if result["frustrated"]:
            reward -= 0.20

        # Repetitive hint penalty
        if result.get("repetitive_hint", False):
            reward -= 0.35

        # Step cost — meaningful time pressure (doubled to 0.02)
        reward -= 0.02

        # Ensure per-step reward is strictly between -0.2 and 1.0
        reward = max(-0.20, min(1.0, reward))

        # ── Termination ──
        num_mastered = sum(self.student.has_ever_mastered.values())
        all_mastered = all(self.student.has_ever_mastered.values())

        terminated = False
        if all_mastered:
            # Completion bonus — efficient completion gets extra
            reward += 1.0  # 3.0 to 5.0
            terminated = True

        truncated = self.steps_taken >= self.max_steps
        self.done = terminated or truncated

        # Penalty for failing when time runs out
        if truncated and not all_mastered:
            mastery_ratio = num_mastered / NUM_SUBTOPICS
            reward -= 1.0 * (1.0 - mastery_ratio)

        reward = float(reward)
        self.cumulative_reward += reward

        # Build observation & info
        observation = self._build_observation()
        info = {
            "subtopic": subtopic,
            "difficulty": difficulty,
            "hint_given": hint_given,
            "learning_gain": learning_gain,
            "mastered_new": result["mastered"],
            "total_mastered": num_mastered,
            "frustration_level": self.student.frustration,
            "correct": result["correct"],
            "all_mastered": all_mastered,
            "frustrated": result["frustrated"],
            "task": self.task_name,
            "cumulative_reward": self.cumulative_reward,
        }

        return observation, reward, self.done, info

    @property
    def state(self) -> TutorState:
        return TutorState(
            student_knowledge=dict(self.student.knowledge),
            steps_taken=self.steps_taken,
            done=self.done,
        )

    def _build_observation(self) -> TutorObservation:
        """Build an observation the agent can see."""
        return TutorObservation(
            knowledge_levels=list(self.student.knowledge.values()),
            has_ever_mastered=list(self.student.has_ever_mastered.values()),
            steps_remaining=self.max_steps - self.steps_taken,
        )
