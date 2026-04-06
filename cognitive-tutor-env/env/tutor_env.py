import random
from openenv.core.env_server import Environment
from models import TutorAction, TutorObservation, TutorState
from .student_simulator import StudentSimulator, ALL_SUBTOPICS, NUM_SUBTOPICS, NUM_DIFFICULTY, MAX_STEPS, MASTERY_THRESHOLD

class TutorEnv(Environment):
    """
    Cognitive Tutor RL Environment.
    The agent (tutor) picks which subtopic + difficulty to present,
    and optionally whether to give a hint.
    A StudentSimulator responds based on current knowledge & frustration.
    """

    def __init__(self):
        super().__init__()
        self.student = StudentSimulator()
        self.steps_taken: int = 0
        self.done: bool = False

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # Initialize student knowledge
        initial_knowledge = {
            sub: round(random.uniform(0.0, 0.25), 2)
            for sub in ALL_SUBTOPICS
        }
        self.student.reset(initial_knowledge)
        self.steps_taken = 0
        self.done = False

        observation = self._build_observation()
        info = {"message": "New tutoring session started"}
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
        gap = difficulty/(NUM_DIFFICULTY-1) - result["knowledge_before"]
        # 3. Calculate reward
        reward = 0.0
        learning_gain = result["knowledge_after"] - result["knowledge_before"]

        # Bonus for progress
        reward += learning_gain * 3.0

        # Correctness incentive
        if result["correct"]:
            reward += 0.1

        # Milestone: One-time Mastery
        if result["mastered"]:
            reward = 1.0  
        if -0.2 <= gap <= 0.3:
            reward += 0.15   # 🎯 optimal teaching zone
        elif gap > 0.5:
            reward -= 0.1    # too hard
        elif gap < -0.3:
            reward -= 0.05   # too easy
        # Anti-farming: Disqualify repeating what is already "Done"
        if self.student.has_ever_mastered[subtopic] and learning_gain < 0.005:
            reward -= 0.1

        # Penalties (Empathetic Tutor Weighting)
        if result["frustrated"]:
            reward -= 0.3 
            
        if result["repetitive_hint"]:
            reward -= 0.2
        if learning_gain < 0.01:
            reward -= 0.15

        num_mastered = sum(self.student.has_ever_mastered.values())
        reward += 0.04 * num_mastered
        
        # Time Penalty
        reward -= 0.02

        # Completion
        all_mastered = all(self.student.has_ever_mastered.values())
        if all_mastered:
            reward = 1.0  
            self.done = True

        truncated = self.steps_taken >= MAX_STEPS
        self.done = self.done or truncated

        if truncated and not all_mastered:
            reward = 0.0
            
        # Clamp reward [0.0, 1.0]
        reward = float(max(0.0, min(1.0, reward)))
        
        # 4. Build observation & info
        observation = self._build_observation()
        info = {
            "subtopic": subtopic,
            "difficulty": difficulty,
            "hint_given": hint_given,
            "student_correct": result["correct"],
            "frustrated": result["frustrated"],
            "mastered": result["mastered"],
            "repetitive_hint": result["repetitive_hint"],
            "knowledge_delta": result["highest_knowledge_delta"],
            "learning_gain": learning_gain,
            "all_mastered": all_mastered,
            "steps_taken": self.steps_taken,
            "frustration_level": self.student.frustration,
        }
        
        return observation, reward, self.done, info

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
            steps_remaining=MAX_STEPS - self.steps_taken,
        )
