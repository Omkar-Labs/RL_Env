import random
from openenv.core.env_server import Environment
from models import TutorAction, TutorObservation, TutorState


# Topics with subtopics — each subtopic has a knowledge level tracked independently
# FRONTEND INPUT: Replace this hardcoded dict with user input from frontend.
# e.g. receive a JSON like {"Math": ["Algebra", "Geometry"], "DSA": ["Arrays", "Trees"]}
# from a POST request and pass it into TutorEnv.__init__(topics=user_topics)
TOPICS = {
    "Math": ["Algebra",  "Calculus"],
    "Science": ["Physics", "Chemistry"]
}

# Flatten all subtopics into a single ordered list for indexing
ALL_SUBTOPICS = []
for subject, subs in TOPICS.items():
    for sub in subs:
        ALL_SUBTOPICS.append(f"{subject}_{sub}")

NUM_SUBTOPICS = len(ALL_SUBTOPICS)   # 9 subtopics total
NUM_DIFFICULTY = 3                    # 0=Easy, 1=Medium, 2=Hard
MAX_STEPS = NUM_SUBTOPICS * 25        # max questions per tutoring session (~25 per topic)
MASTERY_THRESHOLD = 0.8               # knowledge >= 0.8 means mastered
# FRONTEND INPUT: all 4 constants above will auto-update if you make
# TOPICS dynamic. MAX_STEPS & MASTERY_THRESHOLD can also come from user
# e.g. user selects "session length = 40" or "mastery bar = 0.9"


# ======================================================================
# Student Simulator — models a realistic student with knowledge,
# frustration, hint dependency, and learning dynamics.
# ======================================================================
class StudentSimulator:
    """
    Simulates a student responding to tutoring actions.

    Tracks per-subtopic knowledge, frustration level, and hint history
    to produce realistic responses (correct/incorrect, frustrated, etc.).
    """

    def __init__(self, initial_knowledge: dict[str, float] | None = None):
        # Per-subtopic knowledge in [0.0, 1.0]
        self.decay_rate = 0.005
        self.last_practiced: dict[str, int] = {sub: 0 for sub in ALL_SUBTOPICS}
        if initial_knowledge is not None:
            self.knowledge = dict(initial_knowledge)
        else:
            self.knowledge = {
                sub: round(random.uniform(0.0, 0.25), 2)
                for sub in ALL_SUBTOPICS
            }

        # Frustration: rises when questions are too hard, decays when manageable
        self.frustration: float = 0.0          # [0.0, 1.0]
        self.frustration_threshold: float = 0.7  # above this → student is frustrated

        # Hint tracking: how many times each subtopic got hinted consecutively
        self.hint_streak: dict[str, int] = {sub: 0 for sub in ALL_SUBTOPICS}
        self.last_hinted_subtopic: str | None = None

    # ------------------------------------------------------------------
    # respond — the core method: student attempts a question
    # ------------------------------------------------------------------
    def respond(
        self,
        subtopic: str,
        difficulty: int,       # 0=Easy, 1=Medium, 2=Hard
        hint_given: bool = False,
    ) -> dict:
        """
        Returns a dict with:
            correct       : bool  — did the student answer correctly?
            frustrated    : bool  — did the student become frustrated?
            hint_used     : bool  — was a hint consumed?
            mastered      : bool  — did the student just cross the mastery threshold?
            repetitive_hint: bool — is the tutor hinting the same subtopic repeatedly?
            knowledge_before: float
            knowledge_after : float
        """
        
        knowledge_before = self.knowledge[subtopic]
        difficulty_norm = (difficulty + 1) / NUM_DIFFICULTY   # 0.33, 0.67, 1.0
        for sub in ALL_SUBTOPICS:
            if sub == subtopic:
                self.last_practiced[sub] = 0
            else:
                self.last_practiced[sub] += 1
        # --- Probability of correct answer ---
        # Base probability from knowledge vs difficulty
        base_prob = max(0.0, min(1.0, knowledge_before + 0.3 - 0.4 * difficulty_norm))

        # Hints boost probability by ~20%
        if hint_given:
            base_prob = min(1.0, base_prob + 0.2)

        # Frustration drags probability down
        base_prob = max(0.0, base_prob - 0.15 * self.frustration)

        correct = random.random() < base_prob

        # --- Repetitive hinting detection ---
        repetitive_hint = False
        if hint_given:
            if self.last_hinted_subtopic == subtopic:
                self.hint_streak[subtopic] += 1
            else:
                # Reset the old subtopic streak, start new one
                if self.last_hinted_subtopic is not None:
                    self.hint_streak[self.last_hinted_subtopic] = 0
                self.hint_streak[subtopic] = 1
            self.last_hinted_subtopic = subtopic
            # 3+ consecutive hints on same subtopic = repetitive
            if self.hint_streak[subtopic] >= 3:
                repetitive_hint = True
        else:
            # No hint given → reset streak for this subtopic
            if self.last_hinted_subtopic == subtopic:
                self.hint_streak[subtopic] = 0
                self.last_hinted_subtopic = None

        gap = difficulty_norm - knowledge_before

        # --- Learning / knowledge update ---
        if correct:
            if gap > 0.5:
                # Question was way too hard, student guessed lucky -> no learning
                learning_gain = 0.0
            elif gap < -0.2:
                # Question was too easy -> minimal reinforcement
                learning_gain = 0.02
            else:
                # ZPD Zone -> strong learning
                learning_gain = 0.05 + 0.10 * difficulty_norm

            if hint_given:
                # Learning is slightly less if student needed a hint
                learning_gain *= 0.8
        else:
            if gap > 0.5:
                # Way too hard, no learning from attempting
                learning_gain = 0.0
            else:
                # Still learn a little from attempting
                learning_gain = 0.03

        knowledge_after = min(1.0, round(knowledge_before + learning_gain, 4))
        self.knowledge[subtopic] = knowledge_after

        # Did the student just master this subtopic?
        mastered = (knowledge_after >= MASTERY_THRESHOLD and
                    knowledge_before < MASTERY_THRESHOLD)

        # --- Frustration update ---
        # Gap between difficulty and student ability drives frustration
        if gap > 0.3:
            # Question is significantly above student level → frustration rises
            self.frustration = min(1.0, self.frustration + 0.15 * gap)
        elif gap < -0.1:
            # Question is easy → frustration decays
            self.frustration = max(0.0, self.frustration - 0.1)
        else:
            # ZPD zone → slight decay
            self.frustration = max(0.0, self.frustration - 0.05)

        # Extra frustration if student got it wrong
        if not correct:
            self.frustration = min(1.0, self.frustration + 0.1)
        
        # --- Knowledge decay ---
        current_decay_rate = self.decay_rate
        if self.frustration > 0.8:
            current_decay_rate = current_decay_rate * 3.0  # Panic! Fast forgetting

        for sub in ALL_SUBTOPICS:
            if self.last_practiced[sub] > 0:
                if self.knowledge[sub] >= MASTERY_THRESHOLD:
                    self.knowledge[sub] = self.knowledge[sub]*(1.0-current_decay_rate/10.0)
                    self.knowledge[sub] = max(0.0, self.knowledge[sub])
                else:
                    self.knowledge[sub] = self.knowledge[sub]*(1.0-current_decay_rate)
                    self.knowledge[sub] = max(0.0, self.knowledge[sub])
                    

        frustrated = self.frustration >= self.frustration_threshold

        return {
            "correct": correct,
            "frustrated": frustrated,
            "hint_used": hint_given,
            "mastered": mastered,
            "repetitive_hint": repetitive_hint,
            "knowledge_before": knowledge_before,
            "knowledge_after": self.knowledge[subtopic],
        }

    def reset(self, initial_knowledge: dict[str, float] | None = None):
        """Reset student state for a new episode."""
        if initial_knowledge is not None:
            self.knowledge = dict(initial_knowledge)
        else:
            self.knowledge = {
                sub: round(random.uniform(0.0, 0.25), 2)
                for sub in ALL_SUBTOPICS
            }
        self.frustration = 0.0
        self.hint_streak = {sub: 0 for sub in ALL_SUBTOPICS}
        self.last_hinted_subtopic = None
        self.last_practiced = {sub: 0 for sub in ALL_SUBTOPICS}


# ======================================================================
# Tutor Environment
# ======================================================================
class TutorEnv(Environment):
    """
    Cognitive Tutor RL Environment.

    The agent (tutor) picks which subtopic + difficulty to present,
    and optionally whether to give a hint.
    A StudentSimulator responds based on current knowledge & frustration.

    Reward structure:
        +1.0   student masters a concept (crosses mastery threshold)
        +0.4   correct answer after hint
        +0.1   correct answer without hint
        -0.8   student gets frustrated
        -0.2   repetitive hinting (same subtopic 3+ times)
        -0.05  time/step penalty (every step)
        +Δk    bonus proportional to knowledge delta (scaled by 2.0)
    """

    # FRONTEND INPUT: Accept topics, max_steps, mastery_threshold as constructor params
    # e.g. def __init__(self, topics=None, max_steps=50, mastery_threshold=0.8):
    # then override the global TOPICS, MAX_STEPS, MASTERY_THRESHOLD with user values.
    # This way each user session can have its own config without touching globals.
    def __init__(self):
        super().__init__()
        self.student = StudentSimulator()
        self.steps_taken: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------
    # reset  — start a fresh tutoring session
    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # FRONTEND INPUT: If user provides their current skill levels from frontend
        # (e.g. a quiz result), use those instead of random initialization.
        # e.g. initial_knowledge = user_provided_knowledge or default below
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

    # ------------------------------------------------------------------
    # step  — tutor picks an action, student responds, reward is computed
    # ------------------------------------------------------------------
    def step(self, action: TutorAction) :
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self.steps_taken += 1

        # --- 1. Decode the action ---
        subtopic_index = action.subtopic_index % NUM_SUBTOPICS
        difficulty = action.difficulty_level % NUM_DIFFICULTY  # 0, 1, 2
        # Use hint_given if the action dataclass has it, else default False
        hint_given = getattr(action, "hint_given", False)

        subtopic = ALL_SUBTOPICS[subtopic_index]

        # --- 2. Student responds ---
        result = self.student.respond(subtopic, difficulty, hint_given)

        # --- 3. Calculate reward ---
        reward = 0.0

        # (a) Time/step penalty — every step costs a little
        reward -= 0.05

        # (b) Correct answer rewards
        if result["correct"]:
            if result["hint_used"]:
                reward += 0.4      # correct after hint
            else:
                reward += 0.1      # correct without hint
        
        # (c) Mastery bonus — student just crossed the threshold
        if result["mastered"]:
            reward += 1.0

        # (d) Frustration penalty
        if result["frustrated"]:
            reward -= 0.8

        # (e) Repetitive hinting penalty
        if result["repetitive_hint"]:
            reward -= 0.2

        # (f) Knowledge delta bonus — reward improvement in knowledge
        delta_knowledge = result["knowledge_after"] - result["knowledge_before"]
        reward += delta_knowledge * 2.0   # scale factor to make delta meaningful

        reward = round(reward, 4)

        # --- 4. Check termination ---
        all_mastered = all(
            v >= MASTERY_THRESHOLD for v in self.student.knowledge.values()
        )
        truncated = self.steps_taken >= MAX_STEPS
        self.done = all_mastered or truncated

        # --- 5. Build observation & info ---
        observation = self._build_observation()
        info = {
            "subtopic": subtopic,
            "difficulty": difficulty,
            "hint_given": hint_given,
            "student_correct": result["correct"],
            "frustrated": result["frustrated"],
            "mastered": result["mastered"],
            "repetitive_hint": result["repetitive_hint"],
            "knowledge_delta": delta_knowledge,
            "learning_gain": result["knowledge_after"] - result["knowledge_before"],
            "all_mastered": all_mastered,
            "steps_taken": self.steps_taken,
            "frustration_level": self.student.frustration,
        }
        
        return observation, reward, self.done, info

    # ------------------------------------------------------------------
    # state  — full internal state (used by OpenEnv web UI)
    # ------------------------------------------------------------------
    def state(self) -> TutorState:
        return TutorState(
            student_knowledge=dict(self.student.knowledge),
            steps_taken=self.steps_taken,
            done=self.done,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> TutorObservation:
        """Build an observation the agent can see."""
        return TutorObservation(
            knowledge_levels=list(self.student.knowledge.values()),
            steps_remaining=MAX_STEPS - self.steps_taken,
        )
