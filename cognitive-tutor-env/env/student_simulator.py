import random

# Topics with subtopics — each subtopic has a knowledge level tracked independently
TOPICS = {
    "Math": ["Algebra",  "Calculus"],
    "Science": ["Physics", "Chemistry"]
}

# Flatten all subtopics into a single ordered list for indexing
ALL_SUBTOPICS = []
for subject, subs in TOPICS.items():
    for sub in subs:
        ALL_SUBTOPICS.append(f"{subject}_{sub}")

NUM_SUBTOPICS = len(ALL_SUBTOPICS)   # 4 subtopics total
NUM_DIFFICULTY = 3                    # 0=Easy, 1=Medium, 2=Hard
MAX_STEPS = NUM_SUBTOPICS * 50        # Moderated for 50%+ mastery target (200 total)
MASTERY_THRESHOLD = 0.75              # knowledge >= 0.75 means mastered

class StudentSimulator:
    """
    Simulates a student responding to tutoring actions.
    Tracks per-subtopic knowledge, frustration level, and hint history
    to produce realistic responses (correct/incorrect, frustrated, etc.).
    """

    def __init__(self, initial_knowledge: dict[str, float] | None = None):
        # Per-subtopic knowledge in [0.0, 1.0]
        self.decay_rate = 0.015  # Realistic decay (1.5%) for Hackathon calibration
        self.last_practiced: dict[str, int] = {sub: 0 for sub in ALL_SUBTOPICS}
        if initial_knowledge is not None:
            self.knowledge = dict(initial_knowledge)
        else:
            self.knowledge = {
                sub: round(random.uniform(0.0, 0.25), 2)
                for sub in ALL_SUBTOPICS
            }

        # Frustration
        self.frustration: float = 0.0          # [0.0, 1.0]
        self.frustration_threshold: float = 0.7  # above this → student is frustrated

        # Hint tracking
        self.hint_streak: dict[str, int] = {sub: 0 for sub in ALL_SUBTOPICS}
        self.last_hinted_subtopic: str | None = None
        
        # Anti-farming state tracking
        self.has_ever_mastered: dict[str, bool] = {sub: False for sub in ALL_SUBTOPICS}
        self.highest_knowledge: dict[str, float] = dict(self.knowledge)

    def respond(
        self,
        subtopic: str,
        difficulty: int,       # 0=Easy, 1=Medium, 2=Hard
        hint_given: bool = False,
    ) -> dict:
        knowledge_before = self.knowledge[subtopic]
        difficulty_norm = (difficulty + 1) / NUM_DIFFICULTY   # 0.33, 0.67, 1.0
        for sub in ALL_SUBTOPICS:
            if sub == subtopic:
                self.last_practiced[sub] = 0
            else:
                self.last_practiced[sub] += 1

        # Probability of correct answer
        base_prob = max(0.0, min(1.0, knowledge_before + 0.15 - 0.45 * difficulty_norm))
        if hint_given:
            base_prob = min(1.0, base_prob + 0.25)
        base_prob = max(0.0, base_prob - 0.15 * self.frustration)
        correct = random.random() < base_prob

        # Repetitive hinting
        repetitive_hint = False
        if hint_given:
            if self.last_hinted_subtopic == subtopic:
                self.hint_streak[subtopic] += 1
            else:
                if self.last_hinted_subtopic is not None:
                    self.hint_streak[self.last_hinted_subtopic] = 0
                self.hint_streak[subtopic] = 1
            self.last_hinted_subtopic = subtopic
            if self.hint_streak[subtopic] >= 3:
                repetitive_hint = True
        else:
            if self.last_hinted_subtopic == subtopic:
                self.hint_streak[subtopic] = 0
                self.last_hinted_subtopic = None

        gap = difficulty_norm - knowledge_before

        # Learning gain
        if correct:
            if gap > 0.5:
                learning_gain = 0.0
            elif gap < -0.2:
                learning_gain = 0.01
            else:
                learning_gain = 0.02 + 0.05 * difficulty_norm
            if hint_given:
                learning_gain *= 0.8
        else:
            if gap > 0.5:
                learning_gain = 0.0
            else:
                learning_gain = 0.015

        knowledge_after = min(1.0, round(knowledge_before + learning_gain, 4))
        self.knowledge[subtopic] = knowledge_after

        # Anti-farming: Delta high
        old_high = self.highest_knowledge[subtopic]
        delta_high = max(0.0, knowledge_after - old_high)
        self.highest_knowledge[subtopic] = max(old_high, knowledge_after)

        # First-time mastery
        mastered = False
        if knowledge_after >= MASTERY_THRESHOLD and not self.has_ever_mastered[subtopic]:
            self.has_ever_mastered[subtopic] = True
            mastered = True

        # Frustration
        if gap > 0.4:
            self.frustration = min(1.0, self.frustration + 0.15 * gap)
        elif gap < -0.1:
            self.frustration = max(0.0, self.frustration - 0.1)
        else:
            self.frustration = max(0.0, self.frustration - 0.05)
        if not correct:
            self.frustration = min(1.0, self.frustration + 0.12)
        
        # Knowledge decay
        current_decay_rate = self.decay_rate
        if self.frustration > 0.7:
            current_decay_rate *= 0.3
        
        for sub in ALL_SUBTOPICS:
            if self.last_practiced[sub] > 5:
                if self.has_ever_mastered[sub]:
                    self.knowledge[sub] *= (1.0 - current_decay_rate / 15.0)
                else:
                    self.knowledge[sub] *= (1.0 - current_decay_rate)
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
            "highest_knowledge_delta": delta_high,
        }

    def reset(self, initial_knowledge: dict[str, float] | None = None):
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
        self.has_ever_mastered = {sub: False for sub in ALL_SUBTOPICS}
        self.highest_knowledge = dict(self.knowledge)
