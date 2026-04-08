import random

# Topic and indexing setup
TOPICS = {"Math": ["Algebra", "Calculus"], "Science": ["Physics", "Chemistry"]}
ALL_SUBTOPICS = [f"{s}_{sub}" for s, subs in TOPICS.items() for sub in subs]
NUM_SUBTOPICS = len(ALL_SUBTOPICS)
NUM_DIFFICULTY = 3
MASTERY_THRESHOLD = 0.75  # default, overridden per-task

# ── Difficulty-tier configs ──────────────────────────────────────────
# Each tier defines how the simulated student behaves.
# Tighter ZPD + faster decay = harder for random agents = more signal for PPO.
# ZPD math reference:
# difficulty_norm: easy=0.33, medium=0.67, hard=1.0
# gap = difficulty_norm - knowledge
# For ZPD to work, the window [zpd_low, zpd_high] must include the gap
# at the knowledge level where you WANT the agent to use that difficulty.
#
# Easy diff (0.33): useful when knowledge 0.0–0.5  (gap 0.33 to -0.17)
# Med diff  (0.67): useful when knowledge 0.2–0.7  (gap 0.47 to -0.03)
# Hard diff (1.0):  useful when knowledge 0.4–0.85 (gap 0.60 to  0.15)
#
# zpd_high must be >= 0.60 for hard difficulty to EVER be useful!

TASK_CONFIGS = {
    "easy": {
        "max_steps": 80,
        "mastery_threshold": 0.65,
        "decay_rate": 0.002,
        "base_learn_zpd": 0.07,       # learning gain when in ZPD
        "base_learn_easy": 0.02,       # learning gain when task too easy
        "base_learn_lucky": 0.03,      # learning gain on lucky guess
        "base_learn_fail_close": 0.02, # learning from mistakes (close gap)
        "base_learn_fail_far": 0.008,  # learning from mistakes (far gap)
        "zpd_low": -0.20,             # ZPD lower bound
        "zpd_high": 0.65,             # ZPD upper bound — covers all 3 difficulties
        "frustration_recovery": 0.10,
        "frustration_gain": 0.10,
        "decay_after_steps": 15,
        "initial_knowledge_range": (0.05, 0.25),
    },
    "medium": {
        "max_steps": 90,
        "mastery_threshold": 0.70,
        "decay_rate": 0.004,
        "base_learn_zpd": 0.065,
        "base_learn_easy": 0.015,
        "base_learn_lucky": 0.025,
        "base_learn_fail_close": 0.015,
        "base_learn_fail_far": 0.005,
        "zpd_low": -0.15,
        "zpd_high": 0.55,             # covers easy+medium diff; hard is stretch
        "frustration_recovery": 0.08,
        "frustration_gain": 0.15,
        "decay_after_steps": 12,
        "initial_knowledge_range": (0.03, 0.18),
    },
    "hard": {
        "max_steps": 120,
        "mastery_threshold": 0.75,
        "decay_rate": 0.007,
        "base_learn_zpd": 0.05,
        "base_learn_easy": 0.008,
        "base_learn_lucky": 0.015,
        "base_learn_fail_close": 0.010,
        "base_learn_fail_far": 0.003,
        "zpd_low": -0.10,
        "zpd_high": 0.45,             # narrower — must choose difficulty more carefully
        "frustration_recovery": 0.05,
        "frustration_gain": 0.20,
        "decay_after_steps": 8,
        "initial_knowledge_range": (0.0, 0.12),
    },
}

# Expose MAX_STEPS as the medium default for backward compat
MAX_STEPS = TASK_CONFIGS["medium"]["max_steps"]


class StudentSimulator:
    def __init__(self, task_difficulty: str = "medium",
                 initial_knowledge: dict[str, float] | None = None):
        self.task_difficulty = task_difficulty
        self.cfg = TASK_CONFIGS[task_difficulty]

        self.mastery_threshold = self.cfg["mastery_threshold"]
        self.decay_rate = self.cfg["decay_rate"]

        self.last_practiced: dict[str, int] = {sub: 0 for sub in ALL_SUBTOPICS}

        if initial_knowledge:
            self.knowledge = dict(initial_knowledge)
        else:
            lo, hi = self.cfg["initial_knowledge_range"]
            self.knowledge = {
                sub: round(random.uniform(lo, hi), 2)
                for sub in ALL_SUBTOPICS
            }

        self.frustration: float = 0.0
        self.frustration_threshold: float = 0.7
        self.hint_streak: dict[str, int] = {sub: 0 for sub in ALL_SUBTOPICS}
        self.last_hinted_subtopic: str | None = None
        self.has_ever_mastered: dict[str, bool] = {sub: False for sub in ALL_SUBTOPICS}
        self.highest_knowledge: dict[str, float] = dict(self.knowledge)

    def respond(self, subtopic: str, difficulty: int, hint_given: bool = False) -> dict:
        cfg = self.cfg
        knowledge_before = self.knowledge[subtopic]
        difficulty_norm = (difficulty + 1) / NUM_DIFFICULTY  # 0.33, 0.67, 1.0

        # Update practice counters
        for sub in ALL_SUBTOPICS:
            self.last_practiced[sub] = 0 if sub == subtopic else self.last_practiced[sub] + 1

        # ── Success probability ──
        success_chance = max(0.05, min(0.95, (knowledge_before + 0.2) - (0.5 * difficulty_norm)))
        if hint_given:
            success_chance = min(0.98, success_chance + 0.3)
        success_chance -= (0.2 * self.frustration)

        correct = random.random() < success_chance

        # ── Hint repetition check ──
        repetitive_hint = False
        if hint_given:
            self.hint_streak[subtopic] = (
                self.hint_streak[subtopic] + 1
                if self.last_hinted_subtopic == subtopic
                else 1
            )
            if self.hint_streak[subtopic] >= 3:
                repetitive_hint = True
            self.last_hinted_subtopic = subtopic
        else:
            self.hint_streak[subtopic] = 0

        # ── Learning gain (ZPD-gated) ──
        gap = difficulty_norm - knowledge_before
        learning_gain = 0.0

        if correct:
            if cfg["zpd_low"] <= gap <= cfg["zpd_high"]:
                # In the Zone of Proximal Development → best learning
                learning_gain = cfg["base_learn_zpd"] + (0.03 * difficulty_norm)
            elif gap < cfg["zpd_low"]:
                # Too easy → minimal learning
                learning_gain = cfg["base_learn_easy"]
            else:
                # Lucky guess on hard topic
                learning_gain = cfg["base_learn_lucky"]
        else:
            # Failed — learn from mistake
            learning_gain = (
                cfg["base_learn_fail_close"] if gap <= 0.4
                else cfg["base_learn_fail_far"]
            )

        # Apply learning (capped at 1.0)
        knowledge_after = min(1.0, round(knowledge_before + learning_gain, 4))
        self.knowledge[subtopic] = knowledge_after

        # ── Mastery milestone ──
        peak_before = self.highest_knowledge.get(subtopic, 0.0)
        highest_knowledge_delta = max(0.0, knowledge_after - peak_before)

        mastered = False
        if knowledge_after >= self.mastery_threshold and not self.has_ever_mastered[subtopic]:
            self.has_ever_mastered[subtopic] = True
            mastered = True

        # Track highest knowledge
        if knowledge_after > peak_before:
            self.highest_knowledge[subtopic] = knowledge_after

        # ── Frustration ──
        if gap > 0.4 and not correct:
            self.frustration = min(1.0, self.frustration + cfg["frustration_gain"])
        elif correct:
            self.frustration = max(0.0, self.frustration - cfg["frustration_recovery"])
        else:
            self.frustration = max(0.0, self.frustration - 0.02)

        # ── Decay: unpracticed, non-mastered topics lose knowledge ──
        for sub in ALL_SUBTOPICS:
            if (self.last_practiced[sub] > cfg["decay_after_steps"]
                    and not self.has_ever_mastered[sub]):
                self.knowledge[sub] = max(0.0, self.knowledge[sub] - self.decay_rate)

        return {
            "correct": correct,
            "frustrated": self.frustration >= self.frustration_threshold,
            "hint_used": hint_given,
            "mastered": mastered,
            "learning_gain": learning_gain,
            "repetitive_hint": repetitive_hint,
            "knowledge_before": knowledge_before,
            "knowledge_after": self.knowledge[subtopic],
            "highest_knowledge_delta": highest_knowledge_delta,
        }

    def reset(self, initial_knowledge: dict[str, float] | None = None):
        cfg = self.cfg
        if initial_knowledge is not None:
            self.knowledge = dict(initial_knowledge)
        else:
            lo, hi = cfg["initial_knowledge_range"]
            self.knowledge = {
                sub: round(random.uniform(lo, hi), 2)
                for sub in ALL_SUBTOPICS
            }
        self.frustration = 0.0
        self.hint_streak = {sub: 0 for sub in ALL_SUBTOPICS}
        self.last_hinted_subtopic = None
        self.last_practiced = {sub: 0 for sub in ALL_SUBTOPICS}
        self.has_ever_mastered = {sub: False for sub in ALL_SUBTOPICS}
        self.highest_knowledge = dict(self.knowledge)
