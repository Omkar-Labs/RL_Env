# heuristic_agent.py
import numpy as np
from env.student_simulator import NUM_SUBTOPICS, NUM_DIFFICULTY

class HeuristicAgent:
    """
    Focus strategy: stick with one subtopic until mastered, then move to the next.
    Picks difficulty and hints based on current knowledge level.
    """
    def __init__(self):
        self.current_focus = None

    def act(self, obs):
        """
        Calculates the best action based on the state's knowledge levels.
        """
        knowledge_levels = obs[:NUM_SUBTOPICS]
        mastery_flags = obs[NUM_SUBTOPICS:NUM_SUBTOPICS * 2]
        frustration = obs[-1]

        # Pick the unmastered subtopic with the highest knowledge (closest to mastery)
        best_idx = None
        best_k = -1.0

        for i in range(NUM_SUBTOPICS):
            if mastery_flags[i] < 0.5:  # not yet mastered
                if knowledge_levels[i] > best_k:
                    best_k = knowledge_levels[i]
                    best_idx = i

        # Fall back: if all mastered, just pick lowest
        if best_idx is None:
            best_idx = int(np.argmin(knowledge_levels))

        k = knowledge_levels[best_idx]

        # Adaptive difficulty based on knowledge + ZPD alignment
        # Easy ZPD valid: k < ~0.50 (gap = 0.33-k, needs gap in ZPD)
        # Medium ZPD valid: k ~0.15-0.70 (gap = 0.67-k)
        # Hard ZPD valid: k ~0.40+ (gap = 1.0-k)
        if k < 0.25 or frustration > 0.5:
            difficulty = 0  # Easy — low knowledge or frustrated
            hint = 1        # Give hint to reduce frustration
        elif k < 0.50:
            difficulty = 1  # Medium — building up
            hint = 1 if frustration > 0.3 else 0
        else:
            difficulty = 2  # Hard — approaching mastery
            hint = 0        # No hint — let them prove mastery
            
        # Encode: hint + 2 * (difficulty + NUM_DIFFICULTY * subtopic_idx)
        return hint + 2 * (difficulty + NUM_DIFFICULTY * best_idx)

# Stateful instance so focus persists across steps within an episode
_heuristic_instance = HeuristicAgent()

def get_heuristic_action(obs):
    """
    Legacy helper to get a heuristic action.
    """
    return _heuristic_instance.act(obs)
