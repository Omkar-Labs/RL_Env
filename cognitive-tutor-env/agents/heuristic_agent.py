# heuristic_agent.py
import numpy as np
from env.student_simulator import NUM_SUBTOPICS, NUM_DIFFICULTY

class HeuristicAgent:
    """
    Focuses on the subtopic with the absolute lowest knowledge using simple rules.
    """
    def __init__(self):
        pass

    def act(self, obs):
        """
        Calculates the best action based on the state's knowledge levels.
        """
        # obs contains: [knowledge_levels, has_ever_mastered, steps_norm, frustration]
        # knowledge_levels are the first NUM_SUBTOPICS elements
        knowledge_levels = obs[:NUM_SUBTOPICS]
        subtopic_idx = int(np.argmin(knowledge_levels))
        k = knowledge_levels[subtopic_idx]
        
        # Threshold rules for difficulty and hints
        if k < 0.35:
            difficulty = 0  # Easy
            hint = 1        # Give hint
        elif k < 0.65:
            difficulty = 1  # Medium
            hint = 0        # No hint
        else:
            difficulty = 2  # Hard
            hint = 0        # No hint
            
        # Re-encode back into the flat action_space integer (hint + 2 * (diff + 2 * sub_idx))
        return hint + 2 * (difficulty + NUM_DIFFICULTY * subtopic_idx)

def get_heuristic_action(obs):
    """
    Legacy helper to get a heuristic action.
    """
    return HeuristicAgent().act(obs)
