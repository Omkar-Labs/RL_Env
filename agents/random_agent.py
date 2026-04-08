# random_agent.py

class RandomAgent:
    """
    Simply returns a random action from the given action space.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()

def get_random_action(action_space):
    """
    Legacy helper to get a random action.
    """
    return action_space.sample()
