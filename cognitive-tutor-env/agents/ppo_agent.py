# ppo_agent.py
from stable_baselines3 import PPO

class PPOAgent:
    """
    Loads a PPO model and returns a prediction function that returns the action.
    """
    def __init__(self, model_path):
        try:
            self.model = PPO.load(model_path)
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            raise e

    def act(self, obs):
        """
        Calculates the best action based on the state's knowledge levels.
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

def get_ppo_predict_fn(model_path):
    """
    Legacy helper to get a PPO prediction function.
    """
    return PPOAgent(model_path).act
