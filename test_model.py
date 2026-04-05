import numpy as np
from stable_baselines3 import PPO
from train import TutorGymEnv

NUM_SUBTOPICS = 4
NUM_DIFFICULTY = 3

def evaluate_custom(agent_name, predict_fn, num_episodes=100):
    env = TutorGymEnv()
    total_rewards = []
    mastery_hits = 0
    total_frustrations = 0
    
    print(f"\nEvaluating {agent_name}...")
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
            if info.get("frustrated", False):
                total_frustrations += 1
                
        total_rewards.append(ep_reward)
        if info.get("all_mastered", False):
            mastery_hits += 1
            
    avg_reward = sum(total_rewards) / len(total_rewards)
    mastery_pct = (mastery_hits / num_episodes) * 100
    
    print("-" * 50)
    print(f"  Results for: {agent_name}")
    print("-" * 50)
    print(f"  Avg Reward:         {avg_reward:.3f}")
    print(f"  Max Reward:         {max(total_rewards):.3f}")
    print(f"  Mastery Rate:       {mastery_pct:.1f}%")
    print(f"  Total Frustrations: {total_frustrations}")
    print("-" * 50)


# ==========================================================
# 1. Random Agent
# ==========================================================
_dummy_env = TutorGymEnv()
def random_agent(obs):
    # Simply guesses an action completely randomly
    return _dummy_env.action_space.sample()


# ==========================================================
# 2. Simple Heuristic Agent
# ==========================================================
def heuristic_agent(obs):
    # Focuses on the subtopic with the absolute lowest knowledge
    knowledge_levels = obs[:NUM_SUBTOPICS]
    subtopic_idx = int(np.argmin(knowledge_levels))
    k = knowledge_levels[subtopic_idx]
    
    # Simple threshold rules for difficulty and hints
    if k < 0.35:
        difficulty = 0  
        hint = 1        
    elif k < 0.65:
        difficulty = 1  
        hint = 0        
    else:
        difficulty = 2  
        hint = 0        
        
    # Re-encode back into the action_space integer
    return hint + 2 * (difficulty + NUM_DIFFICULTY * subtopic_idx)


# ==========================================================
# 3. Trained Agent Execution
# ==========================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   COGNITIVE TUTOR AGENT SHOWDOWN")
    print("="*50)

    evaluate_custom("Random Agent", random_agent, num_episodes=100)
    
    evaluate_custom("Simple Heuristic Agent", heuristic_agent, num_episodes=100)
    
    try:
        model = PPO.load("ppo_cognitive_tutor")
        
        # PPO predict() returns a tuple (action, state), so we wrap it
        def ppo_agent(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
            
        evaluate_custom("Trained PPO Agent", ppo_agent, num_episodes=100)
        
    except Exception as e:
        print(f"\n[!] Could not load or evaluate Trained PPO model: {e}")
