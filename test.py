import numpy as np
import os
from env.student_simulator import NUM_SUBTOPICS, NUM_DIFFICULTY
from train import TutorGymEnv
from agents import get_random_action, get_heuristic_action, get_ppo_predict_fn

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
            
    avg_reward = np.mean(total_rewards)
    mastery_pct = (mastery_hits / num_episodes) * 100
    
    print("-" * 50)
    print(f"  Results for: {agent_name}")
    print("-" * 50)
    print(f"  Avg Reward:         {avg_reward:.3f}")
    print(f"  Max Reward:         {np.max(total_rewards):.3f}")
    print(f"  Mastery Rate:       {mastery_pct:.1f}%")
    print(f"  Total Frustrations: {total_frustrations}")
    print("-" * 50)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("   COGNITIVE TUTOR AGENT SHOWDOWN")
    print("="*50)

    # 1. Random Agent
    env = TutorGymEnv()
    evaluate_custom("Random Agent", lambda obs: get_random_action(env.action_space), num_episodes=100)
    
    # 2. Heuristic Agent
    evaluate_custom("Simple Heuristic Agent", get_heuristic_action, num_episodes=100)
    
    # 3. Trained PPO Agent
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, "checkpoints", "ppo_model.zip")):
        ppo_fn = get_ppo_predict_fn(os.path.join(script_dir, "checkpoints", "ppo_model"))
        evaluate_custom("Trained PPO Agent", ppo_fn, num_episodes=100)
    else:
        print("\n[!] Skipping PPO Agent: Model not found at checkpoints/ppo_model.")
