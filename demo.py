from env.tutor_env import TutorEnv
from env.student_simulator import NUM_SUBTOPICS, NUM_DIFFICULTY
from models import TutorAction
from agents import HeuristicAgent

# Optional PPO
try:
    from agents import PPOAgent
    USE_PPO = True
except ImportError:
    USE_PPO = False

def decode_action(action_idx: int) -> TutorAction:
    """Convert flat action index → TutorAction(subtopic, difficulty, hint)."""
    hint = action_idx % 2
    remaining = action_idx // 2
    difficulty = remaining % NUM_DIFFICULTY
    subtopic_idx = remaining // NUM_DIFFICULTY
    return TutorAction(
        subtopic_index=subtopic_idx,
        difficulty_level=difficulty,
        hint_given=bool(hint),
    )

def run_demo():
    print("=" * 50)
    print("   COGNITIVE TUTOR DEMO")
    print("=" * 50)

    # Initialize environment
    env = TutorEnv()
    obs_raw, _ = env.reset()
    
    # Simple observation conversion for agents (since they expect knowledge_levels list)
    # The current obs from env.reset() is a TutorObservation object.
    obs = obs_raw.knowledge_levels + obs_raw.has_ever_mastered + [obs_raw.steps_remaining/90.0] + [0.0]

    # Choose agent
    agent = None
    if USE_PPO:
        try:
            agent = PPOAgent("cognitive-tutor-env/checkpoints/ppo_model")
            print("Using PPO Agent\n")
        except:
            print("PPO model not found → using Heuristic Agent\n")
    
    if agent is None:
        agent = HeuristicAgent()
        print("Using Heuristic Agent\n")

    step = 0
    done = False

    while not done:
        step += 1

        # 1. Get action from agent
        action_idx = agent.act(obs)
        
        # 2. Convert to TutorAction
        action = decode_action(int(action_idx))
        
        # 3. Step the environment
        obs_raw, reward, done, info = env.step(action)
        
        # 4. Update observation for next step
        obs = obs_raw.knowledge_levels + obs_raw.has_ever_mastered + [obs_raw.steps_remaining/90.0] + [info.get('frustration_level', 0.0)]

        print(f"\nStep {step}")
        print("-" * 30)
        print(f"Subtopic       : {info['subtopic']}")
        print(f"Difficulty     : {info['difficulty']}")
        print(f"Hint Given     : {info.get('hint_given', False)}")
        print(f"Correct        : {info.get('correct', False)}")
        print(f"Learning Gain  : {round(info['learning_gain'], 4)}")
        print(f"Frustrated     : {info['frustrated']}")
        print(f"Reward         : {round(reward, 3)}")

        if done:
            print("\n" + "=" * 50)
            print("Episode Finished")
            print("=" * 50)
            print(f"All Mastered   : {info.get('all_mastered', False)}")
            print(f"Total Steps    : {info.get('steps_taken', step)}")
            print("=" * 50)

if __name__ == "__main__":
    run_demo()