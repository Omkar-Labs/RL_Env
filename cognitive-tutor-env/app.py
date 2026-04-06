import gradio as gr
from env.tutor_env import TutorEnv
from models.models import TutorAction

env = TutorEnv()
obs, _ = env.reset()

def run_episode(steps=50):
    env.reset()
    total_reward = 0
    
    for _ in range(steps):
        action = TutorAction(
            subtopic_index=0,
            difficulty_level=1,
            hint_given=False
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return f"Total Reward: {total_reward:.2f}", str(env.state())

demo = gr.Interface(
    fn=run_episode,
    inputs=gr.Slider(1, 200, value=50, label="Steps"),
    outputs=["text", "text"],
    title="Cognitive Tutor RL Environment",
    description="Train and evaluate a tutoring AI agent"
)

demo.launch()