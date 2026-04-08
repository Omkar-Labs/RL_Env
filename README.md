
# 🚀 Cognitive Tutor RL Environment (OpenEnv)

An AI-powered tutoring simulation environment built using reinforcement learning principles. This environment models realistic student learning behavior and challenges agents to act as an intelligent tutor that maximizes student mastery while minimizing frustration.

---

## 🎯 Problem Statement

Design an environment where an AI tutor:
1. **Selects what to teach** (subtopic)
2. **Chooses difficulty level** (easy → hard)
3. **Decides whether to provide hints**

### The Goal
*   🧠 **Maximize** student learning and mastery
*   😌 **Minimize** frustration
*   ⚡ **Optimize** teaching efficiency

---

## 🌍 Real-World Relevance

Unlike toy environments, this simulates real educational systems:
*   **Adaptive learning platforms** (like Duolingo, Khan Academy)
*   **Personalized tutoring systems**
*   **AI-driven education assistants**

### 🧠 Core Idea
We simulate a student with cognitive dynamics where:
*   Knowledge evolves over time
*   Forgetting occurs (decay)
*   Frustration impacts performance
*   Hints influence learning

**The agent must discover: “How to teach optimally like a human tutor”**

---

## ⚙️ Environment Design

### 📌 Action Space (`TutorAction`)
*   `subtopic_index`: int — which concept to teach
*   `difficulty_level`: int — 0 = Easy, 1 = Medium, 2 = Hard
*   `hint_given`: bool — whether to provide hint

### 📌 Observation Space (`TutorObservation`)
*   `knowledge_levels`: list[float] — per-subtopic knowledge [0–1]
*   `has_ever_mastered`: list[bool] — mastery history
*   `steps_remaining`: int

### 📌 State (Hidden)
*   `student_knowledge`: dict[str, float]
*   `frustration`: float
*   `hint_streak`: dict
*   `learning_decay`

---

## 🎓 Student Simulator (Key Innovation)

Our student model includes:

*   ✅ **Knowledge Dynamics**: Learning depends on the difficulty vs. ability gap. Strongest learning occurs in the **ZPD (Zone of Proximal Development)**.
*   😤 **Frustration Modeling**: Content that is too hard increases frustration. High frustration reduces student accuracy and learning efficiency.
*   🧠 **Memory & Forgetting**: Unpracticed topics decay over time. Mastered concepts decay significantly slower.
*   🔁 **Anti-Exploitation**: Prevents reward farming on mastered topics by tracking the highest knowledge achieved.

---

## 🏆 Reward Function (0.0 – 1.0)

Designed for dense and stable reinforcement learning.

### ✅ Positive Signals
*   **Learning progress**: reward for knowledge gain
*   **Correct responses**: bonus for student success
*   **Optimal Teaching Zone**: Extra reward for staying within the ZPD
*   **First-time mastery**: high signal for completing a subtopic

### ❌ Penalties
*   **Student frustration**: penalty for overwhelming the student
*   **Repetitive hinting**: penalty for over-assisting
*   **Poor difficulty selection**: penalty for "too easy" or "too hard" tasks
*   **Time inefficiency**: constant step penalty

---

## 📊 Tasks (Agent Graders)

We define three difficulty tiers to evaluate tutor performance:

| Task | Description | Goal |
| :--- | :--- | :--- |
| 🟢 **Easy** | Basic tutoring | Achieve partial learning |
| 🟡 **Medium** | Balanced teaching | Improve multiple topics |
| 🔴 **Hard** | Full curriculum mastery | Master all subtopics |

> Each task returns a normalized reward ∈ [0.0, 1.0].

---

## 🤖 Baseline Agents

We provide three baseline implementations:

1.  🎲 **Random Agent**: Picks random actions. Results in poor learning and high frustration.
2.  📏 **Heuristic Agent**: Follows simple educational rules (e.g., avoid hard topics early). Shows moderate performance.
3.  🧠 **PPO Agent**: Trained using RL. Learns difficulty adaptation, topic scheduling, and frustration management.

### 📈 Example Results

| Agent | Avg Reward | Mastery | Frustration |
| :--- | :--- | :--- | :--- |
| **Random** | Low | ❌ | High |
| **Heuristic** | Medium | ✅ | Medium |
| **PPO** | High | ✅✅ | Low |

---

## 🧪 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train PPO Agent
```bash
python train.py
```

### 3️⃣ Evaluate Agents
```bash
python test.py
```

### 4️⃣ Demo Run
```bash
python demo.py
```

---

## 🐳 Deployment & Setup

### Docker Setup
```bash
docker build -t tutor-env .
docker run -p 7860:7860 tutor-env
```

### 🌐 Hugging Face Spaces Deployment
Includes API-based interaction and visual monitoring of training:
*   Knowledge progression tracking
*   Reward curves
*   Mastery stats

---

## 📁 Project Structure

```text
cognitive-tutor-env/
│
├── env/
│   ├── tutor_env.py          # Main RL environment
│   ├── student_simulator.py  # Student cognitive logic
│   └── __init__.py
│
├── models/
│   ├── models.py             # TutorAction, Observation, State
│   └── __init__.py
│
├── agents/
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   ├── ppo_agent.py          # Trained model loader
│   └── __init__.py
│
├── checkpoints/              # Saved model weights
├── train.py                  # PPO training script
├── test.py                   # Multi-agent evaluation script
├── demo.py                   # Simple interaction demo
├── openenv.yaml              # OpenEnv configuration
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧩 OpenEnv Compliance ✅

*   ✔ **Real-world task**: Modeled after education systems
*   ✔ **API consistency**: `step()`, `reset()`, `state()` implemented
*   ✔ **Typed models**: Full Pydantic/Dataclass support
*   ✔ **Standardized Rewards**: Normalized range [0.0, 1.0]
*   ✔ **Graded challenges**: 3 tasks (easy → hard)
*   ✔ **Standard Baselines**: Baseline evaluation scripts included
*   ✔ **Deployment Ready**: Docker + Hugging Face compatible

---

## 🚀 What Makes This Unique?

*   **🧠 Cognitive Modeling**: Not just Q&A — simulates learning psychology, frustration, and forgetting.
*   **🎯 Pedagogical Intelligence**: Agent learns when to challenge, when to support, and when to review.
*   **⚖️ Balanced RL Design**: Dense rewards, anti-exploitation mechanisms, and realistic constraints.

### 🔮 Future Extensions
*   LLM-powered student responses for natural feedback
*   Multi-student classroom simulations
*   Personalized curriculum generation based on student history
*   Real-world dataset integration

---

## 👨‍💻 Team

Built for **Meta AI Hackathon x Scaler School of Technology**

> **📌 Final Note**: This project demonstrates how reinforcement learning can move beyond games and into human-centric AI systems like education.
