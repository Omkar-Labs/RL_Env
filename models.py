from openenv.core.env_server import Action, Observation, State



class TutorAction(Action):
    """Agent picks a subtopic index (0-8), difficulty level (0-2), and whether to hint."""
    subtopic_index: int = 0
    difficulty_level: int = 0       # 0=Easy, 1=Medium, 2=Hard
    hint_given: bool = False        # whether to provide a hint


class TutorObservation(Observation):
    """What the agent can see after each step."""
    knowledge_levels: list[float] = []   # 9 floats [0.0–1.0]
    steps_remaining: int = 0


class TutorState(State):
    """Full internal state (visible in OpenEnv web UI)."""
    student_knowledge: dict[str, float] = {}
    steps_taken: int = 0
    done: bool = False