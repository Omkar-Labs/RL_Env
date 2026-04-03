from openenv.core.env_server import create_web_interface_app
from my_environment import TutorEnv
from models import TutorAction, TutorObservation


app = create_web_interface_app(TutorEnv, TutorAction, TutorObservation)