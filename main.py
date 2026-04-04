from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from train import train
# from app import app
# from models import TutorAction, TutorObservation, TutorState
# from my_environment import TutorEnv

# This file imports all the required components from the project.

api_app = FastAPI()

# Add CORS Middleware to allow your frontend HTML/JS to connect
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Setup to accept requests from any origin for development
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainConfig(BaseModel):
    total_timesteps: int = 1000

# get for read
# put for update
# post for create
# delete for delete

@api_app.get("/") #slash for using homepage
def greet():
    return "welcome to my RL environment"

@api_app.post("/api/train")
def start_training(config: TrainConfig):
    """
    Runs PPO training synchronously and returns the full evaluation
    results to the frontend once training completes.
    """
    results = train(total_timesteps=config.total_timesteps)
    return {
        "status": "success",
        "message": f"Training complete for {config.total_timesteps} timesteps!",
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run("main:api_app", host="127.0.0.1", port=8000, reload=True)
