from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class CustomerSupportAction(Action):
    """Message sent by the support agent to the customer."""
    message: str = Field(..., description="Agent message text")

class CustomerSupportObservation(Observation):
    """Observation returned by the environment after each step."""
    customer_reply: str = Field(default="", description="Customer response text")
    task_tier: str = Field(default="easy", description="Difficulty tier: easy, medium, hard, expert")
    done: bool = Field(default=False, description="Whether the episode is complete")
    reward: float = Field(default=0.0, description="Grader score from 0.0 to 1.0")
