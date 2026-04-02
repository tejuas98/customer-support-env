from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class CustomerSupportAction(Action):
    """Message sent by the support agent."""
    message: str = Field(..., description="Agent message text")

class CustomerSupportObservation(Observation):
    """Observation containing customer reply and current grader state."""
    customer_reply: str = Field(default="", description="Customer response text")
    task_tier: str = Field(default="easy", description="difficulty: easy, medium, hard")
    done: bool = Field(default=False, description="Whether episode is finished")
    reward: float = Field(default=0.0, description="Grader score (0.0 to 1.0)")
