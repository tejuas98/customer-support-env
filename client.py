# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Customer Support Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CustomerSupportAction, CustomerSupportObservation

class CustomerSupportEnv(
    EnvClient[CustomerSupportAction, CustomerSupportObservation, State]
):
    """
    Client for the Customer Support Environment.
    """

    def _step_payload(self, action: CustomerSupportAction) -> Dict[str, Any]:
        """Convert CustomerSupportAction to JSON payload."""
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CustomerSupportObservation]:
        """Parse server response into StepResult[CustomerSupportObservation]."""
        obs_data = payload.get("observation", {})
        observation = CustomerSupportObservation(
            customer_reply=obs_data.get("customer_reply", ""),
            task_tier=obs_data.get("task_tier", "easy"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
