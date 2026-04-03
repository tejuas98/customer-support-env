from uuid import uuid4
import random
import re

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import CustomerSupportAction, CustomerSupportObservation
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import CustomerSupportAction, CustomerSupportObservation
    except (ImportError, ModuleNotFoundError):
        from customer_support.models import CustomerSupportAction, CustomerSupportObservation


class CustomerSupportEnvironment(Environment):
    """
    Multi-tier Customer Support RL environment with procedural generation.

    Three difficulty tiers:
      easy   - Damaged product refund. Rewards empathy + identity reference + refund offer.
      medium - Software crash. Rewards OS clarification before solution.
      hard   - Billing dispute. Rewards empathy + exact amount + escalation in sequence.

    Reward range: 0.0 to 1.0 (continuous, partial rewards at each step).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Sam", "Drew"]
    PRODUCTS = [
        "Coffee Maker", "Mechanical Keyboard", "Smartwatch",
        "Running Shoes", "Wireless Headphones", "Desk Lamp", "Backpack"
    ]
    SOFTWARE = [
        "Video rendering suite", "Cloud storage sync app",
        "Photo editing software", "Code compiler", "PDF editor"
    ]
    OS_OPTIONS = ["Windows 11", "macOS Sonoma", "Ubuntu 22.04"]

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_tier = "easy"
        self.max_turns = 6
        self.current_reward = 0.0
        self.task_state = {}
        self.context = {}

    def reset(self) -> CustomerSupportObservation:
        """Reset the environment and procedurally generate a new customer scenario."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_reward = 0.0
        self.task_state = {
            "easy_apologized": False,
            "easy_referenced_identity": False,
            "medium_asked_os": False,
            "hard_empathized": False,
            "hard_compensated": False,
        }

        self.context = {
            "name": random.choice(self.NAMES),
            "order_id": f"#{random.randint(10000, 99999)}",
            "product": random.choice(self.PRODUCTS),
            "software": random.choice(self.SOFTWARE),
            "os": random.choice(self.OS_OPTIONS),
            "overcharge_amt": random.randint(50, 500),
        }

        tiers = ["easy", "medium", "hard"]
        if hasattr(self, "forced_tier"):
            self.task_tier = self.forced_tier
        else:
            self.task_tier = random.choice(tiers)

        customer_reply = self._build_opening_message()
        return CustomerSupportObservation(
            customer_reply=customer_reply,
            task_tier=self.task_tier,
            done=False,
            reward=0.0,
        )

    def _build_opening_message(self) -> str:
        if self.task_tier == "easy":
            return (
                f"Hi, I'm {self.context['name']}. My {self.context['product']} "
                f"(Order {self.context['order_id']}) arrived completely smashed. "
                "I want a refund immediately."
            )
        elif self.task_tier == "medium":
            return (
                f"Support, my {self.context['software']} keeps crashing immediately "
                "on launch since this morning. I have a deadline today!"
            )
        else:
            return (
                f"THIS IS UNACCEPTABLE! You double-charged my credit card by "
                f"${self.context['overcharge_amt']} this month. I am furious. "
                "Get me a manager NOW."
            )

    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:  # type: ignore[override]
        """Process one agent turn and update the grader state."""
        self._state.step_count += 1
        agent_msg = action.message.lower()

        customer_reply = ""
        done = False

        if self.task_tier == "easy":
            customer_reply, done = self._process_easy(agent_msg)
        elif self.task_tier == "medium":
            customer_reply, done = self._process_medium(agent_msg)
        elif self.task_tier == "hard":
            customer_reply, done = self._process_hard(agent_msg)

        # Timeout: agent failed to resolve within max turns
        if self._state.step_count >= self.max_turns and not done:
            done = True
            customer_reply = "This is taking too long. I am filing a dispute with my bank. Goodbye."
            # Partial credit is preserved; do not zero it out — diversity matters

        return CustomerSupportObservation(
            customer_reply=customer_reply,
            task_tier=self.task_tier,
            done=done,
            reward=round(self.current_reward, 2),
        )

    def _process_easy(self, msg: str):
        """
        Easy tier grader: 3-step reward progression.
          +0.2 for apology
          +0.2 for referencing name or order ID
          +0.6 for offering refund (capped at 1.0)
        """
        done = False
        reply = "I'm still waiting to hear how you'll resolve this."

        apology_words = ["sorry", "apologize", "apologies", "regret", "sincerely"]
        if any(w in msg for w in apology_words):
            if not self.task_state["easy_apologized"]:
                self.current_reward += 0.2
                self.task_state["easy_apologized"] = True
                reply = "I appreciate the apology — but what about my refund?"

        order_id_str = self.context["order_id"].lstrip("#")
        name_lower = self.context["name"].lower()
        if order_id_str in msg or name_lower in msg:
            if not self.task_state["easy_referenced_identity"]:
                self.current_reward = min(self.current_reward + 0.2, 1.0)
                self.task_state["easy_referenced_identity"] = True

        if any(w in msg for w in ["refund", "replacement", "reimburse", "return your money", "money back"]):
            if self.task_state["easy_apologized"]:
                self.current_reward = min(self.current_reward + 0.6, 1.0)
            else:
                self.current_reward = max(self.current_reward, 0.5)
            reply = "Perfect — a full refund is exactly what I needed. Thank you!"
            done = True

        return reply, done

    def _process_medium(self, msg: str):
        """
        Medium tier grader: 2-step reward progression.
          +0.5 for asking about OS/platform before giving a solution
          +0.5 for providing a fix after OS is known (total = 1.0)
        """
        done = False
        reply = "Are you going to help me fix this crash or not?"

        os_inquiry_words = ["os", "operating system", "windows", "mac", "linux", "platform", "system", "version"]
        fix_words = ["driver", "update", "reinstall", "patch", "reboot", "restart", "clear cache", "fresh install"]

        if not self.task_state["medium_asked_os"]:
            if any(w in msg for w in os_inquiry_words):
                self.current_reward = 0.5
                self.task_state["medium_asked_os"] = True
                reply = f"I'm running {self.context['os']}. Does that help?"
            elif any(w in msg for w in fix_words):
                # Penalize giving a solution without knowing the OS
                self.current_reward = max(self.current_reward, 0.1)
                reply = "You're giving me solutions without even knowing what system I'm on!"
        else:
            if any(w in msg for w in fix_words):
                self.current_reward = 1.0
                reply = "I followed your steps and the software is working again. Thank you!"
                done = True

        return reply, done

    def _process_hard(self, msg: str):
        """
        Hard tier grader: 3-step sequential reward.
          +0.3 for empathy
          +0.4 for referencing the exact overcharge amount AND offering a refund
          +0.3 for escalating to manager AFTER the above two steps (total = 1.0)
        """
        done = False
        reply = "Are you even listening? I SAID I want a manager!"

        empathy_words = ["understand", "frustrating", "sorry", "apologize", "completely understand", "sincerely"]
        refund_words = ["refund", "credit", "return", "reimburse", "reverse the charge"]
        escalation_words = ["manager", "supervisor", "escalate", "transfer you", "higher up"]

        if not self.task_state["hard_empathized"]:
            if any(w in msg for w in empathy_words):
                self.current_reward = min(self.current_reward + 0.3, 1.0)
                self.task_state["hard_empathized"] = True
                reply = "Words are cheap. What are you actually doing about my money?"

        if not self.task_state["hard_compensated"]:
            amount_mentioned = str(self.context["overcharge_amt"]) in msg
            refund_mentioned = any(kw in msg for kw in refund_words)
            if amount_mentioned and refund_mentioned:
                self.current_reward = min(self.current_reward + 0.4, 1.0)
                self.task_state["hard_compensated"] = True
                reply = "Fine, the refund is a start. But I still want to speak to a manager!"
            elif amount_mentioned:
                self.current_reward = min(self.current_reward + 0.1, 1.0)
                reply = "Yes that's the wrong amount. What are you going to do about it?"

        if any(w in msg for w in escalation_words):
            if self.task_state["hard_empathized"] and self.task_state["hard_compensated"]:
                self.current_reward = 1.0
            else:
                # Partial credit for escalation without proper prep
                self.current_reward = min(self.current_reward + 0.2, 0.8)
            reply = "It's about time. Put them on the line."
            done = True

        return reply, done

    @property
    def state(self) -> State:
        return self._state
