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
    Multi-step Customer Support environment with Easy, Medium, and Hard tiers.
    Uses randomized procedural generation for items, names, and IDs.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_tier = "easy"
        self.max_turns = 6
        self.current_reward = 0.0
        self.task_state = {}
        self.context = {}

    def reset(self) -> CustomerSupportObservation:
        """
        Resets environment and generates a new customer problem.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_reward = 0.0
        self.task_state = {
            "easy_apologized": False,
            "easy_referenced_identity": False,
            "medium_asked_os": False,
            "hard_empathized": False,
            "hard_compensated": False
        }

        names = ["Alex", "Jordan", "Taylor", "Casey", "Riley"]
        products = ["Coffee Maker", "Mechanical Keyboard", "Smartwatch", "Running Shoes", "Backpack"]
        software = ["Video rendering suite", "Cloud storage sync", "Photo editor", "Code compiler"]
        os_options = ["Windows 11", "macOS Sonoma", "Ubuntu Desktop"]
        
        self.context = {
            "name": random.choice(names),
            "order_id": f"#{random.randint(10000, 99999)}",
            "product": random.choice(products),
            "software": random.choice(software),
            "os": random.choice(os_options),
            "overcharge_amt": random.randint(150, 500)
        }

        tiers = ["easy", "medium", "hard"]
        if hasattr(self, 'forced_tier'):
            self.task_tier = self.forced_tier
        else:
            self.task_tier = random.choice(tiers)

        if self.task_tier == "easy":
            customer_reply = f"Hi, I'm {self.context['name']}. The {self.context['product']} I ordered (Order {self.context['order_id']}) arrived completely smashed. I want my money back."
        elif self.task_tier == "medium":
            customer_reply = f"Support, my {self.context['software']} keeps crashing immediately on launch since yesterday. I need to get back to work!"
        else:
            customer_reply = f"THIS IS UNACCEPTABLE! You guys double charged my credit card by ${self.context['overcharge_amt']} this month! I am furious. Get me a manager immediately."

        return CustomerSupportObservation(
            customer_reply=customer_reply,
            task_tier=self.task_tier,
            done=False,
            reward=0.0
        )

    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:  # type: ignore[override]
        """
        Processes agent messages and updates the grader state.
        """
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

        if self._state.step_count >= self.max_turns and not done:
            done = True
            customer_reply = "This is taking too long. I'm filing a dispute with my bank. Goodbye."

        return CustomerSupportObservation(
            customer_reply=customer_reply,
            task_tier=self.task_tier,
            done=done,
            reward=round(self.current_reward, 2)
        )

    def _process_easy(self, msg: str):
        done = False
        reply = "I'm still waiting to hear how you'll fix this broken item."
        
        if any(w in msg for w in ["sorry", "apologize", "apologies"]):
            if not self.task_state["easy_apologized"]:
                self.current_reward += 0.2
                self.task_state["easy_apologized"] = True
                reply = "I appreciate the apology, but what about the refund?"

        if str(self.context["order_id"]).lower() in msg or self.context["name"].lower() in msg:
             if not self.task_state["easy_referenced_identity"]:
                  self.current_reward += 0.2
                  self.task_state["easy_referenced_identity"] = True

        if "refund" in msg or "replacement" in msg:
            if self.task_state["easy_apologized"]:
                self.current_reward = min(self.current_reward + 0.6, 1.0)
            else:
                self.current_reward = 0.5 
            reply = "Great, a refund sounds perfect. Thank you for your fast help."
            done = True
            
        return reply, done

    def _process_medium(self, msg: str):
        done = False
        reply = "So are you going to help me fix the crash?"
        
        if not self.task_state["medium_asked_os"]:
            if any(w in msg for w in ["os", "operating system", "windows", "mac", "linux", "platform"]):
                self.current_reward += 0.5
                self.task_state["medium_asked_os"] = True
                reply = f"I am running {self.context['os']}."
            elif any(w in msg for w in ["update", "reinstall", "cache"]):
                self.current_reward += 0.1 
                reply = "I shouldn't have to reinstall without you knowing what system I'm on!"
        else:
            if any(w in msg for w in ["driver", "update", "reinstall", "patch", "terminal", "restart"]):
                self.current_reward = 1.0
                reply = "Okay, updating the software as you suggested fixed it! Thanks."
                done = True

        return reply, done

    def _process_hard(self, msg: str):
        done = False
        reply = "Are you a robot? Did you hear me say I want a manager!?"
        
        if not self.task_state["hard_empathized"]:
            if any(w in msg for w in ["understand", "frustrating", "sorry", "apologize"]):
                self.current_reward += 0.3
                self.task_state["hard_empathized"] = True
                reply = "Don't just apologize, what are you doing about my money!"
        
        if not self.task_state["hard_compensated"]:
            if str(self.context["overcharge_amt"]) in msg and any(kw in msg for kw in ["refund", "credit", "return", "waive"]):
                self.current_reward = min(self.current_reward + 0.4, 0.7)
                self.task_state["hard_compensated"] = True
                reply = "A refund is the bare minimum! I still demand to speak with your manager."
                
        if any(w in msg for w in ["manager", "supervisor", "escalate", "transfer"]):
            if self.task_state["hard_empathized"] and self.task_state["hard_compensated"]:
                self.current_reward = 1.0
            else:
                self.current_reward = min(self.current_reward + 0.3, 0.8)
            reply = "Finally. Put them on."
            done = True
            
        return reply, done

    @property
    def state(self) -> State:
        return self._state
