from uuid import uuid4
import random

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
    Multi-tier Customer Support RL environment with procedural generation
    and curriculum-aware difficulty progression.

    Four difficulty tiers (ordered by complexity):
      easy   - Damaged product: empathy + identity reference + refund offer.
      medium - Software crash: OS diagnosis before solution.
      hard   - Billing dispute: empathy + exact amount + manager escalation.
      expert - Subscription cancellation: diagnose reason + retention offer + resolution.

    Reward range: 0.0 to 1.0 (continuous, partial rewards on each meaningful step).

    Curriculum mode: set `env.curriculum = True` to auto-advance difficulty
    based on rolling success rate. The environment tracks performance and
    matches difficulty to the agent's current capability level.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Sam", "Drew", "Avery", "Quinn"]
    PRODUCTS = [
        "Coffee Maker", "Mechanical Keyboard", "Smartwatch",
        "Running Shoes", "Wireless Headphones", "Desk Lamp",
        "Backpack", "Bluetooth Speaker", "Air Purifier"
    ]
    SOFTWARE = [
        "Video rendering suite", "Cloud storage sync app",
        "Photo editing software", "Code compiler",
        "PDF editor", "Audio production software"
    ]
    OS_OPTIONS = ["Windows 11", "macOS Sonoma", "Ubuntu 22.04"]
    SUBSCRIPTION_PLANS = ["Pro", "Business", "Premium", "Team"]
    CANCEL_REASONS = [
        "too expensive", "not using it enough",
        "switching to a competitor", "missing a key feature"
    ]
    DISCOUNT_OFFERS = [20, 30, 40, 50]

    TIERS = ["easy", "medium", "hard", "expert"]

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_tier = "easy"
        self.max_turns = 8
        self.current_reward = 0.0
        self.task_state = {}
        self.context = {}
        # Curriculum mode: auto-advances difficulty based on rolling success rate
        self.curriculum = False
        self._episode_scores = []
        self._curriculum_tier_idx = 0

    def reset(self) -> CustomerSupportObservation:
        """Reset environment and procedurally generate a new customer scenario."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Track previous episode reward for curriculum adjustment
        if self.current_reward > 0 and self._episode_scores is not None:
            self._episode_scores.append(self.current_reward)
            # Advance tier once agent sustains >0.8 avg over last 3 episodes
            if self.curriculum and len(self._episode_scores) >= 3:
                recent_avg = sum(self._episode_scores[-3:]) / 3
                if recent_avg >= 0.8 and self._curriculum_tier_idx < len(self.TIERS) - 1:
                    self._curriculum_tier_idx += 1
                    self._episode_scores = []

        self.current_reward = 0.0
        self.task_state = {
            "easy_apologized": False,
            "easy_referenced_identity": False,
            "medium_asked_os": False,
            "hard_empathized": False,
            "hard_compensated": False,
            "expert_diagnosed_reason": False,
            "expert_made_offer": False,
        }

        self.context = {
            "name": random.choice(self.NAMES),
            "order_id": f"#{random.randint(10000, 99999)}",
            "product": random.choice(self.PRODUCTS),
            "software": random.choice(self.SOFTWARE),
            "os": random.choice(self.OS_OPTIONS),
            "overcharge_amt": random.randint(50, 500),
            "plan": random.choice(self.SUBSCRIPTION_PLANS),
            "cancel_reason": random.choice(self.CANCEL_REASONS),
            "discount_pct": random.choice(self.DISCOUNT_OFFERS),
            "months_subscribed": random.randint(3, 36),
            "retention_type": random.choice(["discount", "pause", "feature_highlight"]),
        }

        # Determine tier: forced > curriculum > random
        if hasattr(self, "forced_tier") and self.forced_tier in self.TIERS:
            self.task_tier = self.forced_tier
        elif self.curriculum:
            self.task_tier = self.TIERS[self._curriculum_tier_idx]
        else:
            self.task_tier = random.choice(self.TIERS)

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
        elif self.task_tier == "hard":
            return (
                f"THIS IS UNACCEPTABLE! You double-charged my credit card by "
                f"${self.context['overcharge_amt']} this month. I am furious. "
                "Get me a manager NOW."
            )
        else:  # expert
            return (
                f"I need to cancel my {self.context['plan']} subscription. "
                f"I've been a customer for {self.context['months_subscribed']} months "
                f"but it's just {self.context['cancel_reason']}. Please cancel it."
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
        elif self.task_tier == "expert":
            customer_reply, done = self._process_expert(agent_msg)

        if self._state.step_count >= self.max_turns and not done:
            done = True
            customer_reply = "I have been waiting too long. I am filing a formal complaint. Goodbye."

        return CustomerSupportObservation(
            customer_reply=customer_reply,
            task_tier=self.task_tier,
            done=done,
            reward=round(self.current_reward, 2),
        )

    def _process_easy(self, msg: str):
        """
        Easy tier: 3-step reward sequence.
          +0.2  apology detected
          +0.2  customer name or order ID referenced
          +0.6  refund or replacement offered (caps at 1.0)
        """
        done = False
        default_replies = [
            "I'm still waiting to hear how you will fix this.",
            "That's not very helpful. What about my smashed item?",
            "Are you going to help me with my order or just ignore me?",
            "I still need a refund or a replacement. Please focus.",
            "Can we get to the point please?"
        ]
        reply = random.choice(default_replies)

        apology_words = ["sorry", "apologize", "apologies", "regret", "sincerely sorry"]
        if any(w in msg for w in apology_words):
            if not self.task_state["easy_apologized"]:
                self.current_reward = min(self.current_reward + 0.2, 1.0)
                self.task_state["easy_apologized"] = True
                reply = "I appreciate the apology. But what about my money?"

        order_id_digits = self.context["order_id"].lstrip("#")
        name_lower = self.context["name"].lower()
        if order_id_digits in msg or name_lower in msg:
            if not self.task_state["easy_referenced_identity"]:
                self.current_reward = min(self.current_reward + 0.2, 1.0)
                self.task_state["easy_referenced_identity"] = True

        refund_words = ["refund", "replacement", "reimburse", "return your money", "money back", "new one"]
        if any(w in msg for w in refund_words):
            if self.task_state["easy_apologized"]:
                self.current_reward = min(self.current_reward + 0.6, 1.0)
            else:
                self.current_reward = max(self.current_reward, 0.5)
            reply = "Perfect — a full refund is exactly what I needed. Thank you!"
            done = True

        return reply, done

    def _process_medium(self, msg: str):
        """
        Medium tier: 2-step reward sequence.
          +0.5  OS clarification asked before giving a solution
          +0.5  technical fix provided after OS is known (total = 1.0)
        Penalty: giving a fix without knowing the OS caps reward at 0.1.
        """
        done = False
        default_replies = [
            "Are you going to help me fix this crash or not?",
            "I have a deadline today, please focus on the software issue.",
            "I'm still stuck with a non-responsive app.",
            "Can you help me with the crash?"
        ]
        reply = random.choice(default_replies)

        os_words = ["os", "operating system", "windows", "mac", "linux", "ubuntu", "platform", "system", "version"]
        fix_words = ["driver", "update", "reinstall", "patch", "reboot", "restart", "clear cache", "fresh install", "repair"]

        if not self.task_state["medium_asked_os"]:
            if any(w in msg for w in os_words):
                self.current_reward = 0.5
                self.task_state["medium_asked_os"] = True
                reply = f"I am running {self.context['os']}. Does that help?"
            elif any(w in msg for w in fix_words):
                self.current_reward = max(self.current_reward, 0.1)
                reply = "You are giving me fixes without even knowing what system I am on!"
        else:
            if any(w in msg for w in fix_words):
                self.current_reward = 1.0
                reply = "I followed your steps and the software is working again. Thank you!"
                done = True

        return reply, done

    def _process_hard(self, msg: str):
        """
        Hard tier: 3-step sequential reward.
          +0.3  empathy expressed
          +0.4  exact overcharge amount referenced + refund offered
          +0.3  escalation to manager (only full 1.0 if previous steps done)
        """
        done = False
        default_replies = [
            "Did you hear what I said? I want to speak to a MANAGER!",
            "Stop wasting my time and fix the overcharge.",
            "I am furious and I want this resolved now.",
            "Refunding the overcharge is the priority here."
        ]
        reply = random.choice(default_replies)

        empathy_words = ["understand", "frustrating", "sorry", "apologize", "completely understand", "terrible", "awful"]
        refund_words = ["refund", "credit", "return", "reimburse", "reverse the charge", "process the refund"]
        escalation_words = ["manager", "supervisor", "escalate", "transfer you", "higher up", "senior agent"]

        if not self.task_state["hard_empathized"]:
            if any(w in msg for w in empathy_words):
                self.current_reward = min(self.current_reward + 0.3, 1.0)
                self.task_state["hard_empathized"] = True
                reply = "Words are cheap. What are you doing about my money?"

        if not self.task_state["hard_compensated"]:
            amount_mentioned = str(self.context["overcharge_amt"]) in msg
            refund_mentioned = any(kw in msg for kw in refund_words)
            if amount_mentioned and refund_mentioned:
                self.current_reward = min(self.current_reward + 0.4, 1.0)
                self.task_state["hard_compensated"] = True
                reply = "A refund is the bare minimum. Now get me a manager!"
            elif amount_mentioned:
                self.current_reward = min(self.current_reward + 0.1, 1.0)
                reply = "Yes, that is the overcharge. What is your plan?"

        if any(w in msg for w in escalation_words):
            if self.task_state["hard_empathized"] and self.task_state["hard_compensated"]:
                self.current_reward = 1.0
            else:
                self.current_reward = min(self.current_reward + 0.2, 0.8)
            reply = "Finally. Put them on."
            done = True

        return reply, done

    def _process_expert(self, msg: str):
        """
        Expert tier: Subscription cancellation + retention.
          +0.25  Diagnosis: Ask why.
          +0.45  Retention Offer: 
                 - If reason is "too expensive": Reward "discount" or "lower plan".
                 - If reason is "missing feature": Reward "feature highlight" or "roadmap".
                 - If reason is "not using much": Reward "pause subscription".
          +0.3   Resolution: Finalize professionally.
        """
        done = False
        reply = "I just want to cancel. Can you please just process that?"

        reason_words = ["why", "reason", "what's the issue", "tell me more", "happened", "cause"]
        
        # Retention keywords based on context
        discount_words = ["discount", "lower price", "cheaper", f"{self.context['discount_pct']}%"]
        feature_words = ["roadmap", "upcoming", "feature", "new release", "actually we have"]
        pause_words = ["pause", "hold", "vacation mode", "stop for a while"]

        resolution_words = ["cancel", "done", "processed", "stay", "keep", "thanks for helping"]

        if not self.task_state["expert_diagnosed_reason"]:
            if any(w in msg for w in reason_words):
                self.current_reward = min(self.current_reward + 0.25, 1.0)
                self.task_state["expert_diagnosed_reason"] = True
                reply = (
                    f"Honestly, it is just {self.context['cancel_reason']}. "
                    "I do not see the value anymore."
                )
                return reply, done

        if self.task_state["expert_diagnosed_reason"] and not self.task_state["expert_made_offer"]:
            reason = self.context["cancel_reason"]
            offered_correctly = False
            
            if "expensive" in reason and any(w in msg for w in discount_words):
                offered_correctly = True
            elif "feature" in reason and any(w in msg for w in feature_words):
                offered_correctly = True
            elif "using" in reason and any(w in msg for w in pause_words):
                offered_correctly = True
            elif "competitor" in reason and (any(w in msg for w in discount_words) or any(w in msg for w in feature_words)):
                offered_correctly = True

            if offered_correctly:
                self.current_reward = min(self.current_reward + 0.45, 1.0)
                self.task_state["expert_made_offer"] = True
                reply = "That is actually a very good point. I didn't think of that option. What happens next?"
            elif any(w in msg for w in (discount_words + feature_words + pause_words)):
                 # Generic offer but not targeted
                 self.current_reward = min(self.current_reward + 0.2, 1.0)
                 self.task_state["expert_made_offer"] = True
                 reply = "I mean, that's fine, but it doesn't really address my concern. Can we just cancel?"

        if any(w in msg for w in resolution_words):
            if self.task_state["expert_made_offer"]:
                self.current_reward = 1.0
                reply = "You know what, you've been very helpful. Let's keep it for now. Thank you!"
            else:
                self.current_reward = min(self.current_reward + 0.1, 0.6)
                reply = "Okay, it's cancelled. A bit disappointed you didn't even try to keep me as a customer."
            done = True

        return reply, done

    @property
    def state(self) -> State:
        return self._state
