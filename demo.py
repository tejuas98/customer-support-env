import time
import sys
import random
from server.customer_support_environment import CustomerSupportEnvironment
from models import CustomerSupportAction

def slow_print(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.002)
    print()

def run_stress_test():
    print("OPENENV STRESS TEST (V1.1)")
    
    env = CustomerSupportEnvironment()
    tiers = ["easy", "medium", "hard"]
    
    for tier in tiers:
        print(f"\n--- Tier: {tier.upper()} ---")
        env.forced_tier = tier
        obs = env.reset()
        
        slow_print(f"Customer: {obs.customer_reply}")
        
        if tier == "easy":
            responses = [
                 f"I'm so sorry, {env.context['name']}. I'll help you with Order {env.context['order_id']}.",
                 "I'm processing a refund for you right now."
            ]
        elif tier == "medium":
            responses = [
                 "Which operating system are you using on your computer?",
                 "Please update your graphics drivers to fix the crash."
            ]
        else:
            responses = [
                 "I am so sorry for the frustration this overcharge caused.",
                 f"I'm refunding the {env.context['overcharge_amt']} immediately.",
                 "I'm escalating your case to a supervisor now."
            ]

        for i, msg in enumerate(responses):
            slow_print(f"Agent ({i+1}): {msg}")
            obs = env.step(CustomerSupportAction(message=msg))
            print(f"   Reward: {obs.reward}")
            if obs.customer_reply:
                slow_print(f"Customer: {obs.customer_reply}")

        if obs.reward >= 1.0:
            print(f"OK: Tier {tier.upper()} passed.")
        else:
            print(f"FAIL: Tier {tier.upper()} failed. Score: {obs.reward}")
            
    print("\nFINAL QA CHECK: READY")

if __name__ == "__main__":
    run_stress_test()
