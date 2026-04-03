"""
Stress test for the Customer Support RL environment.
Validates all four difficulty tiers with deterministic agent messages.
"""

from server.customer_support_environment import CustomerSupportEnvironment
from models import CustomerSupportAction


def run_tier_steps(env, tier, step_fns):
    """
    Run a tier where step messages can be callables (for dynamic context).
    step_fns: list of str or callable(context) -> str
    """
    env.forced_tier = tier
    obs = env.reset()
    ctx = env.context
    print(f"\n--- Tier: {tier.upper()} ---")
    print(f"Customer: {obs.customer_reply}")
    for i, fn in enumerate(step_fns, 1):
        msg = fn(ctx) if callable(fn) else fn
        action = CustomerSupportAction(message=msg)
        obs = env.step(action)
        print(f"Agent ({i}): {msg}")
        print(f"   Reward: {obs.reward}")
        print(f"Customer: {obs.customer_reply}")
        if obs.done:
            break
    return obs.reward, obs.done


def main():
    env = CustomerSupportEnvironment()
    results = {}

    # Easy: apologize + reference order ID + offer refund (all in 2 turns)
    reward, done = run_tier_steps(env, "easy", [
        lambda ctx: f"I'm so sorry, {ctx['name']}. Let me look into order {ctx['order_id']} right away.",
        "I have processed a full refund for you immediately.",
    ])
    results["easy"] = (reward, done)
    assert reward == 1.0, f"EASY FAILED: reward={reward}"
    print("OK: Tier EASY passed.")

    # Medium: ask OS → provide fix
    reward, done = run_tier_steps(env, "medium", [
        "Could you tell me which operating system you are using?",
        "Please try reinstalling the application on your system.",
    ])
    results["medium"] = (reward, done)
    assert reward == 1.0, f"MEDIUM FAILED: reward={reward}"
    print("OK: Tier MEDIUM passed.")

    # Hard: empathize → quote exact amount + refund → escalate
    reward, done = run_tier_steps(env, "hard", [
        "I completely understand how frustrating this is and I sincerely apologize.",
        lambda ctx: f"I will immediately refund the ${ctx['overcharge_amt']} overcharge to your account.",
        "I am transferring you to a manager right now.",
    ])
    results["hard"] = (reward, done)
    assert reward == 1.0, f"HARD FAILED: reward={reward}"
    print("OK: Tier HARD passed.")

    # Expert: diagnose reason → make targeted offer → resolve
    reward, done = run_tier_steps(env, "expert", [
        "I am sorry to hear that. Could you tell me the reason you are thinking of leaving?",
        lambda ctx: (
            f"I completely understand. What if we offered you a "
            f"{ctx['discount_pct']}% discount on your next 3 months?"
        ),
        "I have processed your cancellation. We hope to serve you again.",
    ])
    results["expert"] = (reward, done)
    assert reward == 1.0, f"EXPERT FAILED: reward={reward}"
    print("OK: Tier EXPERT passed.")

    print("\nFINAL QA CHECK: READY")
    for tier, (r, d) in results.items():
        print(f"  {tier}: {r:.2f} done={d}")


if __name__ == "__main__":
    main()
