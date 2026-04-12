"""
Inference script for the Customer Support OpenEnv environment.

Evaluates an LLM agent across all four difficulty tiers using the
OpenAI-compatible Hugging Face Inference Router.

Environment variables:
    API_BASE_URL:       LLM API endpoint. Defaults to HF router.
    MODEL_NAME:         LLM model. Defaults to Qwen/Qwen2.5-72B-Instruct.
    HF_TOKEN:           Hugging Face token. No default — must be set explicitly.
    LOCAL_IMAGE_NAME:   Optional. Docker image for local testing via from_docker_image().
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI
from client import CustomerSupportEnv
from models import CustomerSupportAction

# Defaults for API_BASE_URL and MODEL_NAME only. HF_TOKEN must never have a default.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — only required when using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SYSTEM_PROMPT = (
    "You are a professional customer support agent. Resolve the customer's issue "
    "efficiently and empathetically in as few turns as possible.\n\n"
    "Rules:\n"
    "1. DAMAGED PRODUCT: Apologize sincerely, reference the customer name or order ID, "
    "and proactively offer a full refund or replacement.\n"
    "2. SOFTWARE CRASH: ALWAYS ask what operating system the customer uses FIRST, "
    "before suggesting any fix.\n"
    "3. BILLING DISPUTE: Acknowledge frustration with empathy, explicitly mention the "
    "exact overcharge dollar amount, offer to refund it, then escalate to a manager.\n"
    "4. SUBSCRIPTION CANCELLATION: Ask why they want to cancel, then offer a targeted "
    "discount or plan downgrade. If they still want to cancel, process it professionally.\n\n"
    "Be concise (2-4 sentences per reply). Never ask multiple questions at once."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_tier(client: OpenAI, env: CustomerSupportEnv, tier: str) -> float:
    """Run one full episode for the given difficulty tier. Returns the final reward."""
    
    log_start(task="customer_support", env=tier, model=MODEL_NAME)

    try:
        result = env.reset(forced_tier=tier)
    except Exception as exc:
        error_msg = str(exc).replace('\n', ' ')
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    trajectory = {
        "tier": tier,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "turns": []
    }
    
    rewards = []
    step_count = 0
    done = result.done
    error_msg = None
    customer_msg = getattr(result.observation, "customer_reply", "")

    while not done:
        step_count += 1
        messages.append({"role": "user", "content": customer_msg})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=150,
            )
            agent_msg = completion.choices[0].message.content or "I am looking into this for you right now."
        except Exception as exc:
            agent_msg = "hello"
            error_msg = str(exc).replace('\n', ' ')

        messages.append({"role": "assistant", "content": agent_msg})
        action = CustomerSupportAction(message=agent_msg)
        
        try:
            result = env.step(action)
            reward = result.reward
            done = result.done
            customer_msg = getattr(result.observation, "customer_reply", "")
        except Exception as exc:
            reward = 0.0
            done = True
            error_msg = str(exc).replace('\n', ' ')
            
        rewards.append(reward)
        action_clean = repr(agent_msg.replace('\n', ' '))
        log_step(step=step_count, action=action_clean, reward=reward, done=done, error=error_msg)
        
        trajectory["turns"].append({
            "step": step_count,
            "customer": customer_msg,
            "agent": agent_msg,
            "reward": reward
        })

    raw_score = rewards[-1] if rewards else 0.0
    score = max(0.001, min(0.999, raw_score))  # Must be strictly (0, 1) per Hackathon validator rules
    success = score >= 0.5
    
    log_end(success=success, steps=step_count, score=score, rewards=rewards)
    
    # Save trajectory to outputs/ (silently, to avoid corrupting STDOUT for evaluator)
    os.makedirs("outputs/trajectories", exist_ok=True)
    filename = f"outputs/trajectories/traj_{tier}_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(trajectory, f, indent=2)
    
    return score


def main():
    # If no HF Token is provided and we require router, exit early silently.
    if not HF_TOKEN and "router" in API_BASE_URL:
        # Avoid printing ERRORs to STDOUT so as not to break standard parsing,
        # but in this case the evaluator will simply fail if the script returns without printing.
        # It's better to just proceed with what we have (API_KEY might be in environment via mock)
        pass

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-token-needed")

    if LOCAL_IMAGE_NAME:
        env_context = CustomerSupportEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env_context = CustomerSupportEnv(base_url="http://localhost:8000")

    try:
        with env_context.sync() as env:
            tiers = ["easy", "medium", "hard", "expert"]
            for tier in tiers:
                run_tier(client, env, tier)
    except Exception as exc:
        print(f"FAILED TO CONNECT TO ENVIRONMENT: {exc}")


if __name__ == "__main__":
    main()
