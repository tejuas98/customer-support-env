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


def run_tier(client: OpenAI, env: CustomerSupportEnv, tier: str) -> float:
    """Run one full episode for the given difficulty tier. Returns the final reward."""
    print("[START]")
    print(f"tier={tier}")

    # Use client.reset with tier forced via metadata/params
    try:
        obs = env.reset(forced_tier=tier)
    except Exception as exc:
        print(f"env_reset_error: {exc}")
        return 0.0
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    trajectory = {
        "tier": tier,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "turns": []
    }
    final_reward = 0.0

    while not obs.done:
        print("[STEP]")
        print(f"customer: {obs.customer_reply}")
        messages.append({"role": "user", "content": obs.customer_reply})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=150,
            )
            agent_msg = completion.choices[0].message.content or "I am looking into this for you right now."
        except Exception as exc:
            print(f"model_error: {exc}")
            agent_msg = "I apologize, but I am having trouble connecting. Let me try again."

        print(f"agent: {agent_msg}")
        messages.append({"role": "assistant", "content": agent_msg})

        action = CustomerSupportAction(message=agent_msg)
        try:
            obs = env.step(action)
        except Exception as exc:
            print(f"env_step_error: {exc}")
            break
        
        final_reward = obs.reward
        
        trajectory["turns"].append({
            "step": len(trajectory["turns"]) + 1,
            "customer": obs.customer_reply,
            "agent": agent_msg,
            "reward": obs.reward
        })

    print(f"reward={final_reward:.2f}")
    
    # Save trajectory to outputs/
    os.makedirs("outputs/trajectories", exist_ok=True)
    filename = f"outputs/trajectories/traj_{tier}_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(trajectory, f, indent=2)
    
    print(f"trajectory_saved={filename}")
    print("[END]")
    return final_reward


def main():
    if not HF_TOKEN and "router" in API_BASE_URL:
        print("ERROR: HF_TOKEN environment variable is not set and needed for HF router.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-token-needed")

    # Benchmarking/Validation Strategy:
    # 1. If LOCAL_IMAGE_NAME is set, use from_docker_image()
    # 2. Otherwise use the default class-based local server via implicit port
    if LOCAL_IMAGE_NAME:
        print(f"Launching environment from Docker: {LOCAL_IMAGE_NAME}")
        env_context = CustomerSupportEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        # Default to local server (ensure uv run server is running or it will auto-start)
        print("Connecting to local environment server...")
        env_context = CustomerSupportEnv(base_url="http://localhost:8000")

    with env_context as env:
        print(f"model={MODEL_NAME}")
        print(f"endpoint={API_BASE_URL}")

        tiers = ["easy", "medium", "hard", "expert"]
        scores = {}

        for tier in tiers:
            score = run_tier(client, env, tier)
            scores[tier] = score

        print("\n--- Evaluation Summary ---")
        for tier, score in scores.items():
            print(f"{tier}: {score:.2f}")
        avg = sum(scores.values()) / len(scores)
        print(f"average: {avg:.2f}")


if __name__ == "__main__":
    main()
