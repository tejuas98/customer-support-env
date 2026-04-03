"""
Inference script for the Customer Support OpenEnv environment.

This script evaluates an LLM agent across all three difficulty tiers
of the Customer Support environment using the OpenAI-compatible API.

Environment variables:
    API_BASE_URL:       LLM API endpoint. Defaults to HF router.
    MODEL_NAME:         LLM model identifier. Defaults to Llama-3.2-3B-Instruct.
    HF_TOKEN:           Hugging Face API key. No default - must be set explicitly.
    LOCAL_IMAGE_NAME:   Optional. Docker image name for local testing via from_docker_image().
"""

import os
from openai import OpenAI
from server.customer_support_environment import CustomerSupportEnvironment
from models import CustomerSupportAction

# Required environment variables - defaults only for API_BASE_URL and MODEL_NAME.
# HF_TOKEN must be set by the caller with no fallback.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional: set this when running the environment from a local Docker image.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def run_tier(client: OpenAI, env: CustomerSupportEnvironment, tier: str) -> float:
    """Run one full episode for the given difficulty tier. Returns the final reward."""
    print("[START]")
    print(f"tier={tier}")

    env.task_tier = tier
    obs = env.reset()
    env.task_tier = tier

    system_prompt = (
        "You are a professional customer support agent. Your goal is to resolve the "
        "customer's issue efficiently and empathetically in as few turns as possible.\n\n"
        "Guidelines:\n"
        "- For broken/damaged product complaints: apologize sincerely, reference the "
        "customer name or order ID, and proactively offer a full refund or replacement.\n"
        "- For software crash issues: ALWAYS ask what operating system the customer is "
        "using FIRST before suggesting any solution.\n"
        "- For billing disputes: acknowledge the frustration with empathy, explicitly "
        "mention the exact overcharge amount in dollars, offer to process the refund, "
        "and then escalate to a manager when requested.\n"
        "Keep responses concise (2-4 sentences). Do not ask multiple questions at once."
    )

    messages = [{"role": "system", "content": system_prompt}]
    final_reward = 0.0

    while not obs.done:
        print(f"[STEP]")
        print(f"customer: {obs.customer_reply}")
        messages.append({"role": "user", "content": obs.customer_reply})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=150,
                stream=False,
            )
            agent_msg = completion.choices[0].message.content or "I am looking into this right now."
        except Exception as exc:
            print(f"model_error: {exc}")
            agent_msg = "I apologize for the inconvenience. Let me escalate this to a manager immediately."

        print(f"agent: {agent_msg}")
        messages.append({"role": "assistant", "content": agent_msg})

        action = CustomerSupportAction(message=agent_msg)
        obs = env.step(action)
        final_reward = obs.reward

    print(f"reward={final_reward:.2f}")
    print("[END]")
    return final_reward


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = CustomerSupportEnvironment()

    print(f"model={MODEL_NAME}")
    print(f"endpoint={API_BASE_URL}")

    tiers = ["easy", "medium", "hard"]
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
