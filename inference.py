"""
Inference script for customer support evaluation.

Required env vars:
    API_BASE_URL: LLM API endpoint.
    MODEL_NAME: LLM model identifier.
    HF_TOKEN: Hugging Face API key.
"""

import os
from openai import OpenAI
from server.customer_support_environment import CustomerSupportEnvironment
from models import CustomerSupportAction

# Set defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

def main():
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is missing.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CustomerSupportEnvironment()
    
    tiers = ["easy", "medium", "hard"]
    
    print(f"Running evaluation with model: {MODEL_NAME}")
    
    for tier in tiers:
        env.task_tier = tier 
        obs = env.reset()
        env.task_tier = tier
        
        # Override initial message for consistent baseline
        if tier == "easy":
            obs.customer_reply = "Hi, the coffee mug I ordered just arrived broken in half. I'm disappointed."
        elif tier == "medium":
            obs.customer_reply = "Your video editing software keeps crashing immediately on launch today."
        else:
            obs.customer_reply = "I was double charged $200 this month! This is absolute theft. Get me a manager immediately."
        obs.task_tier = tier
        print("[START]")
        print(f"--- Starting task: {tier.upper()} ---")
        
        system_prompt = (
            "You are a helpful customer support agent. "
            "For refund requests, apologize, reference the name or Order ID, and offer a refund. "
            "For software crashes, first ask for the user's OS. "
            "For billing errors, be empathetic, quote the exact overcharge amount, and escalate to a manager. "
            "Keep responses short (1-3 sentences)."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        while not obs.done:
            print(f"Customer: {obs.customer_reply}")
            messages.append({"role": "user", "content": obs.customer_reply})
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=200,
                    stream=False
                )
                agent_msg = completion.choices[0].message.content or "noop"
            except Exception as exc:
                print(f"Model error ({exc}). Using default msg.")
                agent_msg = "Please hold while I check my system."
                
            print(f"Agent: {agent_msg}")
            
            messages.append({"role": "assistant", "content": agent_msg})
            
            action = CustomerSupportAction(message=agent_msg)
            print("[STEP]")
            obs = env.step(action)
            
        print(f"Task '{tier.upper()}' done. Score: {obs.reward}")
        print("[END]")

if __name__ == "__main__":
    main()
