---
title: Customer Support OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
---

# Customer Support OpenEnv

A multi-turn, multi-tier RL environment for training AI agents on real-world customer support tasks. Built for the Meta OpenEnv Hackathon.

## Motivation

Customer support is one of the most common real-world tasks deployed with LLMs today. A well-trained support agent must do more than retrieve information — it must empathize, diagnose problems systematically, reference specific details (order IDs, dollar amounts), and know when to escalate. This environment simulates that complexity with procedural generation and curriculum-aware difficulty.

Every `reset()` generates a unique scenario with randomized names, order IDs, overcharge amounts, software types, and OS options, ensuring agents must perform real-time reading comprehension rather than pattern-matching on fixed inputs.

## Four Difficulty Tiers

The environment uses a tiered curriculum that maps to real skill progression for a support agent:

| Tier | Task | Key Skills Tested |
| :--- | :--- | :--- |
| **Easy** | Damaged product refund | Empathy, identity reference, refund offer |
| **Medium** | Software crash troubleshooting | Diagnostic sequencing (OS before solution) |
| **Hard** | Billing dispute escalation | Empathy, exact amount recall, manager escalation |
| **Expert** | Subscription cancellation & retention | Root-cause diagnosis, targeted offer, professional close |

Rewards are continuous (0.0–1.0) with partial credit at each meaningful step, enabling rich reward signals for RL training.

## Curriculum Mode

Set `env.curriculum = True` to auto-advance difficulty. The environment tracks rolling average reward over the last 3 episodes and promotes the agent to the next tier once it sustains ≥0.8 average — matching difficulty to the agent's current capability.

## Space Definitions

### Action Space
| Field | Type | Description |
| :--- | :--- | :--- |
| `message` | `str` | The text message sent by the AI agent to the customer. |

### Observation Space
| Field | Type | Description |
| :--- | :--- | :--- |
| `customer_reply` | `str` | Customer's response (or opening problem statement). |
| `task_tier` | `str` | Current difficulty: `easy`, `medium`, `hard`, `expert`. |
| `done` | `bool` | Whether the episode has completed. |
| `reward` | `float` | Continuous grader score (0.0 to 1.0). |

## Quick Start

```bash
uv sync
uv run server
```

### Docker
```bash
docker build -t support-env -f server/Dockerfile .
docker run -p 8000:8000 support-env
```

## Evaluation

```bash
export HF_TOKEN="your_token"
uv run python3 inference.py
```

Uses `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face Inference Router by default.

### Reference Scores (Qwen2.5-72B-Instruct)
| Tier | Score |
| :--- | :--- |
| Easy | 1.00 |
| Medium | 1.00 |
| Hard | 1.00 |
| Expert | 1.00 |
| **Average** | **1.00** |
