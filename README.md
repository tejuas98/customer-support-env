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

Project for the Meta OpenEnv hackathon. Implements a multi-step customer support environment with tiered difficulty.

## Motivation
The Customer Support OpenEnv environment provides a high-fidelity simulation of real-world agent tasks. It replaces simple "toy" RL games with procedural, multi-turn troubleshooting scenarios that require empathy, precise variable extraction (Order IDs, Refund Amounts), and deterministic escalation logic to solve correctly.

## The Tasks
The environment uses a tiered difficulty scoring system (0.0 - 1.0):

1. **Easy (Difficulty: EASY):** Broken product report. The agent must apologize and offer a refund while referencing the customer's name or Order ID.
2. **Medium (Difficulty: MEDIUM):** Technical software crash. The agent must perform root-cause analysis by identifying the user's OS before providing a solution.
3. **Hard (Difficulty: HARD):** Angry escalation. The agent must show empathy, quote the **exact procedurally generated refund amount**, and only then escalate to a manager for a perfect score.

## Space Definitions

### Action Space
| Field | Type | Description |
| :--- | :--- | :--- |
| `message` | `str` | The text message sent by the AI agent to the customer. |

### Observation Space
| Field | Type | Description |
| :--- | :--- | :--- |
| `customer_reply` | `str` | The customer's response (or initial problem statement). |
| `task_tier` | `str` | Current difficulty tier (`easy`, `medium`, `hard`). |
| `done` | `bool` | Whether the conversation has completed. |
| `reward` | `float` | Continuous grader score representing partial progress (0.0 to 1.0). |

## Quick Start

### 1. Build & Run
```bash
uv sync
uv run server
```

### 2. Docker
To run as a container (for HF Spaces):
```bash
docker build -t support-env -f server/Dockerfile .
docker run -p 8000:8000 support-env
```

## Running Evaluation
Use `inference.py` to run an LLM evaluation across all tasks. 

```bash
export HF_TOKEN="your_token"
python inference.py
```

### Reference Scores
*   Easy: 1.0
*   Medium: 1.0
*   Hard: 1.0
