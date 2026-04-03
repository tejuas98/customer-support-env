---
title: Customer Support OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
---

# 🎧 Customer Support OpenEnv: PREMIUM EDITION

A high-fidelity, multi-tier RL environment for training AI agents on real-world customer support tasks. Optimized for the **Meta OpenEnv Hackathon**.

## ⭐ Premium Hackathon Features
- **Custom Support Dashboard**: A professional Gradio UI redesign at `/web` featuring a real-time Chatbot interface, live reward gauges, and tiered difficulty metrics.
- **Trajectory Logging**: Every evaluation is captured as a high-fidelity JSON trajectory in `outputs/trajectories/`, suitable for jury review and model fine-tuning.
- **Nuanced Expert Tier**: Advanced retention logic requiring **targeted offers** (e.g., offering a discount for price concerns, or roadmap highlights for feature gaps).
- **Curriculum Mode**: Native support for `easy` ➡️ `medium` ➡️ `hard` ➡️ `expert` progression based on agent performance.
- **Docker-First**: Ready-to-deploy `Dockerfile` and `inference.py` optimized for HF Spaces and the OpenEnv framework.

## Motivation

Customer support is one of the most common real-world tasks deployed with LLMs today. A well-trained support agent must do more than retrieve information — it must empathize, diagnose problems systematically, reference specific details (order IDs, dollar amounts), and know when to escalate. This environment simulates that complexity with procedural generation and curriculum-aware difficulty.

## Four Difficulty Tiers

| Tier | Task | Key Skills Tested | Status |
| :--- | :--- | :--- | :--- |
| **Easy** | Damaged product refund | Empathy, identity reference, refund offer | ✅ PASS |
| **Medium** | Software crash troubleshooting | Diagnostic sequencing (OS before solution) | ✅ PASS |
| **Hard** | Billing dispute escalation | Empathy, exact amount recall, manager escalation | ✅ PASS |
| **Expert** | Subscription cancellation & retention | Root-cause diagnosis, targeted offer, professional close | ✅ PASS |

---

## Quick Start

### 1. Local Playground (Custom UI)
```bash
uv sync
uv run server
```
Visit `http://localhost:8000/web` to chat with the environment manually.

### 2. Auto-Evaluation (Target Task)
```bash
export HF_TOKEN="your_token"
uv run python3 inference.py
```
This will evaluate `Qwen2.5-72B-Instruct` and save full session trajectories to `outputs/trajectories/`.

### 3. Docker Deployment
```bash
docker build -t support-env -f server/Dockerfile .
docker run -p 8000:8000 support-env
```

---

## 🏗️ Technical Specifications
- **Model**: `Qwen/Qwen2.5-72B-Instruct` (HF Router)
- **Reward Range**: 0.0 to 1.0 (Continuous)
- **Log Markers**: Strict `[START]`, `[STEP]`, `[END]` compliance for automated grading.
