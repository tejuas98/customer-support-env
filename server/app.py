# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gradio as gr
import os

"""
FastAPI application for the Customer Support Environment.

This module creates an HTTP server that exposes the CustomerSupportEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import CustomerSupportAction, CustomerSupportObservation
    from server.customer_support_environment import CustomerSupportEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from .models import CustomerSupportAction, CustomerSupportObservation
        from .customer_support_environment import CustomerSupportEnvironment
    except (ImportError, ModuleNotFoundError):
        from customer_support.models import CustomerSupportAction, CustomerSupportObservation
        from customer_support.server.customer_support_environment import CustomerSupportEnvironment


def build_custom_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """
    Builds a professional Customer Support Dashboard UI.
    """
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")) as demo:
        gr.Markdown(f"# 🎧 {title} Dashboard")
        gr.Markdown("Training autonomous agents for high-empathy customer resolutions.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Support Conversation", height=500)
                msg = gr.Textbox(label="Agent Response (RL Action)", placeholder="Type your response here...", show_label=False)
                
                with gr.Row():
                    submit = gr.Button("Send message", variant="primary")
                    reset_btn = gr.Button("Reset Episode", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Episode Metrics")
                reward_display = gr.Number(label="Total Reward", value=0.0, precision=2)
                tier_display = gr.Label(label="Current Difficulty Tier")
                step_display = gr.Number(label="Step Count", value=0)
                
                gr.Markdown("---")
                gr.Markdown("### 🎯 Objectives")
                objectives_html = gr.HTML("Follow the 4-rule system prompt for maximum reward.")

        def update_ui():
            state = web_manager.episode_state
            chat_history = []
            
            # Initial customer message
            if state.current_observation:
                chat_history.append({"role": "assistant", "content": state.current_observation.get("customer_reply", "")})
            
            # Action logs
            for log in state.action_logs:
                chat_history.append({"role": "user", "content": log.action.get("message", "")})
                chat_history.append({"role": "assistant", "content": log.observation.get("customer_reply", "")})
            
            reward = 0.0
            if state.action_logs:
                reward = state.action_logs[-1].reward or 0.0
            
            tier = "unknown"
            if state.current_observation:
                tier = state.current_observation.get("task_tier", "easy")

            return chat_history, reward, tier.upper(), state.step_count

        async def handle_reset():
            await web_manager.reset_environment()
            return update_ui()

        async def handle_step(message):
            if not message:
                return update_ui()
            await web_manager.step_environment({"message": message})
            return update_ui()

        # Event handlers
        submit.click(handle_step, inputs=[msg], outputs=[chatbot, reward_display, tier_display, step_display])
        msg.submit(handle_step, inputs=[msg], outputs=[chatbot, reward_display, tier_display, step_display])
        reset_btn.click(handle_reset, outputs=[chatbot, reward_display, tier_display, step_display])
        
        # Initialize
        demo.load(update_ui, outputs=[chatbot, reward_display, tier_display, step_display])

    return demo

# Force Enable Web Interface for development/deployment
os.environ["ENABLE_WEB_INTERFACE"] = "true"

# Create the app with custom Gradio UI
app = create_app(
    CustomerSupportEnvironment,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="customer_support",
    max_concurrent_envs=1,
    gradio_builder=build_custom_ui,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m customer_support.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn customer_support.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
