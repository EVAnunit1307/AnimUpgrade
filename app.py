"""
AnimUpgrade - Web Interface
Gradio app: drop a video, pick a motion type, get enhanced animation back.
Run with: python app.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from pipeline import run_pipeline, MOTION_PRESETS, TIERS


MOTION_OPTIONS = {
    "Auto-detect (generic)": "generic",
    "Flying / aerial scene": "flying",
    "Dialogue / talking heads": "talking",
    "Action / combat": "action",
    "Walking / movement": "walk",
}

QUALITY_OPTIONS = {
    "O3 Pro (best quality)": "o3_pro",
    "O1 Pro (balanced)": "o1_pro",
    "O1 Standard (budget)": "o1_standard",
}


def check_api_key():
    key = os.environ.get("FAL_KEY", "")
    if not key:
        return False, "FAL_KEY not set. Add it in the API Key field below."
    return True, f"API key set ({key[:8]}...)"


def enhance_video(
    video_file,
    motion_label,
    quality_label,
    cfg_scale,
    custom_prompt,
    fal_key_input,
):
    if fal_key_input and fal_key_input.strip():
        os.environ["FAL_KEY"] = fal_key_input.strip()

    if not os.environ.get("FAL_KEY"):
        return None, "No API key. Get one free at fal.ai/dashboard/keys and paste it above.", ""

    if video_file is None:
        return None, "Please upload a video first.", ""

    motion_type = MOTION_OPTIONS.get(motion_label, "generic")
    tier_key = QUALITY_OPTIONS.get(quality_label, "o3_pro")
    prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

    output_dir = tempfile.mkdtemp()

    try:
        result = run_pipeline(
            input_video_path=video_file,
            output_dir=output_dir,
            motion_type=motion_type,
            custom_prompt=prompt,
            tier_key=tier_key,
            cfg_scale=float(cfg_scale),
        )

        status = f"Done! Tier: {result['tier']} | Cost: {result['cost_estimate']}"
        prompt_used = f"Prompt used:\n{result['prompt_used']}"
        return result["output"], status, prompt_used

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return None, "Invalid API key. Check your FAL_KEY.", ""
        elif "quota" in error_msg.lower() or "credit" in error_msg.lower():
            return None, "Insufficient credits. Add credits at fal.ai/dashboard.", ""
        else:
            return None, f"Error: {error_msg}", ""


CSS = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 960px !important; margin: 0 auto; }
"""

DESCRIPTION = """
## AnimUpgrade
**Drop in a flat TV animation clip. Get back a version with real secondary motion.**

Uses Kling video-to-video — preserves your original scene, characters, and art style exactly.
Adds what's missing: hair physics, cape movement, breathing, fabric dynamics.

Cost: ~$0.17/sec of input video. A 5-second clip = ~$0.85.

Get a free API key at [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys)
"""


def build_app():
    with gr.Blocks(title="AnimUpgrade") as app:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. API Key")
                fal_key_input = gr.Textbox(
                    label="fal.ai API Key",
                    placeholder="key-xxxxxxxxxxxxxxxx",
                    type="password",
                    info="Get a free key at fal.ai/dashboard/keys.",
                )
                gr.Markdown(check_api_key()[1])

                gr.Markdown("### 2. Upload video")
                video_input = gr.Video(
                    label="Input clip (MP4, 3-10 seconds recommended)",
                    sources=["upload"],
                )

                gr.Markdown("### 3. Settings")
                motion_input = gr.Dropdown(
                    label="Scene type",
                    choices=list(MOTION_OPTIONS.keys()),
                    value="Auto-detect (generic)",
                    info="Picks the right motion prompt for your scene.",
                )
                quality_input = gr.Dropdown(
                    label="Quality tier",
                    choices=list(QUALITY_OPTIONS.keys()),
                    value="O3 Pro (best quality)",
                )
                cfg_scale_input = gr.Slider(
                    label="Enhancement strength",
                    minimum=0.1,
                    maximum=0.7,
                    value=0.4,
                    step=0.05,
                    info="Lower = more faithful to original. 0.3-0.4 recommended.",
                )
                custom_prompt = gr.Textbox(
                    label="Custom prompt (optional)",
                    placeholder="Describe exactly what secondary motion to add...",
                    lines=3,
                    info="Leave blank to use the scene type preset.",
                )

                run_btn = gr.Button("Enhance Animation", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Result")
                video_output = gr.Video(label="Enhanced clip", interactive=False)
                status_output = gr.Markdown("Upload a clip and click Enhance to get started.")
                prompt_output = gr.Textbox(
                    label="Prompt used",
                    interactive=False,
                    lines=5,
                )

                gr.Markdown("""
---
**Tips:**
- 3-8 second clips work best
- Start with strength 0.4 — go lower if too much changes
- Flying scenes: use "Flying" preset, it targets cape and hair specifically
- The model preserves your art style — it won't go photorealistic
- Kling v2v enhances existing motion, it doesn't invent new scenes
                """)

        run_btn.click(
            fn=enhance_video,
            inputs=[
                video_input,
                motion_input,
                quality_input,
                cfg_scale_input,
                custom_prompt,
                fal_key_input,
            ],
            outputs=[video_output, status_output, prompt_output],
        )

    return app


if __name__ == "__main__":
    print("\n" + "="*50)
    print("AnimUpgrade Web App  |  Kling video-to-video")
    print("="*50)
    has_key, key_msg = check_api_key()
    print(key_msg)
    if not has_key:
        print("Paste your key in the UI to get started.")
    print("="*50 + "\n")

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )
