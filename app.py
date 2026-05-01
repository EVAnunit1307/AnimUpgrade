"""
AnimUpgrade - Web Interface
Run with: python app.py  →  http://localhost:7860
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

# Only confirmed-working endpoints (verified May 2026)
QUALITY_OPTIONS = {
    "O3 Pro — best quality (~$0.17/sec)": "o3_pro",
    "O1 Standard — budget (~$0.17/sec)": "o1_standard",
}


def check_api_key():
    key = os.environ.get("FAL_KEY", "")
    if not key:
        return False, "FAL_KEY not set — paste it in the field below."
    return True, f"API key loaded ({key[:8]}...)"


def enhance_video(video_file, motion_label, quality_label, custom_prompt, fal_key_input):
    if fal_key_input and fal_key_input.strip():
        os.environ["FAL_KEY"] = fal_key_input.strip()

    if not os.environ.get("FAL_KEY"):
        return None, "No API key. Get one at fal.ai/dashboard/keys.", ""

    if video_file is None:
        return None, "Upload a video first.", ""

    motion_type = MOTION_OPTIONS.get(motion_label, "generic")
    tier_key = QUALITY_OPTIONS.get(quality_label, "o3_pro")
    prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

    try:
        result = run_pipeline(
            input_video_path=video_file,
            output_dir=tempfile.mkdtemp(),
            motion_type=motion_type,
            custom_prompt=prompt,
            tier_key=tier_key,
        )
        status = f"Done!  Tier: {result['tier']}  |  Cost: {result['cost_estimate']}"
        return result["output"], status, result["prompt_used"]

    except Exception as e:
        msg = str(e)
        if "401" in msg or "unauthorized" in msg.lower():
            return None, "Invalid API key.", ""
        if "quota" in msg.lower() or "credit" in msg.lower():
            return None, "Out of credits — top up at fal.ai/dashboard.", ""
        return None, f"Error: {msg}", ""


CSS = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 960px !important; margin: 0 auto; }
"""

DESCRIPTION = """
## AnimUpgrade  —  Kling video-to-video

Drop in a flat TV animation clip. Get back the same clip with secondary motion added —
hair physics, cape movement, breathing, fabric dynamics — while the original scene stays intact.

**Clip requirements:** MP4 or MOV · 3–10 seconds · max 200 MB · 720p or higher

Cost: ~$0.17/sec of input  ·  Get a free API key at [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys)
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
                )
                gr.Markdown(check_api_key()[1])

                gr.Markdown("### 2. Upload clip")
                video_input = gr.Video(
                    label="Input (MP4/MOV, 3-10s, max 200MB)",
                    sources=["upload"],
                )

                gr.Markdown("### 3. Settings")
                motion_input = gr.Dropdown(
                    label="Scene type",
                    choices=list(MOTION_OPTIONS.keys()),
                    value="Auto-detect (generic)",
                    info="Selects the right secondary motion prompt for your scene.",
                )
                quality_input = gr.Dropdown(
                    label="Quality tier",
                    choices=list(QUALITY_OPTIONS.keys()),
                    value="O3 Pro — best quality (~$0.17/sec)",
                )
                custom_prompt = gr.Textbox(
                    label="Custom prompt (optional)",
                    placeholder="Describe exactly what secondary motion to add...",
                    lines=4,
                    info="Leave blank to use the scene preset. If you write your own, end with: 'Do not change existing positions, motion, or art style.'",
                )

                run_btn = gr.Button("Enhance Animation", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Result")
                video_output = gr.Video(label="Enhanced clip", interactive=False)
                status_output = gr.Markdown("Upload a clip and click Enhance.")
                prompt_output = gr.Textbox(
                    label="Prompt sent to Kling",
                    interactive=False,
                    lines=6,
                )
                gr.Markdown("""
---
**Tips:**
- 3–8 second clips work best
- Flying scenes: use the "Flying" preset — targets cape and hair specifically
- If too much changes: add "Do not change any existing motion" to a custom prompt
- Audio is preserved automatically
- The model retains original motion structure by design — it adds, doesn't replace
                """)

        run_btn.click(
            fn=enhance_video,
            inputs=[video_input, motion_input, quality_input, custom_prompt, fal_key_input],
            outputs=[video_output, status_output, prompt_output],
        )

    return app


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AnimUpgrade  |  Kling video-to-video")
    print("=" * 50)
    _, key_msg = check_api_key()
    print(key_msg)
    print("=" * 50 + "\n")

    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )
