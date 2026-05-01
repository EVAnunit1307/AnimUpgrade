"""
AnimUpgrade - Web Interface
Gradio app: drop a video, pick a motion type, get upgraded animation back.
Run with: python app.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Ensure pipeline is importable
sys.path.insert(0, str(Path(__file__).parent))

try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    os.system("pip install gradio --break-system-packages -q")
    import gradio as gr

from pipeline import run_pipeline, MOTION_PRESETS


MOTION_OPTIONS = {
    "Auto-detect (generic)": "generic",
    "Flying / aerial scene": "flying",
    "Dialogue / talking heads": "talking",
    "Action / combat": "action",
    "Walking / movement": "walk",
}

QUALITY_OPTIONS = {
    "Pro 1080p (~$0.49/clip)": {"use_pro": True, "use_fast": False},
    "Fast Pro 1080p (~$0.49/clip, quicker)": {"use_pro": True, "use_fast": True},
    "Standard 768p (~$0.27/clip)": {"use_pro": False, "use_fast": False},
}


def check_api_key():
    key = os.environ.get("FAL_KEY", "")
    if not key:
        return False, "❌ FAL_KEY not set. Add it in the API Key field below."
    return True, f"✅ API key set ({key[:8]}...)"


def upgrade_video(
    video_file,
    motion_label,
    quality_label,
    custom_prompt,
    frame_choice,
    fal_key_input,
):
    # Set API key from input if provided
    if fal_key_input and fal_key_input.strip():
        os.environ["FAL_KEY"] = fal_key_input.strip()

    if not os.environ.get("FAL_KEY"):
        return None, "❌ No API key. Get one free at fal.ai/dashboard/keys and paste it above.", ""

    if video_file is None:
        return None, "❌ Please upload a video first.", ""

    motion_type = MOTION_OPTIONS.get(motion_label, "generic")
    quality_opts = QUALITY_OPTIONS.get(quality_label, QUALITY_OPTIONS["Pro 1080p (~$0.49/clip)"])

    frame_number = None
    if frame_choice == "First frame":
        frame_number = 0
    elif frame_choice == "Last frame":
        frame_number = -1
    # else: middle frame (default)

    prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

    output_dir = tempfile.mkdtemp()

    try:
        result = run_pipeline(
            input_video_path=video_file,
            output_dir=output_dir,
            motion_type=motion_type,
            custom_prompt=prompt,
            frame_number=frame_number,
            use_pro=quality_opts["use_pro"],
            use_fast=quality_opts["use_fast"],
        )

        output_video = result["output"]
        status = f"✅ Done! Cost: {result['cost_estimate']} | Tier: {result['tier']}"
        prompt_used = f"Prompt used:\n{result['prompt_used']}"

        return output_video, status, prompt_used

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return None, "❌ Invalid API key. Check your FAL_KEY.", ""
        elif "quota" in error_msg.lower() or "credit" in error_msg.lower():
            return None, "❌ Insufficient credits. Add credits at fal.ai/dashboard.", ""
        else:
            return None, f"❌ Error: {error_msg}", ""


CSS = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 900px !important; margin: 0 auto; }
.main-header { 
    text-align: center; 
    padding: 20px 0 10px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 20px;
}
.main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p { color: #6b7280; margin: 6px 0 0; font-size: 0.95rem; }
.status-box { padding: 12px; border-radius: 8px; font-weight: 500; }
"""

DESCRIPTION = """
## AnimUpgrade
**Drop in a flat TV animation clip. Get back a version with real motion.**

Built for scenes where characters move like PNGs — flying scenes, static dialogue shots, 
background characters sliding across frames. Powered by Hailuo 2.3 Pro.

→ Get a free API key at [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys)
"""

def build_app():
    with gr.Blocks(css=CSS, title="AnimUpgrade") as app:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. API Key")
                fal_key_input = gr.Textbox(
                    label="fal.ai API Key",
                    placeholder="key-xxxxxxxxxxxxxxxx",
                    type="password",
                    info="Get a free key at fal.ai/dashboard/keys. $1 free credit = ~2 clips.",
                )
                key_status = gr.Markdown(check_api_key()[1])

                gr.Markdown("### 2. Upload video")
                video_input = gr.Video(
                    label="Input clip",
                    sources=["upload"],
                )

                gr.Markdown("### 3. Settings")
                motion_input = gr.Dropdown(
                    label="Scene type",
                    choices=list(MOTION_OPTIONS.keys()),
                    value="Auto-detect (generic)",
                    info="Picks the right motion prompt for your scene type.",
                )
                quality_input = gr.Dropdown(
                    label="Quality tier",
                    choices=list(QUALITY_OPTIONS.keys()),
                    value="Fast Pro 1080p (~$0.49/clip, quicker)",
                )
                frame_input = gr.Radio(
                    label="Reference frame",
                    choices=["Middle frame", "First frame", "Last frame"],
                    value="Middle frame",
                    info="Which frame to use as the motion reference.",
                )
                custom_prompt = gr.Textbox(
                    label="Custom prompt (optional)",
                    placeholder="Describe the exact motion you want...",
                    lines=3,
                    info="Leave blank to use the scene type preset.",
                )

                run_btn = gr.Button("Upgrade Animation →", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Result")
                video_output = gr.Video(label="Upgraded clip", interactive=False)
                status_output = gr.Markdown("Upload a clip and click Upgrade to get started.")
                prompt_output = gr.Textbox(
                    label="Prompt used",
                    interactive=False,
                    lines=4,
                )

                gr.Markdown("""
---
**Tips for best results:**
- Use 3-8 second clips for best motion coherence
- Flying scenes respond best to the "Flying" preset
- For Invincible-style: keep "Auto-detect" and add `[Static shot]` to hold the camera
- The model preserves your art style — it won't make it look photorealistic
                """)

        run_btn.click(
            fn=upgrade_video,
            inputs=[
                video_input,
                motion_input,
                quality_input,
                custom_prompt,
                frame_input,
                fal_key_input,
            ],
            outputs=[video_output, status_output, prompt_output],
        )

    return app


if __name__ == "__main__":
    print("\n" + "="*50)
    print("AnimUpgrade Web App")
    print("="*50)
    has_key, key_msg = check_api_key()
    print(key_msg)
    if not has_key:
        print("You can also paste your key in the UI.")
    print("="*50 + "\n")

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
