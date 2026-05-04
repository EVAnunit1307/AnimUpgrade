"""
AnimUpgrade - Web Interface
Stage 1: RIFE local frame interpolation  (pipeline.py)
Stage 2: EbSynth key frame extraction    (ebsynth_prep.py)

Run with: python app.py  →  http://localhost:7860
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from pipeline import (
    run_pipeline,
    find_rife_binary,
    _require_ffmpeg,
)
from ebsynth_prep import extract_key_frames


# ---------------------------------------------------------------------------
# Environment checks (run once at startup, not on every request)
# ---------------------------------------------------------------------------

def _check_env() -> tuple[bool, bool, str]:
    """
    Returns (ffmpeg_ok, rife_ok, status_message).
    Called once at startup to populate the status banner.
    """
    ffmpeg_ok = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

    rife_ok = False
    rife_path = ""
    try:
        p = find_rife_binary()
        rife_ok = True
        rife_path = str(p)
    except SystemExit:
        pass

    parts = []
    if ffmpeg_ok:
        parts.append("ffmpeg OK")
    else:
        parts.append("ffmpeg MISSING — run `winget install ffmpeg` then restart")
    if rife_ok:
        parts.append(f"rife-ncnn-vulkan OK  ({rife_path})")
    else:
        parts.append(
            "rife-ncnn-vulkan MISSING — download pre-built binary from "
            "github.com/nihui/rife-ncnn-vulkan/releases and set RIFE_BIN env var"
        )

    return ffmpeg_ok, rife_ok, "  |  ".join(parts)


_FFMPEG_OK, _RIFE_OK, _ENV_STATUS = _check_env()


# ---------------------------------------------------------------------------
# RIFE models known to ship in the pre-built release ZIP
# ---------------------------------------------------------------------------

RIFE_MODELS = [
    "rife-v4.25",
    "rife-v4.22-lite",
    "rife-v4.6",
    "rife-anime",
]

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

KEYS_DIR = Path("ebsynth_keys")


# ---------------------------------------------------------------------------
# Stage 1 handler
# ---------------------------------------------------------------------------

def run_stage1(
    video_file,
    multiplier,
    model,
    dedup_threshold,
    use_nvenc,
):
    if not _FFMPEG_OK:
        return None, None, "ffmpeg not found — install it and restart the app."
    if not _RIFE_OK:
        return None, None, (
            "rife-ncnn-vulkan binary not found.\n"
            "Download from github.com/nihui/rife-ncnn-vulkan/releases\n"
            "Then set: RIFE_BIN=C:\\path\\to\\rife-ncnn-vulkan.exe"
        )
    if video_file is None:
        return None, None, "Upload a video clip first."

    input_path = Path(video_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(OUTPUTS_DIR / f"{input_path.stem}_rife{multiplier}x_{timestamp}.mp4")

    try:
        run_pipeline(
            input_path=str(video_file),
            output_path=output_path,
            multiplier=int(multiplier),
            model=model,
            dedup_threshold=float(dedup_threshold),
            use_nvenc=use_nvenc,
        )
    except SystemExit as e:
        return None, None, f"Pipeline exited — check terminal for details. (code {e.code})"
    except Exception as e:
        return None, None, f"Error: {e}"

    size_mb = Path(output_path).stat().st_size / 1_048_576
    status = (
        f"Done!  {multiplier}x  |  model: {model}  |  "
        f"dedup: {dedup_threshold}  |  {size_mb:.1f} MB\n"
        f"Saved: {output_path}"
    )
    # Return the output path twice: once for video player, once for State
    # so Stage 2 can auto-populate its enhanced video input.
    return output_path, output_path, status


# ---------------------------------------------------------------------------
# Stage 2 handler
# ---------------------------------------------------------------------------

def run_stage2(original_video, enhanced_video, n_keys, source):
    if original_video is None:
        return [], "Upload the original (pre-RIFE) clip."
    if enhanced_video is None:
        return [], "Upload the Stage 1 enhanced clip (or run Stage 1 first)."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = str(KEYS_DIR / timestamp)

    try:
        extract_key_frames(
            original_video=original_video,
            enhanced_video=enhanced_video,
            n_keys=int(n_keys),
            output_dir=out_dir,
            source=source,
        )
    except Exception as e:
        return [], f"Error: {e}"

    key_pngs = sorted(Path(out_dir).glob("key_*.png"))
    manifest_path = Path(out_dir) / "ebsynth_manifest.json"

    manifest_summary = ""
    if manifest_path.exists():
        with open(manifest_path) as f:
            m = json.load(f)
        manifest_summary = (
            f"\n\nManifest: {manifest_path}\n"
            f"Enhanced FPS: {m['enhanced_fps']:.3f}  |  "
            f"Total frames: {m['total_enhanced_frames']}  |  "
            f"Keys: {m['keys_extracted']}"
        )

    status = (
        f"Extracted {len(key_pngs)} key frames  →  {out_dir}/"
        f"{manifest_summary}\n\n"
        "Next: open PNGs in Photoshop, paint secondary motion, "
        "then feed painted keys into EbSynth 2."
    )

    gallery_items = [(str(p), p.stem) for p in key_pngs]
    return gallery_items, status


# ---------------------------------------------------------------------------
# Auto-populate Stage 2's enhanced video from Stage 1 output
# ---------------------------------------------------------------------------

def on_stage1_complete(enhanced_path):
    """When Stage 1 finishes, pre-fill Stage 2's enhanced video input."""
    if enhanced_path:
        return gr.update(value=enhanced_path)
    return gr.update()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1000px !important; margin: 0 auto; }
"""

ENV_MD = (
    f"**Environment:** {_ENV_STATUS}"
    if _FFMPEG_OK and _RIFE_OK
    else f"**Setup required:** {_ENV_STATUS}"
)


def build_app():
    with gr.Blocks(title="AnimUpgrade") as app:
        gr.Markdown("## AnimUpgrade  |  Local Frame Enhancement Pipeline")
        gr.Markdown(ENV_MD)

        # Shared state: passes Stage 1 output path to Stage 2
        stage1_output_path = gr.State(None)

        with gr.Tabs():

            # ── Stage 1 ────────────────────────────────────────────────────
            with gr.Tab("Stage 1 — RIFE Interpolation"):
                gr.Markdown(
                    "**Removes duplicate holds from 2D animation, then generates smooth "
                    "in-between frames using rife-ncnn-vulkan (Vulkan backend — no CUDA required).**\n\n"
                    "Pipeline: extract frames → dedup holds → RIFE interpolate → reassemble MP4"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        s1_video_in = gr.Video(
                            label="Input clip (MP4 / MOV)",
                            sources=["upload"],
                        )
                        s1_multiplier = gr.Radio(
                            label="Frame rate multiplier",
                            choices=[2, 4],
                            value=2,
                            info="2x = 24→48 fps, 4x = 24→96 fps",
                        )
                        s1_model = gr.Dropdown(
                            label="RIFE model",
                            choices=RIFE_MODELS,
                            value="rife-v4.25",
                            info="Must match a folder inside your rife-ncnn-vulkan models/ directory.",
                        )
                        s1_dedup = gr.Slider(
                            label="Dedup threshold (MAD)",
                            minimum=0.0,
                            maximum=0.10,
                            value=0.02,
                            step=0.005,
                            info=(
                                "0.00 = off  |  0.02 = standard (cel-shading, Invincible)  |  "
                                "0.05 = aggressive (very limited animation)"
                            ),
                        )
                        s1_nvenc = gr.Checkbox(
                            label="GPU encode (h264_nvenc)",
                            value=True,
                            info="Recommended for GTX 1650 — uses dedicated NVENC block, frees GPU compute.",
                        )
                        s1_run = gr.Button("Run Stage 1", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        s1_video_out = gr.Video(
                            label="Enhanced output",
                            interactive=False,
                        )
                        s1_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=4,
                        )
                        gr.Markdown("""
---
**Tips:**
- Dedup 0.02 is correct for most TV animation (removes held frames)
- Use 2x first — 4x needs more VRAM and takes longer
- rife-v4.25 is the best available model in the pre-built release ZIP
- If the model name errors, check your rife-ncnn-vulkan models/ folder
- NVENC falls back to libx264 automatically if not available
                        """)

                s1_run.click(
                    fn=run_stage1,
                    inputs=[s1_video_in, s1_multiplier, s1_model, s1_dedup, s1_nvenc],
                    outputs=[s1_video_out, stage1_output_path, s1_status],
                )

            # ── Stage 2 ────────────────────────────────────────────────────
            with gr.Tab("Stage 2 — EbSynth Key Frames"):
                gr.Markdown(
                    "**Extracts the most dynamic frames from the RIFE-enhanced video "
                    "using dense optical flow.**\n\n"
                    "Open the extracted PNGs in Photoshop, paint secondary motion "
                    "(cape physics, hair, particles, speed lines), then propagate "
                    "the painted keys across all frames using EbSynth 2."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        s2_original = gr.Video(
                            label="Original clip (pre-RIFE)",
                            sources=["upload"],
                        )
                        s2_enhanced = gr.Video(
                            label="Enhanced clip (Stage 1 output — auto-filled after Stage 1 runs)",
                            sources=["upload"],
                        )
                        s2_keys = gr.Slider(
                            label="Number of key frames",
                            minimum=4,
                            maximum=48,
                            value=12,
                            step=1,
                            info=(
                                "How many frames to extract. More keys = finer EbSynth "
                                "propagation but more Photoshop work. 8-16 is typical."
                            ),
                        )
                        s2_source = gr.Radio(
                            label="Extract images from",
                            choices=["enhanced", "original"],
                            value="enhanced",
                            info=(
                                "enhanced = paint on RIFE-smoothed frames (recommended)  |  "
                                "original = paint on raw source frames"
                            ),
                        )
                        s2_run = gr.Button("Extract Key Frames", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        s2_gallery = gr.Gallery(
                            label="Extracted key frames",
                            columns=3,
                            height=400,
                        )
                        s2_status = gr.Textbox(
                            label="Status / EbSynth instructions",
                            interactive=False,
                            lines=10,
                        )

                s2_run.click(
                    fn=run_stage2,
                    inputs=[s2_original, s2_enhanced, s2_keys, s2_source],
                    outputs=[s2_gallery, s2_status],
                )

        # Auto-populate Stage 2 enhanced input when Stage 1 completes
        stage1_output_path.change(
            fn=on_stage1_complete,
            inputs=[stage1_output_path],
            outputs=[s2_enhanced],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("AnimUpgrade  |  Local Frame Enhancement Pipeline")
    print("=" * 65)
    print(_ENV_STATUS)
    print("=" * 65 + "\n")

    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )
