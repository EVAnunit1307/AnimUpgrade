"""
AnimUpgrade - Core Pipeline
Enhances flat TV animation with secondary motion using Kling video-to-video.
Preserves the original scene, characters, and art style — adds what's missing.

Verified endpoints (May 2026):
  fal-ai/kling-video/o3/pro/video-to-video/edit    — best quality
  fal-ai/kling-video/o1/standard/video-to-video/edit — budget

API parameters (confirmed from fal.ai docs):
  prompt      string   required
  video_url   string   required  (MP4/MOV, 3-10s, 720-2160px, max 200MB)
  keep_audio  bool     optional  (default true)
  image_urls  list     optional  (reference images, up to 4)
  elements    list     optional  (character/object references)

Note: there is NO cfg_scale or strength parameter on these endpoints.
Preservation is controlled entirely through the prompt.
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime

try:
    import fal_client
except ImportError:
    print("ERROR: fal-client not installed. Run: pip install fal-client")
    sys.exit(1)

# Prompts describe only what to ADD.
# The model retains original motion structure by design — prompt reinforces it.
MOTION_PRESETS = {
    "flying": (
        "Add natural secondary motion only: cape fabric rippling in wind, hair strands flowing, "
        "subtle muscle tension, atmospheric particles, clothing physics. "
        "Do not change character positions, scene layout, or art style. "
        "Preserve the flat cel-shaded cartoon look exactly."
    ),
    "talking": (
        "Add subtle life only: natural eye blinking, slight chest and shoulder breathing movement, "
        "micro-expressions, hair settling naturally. "
        "Do not change character positions, scene layout, or art style. "
        "Preserve the flat cel-shaded cartoon look exactly."
    ),
    "action": (
        "Enhance impact only: intensify speed lines, add dust and debris particles, "
        "secondary motion on loose clothing, hair and cape physics. "
        "Do not change existing animation, character positions, or art style. "
        "Preserve the flat cel-shaded cartoon look exactly."
    ),
    "walk": (
        "Add natural secondary motion only: hair bounce with each step, fabric settling, "
        "arm swing physics, subtle weight shift. "
        "Do not change character positions, scene layout, or art style. "
        "Preserve the flat cel-shaded cartoon look exactly."
    ),
    "generic": (
        "Add subtle secondary motion only: natural hair physics, fabric movement, "
        "characters breathe and blink naturally, atmospheric depth. "
        "Do not change any existing motion, character positions, or scene composition. "
        "Preserve the flat cel-shaded cartoon art style exactly."
    ),
}

# Confirmed working endpoints only (verified against fal.ai docs May 2026)
TIERS = {
    "o3_pro": {
        "endpoint": "fal-ai/kling-video/o3/pro/video-to-video/edit",
        "label": "O3 Pro",
        "cost": "~$0.17/sec of input",
    },
    "o1_standard": {
        "endpoint": "fal-ai/kling-video/o1/standard/video-to-video/edit",
        "label": "O1 Standard",
        "cost": "~$0.17/sec of input",
    },
}

MAX_FILE_SIZE_MB = 200
MIN_DURATION_S = 3
MAX_DURATION_S = 10


def validate_video(video_path: str) -> None:
    """Raise if the video doesn't meet Kling's input constraints."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0:
            duration = total_frames / fps
            if duration < MIN_DURATION_S:
                raise ValueError(
                    f"Clip is {duration:.1f}s — Kling requires at least {MIN_DURATION_S}s."
                )
            if duration > MAX_DURATION_S:
                raise ValueError(
                    f"Clip is {duration:.1f}s — Kling maximum is {MAX_DURATION_S}s. Trim it first."
                )
    except ImportError:
        pass  # opencv optional for validation only

    size_mb = Path(video_path).stat().st_size / 1_000_000
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File is {size_mb:.0f}MB — Kling maximum is {MAX_FILE_SIZE_MB}MB.")


def upload_video_to_fal(video_path: str) -> str:
    """Upload video to fal.ai storage and return public URL."""
    print("  Uploading video to fal.ai storage...")
    with open(video_path, "rb") as f:
        url = fal_client.upload(f.read(), "video/mp4")
    print(f"  Uploaded: {url}")
    return url


def enhance_with_kling(
    video_url: str,
    motion_type: str = "generic",
    custom_prompt: str = None,
    tier_key: str = "o3_pro",
) -> dict:
    """
    Send video to Kling video-to-video edit via fal.ai.

    The model retains original motion structure by design.
    Prompt controls what secondary motion gets added.
    """
    prompt = custom_prompt or MOTION_PRESETS.get(motion_type, MOTION_PRESETS["generic"])
    tier = TIERS.get(tier_key, TIERS["o3_pro"])

    print(f"  Sending to Kling {tier['label']} ({tier['cost']})...")
    print(f"  Prompt: {prompt[:100]}...")

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in getattr(update, "logs", []):
                msg = log.get("message", "") if isinstance(log, dict) else str(log)
                if msg:
                    print(f"  [fal] {msg}")

    result = fal_client.subscribe(
        tier["endpoint"],
        arguments={
            "video_url": video_url,
            "prompt": prompt,
            "keep_audio": True,
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    video_url_out = result["video"]["url"]
    print(f"  Done! Video URL: {video_url_out}")

    return {
        "video_url": video_url_out,
        "endpoint": tier["endpoint"],
        "tier": tier["label"],
        "cost_estimate": tier["cost"],
        "prompt_used": prompt,
    }


def download_video(url: str, output_path: str) -> str:
    """Download video from URL to local path."""
    print("  Downloading result...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = Path(output_path).stat().st_size / 1_000_000
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def run_pipeline(
    input_video_path: str,
    output_dir: str = "./outputs",
    motion_type: str = "generic",
    custom_prompt: str = None,
    tier_key: str = "o3_pro",
) -> dict:
    """
    Full pipeline: video in → enhanced video out.

    Args:
        input_video_path: Path to input video (MP4/MOV, 3-10s, max 200MB)
        output_dir:       Where to save the enhanced video
        motion_type:      One of: flying, talking, action, walk, generic
        custom_prompt:    Override the motion preset with your own prompt
        tier_key:         "o3_pro" (best) or "o1_standard" (budget)

    Returns:
        Dict with result info and paths
    """
    input_path = Path(input_video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    validate_video(input_video_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{input_path.stem}_enhanced_{timestamp}.mp4"
    out_full_path = str(output_path / out_filename)

    tier_label = TIERS.get(tier_key, TIERS["o3_pro"])["label"]
    print(f"\n{'='*60}")
    print(f"AnimUpgrade  |  Kling video-to-video")
    print(f"{'='*60}")
    print(f"Input:  {input_video_path}")
    print(f"Output: {out_full_path}")
    print(f"Motion: {motion_type}  |  Tier: {tier_label}")
    print(f"{'='*60}\n")

    print("Step 1: Uploading video to fal.ai...")
    video_url = upload_video_to_fal(input_video_path)

    print("\nStep 2: Enhancing with Kling video-to-video (~2-5 min)...")
    result = enhance_with_kling(
        video_url,
        motion_type=motion_type,
        custom_prompt=custom_prompt,
        tier_key=tier_key,
    )

    print("\nStep 3: Downloading result...")
    download_video(result["video_url"], out_full_path)

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"Original: {input_video_path}")
    print(f"Enhanced: {out_full_path}")
    print(f"Cost est: {result['cost_estimate']}")
    print(f"{'='*60}\n")

    return {
        "input": input_video_path,
        "output": out_full_path,
        "video_url": result["video_url"],
        "tier": result["tier"],
        "cost_estimate": result["cost_estimate"],
        "prompt_used": result["prompt_used"],
        "motion_type": motion_type,
        "timestamp": timestamp,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AnimUpgrade: Add secondary motion to flat TV animation"
    )
    parser.add_argument("input", help="Path to input video (MP4/MOV, 3-10s, max 200MB)")
    parser.add_argument("-o", "--output", default="./outputs", help="Output directory")
    parser.add_argument(
        "-m", "--motion",
        default="generic",
        choices=list(MOTION_PRESETS.keys()),
        help="Motion type preset",
    )
    parser.add_argument("-p", "--prompt", help="Custom prompt (overrides preset)")
    parser.add_argument(
        "-t", "--tier",
        default="o3_pro",
        choices=list(TIERS.keys()),
        help="o3_pro = best quality, o1_standard = budget",
    )

    args = parser.parse_args()

    if not os.environ.get("FAL_KEY"):
        print("\nERROR: FAL_KEY not set.")
        print("Get your key at: https://fal.ai/dashboard/keys")
        print("Then: export FAL_KEY=your_key_here\n")
        sys.exit(1)

    result = run_pipeline(
        input_video_path=args.input,
        output_dir=args.output,
        motion_type=args.motion,
        custom_prompt=args.prompt,
        tier_key=args.tier,
    )

    print(json.dumps(result, indent=2))
