"""
AnimUpgrade - Core Pipeline
Enhances flat TV animation with secondary motion using Kling video-to-video.
Preserves the original scene, characters, and art style — adds what's missing.
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

# Prompts describe only what to ADD, not what to replace.
# Kling v2v preserves the original scene — we're just telling it what's missing.
MOTION_PRESETS = {
    "flying": (
        "Add natural secondary motion: cape fabric rippling in wind, hair strands flowing, "
        "subtle muscle tension, atmospheric particles, clothes have natural physics. "
        "Preserve all existing character positions, art style, and scene composition exactly."
    ),
    "talking": (
        "Add subtle life: natural blinking, slight breathing movement in chest and shoulders, "
        "micro-expressions, hair settling naturally. "
        "Preserve all existing character positions, art style, and scene exactly."
    ),
    "action": (
        "Enhance impact: speed lines intensify, dust and debris particles, secondary motion "
        "on loose clothing, hair and cape physics during movement. "
        "Preserve existing animation and art style exactly."
    ),
    "walk": (
        "Add natural secondary motion: hair bounce with each step, fabric settling, "
        "arm swing physics, subtle weight shift. "
        "Preserve all existing character positions, art style, and scene exactly."
    ),
    "generic": (
        "Add subtle secondary motion throughout: natural hair physics, fabric movement, "
        "atmospheric depth, characters breathe and blink naturally. "
        "Preserve all existing positions, motion, and flat cartoon art style exactly."
    ),
}

# Available Kling v2v tiers
TIERS = {
    "o3_pro": {
        "endpoint": "fal-ai/kling-video/o3/pro/video-to-video/edit",
        "label": "O3 Pro",
        "cost": "~$0.17/sec of input",
    },
    "o1_pro": {
        "endpoint": "fal-ai/kling-video/o1/pro/video-to-video/edit",
        "label": "O1 Pro",
        "cost": "~$0.17/sec of input",
    },
    "o1_standard": {
        "endpoint": "fal-ai/kling-video/o1/standard/video-to-video/edit",
        "label": "O1 Standard",
        "cost": "~$0.17/sec of input",
    },
}


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
    cfg_scale: float = 0.4,
) -> dict:
    """
    Send video to Kling video-to-video edit via fal.ai and get back enhanced video.

    cfg_scale controls how strongly the prompt is applied:
      0.2-0.4 = subtle, preserves original almost completely (what we want)
      0.7+    = heavy changes, starts replacing things (avoid)

    Returns dict with keys: video_url, tier, cost_estimate, prompt_used
    """
    prompt = custom_prompt or MOTION_PRESETS.get(motion_type, MOTION_PRESETS["generic"])
    tier = TIERS.get(tier_key, TIERS["o3_pro"])

    print(f"  Sending to Kling {tier['label']} ({tier['cost']})...")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  cfg_scale: {cfg_scale} (lower = more preservation)")

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
            "cfg_scale": cfg_scale,
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
    cfg_scale: float = 0.4,
) -> dict:
    """
    Full pipeline: video in → enhanced video out.

    Args:
        input_video_path: Path to input video (MP4)
        output_dir: Where to save the enhanced video
        motion_type: One of: flying, talking, action, walk, generic
        custom_prompt: Override the motion preset with your own prompt
        tier_key: Quality tier — "o3_pro", "o1_pro", or "o1_standard"
        cfg_scale: How strongly to apply the prompt (0.2-0.4 recommended)

    Returns:
        Dict with result info and paths
    """
    input_path = Path(input_video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{input_path.stem}_enhanced_{timestamp}.mp4"
    out_full_path = str(output_path / out_filename)

    print(f"\n{'='*60}")
    print(f"AnimUpgrade Pipeline  |  Kling video-to-video")
    print(f"{'='*60}")
    print(f"Input:  {input_video_path}")
    print(f"Output: {out_full_path}")
    print(f"Motion: {motion_type}  |  Tier: {TIERS.get(tier_key, TIERS['o3_pro'])['label']}")
    print(f"{'='*60}\n")

    print("Step 1: Uploading video to fal.ai...")
    video_url = upload_video_to_fal(input_video_path)

    print("\nStep 2: Enhancing with Kling video-to-video (~2-5 min)...")
    result = enhance_with_kling(
        video_url,
        motion_type=motion_type,
        custom_prompt=custom_prompt,
        tier_key=tier_key,
        cfg_scale=cfg_scale,
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
        description="AnimUpgrade: Enhance flat TV animation with secondary motion"
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("-o", "--output", default="./outputs", help="Output directory")
    parser.add_argument(
        "-m", "--motion",
        default="generic",
        choices=list(MOTION_PRESETS.keys()),
        help="Motion type preset",
    )
    parser.add_argument("-p", "--prompt", help="Custom motion prompt (overrides preset)")
    parser.add_argument(
        "-t", "--tier",
        default="o3_pro",
        choices=list(TIERS.keys()),
        help="Quality tier: o3_pro (best), o1_pro, o1_standard (budget)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=0.4,
        help="Prompt strength 0.0-1.0 (default 0.4 — low preserves original)",
    )

    args = parser.parse_args()

    if not os.environ.get("FAL_KEY"):
        print("\nERROR: FAL_KEY environment variable not set.")
        print("Get your key at: https://fal.ai/dashboard/keys")
        print("Then run: export FAL_KEY=your_key_here\n")
        sys.exit(1)

    result = run_pipeline(
        input_video_path=args.input,
        output_dir=args.output,
        motion_type=args.motion,
        custom_prompt=args.prompt,
        tier_key=args.tier,
        cfg_scale=args.cfg_scale,

    )

    print(json.dumps(result, indent=2))
