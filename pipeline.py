"""
AnimUpgrade - Core Pipeline
Extracts key frames from a video clip, sends each to Hailuo 2.3 Pro
via fal.ai for motion upgrade, then downloads the result.
"""

import os
import sys
import json
import base64
import shutil
import tempfile
import requests
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import fal_client
except ImportError:
    print("ERROR: fal-client not installed. Run: pip install fal-client")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv not installed. Run: pip install opencv-python-headless")
    sys.exit(1)

# ── Motion prompts tuned for Western TV animation ──────────────────────────
# These are the key to getting good results without style drift.
# Specific to Invincible-style animation — flat cel-shaded, dynamic but clean.

MOTION_PRESETS = {
    "flying": (
        "Cel-shaded cartoon character flying through air, flat animation style, "
        "dynamic cape motion, secondary motion on hair and costume, atmospheric "
        "parallax background, smooth arcing trajectory, [Static shot]"
    ),
    "talking": (
        "Cel-shaded cartoon character talking, subtle head bob, natural blink, "
        "slight shoulder rise with breath, flat animation style, "
        "minimal camera movement, [Static shot]"
    ),
    "action": (
        "Cel-shaded cartoon action scene, impact frames, smear frames, "
        "dynamic motion blur, flat animation style, punchy movement, "
        "secondary motion on loose elements, [Static shot]"
    ),
    "walk": (
        "Cel-shaded cartoon character walking, natural weight shift, "
        "arm swing, subtle bounce, flat animation style, "
        "smooth looping motion, [Static shot]"
    ),
    "generic": (
        "Cel-shaded Western cartoon animation, natural secondary motion, "
        "flat color animation style, smooth fluid movement, "
        "preserve original art style exactly, [Static shot]"
    ),
}


def extract_key_frame(video_path: str, frame_number: int = None) -> str:
    """
    Extract a single key frame from a video as a PNG.
    If frame_number is None, picks the middle frame.
    Returns path to the extracted frame PNG.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_number is None:
        frame_number = total_frames // 2

    frame_number = max(0, min(frame_number, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")

    out_path = video_path.replace(".mp4", f"_frame_{frame_number}.png")
    out_path = str(Path(tempfile.mkdtemp()) / f"frame_{frame_number}.png")
    cv2.imwrite(out_path, frame)

    duration = total_frames / fps if fps > 0 else 0
    print(f"  Extracted frame {frame_number}/{total_frames} ({duration:.1f}s clip, {fps:.0f}fps)")
    return out_path


def image_to_base64_url(image_path: str) -> str:
    """Convert image file to base64 data URL for fal.ai upload."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{data}"


def upload_image_to_fal(image_path: str) -> str:
    """Upload image to fal.ai storage and return public URL."""
    print(f"  Uploading frame to fal.ai storage...")
    with open(image_path, "rb") as f:
        url = fal_client.upload(f.read(), "image/png")
    print(f"  Uploaded: {url}")
    return url


def upgrade_frame_with_hailuo(
    image_url: str,
    motion_type: str = "generic",
    custom_prompt: str = None,
    use_pro: bool = True,
    use_fast: bool = False,
) -> dict:
    """
    Send a frame to Hailuo 2.3 via fal.ai and get back animated video.
    
    Returns dict with keys: video_url, duration, cost_estimate
    """
    prompt = custom_prompt or MOTION_PRESETS.get(motion_type, MOTION_PRESETS["generic"])

    # Pick the right endpoint
    if use_fast and use_pro:
        endpoint = "fal-ai/minimax/hailuo-2.3-fast/pro/image-to-video"
        tier = "Fast Pro (1080p)"
        cost_est = "$0.49"
    elif use_fast:
        endpoint = "fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video"
        tier = "Fast Standard (768p)"
        cost_est = "$0.27"
    elif use_pro:
        endpoint = "fal-ai/minimax/hailuo-2.3/pro/image-to-video"
        tier = "Pro (1080p)"
        cost_est = "$0.49"
    else:
        endpoint = "fal-ai/minimax/hailuo-2.3/standard/image-to-video"
        tier = "Standard (768p)"
        cost_est = "$0.27"

    print(f"  Sending to Hailuo 2.3 {tier} (~{cost_est})...")
    print(f"  Prompt: {prompt[:80]}...")

    def on_queue_update(update):
        if update.status == "IN_PROGRESS":
            for log in getattr(update, "logs", []):
                msg = log.get("message", "") if isinstance(log, dict) else str(log)
                if msg:
                    print(f"  [fal] {msg}")

    result = fal_client.subscribe(
        endpoint,
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "prompt_optimizer": True,
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    video_url = result["video"]["url"]
    print(f"  Done! Video URL: {video_url}")

    return {
        "video_url": video_url,
        "endpoint": endpoint,
        "tier": tier,
        "cost_estimate": cost_est,
        "prompt_used": prompt,
    }


def download_video(url: str, output_path: str) -> str:
    """Download video from URL to local path."""
    print(f"  Downloading result...")
    response = requests.get(url, stream=True, timeout=60)
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
    frame_number: int = None,
    use_pro: bool = True,
    use_fast: bool = False,
) -> dict:
    """
    Full pipeline: video in → upgraded video out.
    
    Args:
        input_video_path: Path to input video (MP4 recommended)
        output_dir: Where to save the upgraded video
        motion_type: One of: flying, talking, action, walk, generic
        custom_prompt: Override the motion preset with your own prompt
        frame_number: Which frame to use (None = middle frame)
        use_pro: Use 1080p Pro tier (True) or 768p Standard (False)
        use_fast: Use Fast variant (lower quality, faster/cheaper)
    
    Returns:
        Dict with result info and paths
    """
    input_path = Path(input_video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{input_path.stem}_upgraded_{timestamp}.mp4"
    out_full_path = str(output_path / out_filename)

    print(f"\n{'='*60}")
    print(f"AnimUpgrade Pipeline")
    print(f"{'='*60}")
    print(f"Input:  {input_video_path}")
    print(f"Output: {out_full_path}")
    print(f"Motion: {motion_type}")
    print(f"{'='*60}\n")

    # Step 1: Extract key frame
    print("Step 1: Extracting key frame...")
    frame_path = extract_key_frame(input_video_path, frame_number)

    # Step 2: Upload to fal.ai
    print("\nStep 2: Uploading to fal.ai...")
    image_url = upload_image_to_fal(frame_path)

    # Step 3: Generate upgraded video
    print("\nStep 3: Generating upgraded animation (this takes ~2-4 min)...")
    result = upgrade_frame_with_hailuo(
        image_url,
        motion_type=motion_type,
        custom_prompt=custom_prompt,
        use_pro=use_pro,
        use_fast=use_fast,
    )

    # Step 4: Download result
    print("\nStep 4: Downloading result...")
    download_video(result["video_url"], out_full_path)

    # Cleanup temp frame
    try:
        os.remove(frame_path)
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"Original:  {input_video_path}")
    print(f"Upgraded:  {out_full_path}")
    print(f"Cost est:  {result['cost_estimate']}")
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
        description="AnimUpgrade: Upgrade flat TV animation using AI"
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
    parser.add_argument("-f", "--frame", type=int, help="Frame number to use (default: middle)")
    parser.add_argument("--standard", action="store_true", help="Use 768p Standard instead of 1080p Pro")
    parser.add_argument("--fast", action="store_true", help="Use Fast variant (cheaper, faster)")

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
        frame_number=args.frame,
        use_pro=not args.standard,
        use_fast=args.fast,
    )

    print(json.dumps(result, indent=2))
