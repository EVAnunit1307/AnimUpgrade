"""
AnimUpgrade â€” Stage 1 Pipeline (RIFE local interpolation)
==========================================================
Takes flat 2D TV animation and increases its effective frame rate using
rife-ncnn-vulkan, a portable Vulkan-based frame interpolation binary that
requires no CUDA, PyTorch, or cloud API.

â”€â”€ Which cloned repos this file uses, and why â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

rife-ncnn-vulkan (cloned at ./rife-ncnn-vulkan/):
  This is SOURCE CODE only â€” CMake project, no pre-built binary inside.
  The models/ subdirectory only goes up to rife-v4.6.
  ACTION REQUIRED: Download the pre-built binary from GitHub Releases:
    https://github.com/nihui/rife-ncnn-vulkan/releases
  The release ZIP contains the executable AND newer models (rife-v4.25).
  See README.md for exact install steps.

AFOptimizer (cloned at ./AFOptimizer/):
  Provides GPU-accelerated frame difference logic (CuPy + cv2.cuda) via
  AFOptimizer/frame_optimization_methods/gpu_acceleration.py and
  frameDifference.py. We borrow the absdiff pattern in utils/dedup.py.
  We do NOT import AFOptimizer directly (it has its own VideoProcessorBase
  dependency chain) â€” instead we replicated the core logic.

MultiPassDedup (cloned at ./MultiPassDedup/):
  Contains infer.py which is a complete dedup+interpolation pipeline using
  GMFSS/RIFE/GIMM models. Requires downloading model weights (~500 MB).
  For the GTX 1650 (4 GB VRAM) this is viable but tight â€” GMFSS is the
  better quality model but may OOM on 1080p. Use rife model flag for safety.
  This is an ALTERNATIVE to our Stage 1 pipeline â€” run it directly if you
  want the multi-pass dedup (smarter hold detection than MAD):
    python MultiPassDedup/infer.py -i input.mp4 -o out.mp4 -np 0 -fps 48 -m rife -s -st 0.3

TheAnimeScripter (cloned at ./TheAnimeScripter/):
  Full production-grade anime processing toolkit. Requires full PyTorch
  CUDA stack (see TheAnimeScripter/extra-requirements-windows.txt).
  Not used here to avoid the heavy dependency install for a simple Stage 1 run.
  Use TAS directly if you want its dedup (DedupSSIMCuda/DedupFlownetS) or
  RIFE interpolation via src/rifearches/:
    python TheAnimeScripter/main.py --input clip.mp4 --output out.mp4
      --interpolate --interpolate-factor 2 --dedup --dedup-method ssim

Practical-RIFE (cloned at ./Practical-RIFE/):
  PyTorch CUDA RIFE inference (inference_video.py). Good quality but requires
  torch+CUDA and downloading model weights. For the GTX 1650, use --fp16
  flag to halve VRAM usage. Uses ssim_matlab from model/pytorch_msssim for
  scene-change detection (SSIM > 0.996 threshold = skip interpolation).
  This is the CUDA alternative to rife-ncnn-vulkan (Vulkan) for this GPU.

flowframes (cloned at ./flowframes/):
  Empty â€” only .git directory present. Not usable.

â”€â”€ Execution path recommendation for GTX 1650 (4 GB VRAM) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  RIFE interpolation: Use rife-ncnn-vulkan with NCNN/Vulkan backend.
    WHY: The 1650 has only 4 GB VRAM. rife-ncnn-vulkan uses NCNN which
    is highly memory-efficient and runs on Vulkan â€” no CUDA overhead.
    Practical-RIFE (PyTorch CUDA) would use ~2.5 GB just for the model
    + CUDA context, leaving little headroom. With rife-ncnn-vulkan,
    the 1650 stays comfortably under 2 GB for 1080p inputs.
    Do NOT use TensorRT â€” the GTX 1650 (Turing TU117) does not support
    TensorRT in a meaningful way for RIFE-scale models.

  Frame deduplication: CPU MAD (this script) or MultiPassDedup SSIM.
    Both are CPU-bound anyway â€” GPU doesn't help much here.

  ffmpeg encoding: Use h264_nvenc for the final encode step (the 1650
    has NVENC) â€” see the --nvenc flag in this script.

  CuPy (optional): pip install cupy-cuda12x accelerates the MAD absdiff
    step in utils/dedup.py. Uses <50 MB VRAM. Safe on the 1650.

â”€â”€ Pipeline flow â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  1. Extract all frames from the input MP4 as PNG (ffmpeg)
  2. De-duplicate near-identical consecutive frames (hold detection)
  3. Interpolate new in-between frames (rife-ncnn-vulkan NCNN/Vulkan)
  4. Reassemble to MP4 at the correct output FPS, preserving audio (ffmpeg)

CLI usage:
  python pipeline.py input.mp4 --output output.mp4 --multiplier 2 --model rife-v4.25 --dedup-threshold 0.02

Environment variables:
  RIFE_BIN   Full path to the rife-ncnn-vulkan executable.
             If not set, the script searches common locations automatically.
             NOTE: The ./rife-ncnn-vulkan/ clone is SOURCE CODE, not a binary.
             Download the pre-built release from GitHub.

Dependencies (Python):
  numpy, opencv-python, tqdm  (pip install -r requirements.txt)
  Optional: cupy-cuda12x      (GPU-accelerated MAD in dedup step)

External binaries required:
  ffmpeg / ffprobe  â€” must be on PATH
  rife-ncnn-vulkan  â€” pre-built binary from GitHub Releases (NOT the clone)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports â€” utils/ lives next to this file.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from utils.dedup import deduplicate_frames  # noqa: E402


# ---------------------------------------------------------------------------
# RIFE binary discovery
# ---------------------------------------------------------------------------

# NOTE: The ./rife-ncnn-vulkan/ directory is the SOURCE CODE clone â€” it does NOT
# contain a built executable. We skip it in the search paths intentionally and
# instead look in the user's tools directory where the pre-built binary lives.
# The pre-built release ZIP contains rife-ncnn-vulkan.exe + all model folders.
_RIFE_SOURCE_CLONE = HERE / "rife-ncnn-vulkan"  # source clone â€” not a binary

RIFE_SEARCH_PATHS = [
    # Pre-built binary placed next to this script (not in the source clone dir)
    HERE / "rife-ncnn-vulkan.exe",          # Windows: dropped in project root
    HERE / "rife_bin" / "rife-ncnn-vulkan.exe",  # Windows: tidy subfolder
    HERE / "rife_bin" / "rife-ncnn-vulkan",      # Mac/Linux: tidy subfolder
    # Common user tool directories (Windows)
    Path("C:/tools/rife-ncnn-vulkan/rife-ncnn-vulkan.exe"),
    Path("C:/rife-ncnn-vulkan/rife-ncnn-vulkan.exe"),
    Path.home() / "tools" / "rife-ncnn-vulkan" / "rife-ncnn-vulkan.exe",
    Path.home() / "rife-ncnn-vulkan" / "rife-ncnn-vulkan.exe",
    # Mac/Linux
    Path.home() / "tools" / "rife-ncnn-vulkan" / "rife-ncnn-vulkan",
    Path.home() / "rife-ncnn-vulkan" / "rife-ncnn-vulkan",
    Path("/usr/local/bin/rife-ncnn-vulkan"),
    Path("/opt/rife-ncnn-vulkan/rife-ncnn-vulkan"),
]


def find_rife_binary() -> Path:
    """
    Find the rife-ncnn-vulkan executable.

    Search order:
      1. RIFE_BIN environment variable (highest priority)
      2. RIFE_SEARCH_PATHS list above
      3. PATH (via shutil.which)

    Raises SystemExit with a clear install message if not found.
    """
    # 1. Explicit env var
    env_bin = os.environ.get("RIFE_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file():
            return p
        print(f"\n[ERROR] RIFE_BIN is set to '{env_bin}' but that file does not exist.")
        print("        Please update RIFE_BIN to point to the rife-ncnn-vulkan executable.\n")
        sys.exit(1)

    # 2. Common locations
    for candidate in RIFE_SEARCH_PATHS:
        if candidate.is_file():
            return candidate

    # 3. PATH
    which = shutil.which("rife-ncnn-vulkan")
    if which:
        return Path(which)

    # Not found â€” print helpful install instructions and exit.
    print("\n" + "=" * 65)
    print("ERROR: rife-ncnn-vulkan binary not found.")
    print("=" * 65)
    print()
    print("IMPORTANT: The ./rife-ncnn-vulkan/ directory in your project is the")
    print("SOURCE CODE clone â€” it does NOT contain a pre-built executable.")
    print("You need the pre-built binary from the GitHub Releases page.")
    print()
    print("Download the latest release for your platform from:")
    print("  https://github.com/nihui/rife-ncnn-vulkan/releases")
    print()
    print("The release ZIP also includes newer models (rife-v4.25 etc.) that")
    print("the source clone does NOT have (source clone only goes to rife-v4.6).")
    print()
    print("â”€â”€â”€ Windows (GTX 1650 â€” NCNN/Vulkan path recommended) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print("  1. Download rife-ncnn-vulkan-YYYYMMDD-windows.zip")
    print("  2. Extract to C:\\tools\\rife-ncnn-vulkan\\")
    print("     The ZIP contains rife-ncnn-vulkan.exe + all model folders.")
    print("  3. Set the environment variable (PowerShell):")
    print('     $env:RIFE_BIN = "C:\\tools\\rife-ncnn-vulkan\\rife-ncnn-vulkan.exe"')
    print("     To persist: [System.Environment]::SetEnvironmentVariable(")
    print('       "RIFE_BIN", "C:\\tools\\rife-ncnn-vulkan\\rife-ncnn-vulkan.exe", "User")')
    print()
    print("  WHY NCNN/Vulkan instead of CUDA for the GTX 1650:")
    print("    - 4 GB VRAM is tight with PyTorch CUDA overhead")
    print("    - rife-ncnn-vulkan uses Vulkan which is more memory-efficient")
    print("    - Same quality output, lower VRAM footprint")
    print("    - Do NOT use TensorRT on the 1650 (TU117 has limited TRT support)")
    print()
    print("â”€â”€â”€ Mac â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print("  1. Download rife-ncnn-vulkan-YYYYMMDD-macos.zip")
    print("  2. Extract to ~/tools/rife-ncnn-vulkan/")
    print("  3. chmod +x ~/tools/rife-ncnn-vulkan/rife-ncnn-vulkan")
    print("  4. export RIFE_BIN=\"$HOME/tools/rife-ncnn-vulkan/rife-ncnn-vulkan\"")
    print("     Add to ~/.zshrc for persistence.")
    print()
    print("â”€â”€â”€ Linux â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print("  1. Download rife-ncnn-vulkan-YYYYMMDD-ubuntu.zip")
    print("  2. chmod +x rife-ncnn-vulkan && sudo mv it to /usr/local/bin/")
    print("     OR: export RIFE_BIN=\"/path/to/rife-ncnn-vulkan\"")
    print("=" * 65 + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _require_ffmpeg() -> None:
    """Exit with a clear message if ffmpeg / ffprobe are missing from PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"\n[ERROR] '{tool}' not found on PATH.")
            print("Install instructions:")
            print("  Windows : winget install ffmpeg")
            print("            OR: choco install ffmpeg")
            print("  Mac     : brew install ffmpeg")
            print("  Linux   : sudo apt install ffmpeg\n")
            sys.exit(1)


def get_video_fps(input_path: str) -> float:
    """
    Use ffprobe to detect the frame rate of the input video.

    Returns a float (e.g. 23.976, 24.0, 29.97, 60.0).
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    raw = result.stdout.strip()  # e.g. "24000/1001" or "24/1"
    fps = float(Fraction(raw))
    return fps


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

def extract_frames(input_path: str, frames_dir: Path) -> None:
    """
    Use ffmpeg to extract every frame of the input MP4 as a PNG.

    Frames are written to frames_dir as zero-padded names:
        00000001.png, 00000002.png, â€¦

    Args:
        input_path: Path to the source MP4/MOV.
        frames_dir: Directory to write PNGs into (created if missing).
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", input_path,
        str(frames_dir / "%08d.png"),
        "-hide_banner",
        "-loglevel", "warning",
    ]
    print(f"[ffmpeg] Extracting frames â†’ {frames_dir}")
    subprocess.run(cmd, check=True)

    count = len(list(frames_dir.glob("*.png")))
    print(f"[ffmpeg] Extracted {count} frames.")


def run_rife(
    rife_bin: Path,
    frames_dir: Path,
    output_dir: Path,
    multiplier: int,
    model: str,
) -> None:
    """
    Call rife-ncnn-vulkan to interpolate in-between frames.

    rife-ncnn-vulkan processes the entire frames_dir directory and writes
    the interpolated sequence to output_dir.

    Flags used:
      -i   input directory of PNG frames
      -o   output directory for interpolated PNG frames
      -m   model name (e.g. rife-v4.25)
      -n   total output frame count = (input_count - 1) * multiplier + 1
           (rife-ncnn-vulkan calculates this correctly when given -n)
      -f   output filename format (rife uses its own numbering)

    Note: rife-ncnn-vulkan uses -n to set the TARGET frame count, not a
    multiplier directly. We calculate it: n_out = (n_in - 1) * M + 1
    which gives exactly M-1 new frames between each pair.

    Args:
        rife_bin:   Path to the rife-ncnn-vulkan executable.
        frames_dir: Input directory containing de-duplicated PNGs.
        output_dir: Directory rife will write interpolated PNGs into.
        multiplier: 2 = double framerate, 4 = quadruple, etc.
        model:      Model subdirectory name (e.g. "rife-v4.25").
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_in = len(list(frames_dir.glob("*.png")))
    if n_in == 0:
        raise RuntimeError(f"No input frames found in {frames_dir}")

    n_out = (n_in - 1) * multiplier + 1

    print(f"\n[rife] Model       : {model}")
    print(f"[rife] Input frames: {n_in}")
    print(f"[rife] Multiplier  : {multiplier}x")
    print(f"[rife] Output target: {n_out} frames")

    cmd = [
        str(rife_bin),
        "-i", str(frames_dir),
        "-o", str(output_dir),
        "-m", model,
        "-n", str(n_out),
    ]

    print(f"[rife] Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    n_written = len(list(output_dir.glob("*.png")))
    print(f"\n[rife] Done. Wrote {n_written} frames to {output_dir}")


def reassemble_video(
    frames_dir: Path,
    input_path: str,
    output_path: str,
    output_fps: float,
    use_nvenc: bool = False,
) -> None:
    """
    Use ffmpeg to reassemble interpolated PNGs into an MP4, preserving audio.

    Encoder selection (mirrors AFOptimizer gpu_acceleration.get_hardware_encoder):
      NVENC (h264_nvenc): preferred on GTX 1650 â€” dedicated HW block, fast,
                          leaves GPU compute free. Probed before use.
      libx264 (CPU):      fallback if nvenc unavailable or not requested.

    Args:
        frames_dir:  Directory containing the interpolated PNG sequence.
        input_path:  Original video â€” its audio stream is remuxed into output.
        output_path: Destination MP4 path.
        output_fps:  Target playback framerate of the output video.
        use_nvenc:   If True, attempt to use h264_nvenc (GTX 1650 NVENC).
                     Falls back to libx264 automatically if nvenc is unavailable.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # rife-ncnn-vulkan may output frames as %08d.png or with its own naming.
    # Detect the pattern by looking at what's actually there.
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No output frames found in {frames_dir}")

    # Build the input pattern. rife-ncnn-vulkan uses %08d.png zero-padded names.
    frame_pattern = str(frames_dir / "%08d.png")

    # GPU encoder selection â€” from AFOptimizer/frame_optimization_methods/gpu_acceleration.py
    # get_hardware_encoder() logic (we inline it here to avoid the dependency chain).
    # For GTX 1650: h264_nvenc is available and recommended â€” NVENC runs on a
    # dedicated hardware block, freeing the GPU compute for other tasks.
    # Fall back to libx264 (CPU) if nvenc probe fails.
    encoder = "libx264"
    encoder_args = ["-crf", "18", "-preset", "slow"]
    if use_nvenc:
        try:
            probe = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=3,
            )
            if probe.returncode == 0 and "h264_nvenc" in probe.stdout:
                encoder = "h264_nvenc"
                encoder_args = ["-rc", "constqp", "-qp", "18"]  # nvenc equiv of CRF 18
                print("[ffmpeg] Using h264_nvenc (GTX 1650 NVENC â€” frees GPU compute)")
            else:
                print("[ffmpeg] h264_nvenc not found â€” falling back to libx264")
        except Exception:
            print("[ffmpeg] nvenc probe failed â€” falling back to libx264")

    cmd = [
        "ffmpeg",
        "-y",                          # overwrite output without asking
        "-framerate", str(output_fps),
        "-i", frame_pattern,           # interpolated frame sequence
        "-i", input_path,              # original video (for audio)
        "-map", "0:v",                 # video from interpolated frames
        "-map", "1:a?",                # audio from original (? = ignore if missing)
        "-c:v", encoder,
        *encoder_args,
        "-pix_fmt", "yuv420p",         # broad compatibility
        output_path,
        "-hide_banner",
        "-loglevel", "warning",
    ]

    print(f"\n[ffmpeg] Reassembling at {output_fps:.3f} fps â†’ {output_path}")
    subprocess.run(cmd, check=True)

    size_mb = Path(output_path).stat().st_size / 1_048_576
    print(f"[ffmpeg] Done. Output: {output_path}  ({size_mb:.1f} MB)  encoder={encoder}")


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    output_path: str,
    multiplier: int = 2,
    model: str = "rife-v4.25",
    dedup_threshold: float = 0.02,
    keep_temp: bool = False,
    use_nvenc: bool = False,
) -> None:
    """
    Full Stage 1 pipeline: extract â†’ dedup â†’ rife interpolate â†’ reassemble.

    Args:
        input_path:       Path to the input MP4/MOV.
        output_path:      Destination path for the enhanced MP4.
        multiplier:       Frame rate multiplier (2 = 2x, 4 = 4x).
        model:            rife-ncnn-vulkan model name (default: rife-v4.25).
        dedup_threshold:  MAD threshold for duplicate frame removal.
                          0.0 = off, 0.02 = standard, 0.05 = aggressive.
        keep_temp:        If True, don't delete temp frame directories after
                          completion (useful for debugging).
        use_nvenc:        If True, attempt h264_nvenc for final encode step.
                          Recommended for GTX 1650 (uses NVENC HW block).
                          Falls back to libx264 automatically.
    """
    _require_ffmpeg()
    rife_bin = find_rife_binary()

    print("\n" + "=" * 65)
    print("AnimUpgrade  |  Stage 1: RIFE Local Interpolation")
    print("=" * 65)
    print(f"  Input      : {input_path}")
    print(f"  Output     : {output_path}")
    print(f"  Multiplier : {multiplier}x")
    print(f"  Model      : {model}")
    print(f"  Dedup thr  : {dedup_threshold}")
    print(f"  RIFE bin   : {rife_bin}")
    print(f"  GPU encode : {'h264_nvenc (GTX 1650 NVENC)' if use_nvenc else 'libx264 (CPU)'}")
    print("=" * 65 + "\n")

    # â”€â”€ Detect input FPS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[pipeline] Detecting input FPS via ffprobe...")
    input_fps = get_video_fps(input_path)
    output_fps = input_fps * multiplier
    print(f"[pipeline] Input FPS  : {input_fps:.4f}")
    print(f"[pipeline] Output FPS : {output_fps:.4f}  ({multiplier}x)")

    # â”€â”€ Create temp workspace â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tmp_root = Path(tempfile.mkdtemp(prefix="animupgrade_"))
    raw_frames_dir   = tmp_root / "input_frames"
    dedup_frames_dir = tmp_root / "dedup_frames"
    rife_output_dir  = tmp_root / "output_frames"

    print(f"\n[pipeline] Temp workspace: {tmp_root}")

    try:
        # â”€â”€ Step 1: Extract frames â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print("\n[pipeline] Step 1/4 â€” Extracting frames...")
        extract_frames(input_path, raw_frames_dir)

        # â”€â”€ Step 2: De-duplicate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print("\n[pipeline] Step 2/4 â€” De-duplicating holds...")
        deduplicate_frames(raw_frames_dir, dedup_frames_dir, threshold=dedup_threshold)

        # â”€â”€ Step 3: RIFE interpolation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print("\n[pipeline] Step 3/4 â€” RIFE interpolation...")
        run_rife(rife_bin, dedup_frames_dir, rife_output_dir, multiplier, model)

        # â”€â”€ Step 4: Reassemble â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print("\n[pipeline] Step 4/4 â€” Reassembling video...")
        reassemble_video(rife_output_dir, input_path, output_path, output_fps, use_nvenc=use_nvenc)

    finally:
        if keep_temp:
            print(f"\n[pipeline] Temp files kept at: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"\n[pipeline] Temp workspace cleaned up.")

    print("\n" + "=" * 65)
    print("DONE!")
    print(f"  Input  : {input_path}  ({input_fps:.4f} fps)")
    print(f"  Output : {output_path}  ({output_fps:.4f} fps)")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description=(
            "AnimUpgrade Stage 1 â€” RIFE local frame interpolation.\n"
            "Removes duplicate holds from 2D animation, then uses\n"
            "rife-ncnn-vulkan to generate smooth in-between frames."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py clip.mp4 --output enhanced.mp4\n"
            "  python pipeline.py clip.mp4 --output out.mp4 --multiplier 4\n"
            "  python pipeline.py clip.mp4 --output out.mp4 --dedup-threshold 0.05\n"
            "  python pipeline.py clip.mp4 --output out.mp4 --keep-temp\n"
        ),
    )

    parser.add_argument(
        "input",
        help="Path to the input video (MP4 or MOV).",
    )
    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Path for the enhanced output MP4. (default: output.mp4)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        choices=[2, 4],
        help=(
            "Frame rate multiplier. 2 = double (e.g. 24â†’48 fps), "
            "4 = quadruple (e.g. 24â†’96 fps). (default: 2)"
        ),
    )
    parser.add_argument(
        "--model",
        default="rife-v4.25",
        help=(
            "rife-ncnn-vulkan model name. Must match a subdirectory "
            "bundled with the binary. (default: rife-v4.25)"
        ),
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.02,
        dest="dedup_threshold",
        help=(
            "Mean absolute difference threshold for hold de-duplication.\n"
            "  0.00 = off (not recommended for 2D animation)\n"
            "  0.02 = standard (default, good for Invincible / cel-shading)\n"
            "  0.05 = aggressive (for very limited / motion comic animation)\n"
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        dest="keep_temp",
        help="Keep intermediate frame directories for debugging.",
    )
    parser.add_argument(
        "--nvenc",
        action="store_true",
        dest="use_nvenc",
        help=(
            "Use h264_nvenc (NVIDIA NVENC) for the final encode step. "
            "Recommended for GTX 1650 and all NVIDIA GPUs â€” the NVENC block "
            "runs independently of GPU compute and is ~4x faster than libx264. "
            "Falls back to libx264 automatically if nvenc is unavailable."
        ),
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"\n[ERROR] Input file not found: {args.input}\n")
        sys.exit(1)

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        multiplier=args.multiplier,
        model=args.model,
        dedup_threshold=args.dedup_threshold,
        keep_temp=args.keep_temp,
        use_nvenc=args.use_nvenc,
    )


if __name__ == "__main__":
    main()

