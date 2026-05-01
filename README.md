# AnimUpgrade

**AI-powered animation upgrade tool.** Takes flat, limited TV animation — the flying PNGs, the sliding character models — and upgrades scenes with real secondary motion.

Built with Hailuo 2.3 Pro via fal.ai. No GPU required.

---

## The problem this solves

TV animation on a budget uses *limited animation* deliberately — characters slide across backgrounds, dialogue scenes have minimal motion, flying scenes are a single pose dragged across the screen. Shows like Invincible Season 3 do this constantly because it's the economic reality of TV budgets.

This tool takes those scenes and adds real motion: cape physics, hair movement, atmospheric parallax, natural secondary motion — while preserving the original art style exactly.

---

## Setup (5 minutes)

### 1. Get a fal.ai API key

Sign up at [fal.ai](https://fal.ai) → Dashboard → API Keys → Create Key

New accounts get ~$1 free credit = about 2 Pro clips to test with.

### 2. Install dependencies

```bash
pip install fal-client opencv-python-headless requests gradio
```

### 3. Set your API key

```bash
export FAL_KEY=your_key_here
```

Or paste it in the web UI.

---

## Usage

### Web app (recommended)

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### Command line

```bash
# Basic usage - auto-detects motion type
python pipeline.py my_clip.mp4

# Specify scene type for better results
python pipeline.py flying_scene.mp4 --motion flying
python pipeline.py dialogue.mp4 --motion talking
python pipeline.py fight.mp4 --motion action

# Custom prompt
python pipeline.py clip.mp4 --prompt "Character floats upward, cape billowing, flat cel-shaded style"

# Use cheaper/faster Standard tier
python pipeline.py clip.mp4 --standard

# Use Fast variant (quicker generation)
python pipeline.py clip.mp4 --fast

# Choose which frame to use as reference
python pipeline.py clip.mp4 --frame 30
```

### As a Python module

```python
from pipeline import run_pipeline

result = run_pipeline(
    input_video_path="invincible_flying.mp4",
    output_dir="./outputs",
    motion_type="flying",
)

print(result["output"])       # Path to upgraded video
print(result["cost_estimate"]) # "$0.49"
print(result["prompt_used"])  # The exact prompt sent to Hailuo
```

---

## How it works

1. **Extract**: Pulls the best key frame from your input clip
2. **Upload**: Sends the frame to fal.ai storage
3. **Generate**: Hailuo 2.3 Pro animates the frame with natural motion, preserving your art style
4. **Download**: Returns the upgraded 6-10 second clip

The model animates from a single reference frame — it understands the cel-shaded, flat-color aesthetic of animation and doesn't try to make it photorealistic.

---

## Motion presets

| Preset | Best for |
|--------|----------|
| `flying` | Aerial scenes, characters in flight, cape/hair physics |
| `talking` | Dialogue scenes, subtle head movement, natural blinks |
| `action` | Fight scenes, impact moments, smear frames |
| `walk` | Walking, running, movement cycles |
| `generic` | Everything else — good default |

---

## Pricing

All pricing through fal.ai:

| Tier | Resolution | Cost per clip |
|------|-----------|---------------|
| Pro | 1080p | ~$0.49 |
| Fast Pro | 1080p | ~$0.49 (faster) |
| Standard | 768p | ~$0.27 |

A 30-clip test batch (enough for a full episode analysis) costs ~$15.

---

## Tips for best results

- **Keep clips short**: 3-8 seconds works best. Longer clips can lose temporal coherence.
- **Pick the right preset**: `flying` for aerial scenes makes a huge difference vs `generic`
- **Add `[Static shot]`** to your custom prompt if you want the camera to hold still
- **Invincible-style tip**: The model responds well to "flat cel-shaded cartoon" in prompts — it locks in the art style
- **Iterate cheap**: Use `--fast --standard` first to test the motion, then run the final version at Pro quality

---

## Project structure

```
animupgrade/
├── pipeline.py    # Core logic — extract, upload, generate, download
├── app.py         # Gradio web interface
├── README.md      # This file
├── uploads/       # Temp upload storage
└── outputs/       # Generated videos saved here
```

---

## Next steps / roadmap

- [ ] Multi-frame extraction — use multiple frames as reference for longer clips
- [ ] Batch processing — upgrade a whole episode's limited animation scenes at once
- [ ] EbSynth integration — use upgraded motion as reference, propagate to original clip for pixel-perfect style lock
- [ ] Kling 3.0 fallback — auto-switch model if Hailuo rate-limits
- [ ] Scene detection — auto-detect scene boundaries and process each scene independently

---

## Contributing / feedback

Drop issues, feedback, and before/after clips. The more Invincible scenes the better.
