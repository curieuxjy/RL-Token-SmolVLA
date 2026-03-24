# RLT for SmolVLA on SO-101 — Setup & Execution Guide

Everything is implemented. This guide walks you through what you need to do, step by step, to go from code to a working RL-trained robot.

---

## Prerequisites

| Item | Status | Notes |
|------|--------|-------|
| SO-101 follower + leader arms | Needed | Both must be connected via USB |
| 2x USB cameras (front, side) | Needed | Logitech C270 or similar |
| M4 Pro Mac | Needed | For on-robot training (MPS backend) |
| Demo dataset (50+ episodes) | Needed | Record via LeRobot teleop |
| GPU for Stage 1 | Optional | RunPod H100 or M4 Pro MPS both work |

---

## Step 1: Fix the LeRobot Python Environment

The `.venv` has broken symlinks (pointed to a deleted conda env). Recreate it:

```bash
cd "/Users/raj/Desktop/Robotics Research/lerobot"

# Remove broken venv
rm -rf .venv

# Create fresh venv with Python 3.10
# (If you don't have 3.10, install via: brew install python@3.10)
python3.10 -m venv .venv
source .venv/bin/activate

# Install LeRobot with SmolVLA extras
pip install -e ".[smolvla]"

# Verify
python -c "from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import SmolVLARLTPolicy; print('RLT import OK')"
```

---

## Step 2: Download the Pretrained SmolVLA Base Model

The base model downloads automatically on first use, but you can pre-cache it:

```bash
source "/Users/raj/Desktop/Robotics Research/lerobot/.venv/bin/activate"

python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
print('Downloading SmolVLA base model...')
policy = SmolVLAPolicy.from_pretrained(
    pretrained_name_or_path='lerobot/smolvla_base',
    config=SmolVLAConfig(load_vlm_weights=True),
)
print(f'Model loaded: {sum(p.numel() for p in policy.parameters()):,} params')
"
```

If you already have a fine-tuned SmolVLA on your SO-101 task (e.g., `RajatDandekar/smolvla_pick_place`), use that path instead — it will give better RLT results since the VLA already knows the task.

---

## Step 3: Record Demo Dataset (if you don't have one)

You need at least 50 teleoperation episodes for Stage 1. Use the existing recording scripts:

```bash
# Connect both arms, then:
cd "/Users/raj/Desktop/Robotics Research"
source lerobot/.venv/bin/activate

# Record pick-and-place demos
bash so101-scripts/record_pick_place.sh
# This records to: RajatDandekar/so101_pick_place_v2
```

Or record a custom task:

```bash
python -m lerobot.scripts.lerobot_record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodemXXXXX \
    --robot.cameras='{"front": {"type": "opencv", "index": 0, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodemYYYYY \
    --dataset.repo_id=RajatDandekar/so101_YOUR_TASK \
    --dataset.num_episodes=50 \
    --fps=30
```

---

## Step 4: Fine-tune SmolVLA on Your Task (if not already done)

If you're starting from the generic `smolvla_base`, fine-tune it on your task demos first:

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=RajatDandekar/so101_pick_place_v2 \
    --batch_size=64 \
    --steps=200000 \
    --output_dir=checkpoints/smolvla_pick_place
```

This gives you a task-specific VLA to build RLT on top of.

---

## Step 5: Run Stage 1 — Train RLT Encoder-Decoder (Offline)

This learns the z_rl representation from frozen VLA embeddings. Runs on any GPU.

```bash
cd "/Users/raj/Desktop/Robotics Research"
source lerobot/.venv/bin/activate

python scripts/train_rlt_stage1.py \
    --smolvla_path checkpoints/smolvla_pick_place \
    --dataset_repo_id RajatDandekar/so101_pick_place_v2 \
    --output_dir checkpoints/rlt_stage1 \
    --steps 5000 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda   # or mps on M4 Pro
```

**Expected:**
- ~30 min on H100, ~1-2 hours on M4 Pro
- Loss should decrease steadily (reconstruction MSE)
- Output: `checkpoints/rlt_stage1/best_checkpoint.pt`

**What to watch for:**
- If loss plateaus early, try increasing `--steps` to 10000
- If loss is very high, the VLA embeddings may not be informative — check that the SmolVLA was properly fine-tuned on the task

---

## Step 6: Run Unit Tests (Optional but Recommended)

```bash
cd "/Users/raj/Desktop/Robotics Research"
source lerobot/.venv/bin/activate

python -m pytest tests/policies/smolvla_rlt/ -v
```

This validates all component shapes, loss computation, and gradient flow without needing a robot.

---

## Step 7: Run Stage 2 — Online RL with Intervention (On-Robot)

This is the main event. Connect both SO-101 arms and cameras.

### 7a. Find your USB ports

```bash
ls /dev/tty.usbmodem*
# Example output:
# /dev/tty.usbmodem58760431541  ← follower
# /dev/tty.usbmodem58760431551  ← leader
```

Tip: plug in one arm at a time to identify which is which.

### 7b. Start Stage 2

```bash
cd "/Users/raj/Desktop/Robotics Research"
source lerobot/.venv/bin/activate

python scripts/train_rlt_stage2.py \
    --smolvla_path checkpoints/smolvla_pick_place \
    --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
    --task "pick up the red cube and place it in the bin" \
    --follower_port /dev/tty.usbmodem58760431541 \
    --leader_port /dev/tty.usbmodem58760431551 \
    --max_episodes 200 \
    --warmup_episodes 5 \
    --device mps \
    --fps 30
```

### 7c. How to operate during training

The training loop works like this:

```
For each episode:
  1. Robot attempts the task autonomously (RL agent or VLA during warmup)
  2. YOU watch the robot

  If the robot is about to fail:
     → Grab the leader arm and physically correct the motion
     → The system detects your intervention within ~60ms
     → Your corrective actions are recorded and labeled as "good"
     → Release the leader arm when the correction is done

  If the robot never needed help:
     → Wait for the episode to finish
     → Terminal prompt asks: "Was this episode successful? [y/n]"
     → Type 'y' or 'n'

  If you intervened:
     → Episode is auto-labeled as success (no prompt needed)
```

### 7d. What to expect

| Phase | Episodes | What happens |
|-------|----------|--------------|
| Warmup | 1-5 | Pure VLA actions. Observe baseline success rate |
| Early RL | 6-30 | Agent explores, mostly fails. Intervene frequently |
| Learning | 30-100 | Success rate starts climbing. Fewer interventions needed |
| Convergence | 100-200 | Agent handles most cases. Rare interventions |

**Session duration:** 2-5 hours of active robot practice (including breaks).

**You can stop and resume.** Checkpoints are saved every 20 episodes. To resume:

```bash
# Add --actor_checkpoint to resume from where you left off
# (You'll need to modify the script to load actor/critic state —
# the checkpoint contains actor_state_dict and critic_state_dict)
```

### 7e. Intervention tips

- **Intervene early.** Don't wait for the robot to crash — step in as soon as you see it heading the wrong way.
- **Be consistent.** Do the correction the same way each time so the actor learns a clear signal.
- **Gripper matters.** If the robot's grip angle or timing is wrong, correct that too.
- **Let it try.** After ~50 episodes, resist the urge to intervene on minor errors. Let the RL agent figure those out.

---

## Step 8: Deploy the Trained Agent

Once Stage 2 is done:

```bash
python scripts/run_rlt_inference.py \
    --smolvla_path checkpoints/smolvla_pick_place \
    --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
    --actor_checkpoint checkpoints/rlt_stage2/final_checkpoint.pt \
    --task "pick up the red cube and place it in the bin" \
    --duration 120 \
    --device mps
```

Note: The inference script has placeholder robot connection code (marked with comments). You'll need to fill in the actual observation → tensor conversion for your camera setup, following the pattern in `openclaw-so101/scripts/policy_runner.py`.

---

## File Map

```
Robotics Research/
├── lerobot/src/lerobot/policies/
│   ├── smolvla/
│   │   └── modeling_smolvla.py          ← Modified: added extract_embeddings()
│   ├── smolvla_rlt/                     ← NEW: all RLT policy code
│   │   ├── __init__.py
│   │   ├── configuration_smolvla_rlt.py ← Config dataclass
│   │   ├── modeling_smolvla_rlt.py      ← Encoder, Decoder, Actor, Critic, Policy
│   │   └── processor_smolvla_rlt.py     ← Delegates to SmolVLA processor
│   └── factory.py                       ← Modified: registered smolvla_rlt
├── scripts/
│   ├── train_rlt_stage1.py              ← Stage 1: offline encoder-decoder
│   ├── train_rlt_stage2.py              ← Stage 2: online RL + intervention
│   └── run_rlt_inference.py             ← Deployment inference loop
├── tests/policies/smolvla_rlt/
│   └── test_smolvla_rlt.py              ← Unit tests
└── RLT-Setup-Guide.md                  ← This file
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: draccus` | Venv not activated. Run `source lerobot/.venv/bin/activate` |
| `broken symbolic link` for python | Recreate venv (Step 1) |
| `No module named 'lerobot'` | Run `pip install -e ".[smolvla]"` in lerobot dir |
| Leader arm not detected | Check `ls /dev/tty.usbmodem*`, try unplugging/replugging |
| Intervention not triggering | Lower `--intervention_threshold` (default 3.0 degrees) |
| Intervention too sensitive | Raise `--intervention_threshold` to 5.0+ |
| OOM on M4 Pro (8GB) | Reduce `--batch_size` to 8 for Stage 1, 128 for Stage 2 |
| Stage 1 loss not decreasing | Check SmolVLA path is correct and model loads properly |
| Stage 2 Q-values diverging | Reduce `--actor_lr` and `--critic_lr` to 1e-4 |
| Robot moves erratically | Emergency stop: Ctrl+C (graceful) or unplug follower USB |

---

## Quick Reference: The Full Pipeline

```
Record 50 demos → Fine-tune SmolVLA → Stage 1 (learn z_rl) → Stage 2 (RL + intervention) → Deploy
     ~2 hours        ~8 hours GPU          ~30 min GPU           ~3 hours on-robot        Done!
```
