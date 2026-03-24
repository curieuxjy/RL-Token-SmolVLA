# RL Token (RLT) for SmolVLA on SO-101

Implementation of the [RL Token paper](https://arxiv.org/abs/2505.13702) (Physical Intelligence) for [SmolVLA](https://huggingface.co/papers/2506.01844) on the SO-101 robot arm, built on top of [LeRobot](https://github.com/huggingface/lerobot) v0.4.4.

## What is RLT?

SmolVLA learns from expert teleoperation demos (supervised fine-tuning), but never learns to recover from failures. RLT solves this by:

1. **Extracting a compact "RL token"** (z_rl, 256-d) from the frozen VLA's internal embeddings
2. **Training a lightweight actor-critic** (~500K params) on top via online RL with human feedback

The robot improves from its own practice with just a few hours of experience, using two forms of human feedback:
- **Sparse reward labels** — human says success/fail after each episode
- **Intervention via teleop** — human grabs the leader arm to correct the robot mid-episode. These corrective demos flow into the replay buffer as high-quality training data.

## Architecture

```
Stage 1 (Offline, ~30 min):              Stage 2 (On-Robot, ~3 hours):

  SmolVLA (frozen)                          SmolVLA (frozen)
       |                                         |
  VLM (576-d) + Expert (432-d)             VLM + Expert embeddings
       |                                         |
  [RLT Encoder] → z_rl (256-d)            [RLT Encoder] (frozen) → z_rl
       |                                         |
  [RLT Decoder] → reconstruction loss      z_rl + state + ref_actions
                                                  |
                                            [Actor MLP] → refined actions
                                            [Critic MLP] → Q-values
                                                  |
                                            Human intervention + reward labels
```

## Repository Structure

```
RL-Token-SmolVLA/
├── smolvla_rlt/                          # RLT policy module (drop into LeRobot)
│   ├── __init__.py
│   ├── configuration_smolvla_rlt.py      # Config dataclass
│   ├── modeling_smolvla_rlt.py           # Encoder, Decoder, Actor, Critic, Policy
│   └── processor_smolvla_rlt.py          # Input/output processors
├── scripts/
│   ├── train_rlt_stage1.py               # Stage 1: offline encoder-decoder training
│   ├── train_rlt_stage2.py               # Stage 2: online RL with human intervention
│   └── run_rlt_inference.py              # Deployment: 30Hz control loop
├── tests/
│   └── test_smolvla_rlt.py               # Unit tests
├── lerobot_patches/                      # Patches to apply to LeRobot v0.4.4
│   ├── modeling_smolvla.patch            # Adds extract_embeddings() to VLAFlowMatching
│   └── factory.patch                     # Registers smolvla_rlt in policy factory
├── install.sh                            # One-command setup script
├── RLT-Setup-Guide.md                    # Detailed step-by-step guide
└── README.md                             # This file
```

## Quick Setup (Mac Mini M4)

```bash
git clone https://github.com/RajatDandekar/RL-Token-SmolVLA.git
cd RL-Token-SmolVLA
bash install.sh /path/to/your/lerobot
```

The install script:
1. Copies `smolvla_rlt/` into LeRobot's policy directory
2. Applies patches to `factory.py` and `modeling_smolvla.py`
3. Copies training/inference scripts

See [RLT-Setup-Guide.md](RLT-Setup-Guide.md) for the full walkthrough.

## Usage

### Stage 1: Learn z_rl (offline, any GPU)

```bash
python scripts/train_rlt_stage1.py \
    --smolvla_path lerobot/smolvla_base \
    --dataset_repo_id RajatDandekar/so101_pick_place_v2 \
    --output_dir checkpoints/rlt_stage1 \
    --steps 5000 \
    --device cuda  # or mps
```

### Stage 2: Online RL with intervention (on-robot)

```bash
python scripts/train_rlt_stage2.py \
    --smolvla_path lerobot/smolvla_base \
    --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
    --task "pick up the red cube and place it in the bin" \
    --follower_port /dev/tty.usbmodemXXXXX \
    --leader_port /dev/tty.usbmodemYYYYY \
    --max_episodes 200 \
    --device mps
```

During training, grab the leader arm at any time to intervene and correct the robot.

### Deploy

```bash
python scripts/run_rlt_inference.py \
    --smolvla_path lerobot/smolvla_base \
    --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
    --actor_checkpoint checkpoints/rlt_stage2/final_checkpoint.pt \
    --task "pick up the red cube and place it in the bin" \
    --device mps
```

## Requirements

- LeRobot v0.4.4 with SmolVLA extras (`pip install -e ".[smolvla]"`)
- Python 3.10+
- SO-101 follower + leader arms (for Stage 2)
- USB cameras (for Stage 2)
- GPU for Stage 1 (H100 recommended, M4 Pro MPS works too)

## Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| z_rl dimension | 256 | Balances expressiveness vs RL sample efficiency |
| Actor architecture | 2-layer MLP | Tiny (~170K params) for fast on-device inference |
| Critic architecture | Twin Q (TD3) | Reduces overestimation bias |
| BC regularization | β=0.1 | Keeps actor close to VLA reference, prevents wild exploration |
| Ref dropout | 50% | Forces actor to rely on z_rl, not just copy ref_actions |
| Intervention detection | 3° joint threshold | Reliable on SO-101 with 2-frame confirmation |
| Intervention reward | Auto-success + 0.5/step bonus | Dense signal from corrective demos |

## References

- [RL Token paper](https://arxiv.org/abs/2505.13702) — Physical Intelligence, 2025
- [SmolVLA paper](https://huggingface.co/papers/2506.01844) — HuggingFace, 2025
- [LeRobot](https://github.com/huggingface/lerobot) — HuggingFace robotics framework
