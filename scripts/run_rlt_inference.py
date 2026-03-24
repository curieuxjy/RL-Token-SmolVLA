#!/usr/bin/env python3
"""RLT Inference on SO-101.

Loads frozen SmolVLA + RLT encoder + trained actor for deployment.
30Hz control loop: observation -> VLA ref + z_rl -> actor -> robot.

Based on openclaw-so101/scripts/policy_runner.py pattern.

Usage:
    python scripts/run_rlt_inference.py \
        --smolvla_path lerobot/smolvla_base \
        --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
        --actor_checkpoint checkpoints/rlt_stage2/final_checkpoint.pt \
        --task "pick up the block and place it in the bin" \
        --duration 60 \
        --device mps
"""

import argparse
import logging
import signal
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SHUTDOWN = False


def signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received.")
    SHUTDOWN = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="RLT Inference on SO-101")
    parser.add_argument("--smolvla_path", type=str, required=True)
    parser.add_argument("--rlt_checkpoint", type=str, required=True,
                        help="Stage 1 encoder checkpoint")
    parser.add_argument("--actor_checkpoint", type=str, required=True,
                        help="Stage 2 actor checkpoint")
    parser.add_argument("--task", type=str, default="pick up the block")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Max inference duration in seconds")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision (CUDA only)")
    # Robot
    parser.add_argument("--robot_port", type=str, default="/dev/tty.usbmodem*")
    return parser.parse_args()


def load_models(args):
    """Load all three model components."""
    device = torch.device(args.device)

    # 1. Frozen SmolVLA
    logger.info(f"Loading SmolVLA from {args.smolvla_path}...")
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    smolvla_policy = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=args.smolvla_path,
        config=SmolVLAConfig(load_vlm_weights=True),
    )
    smolvla_policy.to(device)
    smolvla_policy.eval()

    # 2. RLT encoder (frozen)
    logger.info(f"Loading RLT encoder from {args.rlt_checkpoint}...")
    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP, RLTokenEncoder

    rlt_config = SmolVLARLTConfig(mode="inference")
    encoder = RLTokenEncoder(rlt_config).to(device)

    rlt_ckpt = torch.load(args.rlt_checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(rlt_ckpt["encoder_state_dict"])
    encoder.eval()

    # 3. Trained actor
    logger.info(f"Loading actor from {args.actor_checkpoint}...")
    actor = RLActorMLP(rlt_config).to(device)

    actor_ckpt = torch.load(args.actor_checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(actor_ckpt["actor_state_dict"])
    actor.eval()

    return smolvla_policy, encoder, actor, rlt_config, device


def main():
    args = parse_args()
    smolvla_policy, encoder, actor, rlt_config, device = load_models(args)
    vla_model = smolvla_policy.model

    logger.info("All models loaded. Ready for inference.")
    logger.info(f"Task: {args.task}")
    logger.info(f"Duration: {args.duration}s at {args.fps} FPS")

    # ── Robot setup ─────────────────────────────────────────────────────
    # In real deployment, connect to SO-101 here
    # robot = build_robot(args.robot_port)
    # robot.connect()
    logger.info("Robot connection: implement based on your LeRobot/OpenClaw setup")

    # ── Inference loop ──────────────────────────────────────────────────
    target_dt = 1.0 / args.fps
    start_time = time.time()
    frame_count = 0

    from lerobot.policies.smolvla.modeling_smolvla import pad_vector

    logger.info("Starting inference loop...")

    while not SHUTDOWN:
        loop_start = time.time()
        elapsed = loop_start - start_time

        if elapsed >= args.duration:
            logger.info(f"Duration limit ({args.duration}s) reached.")
            break

        # ── 1. Get observation from robot ───────────────────────────────
        # In real deployment:
        # obs = robot.get_observation()
        # images, state = process_observation(obs)

        # Placeholder — replace with actual robot observation
        # images: list of (B, C, H, W) tensors
        # state: (B, max_state_dim) tensor
        # lang_tokens, lang_masks: from tokenizer

        # ── 2. Run frozen VLA for reference actions + embeddings ────────
        # with torch.no_grad():
        #     ref_actions_full = vla_model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
        #     ref_actions_full = ref_actions_full[:, :, :rlt_config.action_dim]
        #     stride = rlt_config.action_stride
        #     ref_actions_sub = ref_actions_full[:, ::stride, :][:, :rlt_config.n_action_steps_rl, :]
        #
        #     dummy_actions = pad_vector(ref_actions_full, rlt_config.max_action_dim)
        #     vlm_emb, expert_emb = vla_model.extract_embeddings(
        #         images, img_masks, lang_tokens, lang_masks, state, dummy_actions
        #     )

        # ── 3. Encode to z_rl ──────────────────────────────────────────
        # with torch.no_grad():
        #     z_rl = encoder(vlm_emb, expert_emb)

        # ── 4. Actor: refine reference actions ──────────────────────────
        # with torch.no_grad():
        #     state_rl = state_raw[:, :rlt_config.state_dim]
        #     action_mean, _ = actor(z_rl, state_rl, ref_actions_sub)

        # ── 5. Send action to robot ────────────────────────────────────
        # robot.send_action(action_mean[0, 0])  # Send first timestep

        frame_count += 1

        # Progress logging
        if frame_count % 30 == 0:
            logger.info(f"Frame {frame_count} | Elapsed: {elapsed:.1f}s")

        # Maintain target FPS
        loop_dt = time.time() - loop_start
        sleep_time = max(0, target_dt - loop_dt)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ── Cleanup ─────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    actual_fps = frame_count / max(total_time, 1e-6)
    logger.info(f"Inference complete. {frame_count} frames in {total_time:.1f}s ({actual_fps:.1f} FPS)")
    # robot.disconnect()


if __name__ == "__main__":
    main()
