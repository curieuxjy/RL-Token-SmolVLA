#!/usr/bin/env python3
"""Stage 2: Online RL Training on SO-101 with Human Intervention.

Trains a lightweight actor-critic using z_rl from the frozen RLT encoder.
The human provides two forms of feedback:

  1. **Sparse reward labels** — after each episode, human says success/fail
  2. **Intervention via teleop** — during an episode, human can grab the
     leader arm to override RL actions. These corrective demonstrations
     flow into the replay buffer as high-quality transitions, dramatically
     improving sample efficiency.

The intervention mechanism is key: without it, the agent only learns from
sparse binary signals. With it, the critic sees dense corrective demos
that show *what the robot should have done* at each failure state.

Usage:
    python scripts/train_rlt_stage2.py \
        --smolvla_path lerobot/smolvla_base \
        --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
        --task "pick up the block and place it in the bin" \
        --follower_port /dev/tty.usbmodem58760431541 \
        --leader_port /dev/tty.usbmodem58760431551 \
        --max_episodes 200 \
        --device mps

Hardware: M4 Pro Mac connected to SO-101 (follower + leader arms)
"""

import argparse
import json
import logging
import random
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SHUTDOWN = False


def signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received. Finishing current episode...")
    SHUTDOWN = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """Single transition in the replay buffer."""
    z_rl: np.ndarray           # (rlt_hidden_dim,)
    state: np.ndarray          # (state_dim,)
    actions: np.ndarray        # (n_action_steps_rl, action_dim)
    ref_actions: np.ndarray    # (n_action_steps_rl, action_dim)
    reward: float
    next_z_rl: np.ndarray
    next_state: np.ndarray
    next_ref_actions: np.ndarray
    done: float
    is_intervention: bool = False  # True if this action came from human teleop


class ReplayBuffer:
    """Replay buffer with intervention-aware sampling."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            "z_rl": torch.FloatTensor(np.stack([t.z_rl for t in batch])),
            "state": torch.FloatTensor(np.stack([t.state for t in batch])),
            "actions": torch.FloatTensor(np.stack([t.actions for t in batch])),
            "ref_actions": torch.FloatTensor(np.stack([t.ref_actions for t in batch])),
            "rewards": torch.FloatTensor([[t.reward] for t in batch]),
            "next_z_rl": torch.FloatTensor(np.stack([t.next_z_rl for t in batch])),
            "next_state": torch.FloatTensor(np.stack([t.next_state for t in batch])),
            "next_ref_actions": torch.FloatTensor(np.stack([t.next_ref_actions for t in batch])),
            "dones": torch.FloatTensor([[t.done] for t in batch]),
            "is_intervention": torch.FloatTensor([[float(t.is_intervention)] for t in batch]),
        }

    @property
    def intervention_count(self) -> int:
        return sum(1 for t in self.buffer if t.is_intervention)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Intervention detection
# ─────────────────────────────────────────────────────────────────────────────

class InterventionDetector:
    """Detects when a human is actively moving the leader arm.

    Monitors position deltas between consecutive leader readings. If any
    joint moves more than `threshold_degrees` between frames, the human
    is intervening. Uses a small temporal window to avoid single-frame noise.
    """

    # Joints to monitor (skip gripper — it has different units)
    MONITORED_JOINTS = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll",
    ]

    def __init__(
        self,
        threshold_degrees: float = 3.0,
        confirmation_frames: int = 2,
        release_frames: int = 5,
    ):
        """
        Args:
            threshold_degrees: Min joint delta (degrees) to trigger intervention
            confirmation_frames: Consecutive frames above threshold to confirm
            release_frames: Consecutive frames below threshold to release
        """
        self.threshold = threshold_degrees
        self.confirmation_frames = confirmation_frames
        self.release_frames = release_frames

        self._prev_positions: dict[str, float] | None = None
        self._active_count = 0      # Consecutive frames with motion detected
        self._inactive_count = 0    # Consecutive frames without motion
        self._is_intervening = False

    def update(self, leader_positions: dict[str, float]) -> bool:
        """Update with new leader reading. Returns True if human is intervening.

        Args:
            leader_positions: Dict from leader.get_action(), e.g.
                {"shoulder_pan.pos": 45.0, "shoulder_lift.pos": 90.0, ...}
        """
        if self._prev_positions is None:
            self._prev_positions = dict(leader_positions)
            return False

        # Check if any monitored joint moved significantly
        motion_detected = False
        for joint in self.MONITORED_JOINTS:
            key = f"{joint}.pos"
            if key in leader_positions and key in self._prev_positions:
                delta = abs(leader_positions[key] - self._prev_positions[key])
                if delta > self.threshold:
                    motion_detected = True
                    break

        self._prev_positions = dict(leader_positions)

        # State machine: require sustained motion/stillness to transition
        if motion_detected:
            self._active_count += 1
            self._inactive_count = 0
            if self._active_count >= self.confirmation_frames:
                if not self._is_intervening:
                    logger.info("  >> INTERVENTION DETECTED — switching to human teleop")
                self._is_intervening = True
        else:
            self._inactive_count += 1
            self._active_count = 0
            if self._inactive_count >= self.release_frames:
                if self._is_intervening:
                    logger.info("  >> Intervention ended — returning to RL agent")
                self._is_intervening = False

        return self._is_intervening

    def reset(self):
        """Reset state between episodes."""
        self._prev_positions = None
        self._active_count = 0
        self._inactive_count = 0
        self._is_intervening = False


# ─────────────────────────────────────────────────────────────────────────────
# Human feedback
# ─────────────────────────────────────────────────────────────────────────────

def get_human_reward(had_intervention: bool) -> tuple[float, bool]:
    """Prompt human for episode outcome.

    If the human intervened during the episode, the episode is automatically
    considered a success (the human corrected it). Otherwise, ask.

    Returns:
        (reward, should_continue): reward is 1.0 (success) or 0.0 (fail),
        should_continue is False if user wants to quit.
    """
    if had_intervention:
        logger.info("  Episode had human intervention → auto-labeled SUCCESS (reward=1)")
        return 1.0, True

    while True:
        response = input("\n>>> Was this episode successful? [y/n/q(uit)]: ").strip().lower()
        if response in ("y", "yes", "1"):
            return 1.0, True
        elif response in ("n", "no", "0"):
            return 0.0, True
        elif response in ("q", "quit"):
            return 0.0, False
        print("  Please enter 'y', 'n', or 'q'")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RLT Stage 2: Online RL on SO-101 with Intervention")
    # Model paths
    parser.add_argument("--smolvla_path", type=str, required=True,
                        help="Path to pretrained SmolVLA checkpoint")
    parser.add_argument("--rlt_checkpoint", type=str, required=True,
                        help="Path to Stage 1 RLT encoder-decoder checkpoint")
    parser.add_argument("--task", type=str, default="pick up the block",
                        help="Task description for the VLA")
    # Robot connection
    parser.add_argument("--follower_port", type=str, required=True,
                        help="SO-101 follower arm serial port")
    parser.add_argument("--leader_port", type=str, required=True,
                        help="SO-101 leader arm serial port")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["front"],
                        help="Camera names matching training dataset")
    parser.add_argument("--fps", type=float, default=30.0)
    # Training
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--steps_per_episode", type=int, default=50,
                        help="Max action chunks per episode")
    parser.add_argument("--output_dir", type=str, default="checkpoints/rlt_stage2")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--utd_ratio", type=int, default=5)
    parser.add_argument("--critic_per_actor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--bc_weight", type=float, default=0.1)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--warmup_episodes", type=int, default=5,
                        help="Episodes of pure VLA rollout before RL updates")
    parser.add_argument("--buffer_capacity", type=int, default=100_000)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    # Intervention
    parser.add_argument("--intervention_threshold", type=float, default=3.0,
                        help="Joint motion threshold (degrees) for intervention detection")
    parser.add_argument("--intervention_reward_bonus", type=float, default=0.5,
                        help="Per-step reward bonus for intervention transitions")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Load frozen SmolVLA ─────────────────────────────────────────────
    logger.info(f"Loading SmolVLA from {args.smolvla_path}...")
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector

    smolvla_policy = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=args.smolvla_path,
        config=SmolVLAConfig(load_vlm_weights=True),
    )
    smolvla_policy.to(device)
    smolvla_policy.eval()
    for p in smolvla_policy.parameters():
        p.requires_grad = False
    vla_model = smolvla_policy.model

    # ── Load RLT encoder (frozen from Stage 1) ─────────────────────────
    logger.info(f"Loading RLT encoder from {args.rlt_checkpoint}...")
    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import (
        RLActorMLP,
        RLCriticMLP,
        RLTokenEncoder,
    )

    rlt_config = SmolVLARLTConfig(
        mode="online_rl",
        discount=args.discount,
        bc_weight=args.bc_weight,
        target_tau=args.target_tau,
    )

    ckpt = torch.load(args.rlt_checkpoint, map_location=device, weights_only=True)
    encoder = RLTokenEncoder(rlt_config).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    logger.info("RLT encoder loaded and frozen.")

    # ── Create actor + critic ───────────────────────────────────────────
    actor = RLActorMLP(rlt_config).to(device)
    critic = RLCriticMLP(rlt_config).to(device)
    critic_target = RLCriticMLP(rlt_config).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    logger.info(f"Actor: {actor_params:,} params, Critic: {critic_params:,} params (x2 twin)")

    # ── Replay buffer + intervention detector ──────────────────────────
    buffer = ReplayBuffer(args.buffer_capacity)
    intervention_detector = InterventionDetector(
        threshold_degrees=args.intervention_threshold,
    )

    # ── Connect to SO-101 ──────────────────────────────────────────────
    logger.info("Connecting to SO-101 arms...")
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader

    # Import may vary by LeRobot version — adjust if needed
    try:
        follower = SO101Follower(SO101FollowerConfig(port=args.follower_port))
        leader = SO101Leader(SO101LeaderConfig(port=args.leader_port))
        follower.connect()
        leader.connect()
        logger.info("Both arms connected.")
    except Exception as e:
        logger.error(f"Failed to connect to robot: {e}")
        logger.error("Check ports with: ls /dev/tty.usbmodem*")
        return

    # ── Joint name mapping ─────────────────────────────────────────────
    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def leader_action_to_tensor(leader_action: dict) -> torch.Tensor:
        """Convert leader joint positions dict → (action_dim,) tensor."""
        values = [leader_action.get(f"{j}.pos", 0.0) for j in JOINT_NAMES]
        return torch.FloatTensor(values[:rlt_config.action_dim])

    def obs_to_state_tensor(obs: dict) -> torch.Tensor:
        """Convert robot observation dict → (state_dim,) tensor."""
        values = [obs.get(f"{j}.pos", 0.0) for j in JOINT_NAMES]
        return torch.FloatTensor(values[:rlt_config.state_dim])

    def tensor_to_robot_action(action_tensor: torch.Tensor) -> dict:
        """Convert (action_dim,) tensor → robot action dict."""
        action_np = action_tensor.cpu().numpy()
        return {f"{JOINT_NAMES[i]}.pos": float(action_np[i])
                for i in range(min(len(JOINT_NAMES), len(action_np)))}

    # ── Tokenize task once ─────────────────────────────────────────────
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(rlt_config.vlm_model_name)
    task_text = args.task if args.task.endswith("\n") else args.task + "\n"
    tokenized = processor.tokenizer(
        task_text,
        return_tensors="pt",
        padding="max_length",
        max_length=rlt_config.tokenizer_max_length,
        truncation=True,
    )
    lang_tokens = tokenized["input_ids"].to(device)
    lang_masks = tokenized["attention_mask"].to(device).bool()

    # ── Training loop ───────────────────────────────────────────────────
    logger.info(f"\nStarting online RL for {args.max_episodes} episodes...")
    logger.info(f"Intervention threshold: {args.intervention_threshold} degrees")
    logger.info(f"Warmup episodes (pure VLA): {args.warmup_episodes}")
    logger.info(f"Grab the leader arm at any time to intervene!\n")

    episode_rewards = []
    episode_interventions = []
    total_updates = 0
    target_dt = 1.0 / args.fps

    for episode in range(1, args.max_episodes + 1):
        if SHUTDOWN:
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode}/{args.max_episodes}")
        logger.info(f"{'='*60}")

        intervention_detector.reset()
        episode_transitions = []
        episode_had_intervention = False
        intervention_steps = 0

        prev_z_rl = None
        prev_state = None
        prev_ref_actions_sub = None

        for step in range(args.steps_per_episode):
            if SHUTDOWN:
                break
            step_start = time.time()

            # ── 1. Get observation from robot ───────────────────────────
            obs = follower.get_observation()
            state_tensor = obs_to_state_tensor(obs).to(device)

            # Get camera images
            images = []
            img_masks = []
            for cam_name in args.camera_names:
                if cam_name in obs:
                    img = obs[cam_name]  # Tensor from camera
                    if img.ndim == 3:
                        img = img.unsqueeze(0)  # Add batch dim
                    # Normalize to [0,1] if needed, then to [-1,1] for SigLIP
                    if img.max() > 1.0:
                        img = img.float() / 255.0
                    img = img * 2.0 - 1.0
                    images.append(img.to(device))
                    img_masks.append(torch.ones(1, dtype=torch.bool, device=device))

            if not images:
                logger.warning(f"  No camera images at step {step}, skipping")
                continue

            # Pad state for VLA
            state_padded = pad_vector(state_tensor.unsqueeze(0), rlt_config.max_state_dim)

            # ── 2. Read leader arm (for intervention detection) ─────────
            leader_action = leader.get_action()
            is_intervening = intervention_detector.update(leader_action)

            if is_intervening:
                episode_had_intervention = True
                intervention_steps += 1

            # ── 3. Run frozen VLA → ref_actions + embeddings ────────────
            with torch.no_grad():
                # Get reference actions from VLA
                ref_actions_full = vla_model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state_padded,
                )
                ref_actions_full = ref_actions_full[:, :, :rlt_config.action_dim]

                # Subsample to RL chunk size
                stride = rlt_config.action_stride
                ref_actions_sub = ref_actions_full[:, ::stride, :][:, :rlt_config.n_action_steps_rl, :]

                # Extract embeddings for z_rl
                dummy_actions = pad_vector(ref_actions_full, rlt_config.max_action_dim)
                vlm_emb, expert_emb = vla_model.extract_embeddings(
                    images, img_masks, lang_tokens, lang_masks, state_padded, dummy_actions,
                )

                # Encode to z_rl
                z_rl = encoder(vlm_emb, expert_emb)  # (1, 256)

            # ── 4. Decide action: RL actor vs human intervention ────────
            if is_intervening:
                # HUMAN OVERRIDE: use leader arm positions as action
                human_action = leader_action_to_tensor(leader_action).to(device)
                # Expand to chunk format (repeat for all steps)
                executed_action = human_action.unsqueeze(0).unsqueeze(0).expand(
                    1, rlt_config.n_action_steps_rl, -1
                )
                action_source = "HUMAN"
            elif episode <= args.warmup_episodes:
                # WARMUP: use VLA reference actions directly
                executed_action = ref_actions_sub
                action_source = "VLA"
            else:
                # RL AGENT: actor refines reference actions
                with torch.no_grad():
                    state_rl = state_tensor[:rlt_config.state_dim].unsqueeze(0)
                    action_mean, _ = actor(z_rl, state_rl, ref_actions_sub)
                executed_action = action_mean
                action_source = "ACTOR"

            # ── 5. Execute first action step on robot ───────────────────
            first_action = executed_action[0, 0]  # First timestep of chunk
            robot_action = tensor_to_robot_action(first_action)
            follower.send_action(robot_action)

            # ── 6. Store transition (with previous step's data) ─────────
            if prev_z_rl is not None:
                transition = Transition(
                    z_rl=prev_z_rl.cpu().numpy().squeeze(),
                    state=prev_state.cpu().numpy(),
                    actions=prev_executed_action.cpu().numpy().squeeze(),
                    ref_actions=prev_ref_actions_sub.cpu().numpy().squeeze(),
                    reward=0.0,  # Will be filled at episode end
                    next_z_rl=z_rl.cpu().numpy().squeeze(),
                    next_state=state_tensor[:rlt_config.state_dim].cpu().numpy(),
                    next_ref_actions=ref_actions_sub.cpu().numpy().squeeze(),
                    done=0.0,
                    is_intervention=prev_was_intervention,
                )
                episode_transitions.append(transition)

            # Save for next step's transition
            prev_z_rl = z_rl
            prev_state = state_tensor[:rlt_config.state_dim]
            prev_ref_actions_sub = ref_actions_sub
            prev_executed_action = executed_action
            prev_was_intervention = is_intervening

            # Logging
            if step % 10 == 0:
                logger.info(
                    f"  Step {step}/{args.steps_per_episode} | "
                    f"Source: {action_source} | "
                    f"Interventions so far: {intervention_steps}"
                )

            # Maintain target FPS
            elapsed = time.time() - step_start
            sleep_time = max(0, target_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # ── End of episode ──────────────────────────────────────────────

        # Get human reward label
        reward, should_continue = get_human_reward(episode_had_intervention)

        # Assign rewards to transitions
        # Terminal transition gets the episode reward
        # Intervention steps get a per-step bonus (denser signal)
        for i, t in enumerate(episode_transitions):
            is_terminal = (i == len(episode_transitions) - 1)
            t.done = 1.0 if is_terminal else 0.0

            if is_terminal:
                t.reward = reward
            elif t.is_intervention:
                # Intervention transitions get a bonus — they represent
                # corrective actions the human deemed necessary
                t.reward = args.intervention_reward_bonus
            else:
                t.reward = 0.0

            buffer.add(t)

        episode_rewards.append(reward)
        episode_interventions.append(intervention_steps)

        # ── RL Updates ──────────────────────────────────────────────────
        if episode > args.warmup_episodes and len(buffer) >= args.batch_size:
            n_updates = args.utd_ratio * len(episode_transitions)
            if n_updates > 0:
                logger.info(f"  Running {n_updates} RL updates...")

                for update_idx in range(n_updates):
                    batch = buffer.sample(args.batch_size)
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # ── Critic update ───────────────────────────────────
                    with torch.no_grad():
                        next_action_mean, _ = actor(
                            batch["next_z_rl"], batch["next_state"],
                            batch["next_ref_actions"]
                        )
                        target_q = critic_target.q_min(
                            batch["next_z_rl"], batch["next_state"], next_action_mean
                        )
                        td_target = (
                            batch["rewards"]
                            + args.discount * (1.0 - batch["dones"]) * target_q
                        )

                    q1, q2 = critic(batch["z_rl"], batch["state"], batch["actions"])
                    critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    # ── Actor update (every critic_per_actor steps) ─────
                    if update_idx % args.critic_per_actor == 0:
                        action_mean, _ = actor(
                            batch["z_rl"], batch["state"],
                            batch["ref_actions"], training=True
                        )
                        q_value = critic.q_min(
                            batch["z_rl"].detach(), batch["state"], action_mean
                        )
                        q_loss = -q_value.mean()

                        # BC regularization: For intervention transitions,
                        # regularize toward the human's action (not just ref).
                        # For non-intervention, regularize toward VLA ref.
                        bc_target = batch["ref_actions"].clone()
                        interv_mask = batch["is_intervention"].bool().squeeze(-1)
                        if interv_mask.any():
                            # For intervention transitions, the stored "actions"
                            # ARE the human's corrective actions — use those
                            bc_target[interv_mask] = batch["actions"][interv_mask]

                        bc_loss = F.mse_loss(action_mean, bc_target)
                        actor_loss = q_loss + args.bc_weight * bc_loss

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                    # ── EMA target update ───────────────────────────────
                    tau = args.target_tau
                    for p, tp in zip(critic.parameters(), critic_target.parameters()):
                        tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

                    total_updates += 1

                logger.info(
                    f"  Updates done. Critic loss: {critic_loss.item():.4f}, "
                    f"Q mean: {q1.mean().item():.4f}"
                )

        # ── Logging ─────────────────────────────────────────────────────
        recent_rewards = episode_rewards[-20:]
        success_rate = sum(recent_rewards) / len(recent_rewards)
        recent_interventions = episode_interventions[-20:]
        avg_interventions = sum(recent_interventions) / len(recent_interventions)

        logger.info(
            f"  Episode {episode} | Reward: {reward} | "
            f"Interventions: {intervention_steps} steps | "
            f"Success (last 20): {success_rate:.0%} | "
            f"Avg interventions (last 20): {avg_interventions:.1f} | "
            f"Buffer: {len(buffer)} ({buffer.intervention_count} from human) | "
            f"Updates: {total_updates}"
        )

        # ── Save checkpoint ─────────────────────────────────────────────
        if episode % args.save_every == 0:
            ckpt = {
                "episode": episode,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "critic_target_state_dict": critic_target.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "episode_rewards": episode_rewards,
                "episode_interventions": episode_interventions,
                "total_updates": total_updates,
            }
            ckpt_path = output_dir / f"checkpoint_ep{episode}.pt"
            torch.save(ckpt, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

        if not should_continue:
            logger.info("User requested quit.")
            break

    # ── Cleanup ─────────────────────────────────────────────────────────
    logger.info("\nDisconnecting from robot...")
    try:
        follower.disconnect()
        leader.disconnect()
    except Exception:
        pass

    # Final save
    final_ckpt = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "critic_target_state_dict": critic_target.state_dict(),
        "episode_rewards": episode_rewards,
        "episode_interventions": episode_interventions,
        "total_updates": total_updates,
    }
    torch.save(final_ckpt, output_dir / "final_checkpoint.pt")

    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "episode_rewards": episode_rewards,
            "episode_interventions": episode_interventions,
            "total_updates": total_updates,
        }, f, indent=2)

    total_episodes = len(episode_rewards)
    total_interventions = sum(episode_interventions)
    logger.info(f"\nTraining complete!")
    logger.info(f"  Episodes: {total_episodes}")
    logger.info(f"  Total updates: {total_updates}")
    logger.info(f"  Total intervention steps: {total_interventions}")
    logger.info(f"  Final success rate (last 20): "
                f"{sum(episode_rewards[-20:]) / max(1, len(episode_rewards[-20:])):.0%}")
    logger.info(f"  Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
