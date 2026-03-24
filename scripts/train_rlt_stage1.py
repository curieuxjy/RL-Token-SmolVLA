#!/usr/bin/env python3
"""Stage 1: Offline RLT Encoder-Decoder Training.

Trains the RLT encoder and decoder on demonstration data to learn a compact
z_rl representation from frozen SmolVLA embeddings.

Usage:
    python scripts/train_rlt_stage1.py \
        --smolvla_path lerobot/smolvla_base \
        --dataset_repo_id <USER>/svla_so101_pick_place \
        --output_dir checkpoints/rlt_stage1 \
        --steps 5000 \
        --batch_size 16 \
        --device cuda

Hardware: Single GPU (RunPod H100 or M4 Pro MPS)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RLT Stage 1: Encoder-Decoder Training")
    parser.add_argument("--smolvla_path", type=str, required=True,
                        help="Path to pretrained SmolVLA checkpoint")
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                        help="HuggingFace dataset repo ID for demo data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/rlt_stage1")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Load frozen SmolVLA ─────────────────────────────────────────────
    logger.info(f"Loading SmolVLA from {args.smolvla_path}...")
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    smolvla_policy = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=args.smolvla_path,
        config=SmolVLAConfig(load_vlm_weights=True),
    )
    smolvla_policy.to(device)
    smolvla_policy.eval()
    for p in smolvla_policy.parameters():
        p.requires_grad = False
    vla_model = smolvla_policy.model
    logger.info("SmolVLA loaded and frozen.")

    # ── Create RLT encoder + decoder ────────────────────────────────────
    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLTokenDecoder, RLTokenEncoder

    rlt_config = SmolVLARLTConfig(mode="rlt_training")
    encoder = RLTokenEncoder(rlt_config).to(device)
    decoder = RLTokenDecoder(rlt_config).to(device)

    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info(f"RLT Encoder: {enc_params:,} params, Decoder: {dec_params:,} params")

    # ── Optimizer + scheduler ───────────────────────────────────────────
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Load dataset ────────────────────────────────────────────────────
    logger.info(f"Loading dataset {args.dataset_repo_id}...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset_repo_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(dataloader)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # ── Training loop ───────────────────────────────────────────────────
    logger.info(f"Starting training for {args.steps} steps...")
    loss_history = []
    best_loss = float("inf")
    start_time = time.time()

    encoder.train()
    decoder.train()

    for step in range(1, args.steps + 1):
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Prepare inputs for SmolVLA
        images, img_masks = smolvla_policy.prepare_images(batch)
        state = smolvla_policy.prepare_state(batch)
        actions = smolvla_policy.prepare_action(batch)

        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Extract embeddings from frozen VLA
        with torch.no_grad():
            vlm_emb, expert_emb = vla_model.extract_embeddings(
                images, img_masks, lang_tokens, lang_masks, state, actions,
            )

        # RLT forward: encode → z_rl → decode → reconstruction loss
        z_rl = encoder(vlm_emb.detach(), expert_emb.detach())
        vlm_recon, expert_recon = decoder(
            z_rl,
            vlm_target_len=vlm_emb.shape[1],
            expert_target_len=expert_emb.shape[1],
        )

        vlm_loss = F.mse_loss(vlm_recon, vlm_emb.detach())
        expert_loss = F.mse_loss(expert_recon, expert_emb.detach())
        loss = vlm_loss + expert_loss

        # Backward + step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            it_per_sec = step / elapsed
            logger.info(
                f"Step {step}/{args.steps} | Loss: {loss_val:.4f} "
                f"(VLM: {vlm_loss.item():.4f}, Expert: {expert_loss.item():.4f}) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | {it_per_sec:.1f} it/s"
            )

        # Save checkpoint
        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": rlt_config.__dict__,
                "loss": loss_val,
            }
            ckpt_path = output_dir / f"checkpoint_step{step}.pt"
            torch.save(ckpt, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(ckpt, output_dir / "best_checkpoint.pt")
                logger.info(f"New best loss: {best_loss:.4f}")

    # Save final loss history
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    total_time = time.time() - start_time
    logger.info(f"Training complete! {args.steps} steps in {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
