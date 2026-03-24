"""SmolVLA with RL Token (RLT) — core model implementation.

Implements the RLT paper (Physical Intelligence) for SmolVLA:
  Stage 1: Train encoder-decoder to extract z_rl from frozen VLA embeddings
  Stage 2: Train lightweight actor-critic using z_rl for online RL

Architecture:
  Frozen SmolVLA → [RLT Encoder] → z_rl → [Actor MLP] → refined actions
                                        → [Critic MLP] → Q-values
"""

import copy
import math
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
from lerobot.utils.constants import ACTION


# ─────────────────────────────────────────────────────────────────────────────
# RLT Encoder: VLA embeddings → z_rl
# ─────────────────────────────────────────────────────────────────────────────

class RLTokenEncoder(nn.Module):
    """Extracts a compact RL token (z_rl) from frozen VLA embeddings.

    Appends a learnable [RL] token to projected VLA embeddings, runs through
    a transformer encoder, and returns the [RL] token's output as z_rl.
    """

    def __init__(self, config: SmolVLARLTConfig):
        super().__init__()
        d = config.rlt_hidden_dim

        # Project VLM and expert embeddings to shared RLT dimension
        self.vlm_proj = nn.Linear(config.vlm_hidden_dim, d)
        self.expert_proj = nn.Linear(config.expert_hidden_dim, d)

        # Learnable RL token embedding
        self.rl_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.rlt_num_heads,
            dim_feedforward=d * 4,
            dropout=config.rlt_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.rlt_encoder_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        vlm_embeddings: Tensor,     # (B, L_vlm, vlm_hidden_dim)
        expert_embeddings: Tensor,  # (B, L_expert, expert_hidden_dim)
    ) -> Tensor:
        """Returns z_rl of shape (B, rlt_hidden_dim)."""
        B = vlm_embeddings.shape[0]

        # Project to shared dimension
        vlm_proj = self.vlm_proj(vlm_embeddings)      # (B, L_vlm, d)
        expert_proj = self.expert_proj(expert_embeddings)  # (B, L_expert, d)

        # Append learnable RL token
        rl_token = self.rl_token.expand(B, -1, -1)     # (B, 1, d)

        # Concatenate: [vlm_tokens, expert_tokens, rl_token]
        sequence = torch.cat([vlm_proj, expert_proj, rl_token], dim=1)  # (B, L_total+1, d)

        # Transformer encoder
        out = self.encoder(sequence)  # (B, L_total+1, d)

        # Extract RL token output (last position)
        z_rl = self.norm(out[:, -1, :])  # (B, d)

        return z_rl


# ─────────────────────────────────────────────────────────────────────────────
# RLT Decoder: z_rl → reconstructed VLA embeddings
# ─────────────────────────────────────────────────────────────────────────────

class RLTokenDecoder(nn.Module):
    """Reconstructs VLA embeddings from z_rl for training the encoder.

    Uses z_rl as a conditioning signal via cross-attention in a transformer
    decoder to reconstruct the original VLM and expert embeddings.
    """

    def __init__(self, config: SmolVLARLTConfig):
        super().__init__()
        d = config.rlt_hidden_dim

        # Inverse projection from z_rl back to sequence
        self.z_proj = nn.Linear(d, d)

        # Learnable query tokens for reconstruction
        # We use a fixed number of queries and project back to original dims
        self.vlm_queries = nn.Parameter(torch.randn(1, 1, d) * 0.02)  # single query per token
        self.expert_queries = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Transformer decoder (z_rl conditions reconstruction via cross-attn)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.rlt_num_heads,
            dim_feedforward=d * 4,
            dropout=config.rlt_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.rlt_decoder_layers,
        )

        # Project back to original embedding dims
        self.vlm_out_proj = nn.Linear(d, config.vlm_hidden_dim)
        self.expert_out_proj = nn.Linear(d, config.expert_hidden_dim)

        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        z_rl: Tensor,                     # (B, d)
        vlm_target_len: int,               # Number of VLM tokens to reconstruct
        expert_target_len: int,            # Number of expert tokens to reconstruct
    ) -> tuple[Tensor, Tensor]:
        """Returns (vlm_recon, expert_recon) matching original shapes."""
        B = z_rl.shape[0]

        # Expand z_rl as memory for cross-attention
        memory = self.z_proj(z_rl).unsqueeze(1)  # (B, 1, d)

        # Create query sequences
        vlm_q = self.vlm_queries.expand(B, vlm_target_len, -1)      # (B, L_vlm, d)
        expert_q = self.expert_queries.expand(B, expert_target_len, -1)  # (B, L_expert, d)

        # Add positional information via simple learned position encoding
        queries = torch.cat([vlm_q, expert_q], dim=1)  # (B, L_total, d)

        # Decode
        decoded = self.decoder(queries, memory)  # (B, L_total, d)
        decoded = self.norm(decoded)

        # Split and project back to original dims
        vlm_decoded = decoded[:, :vlm_target_len]
        expert_decoded = decoded[:, vlm_target_len:]

        vlm_recon = self.vlm_out_proj(vlm_decoded)        # (B, L_vlm, vlm_hidden)
        expert_recon = self.expert_out_proj(expert_decoded)  # (B, L_expert, expert_hidden)

        return vlm_recon, expert_recon


# ─────────────────────────────────────────────────────────────────────────────
# Actor MLP: z_rl + state + ref_actions → refined actions
# ─────────────────────────────────────────────────────────────────────────────

class RLActorMLP(nn.Module):
    """Lightweight actor that refines VLA reference actions using z_rl.

    Input: z_rl(256) + state(7) + flattened ref_actions(10*7=70) = 333
    Output: action mean (10*7=70) + learnable log_std
    """

    def __init__(self, config: SmolVLARLTConfig):
        super().__init__()
        self.config = config
        action_flat = config.n_action_steps_rl * config.action_dim  # 10 * 7 = 70
        input_dim = config.rlt_hidden_dim + config.state_dim + action_flat  # 256 + 7 + 70 = 333

        layers = []
        prev_dim = input_dim
        for h in config.actor_hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.SiLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_flat))
        self.mlp = nn.Sequential(*layers)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_flat))

        self.action_flat = action_flat
        self.ref_dropout = config.ref_dropout

    def forward(
        self,
        z_rl: Tensor,           # (B, rlt_hidden_dim)
        state: Tensor,           # (B, state_dim)
        ref_actions: Tensor,     # (B, n_action_steps_rl, action_dim)
        training: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Returns (action_mean, log_std) for the RL action chunk.

        action_mean: (B, n_action_steps_rl, action_dim)
        log_std: (action_flat,)
        """
        B = z_rl.shape[0]
        ref_flat = ref_actions.reshape(B, -1)  # (B, 70)

        # Reference action dropout during training
        if training and self.ref_dropout > 0:
            mask = (torch.rand(B, 1, device=ref_flat.device) > self.ref_dropout).float()
            ref_flat = ref_flat * mask

        x = torch.cat([z_rl, state, ref_flat], dim=-1)  # (B, 333)
        action_mean = self.mlp(x)  # (B, 70)

        # Reshape to (B, n_action_steps_rl, action_dim)
        action_mean = action_mean.reshape(B, self.config.n_action_steps_rl, self.config.action_dim)

        return action_mean, self.log_std


# ─────────────────────────────────────────────────────────────────────────────
# Twin Critic MLP: z_rl + state + actions → Q-values
# ─────────────────────────────────────────────────────────────────────────────

class RLCriticMLP(nn.Module):
    """Twin Q-network for TD3-style critic.

    Input: z_rl(256) + state(7) + flattened actions(70) = 333
    Output: Two scalar Q-values (min used for target)
    """

    def __init__(self, config: SmolVLARLTConfig):
        super().__init__()
        action_flat = config.n_action_steps_rl * config.action_dim
        input_dim = config.rlt_hidden_dim + config.state_dim + action_flat

        self.q1 = self._build_mlp(input_dim, config.critic_hidden_dims)
        self.q2 = self._build_mlp(input_dim, config.critic_hidden_dims)

    def _build_mlp(self, input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.SiLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(
        self,
        z_rl: Tensor,       # (B, rlt_hidden_dim)
        state: Tensor,       # (B, state_dim)
        actions: Tensor,     # (B, n_action_steps_rl, action_dim)
    ) -> tuple[Tensor, Tensor]:
        """Returns (Q1, Q2) each of shape (B, 1)."""
        B = z_rl.shape[0]
        actions_flat = actions.reshape(B, -1)
        x = torch.cat([z_rl, state, actions_flat], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(
        self,
        z_rl: Tensor,
        state: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Returns min(Q1, Q2) of shape (B, 1)."""
        q1, q2 = self.forward(z_rl, state, actions)
        return torch.min(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# SmolVLARLTPolicy: wraps everything together
# ─────────────────────────────────────────────────────────────────────────────

class SmolVLARLTPolicy(PreTrainedPolicy):
    """Full RLT policy wrapping frozen SmolVLA + trainable RLT components.

    Supports 3 modes:
      - "rlt_training": Stage 1 — train encoder/decoder on demo data
      - "online_rl": Stage 2 — train actor/critic with online RL
      - "inference": Deploy on robot
    """

    config_class = SmolVLARLTConfig
    name = "smolvla_rlt"

    def __init__(self, config: SmolVLARLTConfig, **kwargs):
        super().__init__(config)
        self.config = config

        # ── RLT encoder + decoder ───────────────────────────────────────
        self.rlt_encoder = RLTokenEncoder(config)
        self.rlt_decoder = RLTokenDecoder(config)

        # ── Actor + Critic ──────────────────────────────────────────────
        self.actor = RLActorMLP(config)
        self.critic = RLCriticMLP(config)
        self.critic_target = RLCriticMLP(config)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Frozen SmolVLA (loaded lazily) ──────────────────────────────
        self._vla_loaded = False
        self._frozen_vla = None

        # ── Action queue for deployment ─────────────────────────────────
        self.reset()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps_rl)}

    def load_frozen_vla(self, vla_model):
        """Attach a frozen SmolVLA model (VLAFlowMatching instance).

        Call this after constructing the policy, passing in the VLAFlowMatching
        model loaded from a pretrained SmolVLA checkpoint.
        """
        self._frozen_vla = vla_model
        self._frozen_vla.eval()
        for p in self._frozen_vla.parameters():
            p.requires_grad = False
        self._vla_loaded = True

    def _ensure_vla(self):
        if not self._vla_loaded:
            raise RuntimeError(
                "Frozen SmolVLA not loaded. Call policy.load_frozen_vla(vla_model) first."
            )

    # ── Embedding extraction from frozen VLA ────────────────────────────

    @torch.no_grad()
    def extract_vla_embeddings(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run frozen VLA forward and return (vlm_out, expert_out) embeddings.

        These are the final-layer hidden states from the VLM prefix and
        action expert suffix, used as input to the RLT encoder.

        Returns:
            vlm_out: (B, L_prefix, vlm_hidden_dim=576)
            expert_out: (B, chunk_size, expert_hidden_dim=432)
        """
        self._ensure_vla()
        return self._frozen_vla.extract_embeddings(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=noise, time=time,
        )

    # ── Stage 1: RLT Training (encoder-decoder) ────────────────────────

    def forward_rlt_training(
        self,
        vlm_embeddings: Tensor,     # (B, L_vlm, 576)
        expert_embeddings: Tensor,  # (B, L_expert, 432)
    ) -> dict[str, Tensor]:
        """Stage 1 forward: encode → z_rl → decode → reconstruction loss."""
        # Detach targets (stop gradient as per RLT paper Eq. 2)
        vlm_target = vlm_embeddings.detach()
        expert_target = expert_embeddings.detach()

        # Encode
        z_rl = self.rlt_encoder(vlm_target, expert_target)  # (B, 256)

        # Decode
        vlm_recon, expert_recon = self.rlt_decoder(
            z_rl,
            vlm_target_len=vlm_target.shape[1],
            expert_target_len=expert_target.shape[1],
        )

        # Reconstruction loss (MSE)
        vlm_loss = F.mse_loss(vlm_recon, vlm_target)
        expert_loss = F.mse_loss(expert_recon, expert_target)
        total_loss = vlm_loss + expert_loss

        return {
            "loss": total_loss,
            "vlm_recon_loss": vlm_loss,
            "expert_recon_loss": expert_loss,
            "z_rl": z_rl,
        }

    # ── Stage 2: Online RL (actor-critic) ───────────────────────────────

    def forward_critic_loss(
        self,
        z_rl: Tensor,           # (B, 256)
        state: Tensor,           # (B, state_dim)
        actions: Tensor,         # (B, n_action_steps_rl, action_dim)
        rewards: Tensor,         # (B, 1)
        next_z_rl: Tensor,       # (B, 256)
        next_state: Tensor,      # (B, state_dim)
        next_ref_actions: Tensor,  # (B, n_action_steps_rl, action_dim)
        dones: Tensor,           # (B, 1)
    ) -> dict[str, Tensor]:
        """Compute TD3-style twin critic loss."""
        with torch.no_grad():
            # Target actions from actor (no exploration noise for simplicity)
            next_action_mean, _ = self.actor(next_z_rl, next_state, next_ref_actions)
            # Target Q from target critic
            target_q = self.critic_target.q_min(next_z_rl, next_state, next_action_mean)
            td_target = rewards + self.config.discount * (1.0 - dones) * target_q

        q1, q2 = self.critic(z_rl, state, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        return {
            "critic_loss": critic_loss,
            "q1_mean": q1.mean(),
            "q2_mean": q2.mean(),
            "td_target_mean": td_target.mean(),
        }

    def forward_actor_loss(
        self,
        z_rl: Tensor,           # (B, 256)
        state: Tensor,           # (B, state_dim)
        ref_actions: Tensor,     # (B, n_action_steps_rl, action_dim)
    ) -> dict[str, Tensor]:
        """Compute actor loss: maximize Q + BC regularization."""
        action_mean, _ = self.actor(z_rl, state, ref_actions, training=True)

        # Q-value maximization
        q_value = self.critic.q_min(z_rl.detach(), state, action_mean)
        q_loss = -q_value.mean()

        # BC regularization: stay close to reference actions
        bc_loss = F.mse_loss(action_mean, ref_actions)

        actor_loss = q_loss + self.config.bc_weight * bc_loss

        return {
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
        }

    def update_target_critic(self):
        """Soft update target critic via EMA."""
        tau = self.config.target_tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

    # ── Inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action for robot execution.

        Manages an action queue: only calls the full pipeline when the
        queue is empty, otherwise pops the next pre-computed action.
        """
        self.eval()

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            # Queue shape: (n_action_steps_rl, batch_size, action_dim)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    @torch.no_grad()
    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Full inference pipeline: VLA → z_rl → actor → actions.

        Returns: (B, n_action_steps_rl, action_dim)
        """
        self._ensure_vla()

        # Import here to avoid circular dependency
        from lerobot.policies.smolvla.modeling_smolvla import pad_vector
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        # Prepare inputs (same as SmolVLA)
        images, img_masks = self._prepare_images(batch)
        state_raw = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state_padded = pad_vector(state_raw, self.config.max_state_dim)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Get VLA reference actions (full 50-step chunk)
        ref_actions_full = self._frozen_vla.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state_padded,
        )
        # Unpad to actual action dim
        ref_actions_full = ref_actions_full[:, :, :self.config.action_dim]

        # Subsample to RL chunk: stride=2, take first n_action_steps_rl
        stride = self.config.action_stride
        ref_actions_sub = ref_actions_full[:, ::stride, :][:, :self.config.n_action_steps_rl, :]

        # Get dummy actions for embedding extraction (use ref as proxy)
        dummy_actions = pad_vector(ref_actions_full, self.config.max_action_dim)

        # Extract embeddings
        vlm_emb, expert_emb = self.extract_vla_embeddings(
            images, img_masks, lang_tokens, lang_masks, state_padded, dummy_actions,
        )

        # Encode to z_rl
        z_rl = self.rlt_encoder(vlm_emb, expert_emb)

        # Actor: refine reference actions
        state_rl = state_raw[:, :self.config.state_dim]  # Original unpadded state
        action_mean, _ = self.actor(z_rl, state_rl, ref_actions_sub)

        return action_mean

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Prepare images using SmolVLA's preprocessing pattern."""
        from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

        images = []
        img_masks = []
        for key in self.config.image_features:
            if key not in batch:
                continue
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=img.device)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    # ── Standard forward (dispatches based on mode) ─────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass — dispatches based on config.mode.

        For rlt_training mode, batch should contain pre-extracted embeddings:
          - "vlm_embeddings": (B, L_vlm, 576)
          - "expert_embeddings": (B, L_expert, 432)

        Returns (loss, loss_dict).
        """
        if self.config.mode == "rlt_training":
            result = self.forward_rlt_training(
                batch["vlm_embeddings"],
                batch["expert_embeddings"],
            )
            return result["loss"], {k: v.item() if isinstance(v, Tensor) and v.ndim == 0 else v for k, v in result.items() if k != "z_rl"}
        else:
            raise ValueError(
                f"forward() not supported for mode='{self.config.mode}'. "
                "Use forward_critic_loss/forward_actor_loss for online_rl."
            )

    def get_optim_params(self) -> dict:
        """Return parameters to optimize based on current mode."""
        if self.config.mode == "rlt_training":
            return list(self.rlt_encoder.parameters()) + list(self.rlt_decoder.parameters())
        elif self.config.mode == "online_rl":
            return list(self.actor.parameters()) + list(self.critic.parameters())
        else:
            return []
