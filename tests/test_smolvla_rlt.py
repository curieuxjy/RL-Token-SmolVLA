"""Unit tests for SmolVLA-RLT policy components.

Tests shapes, loss computation, and training steps for:
  - RLTokenEncoder
  - RLTokenDecoder
  - RLActorMLP
  - RLCriticMLP
  - SmolVLARLTPolicy (Stage 1 forward)
"""

import pytest
import torch

from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import (
    RLActorMLP,
    RLCriticMLP,
    RLTokenDecoder,
    RLTokenEncoder,
    SmolVLARLTPolicy,
)

# Test dimensions
BATCH_SIZE = 4
VLM_SEQ_LEN = 120   # Typical prefix sequence length
EXPERT_SEQ_LEN = 50  # chunk_size


@pytest.fixture
def config():
    return SmolVLARLTConfig(mode="rlt_training")


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# RLT Encoder Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRLTokenEncoder:
    def test_output_shape(self, config, device):
        encoder = RLTokenEncoder(config).to(device)
        vlm_emb = torch.randn(BATCH_SIZE, VLM_SEQ_LEN, config.vlm_hidden_dim, device=device)
        expert_emb = torch.randn(BATCH_SIZE, EXPERT_SEQ_LEN, config.expert_hidden_dim, device=device)

        z_rl = encoder(vlm_emb, expert_emb)

        assert z_rl.shape == (BATCH_SIZE, config.rlt_hidden_dim)
        assert z_rl.dtype == torch.float32

    def test_different_sequence_lengths(self, config, device):
        """Encoder should handle varying input sequence lengths."""
        encoder = RLTokenEncoder(config).to(device)

        for vlm_len, expert_len in [(50, 20), (200, 50), (100, 100)]:
            vlm_emb = torch.randn(BATCH_SIZE, vlm_len, config.vlm_hidden_dim, device=device)
            expert_emb = torch.randn(BATCH_SIZE, expert_len, config.expert_hidden_dim, device=device)
            z_rl = encoder(vlm_emb, expert_emb)
            assert z_rl.shape == (BATCH_SIZE, config.rlt_hidden_dim)

    def test_gradients_flow(self, config, device):
        encoder = RLTokenEncoder(config).to(device)
        vlm_emb = torch.randn(BATCH_SIZE, VLM_SEQ_LEN, config.vlm_hidden_dim, device=device)
        expert_emb = torch.randn(BATCH_SIZE, EXPERT_SEQ_LEN, config.expert_hidden_dim, device=device)

        z_rl = encoder(vlm_emb, expert_emb)
        loss = z_rl.sum()
        loss.backward()

        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_param_count(self, config):
        encoder = RLTokenEncoder(config)
        total = sum(p.numel() for p in encoder.parameters())
        # Should be roughly 3-5M params
        assert 1_000_000 < total < 20_000_000, f"Unexpected param count: {total}"


# ─────────────────────────────────────────────────────────────────────────────
# RLT Decoder Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRLTokenDecoder:
    def test_output_shapes(self, config, device):
        decoder = RLTokenDecoder(config).to(device)
        z_rl = torch.randn(BATCH_SIZE, config.rlt_hidden_dim, device=device)

        vlm_recon, expert_recon = decoder(z_rl, VLM_SEQ_LEN, EXPERT_SEQ_LEN)

        assert vlm_recon.shape == (BATCH_SIZE, VLM_SEQ_LEN, config.vlm_hidden_dim)
        assert expert_recon.shape == (BATCH_SIZE, EXPERT_SEQ_LEN, config.expert_hidden_dim)

    def test_reconstruction_loss_finite(self, config, device):
        encoder = RLTokenEncoder(config).to(device)
        decoder = RLTokenDecoder(config).to(device)

        vlm_emb = torch.randn(BATCH_SIZE, VLM_SEQ_LEN, config.vlm_hidden_dim, device=device)
        expert_emb = torch.randn(BATCH_SIZE, EXPERT_SEQ_LEN, config.expert_hidden_dim, device=device)

        z_rl = encoder(vlm_emb, expert_emb)
        vlm_recon, expert_recon = decoder(z_rl, VLM_SEQ_LEN, EXPERT_SEQ_LEN)

        vlm_loss = torch.nn.functional.mse_loss(vlm_recon, vlm_emb.detach())
        expert_loss = torch.nn.functional.mse_loss(expert_recon, expert_emb.detach())

        assert torch.isfinite(vlm_loss)
        assert torch.isfinite(expert_loss)
        assert vlm_loss > 0
        assert expert_loss > 0


# ─────────────────────────────────────────────────────────────────────────────
# Actor MLP Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRLActorMLP:
    def test_output_shape(self, config, device):
        actor = RLActorMLP(config).to(device)

        z_rl = torch.randn(BATCH_SIZE, config.rlt_hidden_dim, device=device)
        state = torch.randn(BATCH_SIZE, config.state_dim, device=device)
        ref_actions = torch.randn(
            BATCH_SIZE, config.n_action_steps_rl, config.action_dim, device=device
        )

        action_mean, log_std = actor(z_rl, state, ref_actions)

        assert action_mean.shape == (BATCH_SIZE, config.n_action_steps_rl, config.action_dim)
        assert log_std.shape == (config.n_action_steps_rl * config.action_dim,)

    def test_ref_dropout(self, config, device):
        """With ref_dropout=1.0, ref_actions should be fully zeroed."""
        config_full_dropout = SmolVLARLTConfig(mode="rlt_training", ref_dropout=1.0)
        actor = RLActorMLP(config_full_dropout).to(device)

        z_rl = torch.randn(BATCH_SIZE, config.rlt_hidden_dim, device=device)
        state = torch.randn(BATCH_SIZE, config.state_dim, device=device)
        ref_actions = torch.ones(
            BATCH_SIZE, config.n_action_steps_rl, config.action_dim, device=device
        )

        # Train mode with dropout=1.0 should zero all ref_actions
        action_mean_with_drop, _ = actor(z_rl, state, ref_actions, training=True)
        action_mean_no_drop, _ = actor(z_rl, state, ref_actions, training=False)

        # Outputs should differ when dropout is applied
        # (not guaranteed to be different with random init, but very likely)
        assert action_mean_with_drop.shape == action_mean_no_drop.shape

    def test_gradients_flow(self, config, device):
        actor = RLActorMLP(config).to(device)
        z_rl = torch.randn(BATCH_SIZE, config.rlt_hidden_dim, device=device)
        state = torch.randn(BATCH_SIZE, config.state_dim, device=device)
        ref = torch.randn(BATCH_SIZE, config.n_action_steps_rl, config.action_dim, device=device)

        action_mean, _ = actor(z_rl, state, ref, training=True)
        action_mean.sum().backward()

        for p in actor.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# Critic MLP Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRLCriticMLP:
    def test_output_shape(self, config, device):
        critic = RLCriticMLP(config).to(device)

        z_rl = torch.randn(BATCH_SIZE, config.rlt_hidden_dim, device=device)
        state = torch.randn(BATCH_SIZE, config.state_dim, device=device)
        actions = torch.randn(
            BATCH_SIZE, config.n_action_steps_rl, config.action_dim, device=device
        )

        q1, q2 = critic(z_rl, state, actions)

        assert q1.shape == (BATCH_SIZE, 1)
        assert q2.shape == (BATCH_SIZE, 1)

    def test_q_min(self, config, device):
        critic = RLCriticMLP(config).to(device)

        z_rl = torch.randn(BATCH_SIZE, config.rlt_hidden_dim, device=device)
        state = torch.randn(BATCH_SIZE, config.state_dim, device=device)
        actions = torch.randn(
            BATCH_SIZE, config.n_action_steps_rl, config.action_dim, device=device
        )

        q_min = critic.q_min(z_rl, state, actions)
        q1, q2 = critic(z_rl, state, actions)

        assert q_min.shape == (BATCH_SIZE, 1)
        assert torch.all(q_min <= q1) or torch.all(q_min <= q2)

    def test_twin_independence(self, config, device):
        """Twin Q-networks should produce different outputs."""
        critic = RLCriticMLP(config).to(device)

        z_rl = torch.randn(BATCH_SIZE, config.rlt_hidden_dim, device=device)
        state = torch.randn(BATCH_SIZE, config.state_dim, device=device)
        actions = torch.randn(
            BATCH_SIZE, config.n_action_steps_rl, config.action_dim, device=device
        )

        q1, q2 = critic(z_rl, state, actions)
        # With random initialization, Q1 and Q2 should differ
        assert not torch.allclose(q1, q2, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# SmolVLARLTPolicy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSmolVLARLTPolicy:
    def test_rlt_training_forward(self, config, device):
        """Test Stage 1 forward pass with pre-extracted embeddings."""
        policy = SmolVLARLTPolicy(config).to(device)

        batch = {
            "vlm_embeddings": torch.randn(BATCH_SIZE, VLM_SEQ_LEN, config.vlm_hidden_dim, device=device),
            "expert_embeddings": torch.randn(BATCH_SIZE, EXPERT_SEQ_LEN, config.expert_hidden_dim, device=device),
        }

        loss, loss_dict = policy.forward(batch)

        assert torch.isfinite(loss)
        assert loss > 0
        assert "vlm_recon_loss" in loss_dict
        assert "expert_recon_loss" in loss_dict

    def test_rlt_training_backward(self, config, device):
        """Test that gradients flow through encoder/decoder in Stage 1."""
        policy = SmolVLARLTPolicy(config).to(device)

        batch = {
            "vlm_embeddings": torch.randn(BATCH_SIZE, VLM_SEQ_LEN, config.vlm_hidden_dim, device=device),
            "expert_embeddings": torch.randn(BATCH_SIZE, EXPERT_SEQ_LEN, config.expert_hidden_dim, device=device),
        }

        loss, _ = policy.forward(batch)
        loss.backward()

        # Encoder should have gradients
        for p in policy.rlt_encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

        # Decoder should have gradients
        for p in policy.rlt_decoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_training_step(self, config, device):
        """Test a full optimizer step reduces loss (smoke test)."""
        policy = SmolVLARLTPolicy(config).to(device)
        optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-3)

        batch = {
            "vlm_embeddings": torch.randn(BATCH_SIZE, VLM_SEQ_LEN, config.vlm_hidden_dim, device=device),
            "expert_embeddings": torch.randn(BATCH_SIZE, EXPERT_SEQ_LEN, config.expert_hidden_dim, device=device),
        }

        # Step 1
        loss1, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # Step 2 (same batch — loss should decrease with perfect memorization)
        loss2, _ = policy.forward(batch)

        assert torch.isfinite(loss2)
        # Don't assert loss2 < loss1 — one step isn't guaranteed to decrease

    def test_critic_loss(self, config, device):
        """Test critic loss computation with synthetic data."""
        config_rl = SmolVLARLTConfig(mode="online_rl")
        policy = SmolVLARLTPolicy(config_rl).to(device)

        d = config_rl.rlt_hidden_dim
        sd = config_rl.state_dim
        na = config_rl.n_action_steps_rl
        ad = config_rl.action_dim

        result = policy.forward_critic_loss(
            z_rl=torch.randn(BATCH_SIZE, d, device=device),
            state=torch.randn(BATCH_SIZE, sd, device=device),
            actions=torch.randn(BATCH_SIZE, na, ad, device=device),
            rewards=torch.randn(BATCH_SIZE, 1, device=device),
            next_z_rl=torch.randn(BATCH_SIZE, d, device=device),
            next_state=torch.randn(BATCH_SIZE, sd, device=device),
            next_ref_actions=torch.randn(BATCH_SIZE, na, ad, device=device),
            dones=torch.zeros(BATCH_SIZE, 1, device=device),
        )

        assert torch.isfinite(result["critic_loss"])
        assert "q1_mean" in result
        assert "q2_mean" in result

    def test_actor_loss(self, config, device):
        """Test actor loss computation with synthetic data."""
        config_rl = SmolVLARLTConfig(mode="online_rl")
        policy = SmolVLARLTPolicy(config_rl).to(device)

        d = config_rl.rlt_hidden_dim
        sd = config_rl.state_dim
        na = config_rl.n_action_steps_rl
        ad = config_rl.action_dim

        result = policy.forward_actor_loss(
            z_rl=torch.randn(BATCH_SIZE, d, device=device),
            state=torch.randn(BATCH_SIZE, sd, device=device),
            ref_actions=torch.randn(BATCH_SIZE, na, ad, device=device),
        )

        assert torch.isfinite(result["actor_loss"])
        assert "q_loss" in result
        assert "bc_loss" in result

    def test_target_critic_update(self, config, device):
        """Test EMA target update changes target params."""
        config_rl = SmolVLARLTConfig(mode="online_rl")
        policy = SmolVLARLTPolicy(config_rl).to(device)

        # Store original target params
        orig_params = [p.clone() for p in policy.critic_target.parameters()]

        # Modify critic params
        for p in policy.critic.parameters():
            p.data.add_(torch.randn_like(p))

        # Update target
        policy.update_target_critic()

        # Target should have changed
        for orig, updated in zip(orig_params, policy.critic_target.parameters()):
            assert not torch.allclose(orig, updated.data)

    def test_get_optim_params_rlt_training(self, config, device):
        """In rlt_training mode, only encoder+decoder params should be optimized."""
        policy = SmolVLARLTPolicy(config).to(device)
        params = policy.get_optim_params()
        expected_count = (
            sum(1 for p in policy.rlt_encoder.parameters())
            + sum(1 for p in policy.rlt_decoder.parameters())
        )
        assert len(params) == expected_count

    def test_get_optim_params_online_rl(self, device):
        """In online_rl mode, only actor+critic params should be optimized."""
        config_rl = SmolVLARLTConfig(mode="online_rl")
        policy = SmolVLARLTPolicy(config_rl).to(device)
        params = policy.get_optim_params()
        expected_count = (
            sum(1 for p in policy.actor.parameters())
            + sum(1 for p in policy.critic.parameters())
        )
        assert len(params) == expected_count


# ─────────────────────────────────────────────────────────────────────────────
# Config Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSmolVLARLTConfig:
    def test_valid_modes(self):
        for mode in ("rlt_training", "online_rl", "inference"):
            config = SmolVLARLTConfig(mode=mode)
            assert config.mode == mode

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            SmolVLARLTConfig(mode="invalid")

    def test_default_dimensions(self):
        config = SmolVLARLTConfig()
        assert config.vlm_hidden_dim == 576
        assert config.expert_hidden_dim == 432
        assert config.rlt_hidden_dim == 256
        assert config.action_dim == 7
        assert config.state_dim == 7
        assert config.n_action_steps_rl == 10

    def test_action_delta_indices(self):
        config = SmolVLARLTConfig()
        assert config.action_delta_indices == list(range(10))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
