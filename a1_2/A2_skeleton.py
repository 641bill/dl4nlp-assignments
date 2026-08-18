
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False
        )

        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        gate = self.act_fn(gate)

        up = self.up_proj(hidden_states)

        hidden_states = gate * up

        output = self.down_proj(hidden_states)
        return output

# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        # TODO: initalize weights here

    def forward(self, hidden_states, rope_rotations):
        ...

class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: set up W_q, W_k, W_v, W_o here
        # TODO: set up normalizers here
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        assert (
            self.hidden_size % self.num_attention_heads == 0
        )

        self.head_dim = (
            self.hidden_size // self.num_attention_heads
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.q_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        )

        self.k_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        )

    def forward(self, hidden_states, rope_rotations):
        b, m, d = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(
            b,
            m,
            self.num_attention_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            b,
            m,
            self.num_attention_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            b,
            m,
            self.num_attention_heads,
            self.head_dim
        ).transpose(1, 2)

        q, k = apply_rotary_pos_emb(
            q,
            k,
            rope_rotations
        )

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(b, m, d)

        output = self.o_proj(attn_out)
        return output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.self_attn = A2Attention(config)

        self.mlp = A2MLP(config)

        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        )

        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        )

    def forward(self, hidden_states, rope_rotations):
        # Attention
        residual = hidden_states

        hidden_states = self.self_attn(
            hidden_states,
            rope_rotations
        )

        hidden_states = self.post_attention_layernorm(
            hidden_states
        )

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(
            hidden_states
        )

        hidden_states = residual + hidden_states

        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        # TODO: put all transformer decoder layers in a ModuleList.
        self.layers = nn.ModuleList(
            [
                A2DecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=True
        )

        self.unembedding = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        self.loss_func = nn.CrossEntropyLoss(
            ignore_index=-100
        )

        # This line should be called after you have set up all components.
        self.post_init()
        # Trainer would otherwise treat **kwargs as accepting num_items_in_batch.
        self.model_accepts_loss_kwargs = False


    def forward(self, input_ids, labels=None, **kwargs):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                rope_rotations
            )
        hidden_states = self.norm(hidden_states)
        logits = self.unembedding(hidden_states)
        # TODO: Compute the loss as in Assignment 1 if labels is not None.
        loss = None

        if labels is not None:
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]

            shifted_logits = shifted_logits.reshape(
                -1,
                shifted_logits.shape[-1]
            )

            shifted_labels = shifted_labels.reshape(-1)

            loss = self.loss_func(
                shifted_logits,
                shifted_labels
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        rotary = model.rotary_emb
        if rotary.inv_freq.is_meta:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = None
            if device is not None and device.type == "meta":
                device = None
            rotary.inv_freq = rotary._compute_inv_freq(device)
        return model


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        self.rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = self._compute_inv_freq(device)
        # persistent=False: not in checkpoint; recompute after meta from_pretrained.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self, device=None):
        dim = self.rotary_dim
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        if device is not None:
            inv_freq = inv_freq.to(device)
        return inv_freq

    @torch.no_grad()
    def forward(self, x):
        if self.inv_freq.is_meta:
            self.inv_freq = self._compute_inv_freq(x.device)
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin
