# micro-attention-residuals

This is an extension of the original microgpt.py from [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) to include [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) from MoonshotAI. The implementation is kept in the same minimal, dependency-free style as the original.

Since the original microgpt uses `n_layer = 1`, attention residuals don't really have much to work with — the whole point is giving the network a richer way to mix information across many sublayer outputs as depth grows. With 1 layer there are only 3 candidates (embedding, attention output, MLP output) so the learned mixing is near-trivial. But this is useful as a barebones reference implementation to see what the mechanism looks like without any framework abstractions.

There are two variants from the paper, both implemented here:

| File | Variant | Description |
|------|---------|-------------|
| [`micro-full-attention-residuals.py`](micro-full-attention-residuals.py) | Full AttnRes | Attends over every individual sublayer output. Simple but O(2·L) candidates. |
| [`micro-attention-residuals.py`](micro-attention-residuals.py) | Block AttnRes | Groups layers into blocks, attends over block summaries. O(blocks) candidates. |

## Attention Residuals — the idea

In a standard transformer, the residual stream is a running sum — each sublayer just adds its output to what came before:

```python
x = x + attn(rmsnorm(x))    # attention adds to residual
x = x + mlp(rmsnorm(x))     # MLP adds to residual
```

Attention Residuals replace this with a *learned weighted mix* over previous outputs. Before each sublayer, a small projection scores all candidates via softmax, and the model picks how much of each prior output to use as input — rather than always using the sum.

## MoonshotAI's pseudocode

The paper provides this PyTorch-style reference for Block Attention Residuals:

```python
def block_attn_res(blocks: list[Tensor], partial_block: Tensor, proj: Linear, norm: RMSNorm) -> Tensor:
    """
    Inter-block attention: attend over block reps + partial sum.
    blocks:
        N tensors of shape [B, T, D]: completed block representations for each previous block
    partial_block:
        [B, T, D]:    intra-block partial sum (b_n^i)
    """
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h

def forward(self, blocks: list[Tensor], hidden_states: Tensor) -> tuple[list[Tensor], Tensor]:
    partial_block = hidden_states
    # apply block attnres before attn
    # blocks already include token embedding
    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

    # if reaches block boundary, start new block
    # block_size counts ATTN + MLP; each transformer layer has 2
    if self.layer_number % (self.block_size // 2) == 0:
        blocks.append(partial_block)
        partial_block = None

    # self-attention layer
    attn_out = self.attn(self.attn_norm(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out

    # apply block attnres before MLP
    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

    # MLP layer
    mlp_out = self.mlp(self.mlp_norm(h))
    partial_block = partial_block + mlp_out

    return blocks, partial_block
```

The key things to note:
- **Scoring uses normalized candidates** (`rmsnorm`) but the **weighted sum uses raw values**
- **`proj` is zero-initialized** so at init the mix is uniform, recovering standard residual behavior
- **Block boundaries** commit the current partial accumulator and start fresh

## Original microgpt → Full Attention Residuals

In the original microgpt ([`__ignore-microgpt.py`](__ignore-microgpt.py)), the `gpt()` forward pass uses a simple additive residual:

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Attention
        x_residual = x
        x = rmsnorm(x)
        # ... attention computation ...
        x = [a + b for a, b in zip(x, x_residual)]  # additive residual
        # 2) MLP
        x_residual = x
        x = rmsnorm(x)
        # ... MLP computation ...
        x = [a + b for a, b in zip(x, x_residual)]  # additive residual

    logits = linear(x, state_dict["lm_head"])
    return logits
```

In [`micro-full-attention-residuals.py`](micro-full-attention-residuals.py), we replace the additive residual with Full AttnRes — collecting every sublayer output into `layer_outs` and doing a learned mix before each sublayer:

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    layer_outs = [x]  # Full AttnRes: collect all sub-layer outputs

    for li in range(n_layer):
        # 1) Attention — AttnRes: attend over all previous outputs
        w = softmax([sum(p * k for p, k in zip(
            state_dict[f"layer{li}.attn_res_proj"][0], rmsnorm(r)
        )) for r in layer_outs])
        x = [sum(w[i] * layer_outs[i][j] for i in range(len(w))) for j in range(n_embd)]
        x = rmsnorm(x)
        # ... attention computation ...
        layer_outs.append(linear(x_attn, state_dict[f"layer{li}.attn_wo"]))

        # 2) MLP — AttnRes: attend over all previous outputs
        w = softmax([sum(p * k for p, k in zip(
            state_dict[f"layer{li}.mlp_res_proj"][0], rmsnorm(r)
        )) for r in layer_outs])
        x = [sum(w[i] * layer_outs[i][j] for i in range(len(w))) for j in range(n_embd)]
        x = rmsnorm(x)
        # ... MLP computation ...
        layer_outs.append(linear(x, state_dict[f"layer{li}.mlp_fc2"]))

    # Final learned mix over all sublayer outputs
    w = softmax([sum(p * k for p, k in zip(
        state_dict["out_res_proj"][0], rmsnorm(r)
    )) for r in layer_outs])
    x = [sum(w[i] * layer_outs[i][j] for i in range(len(w))) for j in range(n_embd)]
    logits = linear(x, state_dict["lm_head"])
    return logits
```

The candidates list grows as: `[embedding, attn_out_0, mlp_out_0, attn_out_1, mlp_out_1, ...]` — one entry per sublayer output across all layers.

## Full AttnRes → Block Attention Residuals

Full AttnRes works but the candidate set grows as O(2·L) with depth. Block AttnRes ([`micro-attention-residuals.py`](micro-attention-residuals.py)) fixes this by grouping layers into blocks. Instead of `layer_outs`, it tracks `blocks` (committed block summaries) and `partial_block` (the current block's running accumulation):

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    blocks = []          # completed block summaries
    partial_block = x    # current block accumulator (starts as embedding)

    for li in range(n_layer):
        # 1) Attention — Block AttnRes: mix over blocks + current partial
        w = softmax([sum(p * k for p, k in zip(
            state_dict[f"layer{li}.attn_res_proj"][0], rmsnorm(r)
        )) for r in blocks + [partial_block]])
        x = [sum(w[i] * (blocks + [partial_block])[i][j] for i in range(len(w)))
             for j in range(n_embd)]

        # Block boundary: commit partial, start fresh
        if li % layers_per_block == 0:
            blocks.append(partial_block)
            partial_block = None

        x = rmsnorm(x)
        # ... attention computation ...
        partial_block = attn_out if partial_block is None else [a + b for a, b in zip(partial_block, attn_out)]

        # 2) MLP — Block AttnRes: mix over blocks + current partial
        w = softmax([sum(p * k for p, k in zip(
            state_dict[f"layer{li}.mlp_res_proj"][0], rmsnorm(r)
        )) for r in blocks + [partial_block]])
        x = [sum(w[i] * (blocks + [partial_block])[i][j] for i in range(len(w)))
             for j in range(n_embd)]
        x = rmsnorm(x)
        # ... MLP computation ...
        partial_block = [a + b for a, b in zip(partial_block, mlp_out)]

    # Final learned mix over all blocks + last partial
    w = softmax([sum(p * k for p, k in zip(
        state_dict["out_res_proj"][0], rmsnorm(r)
    )) for r in blocks + [partial_block]])
    x = [sum(w[i] * (blocks + [partial_block])[i][j] for i in range(len(w)))
         for j in range(n_embd)]
    logits = linear(x, state_dict["lm_head"])
    return logits
```

The candidates are now: `[block_0_summary, block_1_summary, ..., partial_block]`. At each block boundary the embedding or accumulated partial gets committed and a fresh accumulator starts. With `layers_per_block = 1` (the default), the boundary fires every layer so the trace for `n_layer=2` looks like:

```
layer 0: mix [x] → commit x → attn → mlp → partial = attn_out₀ + mlp_out₀
layer 1: mix [x, partial₀] → commit partial₀ → attn → mlp → partial = attn_out₁ + mlp_out₁
final:   mix [x, partial₀, partial₁]
```

## Usage

No dependencies — just Python 3:

```bash
python micro-attention-residuals.py       # Block AttnRes
python micro-full-attention-residuals.py  # Full AttnRes
```

Both auto-download a names dataset, train for 1000 steps, and generate sample names.

## Credits

- Original microgpt by [@karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- Attention Residuals paper and code by [MoonshotAI](https://github.com/MoonshotAI/Attention-Residuals)
