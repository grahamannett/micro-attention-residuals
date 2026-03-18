# Notes

This is an extension of the original microgpt.py from [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) to include [attention residuals](https://github.com/MoonshotAI/Attention-Residuals). As the original microgpt.py implementation is 1 layer, the residual attentions may not be effective but the aim here is to implement it in the same minimal style for now.

The pseudo-code for pytorch they give is:

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

which we can translate into the microgpt style as following:

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]  # token embedding
    pos_emb = state_dict["wpe"][pos_id]  # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # joint token and position embedding
    layer_outs = [x]  # Full AttnRes: collect all sub-layer outputs

    for li in range(n_layer):
        # 1) Multi-head Attention block — AttnRes: attend over all previous outputs
        w = softmax(
            [
                sum(
                    p * k
                    for p, k in zip(
                        state_dict[f"layer{li}.attn_res_proj"][0], rmsnorm(r)
                    )
                )
                for r in layer_outs
            ]
        )
        x = [sum(w[i] * layer_outs[i][j] for i in range(len(w))) for j in range(n_embd)]
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        layer_outs.append(linear(x_attn, state_dict[f"layer{li}.attn_wo"]))
        # 2) MLP block — AttnRes: attend over all previous outputs
        w = softmax(
            [
                sum(
                    p * k
                    for p, k in zip(
                        state_dict[f"layer{li}.mlp_res_proj"][0], rmsnorm(r)
                    )
                )
                for r in layer_outs
            ]
        )
        x = [sum(w[i] * layer_outs[i][j] for i in range(len(w))) for j in range(n_embd)]
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        layer_outs.append(linear(x, state_dict[f"layer{li}.mlp_fc2"]))

    w = softmax(
        [
            sum(p * k for p, k in zip(state_dict["out_res_proj"][0], rmsnorm(r)))
            for r in layer_outs
        ]
    )
    x = [sum(w[i] * layer_outs[i][j] for i in range(len(w))) for j in range(n_embd)]
    logits = linear(x, state_dict["lm_head"])
    return logits
```

# Credits

- Original microgpt from [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
