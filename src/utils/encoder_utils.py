import torch
import numpy as np

# 文本 → 连续条件嵌入
@torch.no_grad()
def encode_text(
    input_ids,
    attention_mask,
    encoder,
    latent_mean,
    latent_std,
    use_bf16=True,
):
    """Encoder pass from text to latent with normalization."""
    # 缩放公式：(latents - 0.0) / 0.2，把 T5 输出整体放大 5 倍
    autocast_enabled = bool(use_bf16) and input_ids.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
        latents = encoder(input_ids=input_ids, attention_mask=attention_mask, deterministic=True)
    return (latents - latent_mean) / latent_std

# 从两个布尔矩阵造三张掩码:i 是条件 token 时，只能注意条件 token；i 不是条件 token（即生成/摘要段或 pad）时，可以注意任何真实 token。  <----- 条件段 ----->|<-生成段->|<-补位->
def build_self_attn_cond_masks(is_cond, is_valid, xp=np):
    """Build self-attention conditioning masks from cond/valid token flags."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
