from transformers import AutoConfig

config = AutoConfig.from_pretrained("./checkpoints/base")

# See all architecture parameters
print(config)

# Key parameters:
print(f"Encoder layers: {config.encoder_layers}")
print(f"Decoder layers: {config.decoder_layers}")
print(f"Hidden size: {config.d_model}")
print(f"Attention heads: {config.encoder_attention_heads}")
print(f"FFN dimension: {config.encoder_ffn_dim}")
print(f"Vocab size: {config.vocab_size}")