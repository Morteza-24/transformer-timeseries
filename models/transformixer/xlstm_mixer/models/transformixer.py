from torch import nn

from ..layers.Embed import PositionalEmbedding
from ..layers.SelfAttention_Family import AttentionLayer, FullAttention
from ..layers.StandardNorm import Normalize as RevIN
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from .base_model import BaseModel


class Transformixer(BaseModel):
    """NLinear temporal mixing followed by a Transformer variate mixer.

    After the shared NLinear forecast and embedding projection, each variate is a token.
    By default, self-attention is applied without positional encodings and without a
    causal mask so the mixer is permutation-equivariant along the variate axis.

    Ablation flags:
      - ``use_positional_encoding``: add sinusoidal PE on variate tokens (breaks equivariance).
      - ``use_variate_mixer``: if False, return the shared NLinear forecast only (no Transformer).
    """

    def __init__(
        self,
        pred_len: int,
        seq_len: int,
        enc_in: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 1,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = False,
        use_variate_mixer: bool = True,
        max_variates: int = 2048,
    ) -> None:
        super().__init__(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
        d_ff = d_ff or 4 * d_model

        self.use_positional_encoding = use_positional_encoding
        self.use_variate_mixer = use_variate_mixer

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.reversible_instance_norm = RevIN(enc_in, affine=False)

        if use_variate_mixer:
            self.pre_encoding = nn.Linear(self.pred_len, d_model)
            self.pos_embedding = (
                PositionalEmbedding(d_model, max_len=max_variates)
                if use_positional_encoding
                else None
            )
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                False,
                                1,
                                attention_dropout=dropout,
                                output_attention=False,
                            ),
                            d_model,
                            n_heads,
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(e_layers)
                ],
                norm_layer=nn.LayerNorm(d_model),
            )
            self.fc = nn.Linear(d_model, self.pred_len)
        else:
            self.pre_encoding = None
            self.pos_embedding = None
            self.encoder = None
            self.fc = None

    def forecast(self, x_enc, x_mark_enc):
        x_enc = self.reversible_instance_norm(x_enc, "norm")

        seq_last = x_enc[:, -1:, :].detach()
        x = x_enc - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_pre_forecast = (x + seq_last).permute(0, 2, 1)  # [B, C, H]

        if not self.use_variate_mixer:
            return self.reversible_instance_norm(
                x_pre_forecast.permute(0, 2, 1), "denorm"
            )

        x = self.pre_encoding(x_pre_forecast)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(x)
        x, _ = self.encoder(x, attn_mask=None)
        x = self.fc(x).permute(0, 2, 1)

        return self.reversible_instance_norm(x, "denorm")
