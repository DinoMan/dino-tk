import torch.nn as nn

from .encoder import Encoder
from ..utils import make_non_pad_mask


class Conformer(nn.Module):
    def __init__(self, idim, adim=256, aheads=4, elayers=12, attention_dropout_rate=0.1, input_layer="vanilla_linear", eunits=2048, dropout_rate=0.1,
                 positional_dropout_rate=0.1, macaron_style=True, use_cnn_module=True, cnn_module_kernel=31, encoder_attn_layer_type="rel_mha"):
        super(Conformer, self).__init__()

        encoder = Encoder(idim=idim, attention_dim=adim, attention_heads=aheads, linear_units=eunits, num_blocks=elayers, dropout_rate=dropout_rate,
                          positional_dropout_rate=positional_dropout_rate, attention_dropout_rate=attention_dropout_rate, input_layer=input_layer,
                          macaron_style=macaron_style, encoder_attn_layer_type=encoder_attn_layer_type, use_cnn_module=use_cnn_module,
                          cnn_module_kernel=cnn_module_kernel)

        self.encoder = encoder

    def forward(self, x, lengths):
        mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        x = self.encoder(x, mask)[0]
        return x, mask.squeeze().unsqueeze(-1)