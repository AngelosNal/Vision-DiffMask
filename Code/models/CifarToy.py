import torch
from gates import MLPGate, MLPMaxGate, DiffMaskGateInput, DiffMaskGateHidden
from distributions import RectifiedStreched, BinaryConcrete
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
from ViT import ViT

class CifarToyDiffMask(ViT):

    def __init__(self, hparams):
        super().__init__(hparams)
    
        for p in super().parameters():
            p.requires_grad = False
    
        self.alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(()))
                for _ in range(self.net.config.num_hidden_layers + 2)
            ]
        )

        gate = DiffMaskGateInput if self.hparams.gate == "input" else DiffMaskGateHidden

        self.gate = gate(
            hidden_size=self.net.config.hidden_size,
            hidden_attention=self.net.config.hidden_size // 4,
            num_hidden_layers=self.net.config.num_hidden_layers + 2,
            max_position_embeddings=1,
            gate_bias=hparams.gate_bias,
            placeholder=hparams.placeholder,
            init_vector=self.net.bert.embeddings.word_embeddings.weight[
                self.tokenizer.mask_token_id
            ]
            if self.hparams.layer_pred == 0 or self.hparams.gate == "input"
            else None,
        )