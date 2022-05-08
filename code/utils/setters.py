"""
Setters for Transformer hidden layers.

* modified from: https://github.com/nicola-decao/diffmask/blob/master/diffmask/utils/getter_setter.py
"""


from torch import Tensor
from torch.nn import Module
from transformers import ViTForImageClassification
from typing import Callable, List, Optional, Tuple


def vit_setter(
    model: ViTForImageClassification, x: Tensor, hidden_states: List[Optional[Tensor]]
) -> Tuple[Tensor, Tuple[Tensor]]:
    hidden_states_ = []

    def input_post_hook(_: Module, __: tuple, outputs: Optional[tuple] = None) -> Optional[Tensor]:
        input_state = hidden_states[0]
        if input_state is not None:
            hidden_states_.append(input_state[:, 1:])
            return hidden_states_[-1]
        else:
            hidden_states_.append(outputs)

    def get_hook(layer_idx: int, post: bool) -> Callable:
        def hook(_: Module, inputs: tuple, outputs: Optional[tuple] = None) -> Optional[tuple]:
            src_tensor = outputs if post else inputs
            curr_state = hidden_states[layer_idx]

            if curr_state is not None:
                hidden_states_.append(curr_state)
                return (curr_state,) + src_tensor[1:]
            else:
                hidden_states_.append(src_tensor[0])

        return hook

    handles = (
        [model.vit.embeddings.patch_embeddings.register_forward_hook(input_post_hook)]
        + [
            layer.register_forward_pre_hook(get_hook(i, post=False))
            for i, layer in enumerate(model.vit.encoder.layer)
        ]
        + [model.vit.encoder.layer[-1].register_forward_hook(get_hook(-1, post=True))]
    )

    try:
        outputs = model(x).logits
    finally:
        for handle in handles:
            handle.remove()

    # noinspection PyTypeChecker
    return outputs, tuple(hidden_states_)
