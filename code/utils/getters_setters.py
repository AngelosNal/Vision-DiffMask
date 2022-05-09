from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from transformers import ViTForImageClassification
from typing import Callable, Optional, Union


def _add_hooks(
    model: ViTForImageClassification, get_hook: Callable
) -> list[RemovableHandle]:
    handles = (
        [model.vit.embeddings.patch_embeddings.register_forward_hook(get_hook(0))]
        + [
            layer.register_forward_pre_hook(get_hook(i + 1))
            for i, layer in enumerate(model.vit.encoder.layer)
        ]
        + [
            model.vit.encoder.layer[-1].register_forward_hook(
                get_hook(len(model.vit.encoder.layer) + 1)
            )
        ]
    )

    return handles


def vit_getter(
    model: ViTForImageClassification, x: Tensor
) -> tuple[Tensor, list[Tensor]]:
    hidden_states_ = []

    def get_hook(i: int) -> Callable:
        def hook(_: Module, inputs: tuple, outputs: Optional[tuple] = None):
            if i == 0:
                hidden_states_.append(outputs)
            elif 1 <= i <= len(model.vit.encoder.layer):
                hidden_states_.append(inputs[0])
            elif i == len(model.vit.encoder.layer) + 1:
                hidden_states_.append(outputs[0])

        return hook

    handles = _add_hooks(model, get_hook)
    try:
        logits = model(x).logits
    finally:
        for handle in handles:
            handle.remove()

    return logits, hidden_states_


def vit_setter(
    model: ViTForImageClassification, x: Tensor, hidden_states: list[Optional[Tensor]]
) -> tuple[Tensor, list[Tensor]]:
    hidden_states_ = []

    def get_hook(i: int):
        def hook(
            _: Module, inputs: tuple, outputs: Optional[tuple] = None
        ) -> Optional[Union[tuple, Tensor]]:
            if i == 0:
                if hidden_states[i] is not None:
                    # print(hidden_states[i].shape)
                    hidden_states_.append(hidden_states[i][:, 1:])
                    return hidden_states_[-1]
                else:
                    hidden_states_.append(outputs)

            elif 1 <= i <= len(model.vit.encoder.layer):
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return (hidden_states[i],) + inputs[1:]
                else:
                    hidden_states_.append(inputs[0])

            elif i == len(model.vit.encoder.layer) + 1:
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return (hidden_states[i],) + outputs[1:]
                else:
                    hidden_states_.append(outputs[0])

        return hook

    handles = _add_hooks(model, get_hook)

    try:
        logits = model(x).logits
    finally:
        for handle in handles:
            handle.remove()

    return logits, hidden_states_
