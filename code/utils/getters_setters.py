from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from transformers import ViTForImageClassification
from typing import Optional, Union


def _add_hooks(
    model: ViTForImageClassification, get_hook: callable
) -> list[RemovableHandle]:
    """Adds a list of hooks to the model according to the get_hook function provided.

    Args:
        model (ViTForImageClassification): the ViT instance to add hooks to
        get_hook (callable): a function that takes an index and returns a hook

    Returns:
        a list of RemovableHandle instances
    """
    return (
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


def vit_getter(
    model: ViTForImageClassification, x: Tensor
) -> tuple[Tensor, list[Tensor]]:
    """A function that returns the logits and hidden states of the model.

    Args:
        model (ViTForImageClassification): the ViT instance to use for the forward pass
        x (Tensor): the input to the model

    Returns:
        a tuple of the model's logits and hidden states
    """
    hidden_states_ = []

    def get_hook(i: int) -> callable:
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
    """A function that sets some of the model's hidden states and returns its (new) logits
     and hidden states after another forward pass.

    Args:
        model (ViTForImageClassification): the ViT instance to use for the forward pass
        x (Tensor): the input to the model
        hidden_states (list[Optional[Tensor]]): a list, with each element corresponding to
         a hidden state to set or None to calculate anew for that index

    Returns:
        a tuple of the model's logits and (new) hidden states
    """
    hidden_states_ = []

    def get_hook(i: int) -> callable:
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
