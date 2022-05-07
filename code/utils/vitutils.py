import torch


def vit_setter(model, inputs_dict, hidden_states):

    hidden_states_ = []

    def get_hook(i):
        def hook(module, inputs, outputs=None):
            # print(i)
            # print('inputs', inputs)
            # print('outputs', outputs)
            if i == 0:
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i][:, 1:, :])
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
                if hidden_states[-1] is not None:
                    hidden_states_.append(hidden_states[-1])
                    return (hidden_states[-1],) + outputs[1:]
                else:
                    hidden_states_.append(outputs[0])

        return hook

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

    try:
        outputs = model(inputs_dict).logits
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


def accuracy_precision_recall_f1(y_pred, y_true, average=True):
    M = confusion_matrix(y_pred, y_true)

    tp = M.diagonal(dim1=-2, dim2=-1).float()

    precision_den = M.sum(-2)
    precision = torch.where(
        precision_den == 0, torch.zeros_like(tp), tp / precision_den
    )

    recall_den = M.sum(-1)
    recall = torch.where(recall_den == 0, torch.ones_like(tp), tp / recall_den)

    f1_den = precision + recall
    f1 = torch.where(
        f1_den == 0, torch.zeros_like(tp), 2 * (precision * recall) / f1_den
    )

    return ((y_pred == y_true).float().mean(-1),) + (
        tuple(e.mean(-1) for e in (precision, recall, f1))
        if average
        else (precision, recall, f1)
    )


def confusion_matrix(y_pred, y_true):
    device = y_pred.device
    labels = max(y_pred.max().item() + 1, y_true.max().item() + 1)

    return (
        (
            torch.stack((y_true, y_pred), -1).unsqueeze(-2).unsqueeze(-2)
            == torch.stack(
                (
                    torch.arange(labels, device=device).unsqueeze(-1).repeat(1, labels),
                    torch.arange(labels, device=device).unsqueeze(-2).repeat(labels, 1),
                ),
                -1,
            )
        )
        .all(-1)
        .sum(-3)
    )

