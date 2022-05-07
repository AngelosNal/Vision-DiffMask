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
