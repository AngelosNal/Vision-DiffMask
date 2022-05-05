import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from lightningdatamodule import CIFAR10DataModule
from lightningmodule import ImageClassificationNet
import pytorch_lightning as pl

if __name__ == '__main__':
    # How to use ViT
    model = ViTForImageClassification.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10")

    feature_extractor = ViTFeatureExtractor.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
                                                             return_tensors="pt")
    dm = CIFAR10DataModule(batch_size=2, feature_extractor=feature_extractor)
    dm.prepare_data()
    dm.setup()

    dt = iter(dm.train_dataloader())
    input = next(dt)[0]

    with torch.no_grad():
        outputs = model(input, output_hidden_states=False, output_attentions=False)
        print(outputs.logits)


    # # Let's test the model
    trainer = pl.Trainer()
    classification_net = ImageClassificationNet(model, 0.01)
    # TODO: need to fix strange error
    trainer.test(classification_net, dm)
