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
    dm = CIFAR10DataModule(batch_size=64, feature_extractor=feature_extractor)
    dm.prepare_data()
    dm.setup()

    dt = iter(dm.train_dataloader())
    input = next(dt)[0]

    with torch.no_grad():
        outputs = model(input, output_hidden_states=False, output_attentions=False)
        print(outputs.logits)

    # # Let's test the model
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)
    classification_net = ImageClassificationNet(model, 0.01)
    trainer.test(classification_net, dm)
