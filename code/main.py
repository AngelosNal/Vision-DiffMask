import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from lightningdatamodule import CIFAR10DataModule
import pytorch_lightning as pl
from patchdiffmask import ImageInterpretationNet

if __name__ == '__main__':

    model = ViTForImageClassification.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10")
    interpretation_net = ImageInterpretationNet(model)

    feature_extractor = ViTFeatureExtractor.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
                                                             return_tensors="pt")
    dm = CIFAR10DataModule(batch_size=32, feature_extractor=feature_extractor)

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                        logger=pl.loggers.TensorBoardLogger(
            "outputs", name="vit-base-patch16-224-in21k-finetuned-cifar10",),
    )

    trainer.fit(interpretation_net, dm)
