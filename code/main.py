import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from lightningdatamodule import CIFAR10DataModule
from lightningmodule import ImageClassificationNet
import pytorch_lightning as pl
from patchdiffmask import ImageInterpretationNet

if __name__ == '__main__':
    # How to use ViT
    model = ViTForImageClassification.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10")

    # print(model.config.num_hidden_layers)
    # exit()

    feature_extractor = ViTFeatureExtractor.from_pretrained("tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
                                                             return_tensors="pt")
    dm = CIFAR10DataModule(batch_size=1, feature_extractor=feature_extractor)
    # dm.prepare_data()
    # dm.setup()
    # #
    # dt = iter(dm.train_dataloader())
    # input = next(dt)[0]
    #
    # with torch.no_grad():
    #     outputs = model(input, output_hidden_states=True, output_attentions=True)
    #     print(outputs.hidden_states[0].shape)
    #     print(outputs.hidden_states[1].shape)
    #     print(len(outputs.hidden_states))
    #     print(outputs.attentions[0].shape)
    #     print(outputs.attentions[1].shape)
    #     print(len(outputs.attentions))
    #     exit()


    # # Let's test the model
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)
    classification_net = ImageInterpretationNet(model, 0.01)
    trainer.fit(classification_net, dm)
