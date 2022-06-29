import sys
sys.path.insert(0,'../code')
import torch
import torch.nn.functional as F
import numpy as np
from datamodules.transformations import UnNest
from transformers import ViTFeatureExtractor, ViTForImageClassification
from tqdm.auto import tqdm
from datamodules.image_classification import ImageNetDataModule
from models.interpretation import ImageInterpretationNet
from utils.getters_setters import vit_getter, vit_setter
from attributions.grad_cam import grad_cam
from attributions.attention_rollout import attention_rollout
from utils.plot import smoothen
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vit = ViTForImageClassification.from_pretrained("tanlq/vit-base-patch16-224").to(device)

diffmask = ImageInterpretationNet.load_from_checkpoint('../checkpoints/diffmask_imagenet.ckpt').to(device)

feature_extractor=ViTFeatureExtractor.from_pretrained(
    "tanlq/vit-base-patch16-224", return_tensors="pt"
)
feature_extractor = UnNest(feature_extractor)

batch_size = 4

dm = ImageNetDataModule(feature_extractor=feature_extractor, batch_size=batch_size)
dm.prepare_data()
dm.setup('test')
dataloader = dm.test_dataloader()

def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = torch.div(index, dim, rounding_mode='trunc')
        return tuple(reversed(out))

def auc(model, input, labels, attributions, positive=True, k=0.2, num_tokens=1):
    area = []
    B, H, W = attributions.shape
    attributions = attributions.reshape(B, -1)
    # Get hidden states from unmasked input
    logits_orig, hidden_states = vit_getter(model, input)
    probs_orig = logits_orig.softmax(-1)
    if not labels:
        labels = probs_orig.argmax(-1)

    drop_k = int(k * attributions.shape[1])

    for i in range(0, drop_k + 1, 256 * num_tokens):
        keep_k = attributions.shape[1] - i
        idx = attributions.topk(keep_k, dim=1, largest=(not positive)).indices.sort()[0].squeeze(-1)

        torch.manual_seed(123)
        f = torch.rand((B, input.shape[1], H, W), device=device)*2 - 1 # Baseline pixels
        for i in range(B):
            idx_i = idx[i]
            unraveled_idx = unravel_index(idx_i, (H, W))
            f[i, :, unraveled_idx[0], unraveled_idx[1]] = input[i, :, unraveled_idx[0], unraveled_idx[1]]

        # Forward pass through the model to get the logits
        logits, hidden_states = vit_getter(model, f)
        probs = logits.softmax(-1)

        area.append(probs[range(B), labels].mean().item())

    return area

def calculate_auc(positive=True, method='diffmask'):
    auc_positive = []
    for i, (images, labels) in tqdm(enumerate(dataloader), total=50):
        if i == 50: break
        images, labels = images.cuda(), labels.cuda()
        if method == 'diffmask':
            attributions = diffmask.get_mask(images)["mask"].detach()
        elif method == 'rollout':
            attributions = attention_rollout(images=images, vit=vit, device=device)
        elif method == 'gradcam':
            images.requires_grad = True
            attributions = grad_cam(images, vit, True if device=='cuda' else False)
        comp = auc(vit, images, None, attributions, num_tokens=7, k=1, positive=positive)
        auc_positive.append(comp)
    return auc_positive


auc_pos_gradcam = np.asarray(calculate_auc(positive=True, method='gradcam')).mean(0)
auc_neg_gradcam = np.asarray(calculate_auc(positive=False, method='gradcam')).mean(0)

diffmask.set_vision_transformer(vit)

auc_pos_diffmask = np.asarray(calculate_auc(positive=True)).mean(0)
auc_neg_diffmask = np.asarray(calculate_auc(positive=False)).mean(0)

auc_pos_rollout = np.asarray(calculate_auc(positive=True, method='rollout')).mean(0)
auc_neg_rollout = np.asarray(calculate_auc(positive=False, method='rollout')).mean(0)

plt.plot(auc_pos_diffmask, label="DiffMask")
plt.plot(auc_pos_rollout, label="Attention Rollout")
plt.plot(auc_pos_gradcam, label="Grad-CAM")
plt.xlabel("Percentage of positive pixels removed")
len_x = len(auc_pos_diffmask)
x_ticks = np.arange(0, len_x, 3)
x_ticks_labels = [str(int(x * 100 / len_x)) + "%" for x in x_ticks]
plt.xticks(x_ticks, x_ticks_labels)
plt.legend()
plt.savefig("")