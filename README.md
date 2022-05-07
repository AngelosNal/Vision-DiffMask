# Patch-DiffMask

This repo implements an adaptation of DiffMask **[1]** for vision models, and is heavily inspired by its [official PyTorch implementation](https://github.com/nicola-decao/diffmask)

DiffMask is an attribution model for Transformer and RNN architectures in NLP tasks. In particular, given a pretrained architecture, DiffMask predicts how much each part of the input contributed to the final prediction.

We adapted the original idea to work for vision models in CV tasks. Currently only Vision Transformers (ViT) in the image classfication setting are supported.

**[1]** De Cao, N., Schlichtkrull, M. S., Aziz, W., & Titov, I. (2020, November). How do Decisions Emerge across Layers in Neural Models? Interpretation with Differentiable Masking. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ (pp. 3243-3255).
