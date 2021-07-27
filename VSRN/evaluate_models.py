# ----------------------------------------------------------------
# Modified by Yan Gong
# Last revised: July 2021
# Reference: The orignal code is from VSRN: Visual Semantic Reasoning for Image-Text Matching (https://arxiv.org/pdf/1909.02701.pdf).
# The code has been modified from python2 to python3.
# -----------------------------------------------------------------

import torch
from vocab import Vocabulary
import evaluation_models

# for coco
print('Evaluation on COCO:')
evaluation_models.evalrank("pretrain_model/coco/model_coco_1.pth.tar", "pretrain_model/coco/model_coco_2.pth.tar", data_path='data/', split="testall", fold5=True)

# for flickr
print('Evaluation on Flickr30K:')
evaluation_models.evalrank("pretrain_model/flickr/model_fliker_1.pth.tar", "pretrain_model/flickr/model_fliker_2.pth.tar", data_path='data/', split="test", fold5=False)
