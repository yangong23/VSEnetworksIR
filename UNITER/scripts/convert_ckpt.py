# ----------------------------------------------------------------
# Copied by Yan Gong
# Last revised: July 2021
# Reference: The code is copied from UNITER: UNITER: UNiversal Image-TExt Representation Learning (https://arxiv.org/abs/1909.11740).
# -----------------------------------------------------------------

import sys
from collections import OrderedDict

import torch

bert_ckpt, output_ckpt = sys.argv[1:]

bert = torch.load(bert_ckpt)
uniter = OrderedDict()
for k, v in bert.items():
    uniter[k.replace('bert', 'uniter')] = v

torch.save(uniter, output_ckpt)
