# ----------------------------------------------------------------
# Copied by Yan Gong
# Last revised: July 2021
# Reference: The code is copied from UNITER: UNITER: UNiversal Image-TExt Representation Learning (https://arxiv.org/abs/1909.11740).
# -----------------------------------------------------------------

"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
from .sched import noam_schedule, warmup_linear, vqa_schedule, get_lr_sched
from .adamw import AdamW
