# ----------------------------------------------------------------
# Copied by Yan Gong
# Last revised: July 2021
# Reference: The code is copied from UNITER: UNITER: UNiversal Image-TExt Representation Learning (https://arxiv.org/abs/1909.11740).
# -----------------------------------------------------------------

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

BLOB='https://convaisharables.blob.core.windows.net/uniter'

for MODEL in uniter-base uniter-large; do
    # This will overwrite models
    wget $BLOB/pretrained/$MODEL.pt -O $DOWNLOAD/pretrained/$MODEL.pt
done
