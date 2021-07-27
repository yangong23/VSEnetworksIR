# ----------------------------------------------------------------
# Copied by Yan Gong
# Last revised: July 2021
# Reference: The code is copied from UNITER: UNITER: UNiversal Image-TExt Representation Learning (https://arxiv.org/abs/1909.11740).
# -----------------------------------------------------------------

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

IMG_NPY=$1
OUT_DIR=$2

set -e

echo "converting image features ..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
NAME=$(basename $IMG_NPY)
docker run --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUT_DIR,dst=/img_db,type=bind \
    --mount src=$IMG_NPY,dst=/$NAME,type=bind,readonly \
    -w /src chenrocks/uniter \
    python scripts/convert_imgdir.py --img_dir /$NAME --output /img_db

echo "done"
