# ----------------------------------------------------------------
# On the Limitations of Visual-Semantic Embedding Networks for Image-to-Text Information Retrieval
# by Yan Gong, Georgina Cosma, and Hui Fang
# Programmed by Yan Gong
# Last revised: July 2021
# Reference: The code is modified based on UNITER: UNITER: UNiversal Image-TExt Representation Learning (https://arxiv.org/abs/1909.11740). 
# These are the functions for evaluating UNITER and analysing UNITER's limitations.
# Copyright (c) 2021, Yan Gong. All rights reserved.
# -----------------------------------------------------------------

"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Image Text Retrieval evaluation helper
"""
from time import time

import torch
from horovod import torch as hvd
from tqdm import tqdm

from utils.logger import LOGGER
from utils.misc import NoOp
from utils.distributed import all_gather_list
import numpy as np
import pandas as pd
import os

np.seterr(divide='ignore',invalid='ignore')

@torch.no_grad()
def itm_eval(score_matrix, txt_ids, img_ids, txt2img, img2txts):

    sourcePath = './source/'
    Output_dir = './i2t_Results_and_ImageNetTopic_Results'
    if not os.path.exists(Output_dir):
        os.makedirs(Output_dir)

    # image retrieval
    img2j = {i: j for j, i in enumerate(img_ids)}
    _, rank_txt = score_matrix.topk(10, dim=1)

    gt_img_j = torch.LongTensor([img2j[txt2img[txt_id]]
                                 for txt_id in txt_ids],
                                ).to(rank_txt.device
                                     ).unsqueeze(1).expand_as(rank_txt)
    rank = (rank_txt == gt_img_j).nonzero()
    if rank.numel():
        ir_r1 = (rank < 1).sum().item() / len(txt_ids)
        ir_r5 = (rank < 5).sum().item() / len(txt_ids)
        ir_r10 = (rank < 10).sum().item() / len(txt_ids)
    else:
        ir_r1, ir_r5, ir_r10 = 0, 0, 0

    # text retrieval
    txt2i = {t: i for i, t in enumerate(txt_ids)}
    _, rank_img = score_matrix.topk(5000, dim=0)

    tLen = len(txt_ids)
    tr_r = np.zeros(tLen)
    tr_p = np.zeros(tLen)
    tr_f1 = np.zeros(tLen)
    NumTopics = 1000

    if len(img_ids) == 1000:
        mode ='test'
        img_topic_ID = np.loadtxt(sourcePath + 'UNITER_ImageNet_test_img_topic_ID.txt', dtype=int)
        Topic_im_num = np.loadtxt(sourcePath + 'UNITER_ImageNet_Topic_test_im_num.txt', dtype=int)
        imageNames = np.loadtxt(sourcePath + 'UNITER_test_img_name.txt', dtype=str)

    elif len(img_ids) == 1014:
        mode = 'dev'
        img_topic_ID = np.loadtxt(sourcePath + 'UNITER_ImageNet_dev_img_topic_ID.txt', dtype=int)
        Topic_im_num = np.loadtxt(sourcePath + 'UNITER_ImageNet_Topic_dev_im_num.txt', dtype=int)
        imageNames = np.loadtxt(sourcePath + 'UNITER_dev_img_name.txt', dtype=str)

    Topic_tr_r = [np.zeros(tLen) for i in range(NumTopics)]
    Topic_tr_p = [np.zeros(tLen) for i in range(NumTopics)]
    Topic_tr_f1 = [np.zeros(tLen) for i in range(NumTopics)]

    RetrievalResult = []
    RetrievalMark = []
    gtTextID = [np.zeros(5) for i in range(len(img_ids))]
    RetrievalTextID = [np.zeros(5) for i in range(len(img_ids))]

    imageID = []

    # Read Karpathy JSON
    import json
    with open('dataset.json', 'r') as j:
        data = json.load(j)

    gt_ID = []
    for key, value in txt2i.items():
        gt_ID.append(int(key))

    tr_r1_Any1, tr_r5_Any1, tr_r10_Any1, tr_r20_Any1 = 0, 0, 0, 0
    for j, img_id in tqdm(enumerate(img_ids)):

        gt_is = [txt2i[t] for t in img2txts[img_id]]

        ranks = [(rank_img[:, j] == i).nonzero() for i in gt_is]

        rank_Any1 = min([20] + [r.item() for r in ranks if r.numel()])
        if rank_Any1 < 1:
            tr_r1_Any1 += 1
        if rank_Any1 < 5:
            tr_r5_Any1 += 1
        if rank_Any1 < 10:
            tr_r10_Any1 += 1
        if rank_Any1 < 20:
            tr_r20_Any1 += 1


        rank10 = [r.item() for r in ranks if r.numel()]

        top5result = rank_img[:, j][:5].cpu()
        top5result = np.array(top5result)

        for i5 in range(5):
            gtTextID[j][i5] = gt_ID[gt_is[i5]]
            RetrievalTextID[j][i5] = gt_ID[top5result[i5]]

        tt = 0
        for gti in gt_is:
            if top5result[0] == gti:
                tt = 1
        if tt == 1:
            RetrievalMark.append('yes')
        else:
            RetrievalMark.append('no')

        tempCaps = ''
        for t5res in top5result:
            for img in data['images']:
                for sent in img['sentences']:
                    if gt_ID[t5res] == sent['sentid']:
                        tempCaps = tempCaps + sent['raw'] + '\n'

        for img in data['images']:
            if imageNames[j] == img['filename']:
                imageID.append(img['imgid'])

        RetrievalResult.append(tempCaps)

        #  recall precision
        tr_p_temp = np.zeros(tLen)
        tr_f1_temp = np.zeros(tLen)
        RelevantMark = np.zeros(tLen)
        for k in rank10:
            for q in range(tLen):
                if q>=k:
                    RelevantMark[q] = RelevantMark[q] + 1

        # recall
        tr_r_temp = RelevantMark / 5
        tr_r = tr_r_temp + tr_r
        Topic_tr_r[img_topic_ID[j]] = Topic_tr_r[img_topic_ID[j]] + tr_r_temp

        #  precision
        for q in range(tLen):
            tr_p_temp[q] = RelevantMark[q] / (q+1)
        tr_p = tr_p_temp + tr_p
        Topic_tr_p[img_topic_ID[j]] = Topic_tr_p[img_topic_ID[j]] + tr_p_temp

        # F1-Measure
        for q in range(tLen):
            if (tr_r_temp[q] + tr_p_temp[q]) == 0:
                tr_f1_temp[q] = 0
            else:
                tr_f1_temp[q] = 2 * tr_r_temp[q] * tr_p_temp[q] / (tr_r_temp[q] + tr_p_temp[q])
        tr_f1 = tr_f1_temp + tr_f1
        Topic_tr_f1[img_topic_ID[j]] = Topic_tr_f1[img_topic_ID[j]] + tr_f1_temp

    # Average Recall Precision F1-Measure
    tr_r = tr_r / len(img_ids)
    tr_p = tr_p / len(img_ids)
    tr_f1 = tr_f1 / len(img_ids)

    tr_r1_Any1 /= len(img_ids)
    tr_r5_Any1 /= len(img_ids)
    tr_r10_Any1 /= len(img_ids)
    tr_r20_Any1 /= len(img_ids)

    tr_r_Any1 = [tr_r1_Any1, tr_r5_Any1, tr_r10_Any1, tr_r20_Any1]
    AtK_Any1 = [1, 5, 10, 20]




    Topic_r5 = []
    Topic_p1 = []
    for i in range(NumTopics):
        if Topic_im_num[i] == 0:
            Inum = 1
        else:
            Inum = Topic_im_num[i]
        Topic_tr_r[i] = Topic_tr_r[i] / Inum
        Topic_tr_p[i] = Topic_tr_p[i] / Inum
        Topic_tr_f1[i] = Topic_tr_f1[i] / Inum
        Topic_r5.append(Topic_tr_r[i][4])
        Topic_p1.append(Topic_tr_p[i][0])

    tr_r1 = tr_r[0]
    tr_r5 = tr_r[4]
    tr_r10 = tr_r[9]
    tr_r20 = tr_r[19]
    tr_r50 = tr_r[49]
    tr_r100 = tr_r[99]

    ImageNetID = [i for i in range(1000)]
    dataframe = pd.DataFrame({'ImageNet ID': ImageNetID, 'Average Recall@5': Topic_r5,
                              'Average Precision@1': Topic_p1})
    dataframe.to_csv(Output_dir + '/UNITER_' + mode + '_Recall5_Precision1_Per_ImageNetTopic.csv', sep=',')

    dataframe = pd.DataFrame({'Query Image Name': imageNames, 'Query Image ID': imageID, 'Retrieved Results': RetrievalResult, '@1 mark': RetrievalMark})
    dataframe.to_csv(Output_dir + '/UNITER_' + mode + '_RetrievalDetail.csv', sep=',')

    # @K
    AtK = []
    for q in range(tLen):
        AtK.append((q + 1))
    dataframe = pd.DataFrame({'@K': AtK, 'Average Recall RetrievalAll5': tr_r, 'Average Precision RetrievalAll5': tr_p,
                              'Average F1_score RetrievalAll5': tr_f1})
    dataframe.to_csv(Output_dir + '/UNITER_' + mode + '_AveRecallPrecisionF1_Retrieval_All5.csv', sep=',', index=False)

    dataframe = pd.DataFrame({'@K': AtK_Any1, 'Average Recall RetrievalAny1': tr_r_Any1})
    dataframe.to_csv(Output_dir + '/UNITER_' + mode + '_AveRecall_Retrieval_Any1.csv', sep=',', index=False)



    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_log = {'txt_r1': tr_r1,
                'txt_r5': tr_r5,
                'txt_r10': tr_r10,
                'txt_r20': tr_r20,
                'txt_r50': tr_r50,
                'txt_r100': tr_r100,
                'txt_r_mean': tr_mean,
                'img_r1': ir_r1,
                'img_r5': ir_r5,
                'img_r10': ir_r10,
                'img_r_mean': ir_mean,
                'r_mean': r_mean}
    return eval_log


@torch.no_grad()
def evaluate_and_analyseImageNetClass(model, eval_loader):
    st = time()
    LOGGER.info("start running Image/Text Retrieval evaluation ...")
    score_matrix = inference(model, eval_loader)
    dset = eval_loader.dataset
    all_score = hvd.allgather(score_matrix)
    all_txt_ids = [i for ids in all_gather_list(dset.ids)
                   for i in ids]
    all_img_ids = dset.all_img_ids
    assert all_score.size() == (len(all_txt_ids), len(all_img_ids))
    if hvd.rank() != 0:
        return {}

    # NOTE: only use rank0 to compute final scores
    eval_log = itm_eval(all_score, all_txt_ids, all_img_ids,
                        dset.txt2img, dset.img2txts)

    tot_time = time()-st
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds")
    return eval_log


@torch.no_grad()
def inference(model, eval_loader):
    model.eval()
    if hvd.rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()

    score_matrix = torch.zeros(len(eval_loader.dataset),
                               len(eval_loader.dataset.all_img_ids),
                               device=torch.device("cuda"),
                               dtype=torch.float16)

    for i, mini_batches in enumerate(eval_loader):


        j = 0
        for batch in mini_batches:
            scores = model(batch, compute_loss=False)
            bs = scores.size(0)
            score_matrix.data[i, j:j+bs] = scores.data.squeeze(1).half()
            j += bs
        assert j == score_matrix.size(1)
        pbar.update(1)

    model.train()
    pbar.close()
    return score_matrix
