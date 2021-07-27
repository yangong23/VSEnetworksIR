# VSEnetworksIR
## Introduction
This is the PyTorch code for analysing the limitations of VSE networks for image-to-text retrieval, which is described in the paper ["On the Limitations of Visual-Semantic Embedding Networks for Image-to-Text Information Retrieval"](https://www.mdpi.com/2313-433X/7/8/125). The code is modified based on [VSE++](https://github.com/fartashf/vsepp), [SCAN](https://github.com/kuanghuei/SCAN), [VSRN](https://github.com/KunpengLi1994/VSRN), and [UNITER](https://github.com/ChenRocks/UNITER). Please note: VSEpp (named VSE++), SCAN, VSRN, and UNITER in this repository should each be run as independent projects.

The flow of this experiment is as follows:
* Prepare datasets and network models. Dowanload the dataset (VSE++, SCAN, and VSRN use the same data files, UNITER needs different data files). Prepare the model (train new models for VSE++, SCAN, and VSRN, download the pre-trained model for UNITER).
* Evaluate VSE networks (VSE++, SCAN, VSRN, and UNITER). (1) Evaluate the network for retrieving any 1 of 5 descriptions to image queries; (2) evaluate the network for retrieving all 5 descriptions to image queries.
* Analyse limitations of VSE networks (only for VSRN and UNITER). (1) To compute average Recall@5 and Precision@1 for each ImageNet class separately; (2) To get retrieval details for all image queries.

## Run VSE++, SCAN, and VSRN Projects

### Requirements and installation
We recommended the following dependencies.
* ubuntu (>=18.04)

* Python 3.8

* [PyTorch](https://pytorch.org/) (1.7.1)

* [NumPy](https://numpy.org/) (>1.12.1)

* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger) 

* [pycocotools](https://github.com/cocodataset/cocoapi) 

* Punkt Sentence Tokenizer:

``` python
import nltk
nltk.download()
> d punkt
``` 
We used [anaconda](https://www.anaconda.com/) to manage the dependencies, and the code was runing on a NVIDIA RTX 3080 GPU.

### Download data
#### 1 Datasets:
* For VSE++, SCAN, and VSRN:
Image features can be downloaded from [SCAN](https://github.com/kuanghuei/SCAN) by using:
``` 
wget https://iudata.blob.core.windows.net/scan/data.zip
``` 
Extract data from data.zip, and only f30k_precomp is needed by this paper. 

#### 2 Vocabularies:
* Only for SCAN:
Vocabularies can be downloaded from [SCAN](https://github.com/kuanghuei/SCAN) by using:
``` 
wget https://iudata.blob.core.windows.net/scan/vocab.zip
```
* VSE++ and VSRN's Vocabularies(vocab) has been uploaded in the folder of VSEpp and VSRN of this repository.

### Training new models
* For VSE++:
``` 
python train.py --data_path "$DATA_PATH" --data_name f30K_precomp --logger_name runs/flickr_vse++ --max_violation
``` 
* For SCAN:
``` 
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/flickr_scan/log --model_name runs/flickr_scan/log --max_violation --bi_gru --agg_func=LogSumExp --cross_attn=i2t --lambda_lse=5 --lambda_softmax=4
``` 
* For VSRN:
``` 
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --max_violation --lr_update 10  --max_len 60
``` 

### Evaluation
#### 1 Evaluating the network for retrieving any 1 of 5 descriptions to the image query
* For VSE++: 
Modify the model_path and data_path in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py
``` 
* For SCAN: 
Modify the model_path and data_path in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py
``` 
* For VSRN: 
Modify the model_path and data_path in the evaluation.py file. Then Run evaluation.py:
``` 
python evaluation.py
``` 
Results: Get the results on the screen printing. This repeats the work of VSE++, SCAN, and VSRN.

#### 2 Evaluating the network for retrieving all 5 descriptions to the image query
* For VSE++: 
Modify the model_path and data_path in the evaluation_retrieval_all5_and_any1.py file. Then Run evaluation_retrieval_all5_and_any1.py:
``` 
python evaluation_retrieval_all5_and_any1.py
``` 
* For SCAN: 
Modify the model_path and data_path in the evaluation_retrieval_all5_and_any1.py file. Then Run evaluation_retrieval_all5_and_any1.py:
``` 
python evaluation_retrieval_all5_and_any1.py
``` 
* For VSRN: 
Modify the model_path and data_path in the evaluation_retrieval_all5_and_any1.py file. Then Run evaluation_retrieval_all5_and_any1.py:
``` 
python evaluation_retrieval_all5_and_any1.py
``` 
Results: Get the results in the folder of "i2t_Results". (1) "AveRecallPrecisionF1.csv" saves average Recall, average Precision, and average F1-score for all ranks.

### Limitation analysis
* Only for VSRN: 
Modify the model_path and data_path in the evaluation_ImageNetClass.py file. Then Run evaluation_ImageNetClass.py:
``` 
python evaluation_ImageNetClass.py
``` 
Results: Get the results in the folder of "ImageNetTopic_Results". (1) "VSRN_mode_Recall5_Precision1_Per_ImageNetClass.csv" saves Recall@5 and Precision@1 for every ImageNet class; (2) "VSRN_mode_RetrievalDetail.csv" saves the top 5 ranked retrieved descriptions to the image query for all data in the set.


## Run UNITER Project
### Requirements and installation

The requirements and installation can be found in the instructions provided by [UNITER](https://github.com/ChenRocks/UNITER). These require installation of the following:

* [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+)

* [Docker](https://docs.docker.com/engine/install/ubuntu/) (19.03+)

* [vidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)

The scripts require the user to have the [docker group membership](https://docs.docker.com/engine/install/linux-postinstall/) so that docker commands can be run without sudo. Only Linux with NVIDIA GPUs is supported. Some code in this repo are copied/modified from opensource implementations made available by [PyTorch](https://github.com/pytorch/pytorch), [HuggingFace](https://github.com/huggingface/transformers), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), and [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch). The image features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention).

We used Python 3.8 and Pytorch 1.7.1 in the [anaconda](https://www.anaconda.com/) environment, and the code was runing on a NVIDIA RTX 3080 GPU with the system of ubuntu 18.04.

### Download data and pre-trained models
* 1 Download image features and trained models:
``` 
bash scripts/download_itm.sh $PATH_TO_STORAGE
```
Only Flirk30K is needed by this paper. The model is downloaded, we do not need to train a new model.

* 2 Download [the JSON file of Flickr30K](https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip) produced by [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/deepimagesent/). Extract "dataset.json" file from flickr30k.zip and copy "dataset.json" file into the root of UNITER project.

### Evaluation and limitation analysis
* Evaluating UNITER for retrieving any 1 of 5 descriptions, evaluating UNITER for retrieving all 5 descriptions, and analysing limitations of UNITER can be finished by runing one python file.
Modify the model_path and data_path in config/evaluation-itm-flickr-base-8gpu.json file. Then Run test_itm_evaluation_and_analyseImageNetClass.py:
``` 
python test_itm_evaluation_and_analyseImageNetClass.py 
``` 
Results: Get results in the folder of "i2t_Results_and_ImageNetTopic_Results".  (1) "AveRecall_Retrieval_Any1.csv" saves average Recall @1 @5 @10 @20 for retrieving any 1 of 5 descriptions; (2) "AveRecallPrecisionF1_Retrieval_All5.csv" saves average Recall, average Precision, and average F1-score with all ranks for retrieving all 5 descriptions; (3) "UNITER_mode_Recall5_Precision1_Per_ImageNetClass.csv" saves Recall@5 and Precision@1 for every ImageNet class; (4) "UNITER_mode_RetrievalDetail.csv" saves the top 5 ranked retrieved descriptions to the image query for all data in the set.

## Reference
If you found this code useful, please cite the following paper:

*Gong Y, Cosma G, Fang H. On the Limitations of Visual-Semantic Embedding Networks for Image-to-Text Information Retrieval[J]. Journal of Imaging, 2021, 7(8): 125.*

BibTeX:

    @inproceedings{gong2021limivse,
      title={On the Limitations of Visual-Semantic Embedding Networks for Image-to-Text Information Retrieval},
      author={Gong, Yan and Cosma, Georgina and Fang, Hui},
      journal={Journal of Imaging},
      volume={7},
      number={8},
      pages={125},
      year={2021},
      publisher={Multidisciplinary Digital Publishing Institute}
    }
      


