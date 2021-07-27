# VSEnetworksIR
Code will be uploaded very soon (by the 30th July 2021). 

Notice: VSEpp(named VSE++), SCAN, VSRN, and UNITER in this repository should be the independent project for running.

## VSE++, SCAN, and VSRN:

### Requirements and Installation
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
We used anaconda3 to manage the dependencies, and the code was runing on a NVIDIA RTX 3080 GPU.

### Download Data
* For VSE++, SCAN, and VSRN:
All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from SCAN by using:
``` 
wget https://iudata.blob.core.windows.net/scan/data.zip
wget https://iudata.blob.core.windows.net/scan/vocab.zip
``` 

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
Results: Get the results in the folder of "i2t_Results". (1) "AveRecallPrecisionF1.csv" saves average Recall, average Precision, and average F1-Measure with all ranks.

### Limitation Analysis
* Only for VSRN: 
Modify the model_path and data_path in the evaluation_ImageNetClass.py file. Then Run evaluation_ImageNetClass.py:
``` 
python evaluation_ImageNetClass.py
``` 
Results: Get the results in the folder of "ImageNetTopic_Results". (1) "VSRN_mode_Recall5_Precision1_Per_ImageNetClass.csv" saves Recall@5 and Precision@1 for every ImageNet class; (2) "VSRN_mode_RetrievalDetail.csv" saves the top 5 ranked retrieved descriptions to the image query for all data in the set.


## UNITER:
### Requirements and Installation

The requirements and installation can be followed by the official introduction of [UNITER](https://github.com/ChenRocks/UNITER). Official introduction requires to install the following:

* [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+)

* [Docker](https://docs.docker.com/engine/install/ubuntu/) (19.03+)

* [vidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)

The scripts require the user to have the [docker group membership](https://docs.docker.com/engine/install/linux-postinstall/) so that docker commands can be run without sudo. Only Linux with NVIDIA GPUs is supported. Some code in this repo are copied/modified from opensource implementations made available by [PyTorch](https://github.com/pytorch/pytorch), [HuggingFace](https://github.com/huggingface/transformers), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), and [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch). The image features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention).

We used Python 3.8 and Pytorch 1.7.1 with anaconda3 environment, and the code was runing on a NVIDIA RTX 3080 GPU with the system of ubuntu 18.04.

### Download Data and Trained Models
* Download data and trained models:
``` 
bash scripts/download_itm.sh $PATH_TO_STORAGE
``` 

* Download [the JSON file of Flickr30K produced by Andrej Karpathy](https://cs.stanford.edu/people/karpathy/deepimagesent/). Extract zip file to get "dataset.json" and copy "dataset.json" into the root of UNITER project.

### Evaluation and Limitation Analysis
* Evaluating UNITER for retrieving any 1 of 5 descriptions, evaluating UNITER for retrieving all 5 descriptions, and analysing limitations of UNITER can be finished by runing one python file.
Modify the model_path and data_path in config/evaluation-itm-flickr-base-8gpu.json file. Then Run test_itm_evaluation_and_analyseImageNetClass.py:
``` 
python test_itm_evaluation_and_analyseImageNetClass.py 
``` 
Results: Get results in the folder of "i2t_Results_and_ImageNetTopic_Results".  (1) "AveRecall_Retrieval_Any1.csv" saves average Recall @1 @5 @10 @20 for retrieving any 1 of 5 descriptions; (2) "AveRecallPrecisionF1_Retrieval_All5.csv" saves average Recall, average Precision, and average F1-Measure with all ranks for retrieving all 5 descriptions; (3) "UNITER_mode_Recall5_Precision1_Per_ImageNetClass.csv" saves Recall@5 and Precision@1 for every ImageNet class; (4) "UNITER_mode_RetrievalDetail.csv" saves the top 5 ranked retrieved descriptions to the image query for all data in the set.




