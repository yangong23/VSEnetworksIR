# VSEnetworksIR
Code will be uploaded very soon (by the 30th July 2021).

## VSE++, SCAN, and VSRN:

### Requirements and Installation
We recommended the following dependencies.

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

## UNITER:
### Requirements and Installation

The requirements and installation can be followed by the official introduction of [UNITER](https://github.com/ChenRocks/UNITER). Please install the following:

* [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+)

* [Docker](https://docs.docker.com/engine/install/ubuntu/) (19.03+)

* [vidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)

The scripts require the user to have the [docker group membership](https://docs.docker.com/engine/install/linux-postinstall/) so that docker commands can be run without sudo. Only Linux with NVIDIA GPUs is supported. Some code in this repo are copied/modified from opensource implementations made available by [PyTorch](https://github.com/pytorch/pytorch), [HuggingFace](https://github.com/huggingface/transformers), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), and [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch). The image features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention).

### Download Data and Trained Models
* Download data and trained models:
``` 
bash scripts/download_itm.sh $PATH_TO_STORAGE
``` 

* Download [the JSON file of Flickr30K produced by Andrej Karpathy](https://cs.stanford.edu/people/karpathy/deepimagesent/). Extract "dataset.json" and copy it into the root of UNITER project.

### Evaluation and Limitation Analysis
* Open "config/evaluation-itm-flickr-base-8gpu.json" and set DATA PATH and MODEL PATH.
* Evaluating UNITER for retrieving any 1 of 5 descriptions, evaluating UNITER for retrieving all 5 descriptions, and analysing limitations of UNITER can be finished by runing one python file:
``` 
python test_itm_evaluation_and_analyseImageNetTopics.py 
``` 
* Find results in the folder of "i2t_Results_and_ImageNetTopic_Results".




