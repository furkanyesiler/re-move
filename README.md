# Less is more: Faster and better music version identification with embedding distillation

This repository contains the code and hyperparameters used for the experiments described in the following paper:

> F. Yesiler, J. Serrà, and E. Gómez. Less is more: Faster and better music version identification with embedding distillation. In Proc. of the Int. Soc. for Music Information Retrieval Conf. (ISMIR), 2020

## Da-TACOS training subset
The instructions on how to download the newly-contributed Da-TACOS training set will be shared soon. In the meantime, you can check our publication to find out more about this dataset.

## Using the code
Below, we specify some use cases and explain the important steps you need to follow for using the code in this repository.

* Evaluating the pre-trained models on Da-TACOS benchmark set
* Evaluating the pre-trained models on a private dataset
* Training a model with the Da-TACOS training set (soon)
* Training a model with a private dataset (soon)

### 0 - Overview of the code
Both the training and the evaluation code in this repository are called through [main.py](https://github.com/furkanyesiler/re-move/blob/master/main.py). The arguments for that script can be seen as the following:
```bash
python main.py
```
```
usage: main.py [-h] [-rt {train,test}] [-mp MAIN_PATH]
               [--exp_type {lsr,kd,pruning,baseline}]
               [-pri PRUNING_ITERATIONS] [-tf TRAIN_FILE] [-ch CHUNKS]
               [-vf VAL_FILE] [-noe NUM_OF_EPOCHS] [-emb EMB_SIZE]
               [-bs BATCH_SIZE] [-l {0,1,2,3}] [-kdl {distance,cluster}]
               [-ms {0,1,2,3}] [-ma MARGIN] [-o {0,1}] [-lr LEARNING_RATE]
               [-flr FINETUNE_LEARNING_RATE] [-mo MOMENTUM]
               [-lrs LR_SCHEDULE [LR_SCHEDULE ...]] [-lrg LR_GAMMA]
               [-pl PATCH_LEN] [-da {0,1}] [-nw NUM_WORKERS]
               [-ofb OVERFIT_BATCH]

Training and evaluation code for Re-MOVE experiments.

optional arguments:
  -h, --help            show this help message and exit
  -rt {train,test}, --run_type {train,test}
                        Whether to run the training or the evaluation script.
  -mp MAIN_PATH, --main_path MAIN_PATH
                        Path to the working directory.
  --exp_type {lsr,kd,pruning,baseline}
                        Type of experiment to run.
  -pri PRUNING_ITERATIONS, --pruning_iterations PRUNING_ITERATIONS
                        Number of iterations for pruning.
  -tf TRAIN_FILE, --train_file TRAIN_FILE
                        Path for training file. If more than one file are
                        used, write only the common part.
  -ch CHUNKS, --chunks CHUNKS
                        Number of chunks for training set.
  -vf VAL_FILE, --val_file VAL_FILE
                        Path for validation data.
  -noe NUM_OF_EPOCHS, --num_of_epochs NUM_OF_EPOCHS
                        Number of epochs for training.
  -emb EMB_SIZE, --emb_size EMB_SIZE
                        Size of the final embeddings.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training iterations.
  -l {0,1,2,3}, --loss {0,1,2,3}
                        Which loss to use for training. 0 for Triplet, 1 for
                        ProxyNCA, 2 for NormalizedSoftmax, and 3 for Group
                        loss.
  -kdl {distance,cluster}, --kd_loss {distance,cluster}
                        Which loss to use for Knowledge Distillation training.
  -ms {0,1,2,3}, --mining_strategy {0,1,2,3}
                        Mining strategy for Triplet loss. 0 for random, 1 for
                        semi-hard, 2 for hard, 3 for semi-hard with using all
                        positives.
  -ma MARGIN, --margin MARGIN
                        Margin for Triplet loss.
  -o {0,1}, --optimizer {0,1}
                        Optimizer for training. 0 for SGD, 1 for Ranger.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Base learning rate for the optimizer.
  -flr FINETUNE_LEARNING_RATE, --finetune_learning_rate FINETUNE_LEARNING_RATE
                        Learning rate for finetuning the feature extractor for
                        LSR training.
  -mo MOMENTUM, --momentum MOMENTUM
                        Value for momentum parameter for SGD.
  -lrs LR_SCHEDULE [LR_SCHEDULE ...], --lr_schedule LR_SCHEDULE [LR_SCHEDULE ...]
                        Epochs for reducing the learning rate. Multiple
                        arguments are appended in a list.
  -lrg LR_GAMMA, --lr_gamma LR_GAMMA
                        Step size for learning rate scheduler.
  -pl PATCH_LEN, --patch_len PATCH_LEN
                        Number of frames for each input in time dimension
                        (only for training).
  -da {0,1}, --data_aug {0,1}
                        Whether to use data augmentation while training.
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        Num of workers for the data loader.
  -ofb OVERFIT_BATCH, --overfit_batch OVERFIT_BATCH
                        Whether to overfit a single batch. It may help with
                        revealing problems with training.
```

There are 4 types of experiments you can create/evaluate: (1) latent space reconfiguration (lsr), (2) knowledge distillation, (3) pruning, and (4) baseline. Which experiment setting to use can be changed with the `--exp_type` argument. For each experiment type, one set of default values are created and stored in the `data` folder.

To re-create/evaluate the experiments in the paper, you can simply play with `--exp_type`, `--emb_size`, `--kd_loss`, `--loss` arguments.

The exact configurations used for each experiment presented in the paper will be shared soon.

### 1 - Evaluating the pre-trained models on Da-TACOS benchmark subset
To facilitate the benchmarking process and to present a pipeline for evaluating the pre-trained Re-MOVE models, we have prepared [benchmark_da-tacos.py](https://github.com/furkanyesiler/move/blob/master/benchmark_da-tacos.py). To use the script, you can follow the steps below:

#### 1.1 - Requirements
* Python 3.6+
* Create a virtual enviroment and install requirements
```bash
git clone https://github.com/furkanyesiler/re-move.git
cd re-move
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_benchmark.txt
```

#### 1.2 - Running benchmark_da-tacos.py
After creating the virtual environment and installing the required packages, you can simply run
```bash
python benchmark_da-tacos.py --unpack --remove
```
```
usage: benchmark_da-tacos.py [-h] [--outputdir OUTPUTDIR] [--unpack]
                             [--remove]

Downloading and preprocessing cremaPCP features of the Da-TACOS benchmark
subset

optional arguments:
  -h, --help            show this help message and exit
  --outputdir OUTPUTDIR
                        Directory to store the dataset (default: ./data)
  --unpack              Unpack the zip files (default: False)
  --remove              Remove zip files after unpacking (default: False)
```

This script downloads the metadata and the cremaPCP features of the Da-TACOS benchmark set, and preprocesses them to work with our evaluation setting. Specifically, after downloading the files:
* it downsamples the cremaPCP features by 8, 
* reshapes them from Tx12 to 1x23xT (for the intuition behind this step, you can check our paper), 
* stores them in a dictionary which is saved as a `.pt` file,
* creates ground truth annotations to be used by our evaluation function.

Both the data and the ground truth annotations (named `benchmark_crema.pt` and `ytrue_benchmark.pt`, respectively) are stored in the `data` folder.

#### 1.3 - Running the evaluation script
After the features are downloaded and preprocessed, you can use the script below to evaluate the pre-trained Re-MOVE model on the Da-TACOS benchmark subset:

```bash
python main.py -rt test 
```
or
```bash
python main.py -rt test
```


### 2 - Evaluating the pre-trained models on a private dataset
For this use case, we would like to point out a number of requirements you must follow.

#### 2.1 - Requirements
* Python 3.6+
* Create a virtual enviroment and install requirements
```bash
git clone https://github.com/furkanyesiler/re-move.git
cd re-move
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2.2 - Setting up the dataset and annotations
Re-MOVE was trained using cremaPCP features, and therefore, it may underperform with other Pitch Class Profile (PCP, or chroma) variants. To extract cremaPCP features for your audio collection, please refer to [acoss](https://github.com/furkanyesiler/acoss).

cremaPCP features used for training are created using non-overlapping frames and a hop size of 4096 on audio tracks sampled at 44.1 kHz (about 93 ms per frame).

After obtaining cremaPCP features for your dataset, you should cast them as a torch.Tensor, and reshape them from Tx12 to 1x23xT. For this step, you can use the following code snippet:

```python
import numpy as np
import torch

# this variable represents your cremaPCP feature
cremaPCP = np.random.rand(100, 12)

# this is casting the crema feature to a torch.Tensor type
cremaPCP_tensor = torch.from_numpy(cremaPCP).t()

# this is the resulting cremaPCP feature
cremaPCP_reshaped = torch.cat((cremaPCP_tensor, cremaPCP_tensor))[:23].unsqueeze(0)

```

##### 2.2.1 - Dataset file

When the cremaPCP features per song are ready, you need to create a dataset file and a ground truth annotations file. The dataset file should be a python dictionary with 2 keys: `data` and `labels`. Each key (i.e. 'data' or 'labels') should point to a python list which contains the respective cremaPCP features and label of that track. Specifically, let `dataset_dict` be our dataset dictionary, and `dataset_dict['data']` and `dataset_dict['labels']` be our lists. The label of the song `dataset_dict['data'][42]` should be `dataset_dict['labels'][42]`. Finally, the dataset file should be saved under `data` folder, and should be named `benchmark_crema.pt`. An example code is shown below:

```python
import os

root_dir = '/your/root/directory/of/move'

data = []
labels = []
for i in range(dataset_size):
	cremaPCP = load_cremaPCP(i)  # loading the cremaPCP features for the ith song of your dataset
	label = load_label(i)  # loading the label of the ith song of your dataset

	data.append(cremaPCP)
	labels.append(label)

dataset_dict = {'data': data, 'labels': label}

torch.save(dataset_dict, os.path.join(root_dir, 'data', 'benchmark_crema.pt'))

```

##### 2.2.2 - Annotations file

When your dataset file ('benchmark_crema.pt') is ready, you have to create a ground truth annotations file which is stored in `data` folder, and should be named `ytrue_benchmark.pt`. This file should be a torch.Tensor with the shape NxN (N is the size of your dataset). Finally, the diagonal of this matrix should be 0. You can find an example code below:

```python
import os

import torch

data_dir = '/your/root/directory/of/move/data/'

labels = torch.load(os.path.join(data_dir, 'benchmark_crema.pt'))['labels']

ytrue = []

for i in range(len(labels)):
	main_label = labels[i]  # label of the ith song
	sub_ytrue = []
	for j in range(len(labels)):
		if labels[j] == main_label and i!= j:  # checking whether the ith and jth song has the same label
			sub_ytrue.append(1)
		else:
			sub_ytrue.append(0)
	ytrue.append(sub_ytrue)

ytrue = torch.Tensor(ytrue)
torch.save(ytrue, os.path.join(data_dir, 'ytrue_benchmark.pt'))
```

#### 2.3 - Running the evaluation script
After you prepared your dataset and annotation files, you can use the script below to evaluate the pre-trained MOVE model on your dataset:

```bash
python main.py -rt test
```
or
```bash
python main.py -rt test
```

### 3 - Training a model with the Da-TACOS training subset
Coming soon...

### 4 - Training a model with a private training set
Coming soon...

## Questions
For any questions you may have, feel free to create an issue or contact [me](mailto:furkan.yesiler@upf.edu).

## License
The code in this repository is licensed under [Affero GPL v3](https://www.gnu.org/licenses/agpl-3.0.en.html).

## References
Please cite our reference if you plan to use the code in this repository:
```
@inproceedings{yesiler2020,
    author = "Furkan Yesiler and Joan Serrà and Emilia Gómez",
    title = "Less is more: {Faster} and better music version identification with embedding distillation",
    booktitle = "Proc. of the Int. Soc. for Music Information Retrieval Conf. (ISMIR)",
    year = "2020"
}
```

## Acknowledgments

This work has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068 (MIP-Frontiers).

This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 770376 (TROMPA).

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" height="64" hspace="20">
