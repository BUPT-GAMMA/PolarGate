# PolarGate: Breaking the Functionality Representation Bottleneck of And-Inverter Graph Neural Network

Official code repository for the paper:
PolarGate: Breaking the Functionality Representation Bottleneck of And-Inverter Graph Neural Network

PolarGate can serve as an encoder to embed standardized logic circuit (i.e., and-inverter graph) into gate-level embedding vectors. Such general embeddings with rich functional information can be applied to various downstream tasks, including testability analysis, SAT problem solving and logic synthesis. If you plan to explore more potential downstream tasks, please feel free to discuss with us (Email: liu_jiawei@bupt.edu.cn). We are looking forward to collaborate with you!


## Abstract
Understanding the functionality of Boolean networks is crucial for processes such as functional equivalence checking, logic synthesis and malicious logic identification. 
With the proliferation of deep learning in electronic design automation (EDA), graph neural networks (GNNs) are widely used for embedding the and-inverter graphs (AIGs), a standard form of Boolean networks, into vectorized representation. 
A key challenge in the use of GNN for Boolean representation is that although GNNs can well encapsulate the structural properties of AIGs, they usually fail to fully capture the functionality of Boolean logic. 
Moreover, most GNNs designed for AIGs (also called AIGNNs) either rely on a large amount of training data or require complex supervisory tasks, making it difficult to maintain high training efficiency and prediction accuracy. 
In this work, for the first time, we focus on breaking the bottleneck of AIGNNs by augmenting their capability of functional representation, providing an efficient solution called PolarGate, which naturally aligns the message passing process with the logical functionality of AIGs. 
Specifically, we map the behavior of the logic gate into an ambipolar state space, customize differentiable logical operators, and design a functionality-aware message passing strategy. 
Experimental results on two logically related tasks (i.e., signal probability prediction and truth-table distance prediction) show that PolarGate outperforms the state-of-the-art GNN-based methods for Boolean representation, **with an improvement of 62.1% (40.6%) in learning capability and 79.5% (85.6%) in efficiency on two tasks.**

## Installation
```shell
conda create -n polargate python=3.8.10
conda activate polargate
pip install -r requirements.txt
```

## Directory Structure

```
AIGDataset
  ©À©¤©¤ PolarGate_raw
  ©À©¤©¤ PolarGate_processed
       ©À©¤©¤ npz
           ©À©¤©¤ pi_edges.npz
           ©À©¤©¤ labels.npz
       ©À©¤©¤ [circuit_name]
           ©À©¤©¤ raw
               ©À©¤©¤ node-feat.csv
               ©À©¤©¤ prob.csv
               ©À©¤©¤ signed_edge.csv
       ©À©¤©¤ split
           ©À©¤©¤ 0.01-0.01-0.98
           ©À©¤©¤ 0.02-0.02-0.96
           ©À©¤©¤ 0.05-0.05-0.9
           ©À©¤©¤ 0.1-0.1-0.8
PolarGate
  ©À©¤©¤ layers.py
  ©À©¤©¤ model.py
  ©À©¤©¤ load_data.py
  ©À©¤©¤ preprocess_data.py
  ©À©¤©¤ train.py
  ©À©¤©¤ train.sh
```

## Prepare Dataset
```bash
mkdir AIGDataset
cd AIGDataset
```
If you want a preprocessed dataset (ready-to-use), you can download it from here:
```bash
wget https://github.com/BUPT-GAMMA/PolarGate/releases/download/dataset/PolarGate_processed.zip
unzip PolarGate_processed.zip 
```
If you want the raw dataset (in .bench format), you can download it from here. Then, you can use `preprocess_data.py` to generate data input supported by PolarGate.
```bash
wget https://github.com/BUPT-GAMMA/PolarGate/releases/download/dataset/PolarGate_raw.zip
unzip PolarGate_raw.zip 
```

## Model training
To train the model for signal probability prediction, use the following command:
```bash
python train.py --task 'prob' --model 'PolarGate' --device 0 --batch_size 256 --eval_step 1 \
       --split_file 0.05-0.05-0.9 --layer_num 9 --feature_type 'one-hot' --in_dim 3
```

**Explanation:**
- `--task 'prob'`: Specifies the task as signal probability prediction.
- `--model 'PolarGate'`: Specifies the model name as PolarGate.
- `--device 0`: Indicates the GPU device number to use.
- `--batch_size 256`: Sets the batch size for training.
- `--eval_step 1`: Specifies the evaluation step interval.
- `--split_file 0.05-0.05-0.9`: Indicates the data split proportions for training, validation, and testing.
- `--layer_num 9`: Sets the number of layers in the model.
- `--feature_type 'one-hot'`: Specifies the feature type as one-hot encoding.
- `--in_dim 3`: Sets the input dimension size.

Similarly, to train the model for truth-table distance prediction, use the following command:

```bash
python train.py --task 'tt' --model 'PolarGate' --device 0 --batch_size 256 --eval_step 1 \
       --split_file 0.05-0.05-0.9 --layer_num 9 --feature_type 'one-hot' --in_dim 3
```

## Cite PolarGate
If PolarGate could help your project, please cite our work:
```bash
@inproceedings{PolarGate,
  author={Liu, Jiawei and Zhai, Jianwang and Zhao, Mingyu and Lin, Zhe and Yu, Bei and Shi, Chuan},
  booktitle={2024 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)}, 
  title={PolarGate: Breaking the Functionality Representation Bottleneck of And-Inverter Graph Neural Network}, 
  year={2024}}
```